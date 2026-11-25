"""Phase 4 C3: durable SQLite Quota Engine — atomic, persistent quota reservations
shared across runs and processes.

Failing-first: written against the C3 spec before quota_engine.py / schema v6
existed. Covers: v5->v6 migration with data preservation, persistence across
connections, cross-connection/thread concurrency exactness, atomic all-or-nothing
rollback, quota exhaustion, idempotent vs conflicting duplicate dispatch IDs,
subprocess crash recovery (reservation retained, never auto-refunded), stale-worker
lease fencing, period boundaries with retained historical windows, cross-account
pool isolation, shared account pool across models, strict integer accounting,
fail-closed unknown/future reset anchors, and config changes never resetting
consumed quotas."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import pytest

from geosgan_hive.checkpoints import (
    SCHEMA_VERSION,
    CheckpointStore,
    PersistMode,
    quota_window_for,
)
from geosgan_hive.cloud_profiles import (
    CloudProfile,
    CloudProfileRegistry,
    TokenEstimate,
)
from geosgan_hive.errors import (
    CheckpointError,
    CloudConfigError,
    CostUnknownError,
    QuotaExhaustedError,
)
from geosgan_hive.quota_engine import QuotaEngine

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
ANCHOR = datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC)  # NOW - 1 day
FP_A = "a" * 64
FP_B = "b" * 64
POOL_A = "gemini:acct:" + FP_A
POOL_B = "gemini:acct:" + FP_B
MODEL = "gemini/gemini-2.0-flash"


def free_profile(**overrides) -> dict:
    base = dict(
        provider="gemini",
        models=[MODEL],
        billing="free_tier_verified",
        verified_free=True,
        overage_billing_disabled=True,
        billing_verified=True,
        verified_at=(NOW - timedelta(days=1)).isoformat(),
        account_label="main",
        account_fingerprint=FP_A,
        quota_window_start=ANCHOR.isoformat(),
        verification_evidence={
            "source": "provider_billing_console_export",
            "evidence_ref": "ev-2026-01-14",
            "collected_at": (NOW - timedelta(days=1)).isoformat(),
            "collected_by": "operator-1",
            "credential_bound": True,
            "account_fingerprint": FP_A,
        },
        quotas=[{"pool": POOL_A, "max_requests": 1500, "max_tokens": 1_000_000,
                 "period": "day"}],
    )
    base.update(overrides)
    return base


def registry(profiles, *, zero_cost=True, trusted=False, **kw) -> CloudProfileRegistry:
    kw.setdefault("max_spend_usd", 0.0 if zero_cost else None)
    kw.setdefault("trusted_evidence_binding", trusted)
    return CloudProfileRegistry(
        profiles=[CloudProfile.model_validate(p) for p in profiles],
        zero_cost=zero_cost,
        **kw,
    )


def make_engine(tmp_path, profiles, *, zero_cost=True, now=NOW, name="q.db", **kw):
    store = CheckpointStore(str(tmp_path / name), utcnow=lambda: now)
    reg = registry(profiles, zero_cost=zero_cost, **kw)
    eng = QuotaEngine(store, reg, utcnow=lambda: now)
    return eng, store, reg


def day_items(amount=1):
    return [{"kind": "requests", "period": "day", "amount": amount}]


# ---- window computation (never invents reset times; fails closed) ----

def test_window_day_boundaries():
    a = "2026-01-01T00:00:00+00:00"
    assert quota_window_for(a, "day", "2026-01-01T00:00:00+00:00") == \
        "2026-01-01T00:00:00.000+00:00"
    assert quota_window_for(a, "day", "2026-01-03T12:00:00+00:00") == \
        "2026-01-03T00:00:00.000+00:00"
    # exactly on the boundary -> new window
    assert quota_window_for(a, "day", "2026-01-02T00:00:00.000+00:00") == \
        "2026-01-02T00:00:00.000+00:00"


def test_window_week_boundaries():
    a = "2026-01-01T00:00:00+00:00"
    assert quota_window_for(a, "week", "2026-01-14T23:59:59+00:00") == \
        "2026-01-08T00:00:00.000+00:00"
    assert quota_window_for(a, "week", "2026-01-15T00:00:00+00:00") == \
        "2026-01-15T00:00:00.000+00:00"


def test_window_month_calendar_arithmetic_and_clamping():
    assert quota_window_for("2026-01-01T00:00:00+00:00", "month",
                            "2026-03-15T00:00:00+00:00") == "2026-03-01T00:00:00.000+00:00"
    # Jan 31 anchor: Feb has no 31st -> clamped to Feb 28, window containing Mar 1 is Feb 28
    assert quota_window_for("2026-01-31T00:00:00+00:00", "month",
                            "2026-03-01T00:00:00+00:00") == "2026-02-28T00:00:00.000+00:00"


def test_window_fail_closed_on_future_anchor_naive_and_unknown_period():
    with pytest.raises(CheckpointError):
        quota_window_for("2026-02-01T00:00:00+00:00", "day", "2026-01-15T12:00:00+00:00")
    with pytest.raises(CheckpointError):
        quota_window_for("2026-01-01T00:00:00", "day", "2026-01-15T12:00:00+00:00")  # naive
    with pytest.raises(CheckpointError):
        quota_window_for("2026-01-01T00:00:00+00:00", "hour", "2026-01-15T12:00:00+00:00")
    with pytest.raises(CheckpointError):
        quota_window_for("not-a-date", "day", "2026-01-15T12:00:00+00:00")


# ---- schema v6 migration ----

def _make_v5_db(path):
    from geosgan_hive.checkpoints import _LEDGER_SCHEMA, _SCHEMA

    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.executescript(_LEDGER_SCHEMA)
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '5')")
    conn.execute(
        "INSERT INTO runs (run_id, state, persist_mode, task, agent, model, "
        "lease_owner, lease_expires_at, created_at, updated_at, physical_reserved) "
        "VALUES ('r5', 'running', 'metadata_only', 'task-text', 'sup', 'ollama/m', "
        "'w', '2999-01-01T00:00:00.000+00:00', '2026-01-01T00:00:00.000+00:00', "
        "'2026-01-01T00:00:00.000+00:00', 3)"
    )
    conn.execute(
        "INSERT INTO tool_executions (execution_id, run_id, agent, tool, args_sha256, "
        "args_json, status, requested_at) VALUES ('e5', 'r5', 'a', 'fs_read', 'h', '{}', "
        "'completed', '2026-01-01T00:00:00.000+00:00')"
    )
    conn.commit()
    conn.close()


def test_schema_version_is_6_and_fresh_store_has_quota_tables(tmp_path):
    assert SCHEMA_VERSION == 6
    store = CheckpointStore(str(tmp_path / "fresh.db"))
    with store._lock:
        names = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"quota_reservations", "quota_usage"} <= names
    store.close()


def test_migrate_v5_to_v6_preserves_all_data(tmp_path):
    db = str(tmp_path / "v5.db")
    _make_v5_db(db)
    store = CheckpointStore(db)  # triggers v5 -> v6 migration
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        names = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "6"
    assert {"quota_reservations", "quota_usage"} <= names
    cp = store.get("r5")
    assert cp is not None and cp.task == "task-text" and cp.physical_reserved == 3
    assert cp.state.value == "running"  # live lease preserved, NOT marked FAILED
    row = store.ledger_get("e5")
    assert row is not None and row["status"] == "completed"
    # quota engine works on the migrated DB
    ok = store.reserve_quota_batch(
        "d-mig", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
        limits={("requests", "day"): 10}, now_iso=NOW.isoformat(timespec="milliseconds"))
    assert ok is True
    store.close()


def test_schema_above_v6_rejected(tmp_path):
    path = str(tmp_path / "bad.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '7')")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="schema_version"):
        CheckpointStore(path)


# ---- persistence across connections ----

def test_reservation_persists_across_connections(tmp_path):
    db = str(tmp_path / "p.db")
    s1 = CheckpointStore(db)
    s1.reserve_quota_batch("d1", pool=POOL_A, items=day_items(1),
                           anchor_iso=ANCHOR.isoformat(),
                           limits={("requests", "day"): 10},
                           now_iso=NOW.isoformat(timespec="milliseconds"))
    s1.close()
    s2 = CheckpointStore(db)
    res = s2.get_reservation("d1")
    assert res is not None and res["status"] == "reserved"
    assert res["items"] == [{"kind": "requests", "period": "day", "amount": 1}]
    rows = s2.quota_status(POOL_A)
    assert len(rows) == 1 and rows[0]["used"] == 1 and rows[0]["kind"] == "requests"
    s2.close()


# ---- engine happy path: ALL quotas enforced, one batch ----

def test_engine_reserve_enforces_all_quotas_in_one_batch(tmp_path):
    quotas = [
        {"pool": POOL_A, "max_requests": 10, "max_tokens": 1000, "period": "day"},
        {"pool": POOL_A, "max_requests": 100, "max_tokens": 50000, "period": "month"},
    ]
    eng, store, _ = make_engine(tmp_path, [free_profile(quotas=quotas)])
    res = eng.reserve_for_model(MODEL, "d1", TokenEstimate(prompt_tokens=100,
                                                            max_output_tokens=50), now=NOW)
    assert res.status == "reserved" and res.pool == POOL_A
    kinds = {(i["kind"], i["period"], i["amount"]) for i in res.items}
    assert kinds == {("requests", "day", 1), ("tokens", "day", 150),
                     ("requests", "month", 1), ("tokens", "month", 150)}
    rows = store.quota_status(POOL_A)
    assert len(rows) == 4  # all written atomically in one transaction
    st = eng.quota_status(MODEL, now=NOW)
    day_w = quota_window_for(ANCHOR.isoformat(), "day", NOW.isoformat(timespec="milliseconds"))
    mon_w = quota_window_for(ANCHOR.isoformat(), "month", NOW.isoformat(timespec="milliseconds"))
    assert st[("requests", "day")] == {"used": 1, "limit": 10, "window_start": day_w,
                                        "pool": POOL_A}
    assert st[("tokens", "day")]["used"] == 150 and st[("tokens", "day")]["limit"] == 1000
    assert st[("requests", "month")]["window_start"] == mon_w
    store.close()


# ---- exhaustion + atomic rollback ----

def test_exhaustion_raises_and_persists_nothing(tmp_path):
    quotas = [{"pool": POOL_A, "max_requests": 2, "max_tokens": 100000, "period": "day"}]
    eng, store, _ = make_engine(tmp_path, [free_profile(quotas=quotas)])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    eng.reserve_for_model(MODEL, "d2", TokenEstimate(1, 1), now=NOW)
    with pytest.raises(QuotaExhaustedError):
        eng.reserve_for_model(MODEL, "d3", TokenEstimate(1, 1), now=NOW)
    assert store.get_reservation("d3") is None
    rows = store.quota_status(POOL_A)
    assert [r for r in rows if r["kind"] == "requests"][0]["used"] == 2
    store.close()


def test_atomic_rollback_when_any_item_exceeds(tmp_path):
    # requests/day passes for d2 but tokens/day fails -> NOTHING may be written
    quotas = [{"pool": POOL_A, "max_requests": 10, "max_tokens": 100, "period": "day"}]
    eng, store, _ = make_engine(tmp_path, [free_profile(quotas=quotas)])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(0, 90), now=NOW)  # tokens used 90
    with pytest.raises(QuotaExhaustedError):
        eng.reserve_for_model(MODEL, "d2", TokenEstimate(0, 50), now=NOW)  # 90+50 > 100
    assert store.get_reservation("d2") is None
    rows = {(r["kind"]): r["used"] for r in store.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 90}  # no partial write, no refund
    store.close()


# ---- duplicate dispatch IDs ----

def test_duplicate_dispatch_id_idempotent_not_double_counted(tmp_path):
    store = CheckpointStore(str(tmp_path / "d.db"))
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=NOW.isoformat(timespec="milliseconds"))
    assert store.reserve_quota_batch("d1", **kw) is True
    assert store.reserve_quota_batch("d1", **kw) is True  # idempotent replay
    rows = store.quota_status(POOL_A)
    assert rows[0]["used"] == 1  # never double-counted
    store.close()


def test_conflicting_duplicate_dispatch_id_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "d.db"))
    now = NOW.isoformat(timespec="milliseconds")
    assert store.reserve_quota_batch(
        "d1", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
        limits={("requests", "day"): 10}, now_iso=now) is True
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        store.reserve_quota_batch(
            "d1", pool=POOL_A, items=day_items(5), anchor_iso=ANCHOR.isoformat(),
            limits={("requests", "day"): 10}, now_iso=now)
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        store.reserve_quota_batch(
            "d1", pool=POOL_B, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
            limits={("requests", "day"): 10}, now_iso=now)
    assert store.quota_status(POOL_A)[0]["used"] == 1  # unchanged
    store.close()


# ---- strict integer accounting ----

@pytest.mark.parametrize("bad_amount", [-1, 1.5, True, False, "3", None,
                                        float("inf"), float("nan")])
def test_non_strict_int_amounts_rejected(tmp_path, bad_amount):
    store = CheckpointStore(str(tmp_path / "v.db"))
    with pytest.raises(CheckpointError):
        store.reserve_quota_batch(
            "dx", pool=POOL_A, items=[{"kind": "requests", "period": "day",
                                      "amount": bad_amount}],
            anchor_iso=ANCHOR.isoformat(),
            limits={("requests", "day"): 10},
            now_iso=NOW.isoformat(timespec="milliseconds"))
    assert store.quota_status(POOL_A) == []
    store.close()


def test_empty_items_missing_limit_bad_kind_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "v.db"))
    now = NOW.isoformat(timespec="milliseconds")
    with pytest.raises(CheckpointError):
        store.reserve_quota_batch("d1", pool=POOL_A, items=[], anchor_iso=ANCHOR.isoformat(),
                                  limits={}, now_iso=now)
    with pytest.raises(CheckpointError):  # item without a limit -> fail closed
        store.reserve_quota_batch("d1", pool=POOL_A, items=day_items(1),
                                  anchor_iso=ANCHOR.isoformat(), limits={}, now_iso=now)
    with pytest.raises(CheckpointError):  # unknown kind
        store.reserve_quota_batch("d1", pool=POOL_A,
                                  items=[{"kind": "dollars", "period": "day", "amount": 1}],
                                  anchor_iso=ANCHOR.isoformat(),
                                  limits={("dollars", "day"): 5}, now_iso=now)
    with pytest.raises(CheckpointError):  # bad limit value
        store.reserve_quota_batch("d1", pool=POOL_A, items=day_items(1),
                                  anchor_iso=ANCHOR.isoformat(),
                                  limits={("requests", "day"): 2.5}, now_iso=now)
    with pytest.raises(CheckpointError):  # conflicting duplicate items in one batch
        store.reserve_quota_batch("d1", pool=POOL_A,
                                  items=day_items(1) + day_items(2),
                                  anchor_iso=ANCHOR.isoformat(),
                                  limits={("requests", "day"): 10}, now_iso=now)
    store.close()


def test_duplicate_items_same_amount_deduped(tmp_path):
    store = CheckpointStore(str(tmp_path / "v.db"))
    ok = store.reserve_quota_batch(
        "d1", pool=POOL_A, items=day_items(1) + day_items(1),
        anchor_iso=ANCHOR.isoformat(), limits={("requests", "day"): 10},
        now_iso=NOW.isoformat(timespec="milliseconds"))
    assert ok is True
    assert store.quota_status(POOL_A)[0]["used"] == 1  # deduped, not 2
    store.close()


@pytest.mark.parametrize("est", [
    TokenEstimate(prompt_tokens=1.5, max_output_tokens=10),
    TokenEstimate(prompt_tokens=True, max_output_tokens=10),
    TokenEstimate(prompt_tokens=-1, max_output_tokens=10),
    TokenEstimate(prompt_tokens=0, max_output_tokens=0),    # unbounded output
    TokenEstimate(prompt_tokens=0, max_output_tokens=-3),
])
def test_engine_rejects_non_strict_or_unbounded_estimates(tmp_path, est):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    with pytest.raises(CostUnknownError):
        eng.reserve_for_model(MODEL, "d1", est, now=NOW)
    assert store.quota_status(POOL_A) == []
    store.close()


# ---- fail-closed unknown/future reset semantics ----

def test_engine_fail_closed_without_window_anchor(tmp_path):
    prof = dict(provider="gemini", models=[MODEL], billing="paid",
                account_label="paid-acct", account_fingerprint="c" * 64,
                quotas=[{"pool": "gemini:acct:" + "c" * 64, "max_requests": 10,
                         "period": "day"}])
    eng, store, _ = make_engine(tmp_path, [prof])
    with pytest.raises(CloudConfigError, match="quota_window_start"):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    store.close()


def test_engine_fail_closed_on_future_window_anchor(tmp_path):
    prof = free_profile(quota_window_start=(NOW + timedelta(days=1)).isoformat())
    eng, store, _ = make_engine(tmp_path, [prof])
    with pytest.raises(CloudConfigError, match="future"):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    store.close()


def test_engine_fail_closed_on_unknown_model_and_missing_quotas(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    with pytest.raises(CloudConfigError):
        eng.reserve_for_model("gemini/not-registered", "d1", TokenEstimate(1, 1), now=NOW)
    prof = dict(provider="gemini", models=["gemini/p2"], billing="paid",
                account_label="noq", account_fingerprint="d" * 64)
    eng2, store2, _ = make_engine(tmp_path, [prof], name="q2.db")
    with pytest.raises(CloudConfigError):
        eng2.reserve_for_model("gemini/p2", "d1", TokenEstimate(1, 1), now=NOW)
    store.close()
    store2.close()


def test_engine_rejects_naive_clock(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    with pytest.raises(CloudConfigError):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1),
                              now=datetime(2026, 1, 15, 12, 0, 0))  # naive
    store.close()


# ---- lease fencing / stale workers ----

def test_stale_worker_fenced_by_owner_and_generation(tmp_path):
    db = str(tmp_path / "lease.db")
    eng, store, _ = make_engine(tmp_path, [free_profile()], name="lease.db")
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=MODEL,
                     root_task_id="root")
    res = eng.reserve_for_model(MODEL, "d1", TokenEstimate(10, 20), now=NOW,
                                run_id="r", lease_owner="w1", lease_generation=0)
    assert res.status == "reserved"
    # another worker acquires a fresh lease (generation bumped)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET lease_owner='w2', lease_generation=1, "
                 "lease_expires_at='2999-01-01T00:00:00.000+00:00' WHERE run_id='r'")
    conn.commit()
    conn.close()
    # stale worker w1/gen0 can neither reserve nor mark dispatched
    with pytest.raises(CheckpointError, match="fencing"):
        eng.reserve_for_model(MODEL, "d2", TokenEstimate(1, 1), now=NOW,
                              run_id="r", lease_owner="w1", lease_generation=0)
    assert eng.mark_dispatched("d1", run_id="r", lease_owner="w1",
                               lease_generation=0) is False
    assert store.get_reservation("d2") is None
    # the new owner can reserve, and can mark ITS OWN reservation dispatched
    eng.reserve_for_model(MODEL, "d2", TokenEstimate(1, 1), now=NOW,
                          run_id="r", lease_owner="w2", lease_generation=1)
    # but w2 must not advance w1's reservation (row-bound fencing, like ledger_update)
    assert eng.mark_dispatched("d1", run_id="r", lease_owner="w2",
                               lease_generation=1) is False
    assert eng.mark_dispatched("d2", run_id="r", lease_owner="w2",
                               lease_generation=1) is True
    assert store.get_reservation("d1")["status"] == "reserved"  # untouched by stale/other worker
    assert store.get_reservation("d2")["status"] == "dispatched"
    store.close()


def test_reservation_after_run_terminal_is_fenced(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=MODEL,
                     root_task_id="root")
    store.mark_failed("r", owner="w1", error="boom")
    with pytest.raises(CheckpointError):
        eng.reserve_for_model(MODEL, "d9", TokenEstimate(1, 1), now=NOW,
                              run_id="r", lease_owner="w1", lease_generation=0)
    store.close()


# ---- mark_dispatched lifecycle ----

def test_mark_dispatched_lifecycle_idempotent_no_refund(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(10, 20), now=NOW)
    assert eng.mark_dispatched("nope") is False
    assert eng.mark_dispatched("d1") is True
    assert eng.mark_dispatched("d1") is True  # idempotent
    assert store.get_reservation("d1")["status"] == "dispatched"
    rows = {(r["kind"]): r["used"] for r in store.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 30}  # never refunded/decremented
    store.close()


# ---- period boundaries: rollover keeps historical windows ----

def test_period_rollover_retains_old_windows(tmp_path):
    quotas = [{"pool": POOL_A, "max_requests": 100, "max_tokens": 10**9, "period": "day"},
              {"pool": POOL_A, "max_requests": 1000, "max_tokens": 10**12, "period": "month"}]
    eng, store, _ = make_engine(tmp_path, [free_profile(quotas=quotas)])
    t1 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
    anchor = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    prof = free_profile(quotas=quotas, quota_window_start=anchor.isoformat())
    eng, store, _ = make_engine(tmp_path, [prof], name="roll.db")
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=t1)
    eng.reserve_for_model(MODEL, "d2", TokenEstimate(1, 1), now=t2)
    rows = [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"
            and r["period"] == "day"]
    windows = {r["window_start"]: r["used"] for r in rows}
    assert windows == {"2026-01-02T00:00:00.000+00:00": 1,
                       "2026-01-03T00:00:00.000+00:00": 1}  # old window retained
    st = eng.quota_status(MODEL, now=t2)
    assert st[("requests", "day")]["window_start"] == "2026-01-03T00:00:00.000+00:00"
    assert st[("requests", "day")]["used"] == 1  # per-window, not cumulative-reset
    assert st[("requests", "month")]["window_start"] == "2026-01-01T00:00:00.000+00:00"
    assert st[("requests", "month")]["used"] == 2  # month window accumulates both
    store.close()


# ---- cross-account isolation + shared account pool ----

def _acct_profile(fp, label, model, pool, max_requests):
    p = free_profile(models=[model], account_label=label, account_fingerprint=fp,
                     quota_window_start=ANCHOR.isoformat(),
                     quotas=[{"pool": pool, "max_requests": max_requests,
                              "max_tokens": 10**9, "period": "day"}])
    p["verification_evidence"]["account_fingerprint"] = fp
    return p


def test_cross_account_pools_are_isolated(tmp_path):
    profs = [_acct_profile(FP_A, "acct-a", "gemini/model-a", POOL_A, 1),
             _acct_profile(FP_B, "acct-b", "gemini/model-b", POOL_B, 1)]
    eng, store, _ = make_engine(tmp_path, profs)
    eng.reserve_for_model("gemini/model-a", "a1", TokenEstimate(1, 1), now=NOW)
    with pytest.raises(QuotaExhaustedError):
        eng.reserve_for_model("gemini/model-a", "a2", TokenEstimate(1, 1), now=NOW)
    # account B is unaffected by account A's exhaustion
    eng.reserve_for_model("gemini/model-b", "b1", TokenEstimate(1, 1), now=NOW)
    assert store.quota_status(POOL_A)[0]["used"] == 1
    assert store.quota_status(POOL_B)[0]["used"] == 1
    store.close()


def test_shared_account_pool_across_models(tmp_path):
    prof = free_profile(models=["gemini/m1", "gemini/m2"],
                        quotas=[{"pool": POOL_A, "max_requests": 2,
                                 "max_tokens": 10**9, "period": "day"}])
    eng, store, _ = make_engine(tmp_path, [prof])
    eng.reserve_for_model("gemini/m1", "d1", TokenEstimate(1, 1), now=NOW)
    eng.reserve_for_model("gemini/m2", "d2", TokenEstimate(1, 1), now=NOW)
    with pytest.raises(QuotaExhaustedError):
        eng.reserve_for_model("gemini/m1", "d3", TokenEstimate(1, 1), now=NOW)
    assert store.quota_status(POOL_A)[0]["used"] == 2  # one shared pool
    store.close()


# ---- config changes never reset consumed quotas ----

def test_config_limit_change_does_not_reset_used(tmp_path):
    db = str(tmp_path / "cfg.db")
    store = CheckpointStore(db, utcnow=lambda: NOW)
    reg1 = registry([free_profile(quotas=[{"pool": POOL_A, "max_requests": 5,
                                           "max_tokens": 10**9, "period": "day"}])])
    eng1 = QuotaEngine(store, reg1, utcnow=lambda: NOW)
    for i in range(3):
        eng1.reserve_for_model(MODEL, f"d{i}", TokenEstimate(1, 1), now=NOW)
    # operator edits config: same account fingerprint/pool, higher limit
    reg2 = registry([free_profile(quotas=[{"pool": POOL_A, "max_requests": 100,
                                           "max_tokens": 10**9, "period": "day"}])])
    eng2 = QuotaEngine(store, reg2, utcnow=lambda: NOW)
    st = eng2.quota_status(MODEL, now=NOW)
    assert st[("requests", "day")]["used"] == 3 and st[("requests", "day")]["limit"] == 100
    eng2.reserve_for_model(MODEL, "d3", TokenEstimate(1, 1), now=NOW)  # allowed by new limit
    assert eng2.quota_status(MODEL, now=NOW)[("requests", "day")]["used"] == 4
    store.close()


# ---- concurrency: threads + two independent connections ----

def test_concurrent_reservations_exact_under_contention(tmp_path):
    db = str(tmp_path / "conc.db")
    s1 = CheckpointStore(db)
    s2 = CheckpointStore(db)
    anchor = ANCHOR.isoformat()
    now = NOW.isoformat(timespec="milliseconds")
    limits = {("requests", "day"): 20}
    outcomes = []
    lock = threading.Lock()
    barrier = threading.Barrier(30)

    def worker(i):
        store = s1 if i % 2 == 0 else s2
        barrier.wait()
        try:
            ok = store.reserve_quota_batch(f"d{i}", pool=POOL_A, items=day_items(1),
                                           anchor_iso=anchor, limits=limits, now_iso=now)
            outcome = "ok" if ok else "fenced"
        except QuotaExhaustedError:
            outcome = "exhausted"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 20
    assert outcomes.count("exhausted") == 10
    assert s1.quota_status(POOL_A)[0]["used"] == 20  # exact, never over-counted
    assert len([r for r in s1.list_runs()]) == 0
    s1.close()
    s2.close()


def test_concurrent_token_reservations_exact(tmp_path):
    db = str(tmp_path / "conc2.db")
    s1 = CheckpointStore(db)
    s2 = CheckpointStore(db)
    anchor = ANCHOR.isoformat()
    now = NOW.isoformat(timespec="milliseconds")
    items = [{"kind": "tokens", "period": "day", "amount": 150}]
    limits = {("tokens", "day"): 1000}
    outcomes = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker(i):
        store = s1 if i % 2 == 0 else s2
        barrier.wait()
        try:
            store.reserve_quota_batch(f"t{i}", pool=POOL_A, items=items,
                                      anchor_iso=anchor, limits=limits, now_iso=now)
            outcome = "ok"
        except QuotaExhaustedError:
            outcome = "exhausted"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 6  # 6*150=900 <= 1000, 7th would exceed
    assert s1.quota_status(POOL_A)[0]["used"] == 900
    s1.close()
    s2.close()


# ---- subprocess crash recovery: reservation retained, never auto-refunded ----

CRASH_AFTER_RESERVE = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode

db, anchor, now = sys.argv[1], sys.argv[2], sys.argv[3]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="s", model="m",
                 root_task_id="r")
ok = store.reserve_quota_batch(
    "d-crash", pool="gemini:acct:" + "a" * 64,
    items=[{"kind": "requests", "period": "day", "amount": 1},
           {"kind": "tokens", "period": "day", "amount": 150}],
    anchor_iso=anchor, limits={("requests", "day"): 10, ("tokens", "day"): 1000},
    now_iso=now, run_id="crash", lease_owner="w", lease_generation=0)
assert ok, "reservation must succeed before the crash"
os._exit(9)  # hard crash AFTER the durable reservation, BEFORE any dispatch
"""


def test_crash_after_reservation_retains_it_without_refund(tmp_path):
    db = str(tmp_path / "crash.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    anchor = ANCHOR.isoformat()
    now = NOW.isoformat(timespec="milliseconds")
    proc = subprocess.run([sys.executable, "-c", CRASH_AFTER_RESERVE, db, anchor, now],
                          cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 9  # the subprocess really crashed
    s2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    res = s2.get_reservation("d-crash")
    assert res is not None and res["status"] == "reserved"  # persisted, not lost
    assert res["run_id"] == "crash" and res["lease_owner"] == "w"
    rows = {(r["kind"]): r["used"] for r in s2.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 150}  # conservatively retained
    # reconciliation: same dispatch_id is idempotent, never double-counted
    assert s2.reserve_quota_batch(
        "d-crash", pool=POOL_A,
        items=[{"kind": "requests", "period": "day", "amount": 1},
               {"kind": "tokens", "period": "day", "amount": 150}],
        anchor_iso=anchor, limits={("requests", "day"): 10, ("tokens", "day"): 1000},
        now_iso=now, run_id="crash", lease_owner="w", lease_generation=0) is True
    rows = {(r["kind"]): r["used"] for r in s2.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 150}  # no double count, no auto-refund
    # lifecycle can still complete; usage never decreases
    assert s2.mark_dispatched("d-crash") is True
    rows = {(r["kind"]): r["used"] for r in s2.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 150}
    # a new dispatch still works against the retained usage
    assert s2.reserve_quota_batch(
        "d-next", pool=POOL_A, items=day_items(1), anchor_iso=anchor,
        limits={("requests", "day"): 10}, now_iso=now) is True
    assert [r for r in s2.quota_status(POOL_A) if r["kind"] == "requests"][0]["used"] == 2
    s2.close()


def test_reservation_json_is_stored_safely(tmp_path):
    store = CheckpointStore(str(tmp_path / "j.db"))
    store.reserve_quota_batch("d1", pool=POOL_A, items=day_items(2),
                              anchor_iso=ANCHOR.isoformat(),
                              limits={("requests", "day"): 10},
                              now_iso=NOW.isoformat(timespec="milliseconds"))
    with store._lock:
        raw = store._conn.execute(
            "SELECT items_json FROM quota_reservations WHERE dispatch_id='d1'"
        ).fetchone()[0]
    assert json.loads(raw) == [{"kind": "requests", "period": "day", "amount": 2}]
    store.close()
