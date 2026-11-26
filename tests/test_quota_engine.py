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
    QuotaReservationOutcome,
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
        quota_reset_semantics="provider_documented",
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


def test_schema_version_is_11_and_fresh_store_has_quota_and_benchmark_tables(tmp_path):
    assert SCHEMA_VERSION == 11
    store = CheckpointStore(str(tmp_path / "fresh.db"))
    with store._lock:
        names = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"quota_reservations", "quota_usage", "quota_policies",
            "quota_policy_audit", "api_reservations", "api_usage", "api_audit"} <= names
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
    assert v == "11"
    assert {"quota_reservations", "quota_usage", "quota_policies",
            "quota_policy_audit"} <= names
    cp = store.get("r5")
    assert cp is not None and cp.task == "task-text" and cp.physical_reserved == 3
    assert cp.state.value == "running"  # live lease preserved, NOT marked FAILED
    row = store.ledger_get("e5")
    assert row is not None and row["status"] == "completed"
    # quota engine works on the migrated DB
    ok = store.reserve_quota_batch(
        "d-mig", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
        limits={("requests", "day"): 10}, now_iso=NOW.isoformat(timespec="milliseconds"))
    assert ok == QuotaReservationOutcome.CREATED
    store.close()


def test_schema_above_v9_rejected(tmp_path):
    path = str(tmp_path / "bad.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '999')")
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
    assert res["items"][0]["kind"] == "requests" and res["items"][0]["amount"] == 1
    assert "window_start" in res["items"][0] and res["policy_hash"]
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
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    # idempotent replay: reconciles WITHOUT granting new dispatch authorization
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.REPLAYED
    rows = store.quota_status(POOL_A)
    assert rows[0]["used"] == 1  # never double-counted
    store.close()


def test_conflicting_duplicate_dispatch_id_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "d.db"))
    now = NOW.isoformat(timespec="milliseconds")
    assert store.reserve_quota_batch(
        "d1", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
        limits={("requests", "day"): 10},
        now_iso=now) == QuotaReservationOutcome.CREATED
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
    assert ok == QuotaReservationOutcome.CREATED
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
    assert eng.mark_dispatched("d1") is True  # exactly one caller gets authorization
    assert eng.mark_dispatched("d1") is False  # replay NEVER re-authorizes a dispatch
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

def test_config_limit_change_requires_explicit_migration_and_keeps_used(tmp_path):
    db = str(tmp_path / "cfg.db")
    store = CheckpointStore(db, utcnow=lambda: NOW)
    reg1 = registry([free_profile(quotas=[{"pool": POOL_A, "max_requests": 5,
                                           "max_tokens": 10**9, "period": "day"}])])
    eng1 = QuotaEngine(store, reg1, utcnow=lambda: NOW)
    for i in range(3):
        eng1.reserve_for_model(MODEL, f"d{i}", TokenEstimate(1, 1), now=NOW)
    # operator edits config: same account, higher limit -> incompatible policy,
    # rejected across configurations until migrated explicitly
    reg2 = registry([free_profile(quotas=[{"pool": POOL_A, "max_requests": 100,
                                           "max_tokens": 10**9, "period": "day"}])])
    eng2 = QuotaEngine(store, reg2, utcnow=lambda: NOW)
    with pytest.raises(CheckpointError, match="policy"):
        eng2.reserve_for_model(MODEL, "d3", TokenEstimate(1, 1), now=NOW)
    assert eng2.migrate_quota_policy(MODEL, anchor=ANCHOR, operator="op") is True
    st = eng2.quota_status(MODEL, now=NOW)
    assert st[("requests", "day")]["used"] == 3 and st[("requests", "day")]["limit"] == 100
    eng2.reserve_for_model(MODEL, "d3", TokenEstimate(1, 1), now=NOW)  # allowed now
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
            res = store.reserve_quota_batch(f"d{i}", pool=POOL_A, items=day_items(1),
                                            anchor_iso=anchor, limits=limits, now_iso=now)
            outcome = res.value
        except QuotaExhaustedError:
            outcome = "exhausted"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("created") == 20
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
            res = store.reserve_quota_batch(f"t{i}", pool=POOL_A, items=items,
                                            anchor_iso=anchor, limits=limits, now_iso=now)
            outcome = res.value
        except QuotaExhaustedError:
            outcome = "exhausted"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("created") == 6  # 6*150=900 <= 1000, 7th would exceed
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
res = store.reserve_quota_batch(
    "d-crash", pool="gemini:acct:" + "a" * 64,
    items=[{"kind": "requests", "period": "day", "amount": 1},
           {"kind": "tokens", "period": "day", "amount": 150}],
    anchor_iso=anchor, limits={("requests", "day"): 10, ("tokens", "day"): 1000},
    now_iso=now, run_id="crash", lease_owner="w", lease_generation=0)
assert res.value == "created", "reservation must succeed before the crash"
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
    # reconciliation: same dispatch_id replays idempotently (no new authorization),
    # never double-counted
    assert s2.reserve_quota_batch(
        "d-crash", pool=POOL_A,
        items=[{"kind": "requests", "period": "day", "amount": 1},
               {"kind": "tokens", "period": "day", "amount": 150}],
        anchor_iso=anchor, limits={("requests", "day"): 10, ("tokens", "day"): 1000},
        now_iso=now, run_id="crash", lease_owner="w", lease_generation=0
    ) == QuotaReservationOutcome.REPLAYED
    rows = {(r["kind"]): r["used"] for r in s2.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 150}  # no double count, no auto-refund
    # lifecycle can still complete exactly once; usage never decreases
    assert s2.mark_dispatched("d-crash", run_id="crash", lease_owner="w",
                              lease_generation=0, now_iso=now) is True
    assert s2.mark_dispatched("d-crash", run_id="crash", lease_owner="w",
                              lease_generation=0, now_iso=now) is False  # no re-authz
    assert s2.mark_dispatched("d-crash", now_iso=now) is False  # unbound caller rejected
    rows = {(r["kind"]): r["used"] for r in s2.quota_status(POOL_A)}
    assert rows == {"requests": 1, "tokens": 150}
    # a new dispatch still works against the retained usage
    assert s2.reserve_quota_batch(
        "d-next", pool=POOL_A, items=day_items(1), anchor_iso=anchor,
        limits={("requests", "day"): 10}, now_iso=now) == QuotaReservationOutcome.CREATED
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
    stored = json.loads(raw)
    assert stored[0]["kind"] == "requests" and stored[0]["amount"] == 2
    assert "window_start" in stored[0]  # reservations are bound to actual windows
    store.close()


# ---- C3 hardening: complete lease binding (audit point 1) ----

@pytest.mark.parametrize("kw", [
    dict(run_id="r"),                                             # run without owner
    dict(run_id="r", lease_owner="w"),                            # missing generation
    dict(run_id="r", lease_generation=0),                         # missing owner
    dict(lease_owner="w"),                                        # owner without run
    dict(lease_owner="w", lease_generation=0),                    # owner without run
    dict(lease_generation=0),                                     # generation alone
    dict(run_id="r", lease_owner="w", lease_generation=-1),       # negative generation
    dict(run_id="r", lease_owner="w", lease_generation=True),     # bool generation
    dict(run_id="", lease_owner="w", lease_generation=0),         # empty run id
])
def test_partial_or_malformed_lease_binding_rejected(tmp_path, kw):
    store = CheckpointStore(str(tmp_path / "b.db"))
    now = NOW.isoformat(timespec="milliseconds")
    with pytest.raises(CheckpointError, match="binding|generation|run_id"):
        store.reserve_quota_batch("d1", pool=POOL_A, items=day_items(1),
                                  anchor_iso=ANCHOR.isoformat(),
                                  limits={("requests", "day"): 10}, now_iso=now, **kw)
    with pytest.raises(CheckpointError):
        store.mark_dispatched("d1", **kw)
    assert store.quota_status(POOL_A) == []
    store.close()


def test_stale_worker_rejected_on_identical_replay(tmp_path):
    """Audit point 4: lease ownership is validated on DUPLICATE paths too."""
    db = str(tmp_path / "s.db")
    eng, store, _ = make_engine(tmp_path, [free_profile()], name="s.db")
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=MODEL,
                     root_task_id="root")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10},
              now_iso=NOW.isoformat(timespec="milliseconds"),
              run_id="r", lease_owner="w1", lease_generation=0)
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    # lease takeover by w2
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET lease_owner='w2', lease_generation=1, "
                 "lease_expires_at='2999-01-01T00:00:00.000+00:00' WHERE run_id='r'")
    conn.commit()
    conn.close()
    # identical replay by the STALE worker must be rejected, not silently accepted
    with pytest.raises(CheckpointError, match="fencing"):
        store.reserve_quota_batch("d1", **kw)
    # a replay with a DIFFERENT binding is a conflicting duplicate, not a reconciliation
    kw2 = {**kw, "lease_owner": "w2", "lease_generation": 1}
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        store.reserve_quota_batch("d1", **kw2)
    assert store.quota_status(POOL_A)[0]["used"] == 1
    assert store.get_reservation("d1")["status"] == "reserved"  # readable, unchanged
    store.close()


# ---- C3 hardening: replay never authorizes a new dispatch (audit point 2) ----

def test_engine_replay_of_used_dispatch_id_never_authorizes(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    with pytest.raises(CheckpointError, match="never authorizes"):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    eng.mark_dispatched("d1")
    # even after dispatch, a replay of the same id is rejected by the engine
    with pytest.raises(CheckpointError, match="never authorizes"):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    assert store.quota_status(POOL_A)[0]["used"] == 1  # still no double count
    store.close()


def test_concurrent_dispatch_authorization_single_winner(tmp_path):
    """Audit point 3: exactly one caller may obtain dispatch authorization."""
    db = str(tmp_path / "mw.db")
    s1 = CheckpointStore(db)
    s2 = CheckpointStore(db)
    now = NOW.isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=now)
    assert s1.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(i):
        store = s1 if i % 2 == 0 else s2
        barrier.wait()
        r = store.mark_dispatched("d1", now_iso=now)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1  # single winner only
    assert s1.get_reservation("d1")["status"] == "dispatched"
    s1.close()
    s2.close()


# ---- C3 hardening: persisted policy identity + safe migration (points 5, 6) ----

def test_anchor_change_rejected_until_explicit_migration(tmp_path):
    store = CheckpointStore(str(tmp_path / "p.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now2 = (NOW + timedelta(hours=6)).isoformat(timespec="milliseconds")
    anchor2 = (ANCHOR + timedelta(hours=1)).isoformat()  # forward-moving window
    kw = dict(pool=POOL_A, items=day_items(1), limits={("requests", "day"): 10},
              now_iso=now)
    assert store.reserve_quota_batch("d1", anchor_iso=ANCHOR.isoformat(),
                                     **kw) == QuotaReservationOutcome.CREATED
    pol = store.quota_policy(POOL_A)
    assert pol is not None and pol["anchor_iso"].startswith("2026-01-14T12")
    # changing the anchor without explicit migration is rejected (no silent reset)
    with pytest.raises(CheckpointError, match="policy"):
        store.reserve_quota_batch("d2", anchor_iso=anchor2, **{**kw, "now_iso": now2})
    assert store.get_reservation("d2") is None
    # explicit operator migration: allowed, ACTIVE usage carried forward
    assert store.migrate_quota_policy(POOL_A, anchor_iso=anchor2, periods=["day"],
                                      operator="op-1", now_iso=now2) is True
    assert store.migrate_quota_policy(POOL_A, anchor_iso=anchor2, periods=["day"],
                                      operator="op-1", now_iso=now2) is False  # no-op
    rows = {r["window_start"]: r["used"] for r in store.quota_status(POOL_A)}
    assert rows["2026-01-15T13:00:00.000+00:00"] == 1  # carried into the new window
    assert rows["2026-01-15T12:00:00.000+00:00"] == 1  # history retained, never reset
    assert store.reserve_quota_batch("d2", anchor_iso=anchor2,
                                     **{**kw, "now_iso": now2}
                                     ) == QuotaReservationOutcome.CREATED
    assert store.quota_policy(POOL_A)["anchor_iso"].startswith("2026-01-14T13")
    assert store.quota_policy(POOL_A)["operator"] == "op-1"
    store.close()


def test_replay_conflicting_across_window_limit_or_policy_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "c.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now_next_day = (NOW + timedelta(days=1)).isoformat(timespec="milliseconds")
    base = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
                now_iso=now)
    assert store.reserve_quota_batch("d1", limits={("requests", "day"): 10},
                                     **base) == QuotaReservationOutcome.CREATED
    # same id, later window -> conflicting replay across windows
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        store.reserve_quota_batch("d1", limits={("requests", "day"): 10},
                                  **{**base, "now_iso": now_next_day})
    # same id, different limits -> conflicting replay across limits
    with pytest.raises(CheckpointError, match="conflicting duplicate"):
        store.reserve_quota_batch("d1", limits={("requests", "day"): 99}, **base)
    # same id, different anchor (policy) -> conflicting replay across policies
    with pytest.raises(CheckpointError):
        store.reserve_quota_batch(
            "d1", limits={("requests", "day"): 10},
            **{**base, "anchor_iso": (ANCHOR + timedelta(days=3)).isoformat()})
    assert store.quota_status(POOL_A)[0]["used"] == 1
    store.close()


def test_reservation_carries_policy_hash_and_windows(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(5, 5), now=NOW)
    res = store.get_reservation("d1")
    assert res["policy_hash"] and len(res["policy_hash"]) == 64
    assert res["items"][0]["window_start"] == quota_window_for(
        ANCHOR.isoformat(), "day", NOW.isoformat(timespec="milliseconds"))
    store.close()


# ---- C3 hardening: unknown reset semantics fail closed (point 7) ----

def test_engine_fail_closed_on_unknown_reset_semantics(tmp_path):
    prof = free_profile(quota_reset_semantics="unknown")  # anchor alone is not evidence
    eng, store, _ = make_engine(tmp_path, [prof])
    with pytest.raises(CloudConfigError, match="reset semantics"):
        eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    assert store.quota_status(POOL_A) == []
    store.close()


# ---- C3 hardening: SQLite integer bounds, zero requests (point 8) ----

@pytest.mark.parametrize("bad_amount", [2**63, 2**63 - 1 + 1, -(2**63)])
def test_overflow_amounts_rejected(tmp_path, bad_amount):
    store = CheckpointStore(str(tmp_path / "o.db"))
    with pytest.raises(CheckpointError, match="rejected|int64|overflow|bounds"):
        store.reserve_quota_batch(
            "d1", pool=POOL_A,
            items=[{"kind": "tokens", "period": "day", "amount": bad_amount}],
            anchor_iso=ANCHOR.isoformat(), limits={("tokens", "day"): 2**63 - 1},
            now_iso=NOW.isoformat(timespec="milliseconds"))
    store.close()


def test_zero_request_reservation_and_overflow_limit_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "z.db"))
    now = NOW.isoformat(timespec="milliseconds")
    with pytest.raises(CheckpointError, match="zero|at least 1"):
        store.reserve_quota_batch(
            "d1", pool=POOL_A,
            items=[{"kind": "requests", "period": "day", "amount": 0}],
            anchor_iso=ANCHOR.isoformat(), limits={("requests", "day"): 10}, now_iso=now)
    with pytest.raises(CheckpointError, match="int64|overflow|bounds"):
        store.reserve_quota_batch(
            "d1", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
            limits={("requests", "day"): 2**63}, now_iso=now)
    assert store.quota_status(POOL_A) == []
    store.close()


# ---- C3 hardening: atomic, recoverable migrations (point 9) ----

def test_interrupted_migration_rolls_back_and_recovers(tmp_path, monkeypatch):
    from geosgan_hive.checkpoints import _QUOTA_SCHEMA, _exec_schema

    db = str(tmp_path / "int.db")
    _make_v5_db(db)
    orig = CheckpointStore._migrate_5_to_6

    def boom(self):
        _exec_schema(self._conn, _QUOTA_SCHEMA)  # partial work, then "crash"
        raise RuntimeError("interrupted mid-migration")

    monkeypatch.setattr(CheckpointStore, "_migrate_5_to_6", boom)
    with pytest.raises(RuntimeError, match="interrupted"):
        CheckpointStore(db)
    # the whole migration transaction rolled back: still v5, no quota tables
    conn = sqlite3.connect(db)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert v == "5" and "quota_reservations" not in tables and "quota_usage" not in tables
    # reopening without the fault migrates cleanly (recoverable)
    monkeypatch.setattr(CheckpointStore, "_migrate_5_to_6", orig)
    store = CheckpointStore(db)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "11" and {"quota_reservations", "quota_usage", "quota_policies",
                         "api_reservations", "benchmark_evidence", "routing_decisions",
                         "quota_policy_audit"} <= tables
    assert store.get("r5") is not None
    store.close()


# ---- C3 hardening: increment-only accounting (point 10) ----

def test_usage_is_increment_only_across_operations(tmp_path):
    store = CheckpointStore(str(tmp_path / "mono.db"))
    now = NOW.isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 100}, now_iso=now)
    for i in range(5):
        assert store.reserve_quota_batch(f"d{i}", **kw) == QuotaReservationOutcome.CREATED
    store.mark_dispatched("d0", now_iso=now)
    store.mark_dispatched("d0", now_iso=now)  # rejected replay: no effect
    used = [r["used"] for r in store.quota_status(POOL_A)]
    assert used == [5]  # monotonic, never decremented, no refund path
    store.close()


# ---- C3 residual: binding match incl. all-None (audit point 1) ----

def test_run_bound_reservation_never_authorized_by_unbound_caller(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=MODEL,
                     root_task_id="root")
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW,
                          run_id="r", lease_owner="w1", lease_generation=0)
    assert store.mark_dispatched("d1", now_iso=NOW.isoformat(timespec="milliseconds")) is False
    assert store.mark_dispatched("d1", run_id="r", lease_owner="w2", lease_generation=1,
                                 now_iso=NOW.isoformat(timespec="milliseconds")) is False
    assert store.mark_dispatched("d1", run_id="r", lease_owner="w1", lease_generation=0,
                                 now_iso=NOW.isoformat(timespec="milliseconds")) is True
    store.close()


def test_unbound_reservation_rejects_bound_caller(tmp_path):
    store = CheckpointStore(str(tmp_path / "u.db"))
    now = NOW.isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=now)
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    assert store.mark_dispatched("d1", run_id="r", lease_owner="w", lease_generation=0,
                                 now_iso=now) is False  # bound caller on unbound row
    assert store.mark_dispatched("d1", now_iso=now) is True  # matching all-None binding
    store.close()


# ---- C3 residual: window + policy revalidation at authorization (point 2) ----

def test_mark_dispatched_rejects_rolled_over_window(tmp_path):
    store = CheckpointStore(str(tmp_path / "w.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now_next = (NOW + timedelta(days=1)).isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=now)
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    assert store.reserve_quota_batch("d2", **kw) == QuotaReservationOutcome.CREATED
    # d1 in the same window -> authorized; d2 "dispatched" a day later -> NOT
    assert store.mark_dispatched("d1", now_iso=now) is True
    assert store.mark_dispatched("d2", now_iso=now_next) is False
    assert store.get_reservation("d2")["status"] == "reserved"  # untouched
    store.close()


def test_mark_dispatched_rejects_after_policy_change(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    pol = store.quota_policy(POOL_A)
    # explicit policy change (extra period, same constraints) -> audited migration
    assert store.migrate_quota_policy(
        POOL_A, anchor_iso=ANCHOR.isoformat(), periods=["day", "week"],
        operator="op", constraints=pol["constraints"],
        reset_semantics=pol["reset_semantics"]) is True
    assert eng.mark_dispatched("d1", now=NOW) is False  # policy no longer current
    store.close()


# ---- C3 residual: safe, audited, usage-carrying migration (points 3, 5) ----

def test_migration_preserves_effective_usage_for_enforcement(tmp_path):
    store = CheckpointStore(str(tmp_path / "carry.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now2 = (NOW + timedelta(hours=6)).isoformat(timespec="milliseconds")
    anchor2 = (ANCHOR + timedelta(hours=1)).isoformat()
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 5}, now_iso=now)
    for i in range(3):
        assert store.reserve_quota_batch(f"d{i}", **kw) == QuotaReservationOutcome.CREATED
    assert store.migrate_quota_policy(POOL_A, anchor_iso=anchor2, periods=["day"],
                                      operator="op", now_iso=now2) is True
    kw2 = {**kw, "anchor_iso": anchor2, "now_iso": now2}
    # carried usage still counts against the limit in the new window: 2 more OK, 3rd not
    assert store.reserve_quota_batch("d3", **kw2) == QuotaReservationOutcome.CREATED
    assert store.reserve_quota_batch("d4", **kw2) == QuotaReservationOutcome.CREATED
    with pytest.raises(QuotaExhaustedError):
        store.reserve_quota_batch("d5", **kw2)
    rows = {r["window_start"]: r["used"] for r in store.quota_status(POOL_A)}
    assert rows["2026-01-15T13:00:00.000+00:00"] == 5  # 3 carried + 2 new
    assert rows["2026-01-15T12:00:00.000+00:00"] == 3  # history retained
    store.close()


def test_unsafe_migrations_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "unsafe.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now2 = (NOW + timedelta(hours=6)).isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=now)
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    # future anchor -> fail closed
    with pytest.raises(CheckpointError):
        store.migrate_quota_policy(POOL_A, anchor_iso=(NOW + timedelta(days=2)).isoformat(),
                                   periods=["day"], operator="op", now_iso=now2)
    # backward-moving active window -> unsafe, rejected
    with pytest.raises(CheckpointError, match="unsafe"):
        store.migrate_quota_policy(POOL_A, anchor_iso=(ANCHOR + timedelta(hours=1)).isoformat(),
                                   periods=["day"], operator="op", now_iso=now)
    assert store.quota_policy(POOL_A)["anchor_iso"].startswith("2026-01-14T12")
    store.close()


def test_migration_is_transactional_and_audited(tmp_path):
    store = CheckpointStore(str(tmp_path / "audit.db"))
    now = NOW.isoformat(timespec="milliseconds")
    now2 = (NOW + timedelta(hours=6)).isoformat(timespec="milliseconds")
    anchor2 = (ANCHOR + timedelta(hours=1)).isoformat()
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              limits={("requests", "day"): 10}, now_iso=now)
    assert store.reserve_quota_batch("d1", **kw) == QuotaReservationOutcome.CREATED
    old_hash = store.quota_policy(POOL_A)["policy_hash"]
    assert store.migrate_quota_policy(POOL_A, anchor_iso=anchor2, periods=["day"],
                                      operator="op-9", now_iso=now2) is True
    audit = store.policy_audit(POOL_A)
    assert len(audit) == 1
    assert audit[0]["operator"] == "op-9"
    assert audit[0]["old_policy_hash"] == old_hash
    assert audit[0]["new_policy_hash"] == store.quota_policy(POOL_A)["policy_hash"]
    assert audit[0]["carried"] == [{"kind": "requests", "period": "day",
                                    "from": "2026-01-15T12:00:00.000+00:00",
                                    "to": "2026-01-15T13:00:00.000+00:00", "used": 1}]
    store.close()


# ---- C3 residual: complete canonical constraints persisted (point 4) ----

def test_policy_persists_constraints_and_reset_semantics(tmp_path):
    eng, store, _ = make_engine(tmp_path, [free_profile()])
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    pol = store.quota_policy(POOL_A)
    assert pol["constraints"] == [
        {"kind": "requests", "period": "day", "limit": 1500},
        {"kind": "tokens", "period": "day", "limit": 1_000_000},
    ]
    assert pol["reset_semantics"] == "provider_documented"
    store.close()


def test_incompatible_config_rejected_across_connections(tmp_path):
    db = str(tmp_path / "xproc.db")
    s1 = CheckpointStore(db)
    s2 = CheckpointStore(db)  # independent connection = "another process"
    now = NOW.isoformat(timespec="milliseconds")
    kw = dict(pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
              now_iso=now)
    assert s1.reserve_quota_batch(
        "d1", limits={("requests", "day"): 10}, **kw) == QuotaReservationOutcome.CREATED
    # a different canonical constraint set from another process is incompatible
    with pytest.raises(CheckpointError, match="policy"):
        s2.reserve_quota_batch(
            "d2", limits={("requests", "day"): 20},
            constraints=[{"kind": "requests", "period": "day", "limit": 20}], **kw)
    assert s2.get_reservation("d2") is None
    s1.close()
    s2.close()


# ---- C3 residual: dev-v6 databases handled explicitly (point 6) ----

def test_dev_v6_database_migrates_to_v7_explicitly(tmp_path):
    db = str(tmp_path / "dev6.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '6')")
    # dev-v6 quota_policies WITHOUT constraints_json / reset_semantics columns
    conn.execute(
        "CREATE TABLE quota_policies (pool TEXT PRIMARY KEY, anchor_iso TEXT NOT NULL, "
        "periods_json TEXT NOT NULL, policy_hash TEXT NOT NULL, operator TEXT, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE quota_reservations (dispatch_id TEXT PRIMARY KEY, pool TEXT NOT NULL, "
        "anchor_iso TEXT NOT NULL, items_json TEXT NOT NULL, limits_json TEXT NOT NULL, "
        "policy_hash TEXT NOT NULL, run_id TEXT, lease_owner TEXT, lease_generation INTEGER, "
        "status TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE quota_usage (pool TEXT NOT NULL, kind TEXT NOT NULL, "
        "period TEXT NOT NULL, window_start TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (pool, kind, period, window_start))")
    anchor_canon = "2026-01-14T12:00:00.000+00:00"
    window = quota_window_for(anchor_canon, "day", NOW.isoformat(timespec="milliseconds"))
    conn.execute("INSERT INTO quota_policies VALUES (?, ?, ?, ?, NULL, ?, ?)",
                 (POOL_A, anchor_canon, '["day"]', "deadbeef" * 8,
                  NOW.isoformat(), NOW.isoformat()))
    conn.execute("INSERT INTO quota_reservations VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)",
                 ("d-old", POOL_A, anchor_canon,
                  json.dumps([{"kind": "requests", "period": "day", "amount": 1,
                               "window_start": window}]),
                  json.dumps([{"kind": "requests", "period": "day", "limit": 10}]),
                  "deadbeef" * 8, "reserved", NOW.isoformat()))
    conn.execute("INSERT INTO quota_usage VALUES (?, 'requests', 'day', ?, 1)",
                 (POOL_A, window))
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)  # triggers v6 -> v7
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        cols = {r["name"] for r in store._conn.execute(
            "PRAGMA table_info(quota_policies)").fetchall()}
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "11"
    assert {"constraints_json", "reset_semantics"} <= cols
    assert "quota_policy_audit" in tables
    # stored rows preserved and hashes recomputed consistently
    res = store.get_reservation("d-old")
    pol = store.quota_policy(POOL_A)
    assert res["policy_hash"] == pol["policy_hash"] != "deadbeef" * 8
    assert store.quota_status(POOL_A)[0]["used"] == 1
    # the old reservation can still be authorized in its own window
    assert store.mark_dispatched("d-old", now_iso=NOW.isoformat(timespec="milliseconds")) is True
    # and new reservations work on the migrated schema
    now = NOW.isoformat(timespec="milliseconds")
    assert store.reserve_quota_batch(
        "d-new", pool=POOL_A, items=day_items(1), anchor_iso=ANCHOR.isoformat(),
        limits={("requests", "day"): 10}, now_iso=now) == QuotaReservationOutcome.CREATED
    store.close()


# ---- C3 residual: missing lease expiry rejected (point 7) ----

def test_missing_lease_expiry_rejects_run_bound_dispatch(tmp_path):
    db = str(tmp_path / "nl.db")
    eng, store, _ = make_engine(tmp_path, [free_profile()], name="nl.db")
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=MODEL,
                     root_task_id="root")
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW,
                          run_id="r", lease_owner="w1", lease_generation=0)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET lease_expires_at=NULL WHERE run_id='r'")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="fencing"):
        eng.reserve_for_model(MODEL, "d2", TokenEstimate(1, 1), now=NOW,
                              run_id="r", lease_owner="w1", lease_generation=0)
    assert eng.mark_dispatched("d1", run_id="r", lease_owner="w1",
                               lease_generation=0, now=NOW) is False
    store.close()
