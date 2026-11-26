"""C5 targeted adversarial audit (of 0956e9b) — failing-first regression tests.

Covers: recovery state correctness (permanent billing/evidence failures must not
stay RUNNING; atomic fenced terminal transition; no recovery busy loops), strict
FREE_ONLY billing-enforcement evidence (trusted flag alone is insufficient),
reservation substitution integrity, Retry-After edge cases, fresh-subprocess
offline startup with network blocked BEFORE imports, v7->v8 migration, and crash
after mark_dispatched (no refund, no re-authorization).
"""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import litellm
import pytest

from geosgan_hive.checkpoints import (
    CheckpointStore,
    PersistMode,
    QuotaReservationOutcome,
)
from geosgan_hive.cloud_gate import ZeroCostGate, sanitized_request_digest
from geosgan_hive.cloud_profiles import (
    CloudProfile,
    CloudProfileRegistry,
    TokenEstimate,
)
from geosgan_hive.errors import CheckpointError, CloudConfigError, QuotaExhaustedError
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway, _retry_after_seconds
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
ANCHOR = datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC)
FP_A = "a" * 64
POOL_A = "gemini:acct:" + FP_A
M1 = "gemini/gemini-2.0-flash"


class ExactAdapter:
    name = "exact"
    allow_network = False
    is_test_only = False
    from geosgan_hive.cloud_profiles import AdapterCapabilities, TokenEstimationMode
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.EXACT)

    def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
        return 0.0

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        from geosgan_hive.cloud_profiles import TokenEstimate as TE
        return TE(prompt_tokens=prompt_chars, max_output_tokens=max_output_tokens)

    def quota_pool(self, model):
        return POOL_A

    def dispatch(self, model, messages):
        raise RuntimeError("offline")


def evidence(fp=FP_A, enforcement="provider_hard_limit", **kw):
    base = dict(source="provider_billing_console_export", evidence_ref="ev-1",
                collected_at=(NOW - timedelta(days=1)).isoformat(),
                collected_by="operator-1", credential_bound=True, account_fingerprint=fp,
                billing_enforcement=enforcement)
    base.update(kw)
    return base


def free_profile(model=M1, fp=FP_A, enforcement="provider_hard_limit", max_requests=100,
                 **overrides):
    pool = "gemini:acct:" + fp
    base = dict(
        provider="gemini", models=[model], billing="free_tier_verified",
        verified_free=True, overage_billing_disabled=True, billing_verified=True,
        verified_at=(NOW - timedelta(days=1)).isoformat(), account_label="main",
        account_fingerprint=fp, quota_window_start=ANCHOR.isoformat(),
        quota_reset_semantics="provider_documented",
        verification_evidence=evidence(fp, enforcement),
        quotas=[{"pool": pool, "max_requests": max_requests, "max_tokens": 10**9,
                 "period": "day"}],
    )
    base.update(overrides)
    return base


def registry(profiles, *, trusted=True, zero_cost=True):
    return CloudProfileRegistry(
        profiles=[CloudProfile.model_validate(p) for p in profiles],
        zero_cost=zero_cost, max_spend_usd=0.0 if zero_cost else None,
        trusted_evidence_binding=trusted)


def make_gate(tmp_path, profiles, *, trusted=True, clock=None, name="a.db"):
    clock = clock or (lambda: NOW)
    store = CheckpointStore(str(tmp_path / name), utcnow=clock)
    gate = ZeroCostGate(registry(profiles, trusted=trusted), store,
                        adapter=ExactAdapter(), utcnow=clock)
    return gate, store


# ---- 1. recovery state correctness ----

def test_permanent_evidence_failure_marks_failed_across_connections(tmp_path):
    """Expired evidence on resume -> FAILED via an independent connection, lease
    released atomically, and the run never re-enters the eligible set."""
    gate, store = make_gate(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="x")
    store.mark_waiting("r", owner="w1", error="quota", counters_json="{}", steps_json="[]",
                       route_json="[]", transcript_json='[{"role":"user","content":"x"}]',
                       step_count=0,
                       next_attempt_at=(NOW - timedelta(seconds=1)).isoformat(),
                       deadline_at=(NOW + timedelta(days=30)).isoformat())
    store.close()
    # independent connection (separate process view) with an EXPIRED evidence profile
    expired = free_profile()
    expired["verified_at"] = (NOW - timedelta(days=100)).isoformat()
    expired["verification_ttl_days"] = 30
    expired["verification_evidence"]["collected_at"] = (NOW - timedelta(days=100)).isoformat()
    store2 = CheckpointStore(store.path, utcnow=lambda: NOW)
    reg2 = registry([expired])
    gate2 = ZeroCostGate(reg2, store2, adapter=ExactAdapter(), utcnow=lambda: NOW)
    ok, reason = gate2.registry.live_dispatch_eligible(M1, now=NOW)
    assert not ok and "expired" in reason
    # the terminal transition is atomic + fenced: claim, then fail
    assert store2.claim_lease_gen("r", "w2") == 1
    store2.mark_failed("r", owner="w2", error="expired evidence")
    cp = store2.get("r")
    assert cp.state.value == "failed" and cp.lease_owner is None
    assert store2.list_eligible() == []  # no busy loop for the recovery worker
    store2.close()


def test_state_classification_distinct(tmp_path):
    """Temporary quota exhaustion vs permanent billing vs auth vs unavailability."""
    from geosgan_hive.errors import (
        AuthenticationError,
        ProviderUnavailableError,
        RateLimitError,
        classify_error,
    )
    assert classify_error(QuotaExhaustedError("q"))[1:] == (False, True, False)
    assert classify_error(RateLimitError("429"))[1:] == (True, True, False)
    assert classify_error(AuthenticationError("401"))[2] is False  # permanent
    assert classify_error(ProviderUnavailableError("wait"))[0].value == "provider_unavailable"
    # permanent billing-policy rejection: not retried, no fallback
    assert classify_error(CloudConfigError("billing"))[1:] == (False, False, False)


# ---- 2. strict FREE_ONLY billing enforcement ----

def test_billing_enforcement_unknown_denies_despite_trusted_flag(tmp_path):
    # the manual trusted_evidence_binding flag alone is NOT proof of billing safety
    gate, store = make_gate(tmp_path, [free_profile(enforcement="unknown")], trusted=True)
    ok, reason = gate.registry.live_dispatch_eligible(M1, now=NOW)
    assert not ok and "UNKNOWN" in reason
    store.close()


@pytest.mark.parametrize("enforcement,cred_bound,expected_ok", [
    ("provider_hard_limit", True, True),
    ("free_allowance_no_overage", True, True),    # allowance + overage disabled + bound
    ("free_allowance_no_overage", False, False),  # unbound allowance evidence: deny
    ("local_inference", True, False),             # not a local provider: deny
    ("unknown", True, False),
])
def test_enforcement_categories(tmp_path, enforcement, cred_bound, expected_ok):
    prof = free_profile(enforcement=enforcement)
    prof["verification_evidence"]["credential_bound"] = cred_bound
    gate, store = make_gate(tmp_path, [prof])
    ok, _reason = gate.registry.live_dispatch_eligible(M1, now=NOW)
    assert ok is expected_ok
    store.close()


def test_local_inference_allowed_only_for_ollama(tmp_path):
    prof = dict(provider="ollama", models=["ollama/llama3.2"], billing="free_tier_verified",
                verified_free=True, overage_billing_disabled=True, billing_verified=True,
                verified_at=(NOW - timedelta(days=1)).isoformat(), account_label="loc",
                account_fingerprint=FP_A, quota_window_start=ANCHOR.isoformat(),
                quota_reset_semantics="provider_documented",
                verification_evidence=evidence(enforcement="local_inference"),
                quotas=[{"pool": "ollama:acct:" + FP_A, "max_requests": 10,
                         "max_tokens": 10**6, "period": "day"}])
    gate, store = make_gate(tmp_path, [prof])
    ok, _ = gate.registry.live_dispatch_eligible("ollama/llama3.2", now=NOW)
    assert ok is True
    store.close()


# ---- 3. reservation substitution integrity ----

def _kw(**over):
    kw = dict(model=M1, messages=[Message("user", "orig").to_openai()], max_tokens=16,
              temperature=0.2, timeout=60.0, num_retries=0, api_key="sk-test")
    kw.update(over)
    return kw


def test_substitution_after_reservation_rejected(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    base = _kw()
    did = gate.prepare_dispatch(M1, dict(base))
    for changed in (
        _kw(messages=[Message("user", "CHANGED").to_openai()]),  # payload
        _kw(model="gemini/other-model"),                          # model/routing
        _kw(max_tokens=999),                                      # token bound
        _kw(api_base="https://evil.example"),                     # endpoint routing
        _kw(temperature=0.9),                                     # any semantic param
    ):
        with pytest.raises(CheckpointError, match="authorization denied"):
            gate.finalize_dispatch(M1, did, kwargs=changed)
    gate.finalize_dispatch(M1, did, kwargs=base)  # original payload still fine
    store.close()


def test_credential_stripping_keeps_account_binding(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    kw = _kw()
    d1 = sanitized_request_digest(kw)
    d2 = sanitized_request_digest(_kw(api_key="sk-DIFFERENT"))  # credential swap
    assert d1 == d2  # api_key is not part of the digest...
    # ...but account binding is enforced via the pool fingerprint + lease, not the key
    did = gate.prepare_dispatch(M1, dict(kw))
    assert did
    with store._lock:
        row = store._conn.execute("SELECT pool, model FROM quota_reservations").fetchone()
    assert row["pool"] == POOL_A and row["model"] == M1
    store.close()


def test_reservation_cannot_be_replayed_or_transferred(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    did = gate.prepare_dispatch(M1, dict(_kw()))
    gate.finalize_dispatch(M1, did, kwargs=_kw())
    assert store.mark_dispatched(did, now_iso=NOW.isoformat(timespec="milliseconds")) is False
    # transfer to another pool/model is a conflicting duplicate
    with pytest.raises(CheckpointError):
        store.reserve_quota_batch(did, pool="gemini:acct:" + "b" * 64,
                                  items=[{"kind": "requests", "period": "day", "amount": 1}],
                                  anchor_iso=ANCHOR.isoformat(),
                                  limits={("requests", "day"): 10},
                                  now_iso=NOW.isoformat(timespec="milliseconds"))
    store.close()


# ---- 4. Retry-After edge cases ----

@pytest.mark.parametrize("raw,expected", [
    ("abc", None), ("", None), ("-5", None), ("1e999", None), ("nan", None),
    ("5", 5.0), ("0.5", 0.5), ("99999999", 86400.0),  # huge values capped at 24h
])
def test_retry_after_parsing(raw, expected):
    exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": raw}))
    assert _retry_after_seconds(exc) == expected


def test_no_free_provider_remaining_suspends_not_paid(tmp_path, monkeypatch):
    """FREE_ONLY: when no authorized free provider remains, WAIT suspends; no paid
    model is ever dispatched."""
    paid = dict(provider="gemini", models=["gemini/paid-m"], billing="paid",
                account_label="paid", account_fingerprint="c" * 64)
    gate, store = make_gate(tmp_path, [free_profile(max_requests=1), paid])
    # exhaust the only free quota
    gate.engine.reserve_for_model(M1, "pre", TokenEstimate(1, 1), now=NOW)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs["model"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
                {"next": "FINISH", "answer": "x"})))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs.get("model", M1))

    monkeypatch.setattr(litellm, "completion", fake)
    monkeypatch.setattr(litellm, "completion_cost", lambda *a, **k: None)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    policy = FallbackPolicy(mode=FallbackMode.WAIT, fallback_models=["gemini/paid-m"],
                            max_fallback_attempts=3)
    from geosgan_hive.errors import ProviderUnavailableError
    with pytest.raises(ProviderUnavailableError):
        gw.complete([Message("user", "x")], model=M1, max_tokens=8, fallback=policy)
    assert calls == []  # paid candidate never dispatched
    store.close()


def test_provider_429_after_local_window_reset_still_suspends(tmp_path, monkeypatch):
    """Local window reset is NOT treated as a provider reset: if the provider still
    429s, the run suspends again on the Retry-After evidence."""
    clock = {"now": NOW}
    gate, store = make_gate(tmp_path, [free_profile(max_requests=100)], clock=lambda: clock["now"])

    def rl(**kwargs):
        exc = litellm.exceptions.RateLimitError("429", llm_provider="gemini", model=M1)
        exc.response = SimpleNamespace(headers={"retry-after": "300"})
        raise exc

    monkeypatch.setattr(litellm, "completion", rl)
    monkeypatch.setattr(litellm, "completion_cost", lambda *a, **k: None)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    policy = FallbackPolicy(mode=FallbackMode.WAIT, max_fallback_attempts=1)
    from geosgan_hive.errors import ProviderUnavailableError
    # local window still active + provider 429 -> suspend with Retry-After evidence
    with pytest.raises(ProviderUnavailableError):
        gw.complete([Message("user", "x")], model=M1, max_tokens=8, fallback=policy)
    assert gw.retry_after_hint("gemini") == 300.0
    # after the LOCAL window resets, local quota allows a retry (provider may still 429)
    clock["now"] = NOW + timedelta(days=1)
    with pytest.raises(ProviderUnavailableError):  # provider still rate-limited
        gw.complete([Message("user", "x")], model=M1, max_tokens=8, fallback=policy)
    rows = [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"]
    assert len(rows) == 2 and all(r["used"] == 1 for r in rows)  # both windows, no refund
    store.close()


# ---- 5. fresh-subprocess offline startup (network blocked BEFORE imports) ----

OFFLINE_STARTUP_SCRIPT = """
import socket, sys

def _blocked(*a, **k):
    raise AssertionError("outbound network during offline startup")

# network restrictions active BEFORE importing HIVE or LiteLLM
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

import geosgan_hive  # noqa: E402  (must not fetch anything)
assert geosgan_hive is not None
import os
assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
import litellm  # noqa: E402  (must use the bundled local cost map, no fetch)
from geosgan_hive.llm import FakeLLM
from geosgan_hive.messages import Message

client = FakeLLM(responses=["pong"])
out = client.complete([Message("user", "ping")])
assert out == "pong"
print("OFFLINE-STARTUP-OK")
"""


def test_fresh_subprocess_offline_startup_no_network(tmp_path):
    repo = os.getcwd()
    proc = subprocess.run(
        [sys.executable, "-c", OFFLINE_STARTUP_SCRIPT],
        cwd=repo, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OFFLINE-STARTUP-OK" in proc.stdout


# ---- 6. v7 -> v8 migration + crash after mark_dispatched ----

def test_dev_v7_database_migrates_to_v8(tmp_path):
    db = str(tmp_path / "v7.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '7')")
    conn.execute(
        "CREATE TABLE quota_reservations (dispatch_id TEXT PRIMARY KEY, pool TEXT NOT NULL, "
        "anchor_iso TEXT NOT NULL, items_json TEXT NOT NULL, limits_json TEXT NOT NULL, "
        "policy_hash TEXT NOT NULL, run_id TEXT, lease_owner TEXT, lease_generation INTEGER, "
        "status TEXT NOT NULL, created_at TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO quota_reservations VALUES "
        "('d-old', ?, ?, '[]', '[]', 'h', NULL, NULL, NULL, 'reserved', 't')",
        (POOL_A, ANCHOR.isoformat(timespec="milliseconds")))
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        cols = {r["name"] for r in
                store._conn.execute("PRAGMA table_info(quota_reservations)").fetchall()}
    assert v == "8"
    assert {"model", "request_sha256", "prompt_tokens", "max_output_tokens", "cost_usd",
            "endpoint"} <= cols
    assert store.get_reservation("d-old")["dispatch_id"] == "d-old"  # preserved
    store.close()


CRASH_AFTER_DISPATCH = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode

db, anchor, now = sys.argv[1], sys.argv[2], sys.argv[3]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
res = store.reserve_quota_batch(
    "d-crash", pool="gemini:acct:" + "a" * 64,
    items=[{"kind": "requests", "period": "day", "amount": 1}],
    anchor_iso=anchor, limits={("requests", "day"): 10}, now_iso=now,
    model="gemini/gemini-2.0-flash", request_sha256="ab" * 32, prompt_tokens=5,
    max_output_tokens=10, cost_usd=0.0, endpoint=None)
assert res.value == "created"
assert store.mark_dispatched("d-crash", now_iso=now) is True  # authorized...
os._exit(9)  # ...hard crash AFTER authorization, BEFORE the provider call
"""


def test_crash_after_dispatch_authorization_no_refund_no_reauth(tmp_path):
    db = str(tmp_path / "crash2.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    anchor = ANCHOR.isoformat()
    now = NOW.isoformat(timespec="milliseconds")
    proc = subprocess.run([sys.executable, "-c", CRASH_AFTER_DISPATCH, db, anchor, now],
                          cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 9
    s2 = CheckpointStore(db, utcnow=lambda: NOW)
    res = s2.get_reservation("d-crash")
    assert res["status"] == "dispatched" and res["request_sha256"] == "ab" * 32
    assert res["model"] == M1 and res["max_output_tokens"] == 10 and res["cost_usd"] == 0.0
    assert s2.mark_dispatched("d-crash", now_iso=now) is False  # never re-authorized
    rows = s2.quota_status(POOL_A)
    assert rows[0]["used"] == 1  # consumed reservation conservatively retained
    # a new dispatch still works and is counted separately
    assert s2.reserve_quota_batch(
        "d-next", pool=POOL_A, items=[{"kind": "requests", "period": "day", "amount": 1}],
        anchor_iso=anchor, limits={("requests", "day"): 10}, now_iso=now
    ) == QuotaReservationOutcome.CREATED
    assert s2.quota_status(POOL_A)[0]["used"] == 2
    s2.close()
