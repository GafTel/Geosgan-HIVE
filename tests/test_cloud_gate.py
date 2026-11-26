"""Phase 4 C4: zero-cost gate at the gateway dispatch boundary + fallback integration.

Failing-first, fully offline: LiteLLM is monkeypatched (mocked provider), a socket
spy proves no network is attempted, and the durable QuotaEngine runs on real
SQLite. Covers: gate wiring/fail-closed without a store, eligible dispatch with
durable reservation + single-use authorization, max_tokens enforcement, complete
prompt accounting, retry (429 + Retry-After) vs quota exhaustion, fallback to
free-verified candidates only (no paid fallback), lease fencing on resumed runs,
async path, and rejection of test-only mock adapters for live estimation."""

import asyncio
import json
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.cloud_gate import ZeroCostGate
from geosgan_hive.cloud_profiles import (
    AdapterCapabilities,
    CloudProfile,
    CloudProfileRegistry,
    TokenEstimate,
    TokenEstimationMode,
)
from geosgan_hive.errors import (
    CheckpointError,
    CloudConfigError,
    CloudOfflineError,
    CostUnknownError,
    FallbackExhaustedError,
    QuotaExhaustedError,
)
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
ANCHOR = datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC)
FP_A = "a" * 64
FP_B = "b" * 64
POOL_A = "gemini:acct:" + FP_A
POOL_B = "gemini:acct:" + FP_B
M1 = "gemini/gemini-2.0-flash"
M2 = "gemini/gemini-2.5-flash"


class LiveStubAdapter:
    """Live-capable but strictly offline stub: EXACT token estimation (1 char =
    1 token) so prompt accounting is deterministic; never performs network I/O."""

    name = "live-stub"
    allow_network = False
    is_test_only = False
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.EXACT)

    def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
        return 0.0 if model.startswith("gemini/") else None

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        if not model.startswith("gemini/") or max_output_tokens < 1:
            return None
        return TokenEstimate(prompt_tokens=prompt_chars, max_output_tokens=max_output_tokens)

    def quota_pool(self, model):
        return POOL_A if model == M1 else POOL_B

    def dispatch(self, model, messages):
        raise CloudOfflineError("live-stub is offline-only; no network request made")


def free_profile(model=M1, fp=FP_A, label="main", max_requests=100,
                 max_tokens=10**9, **overrides) -> dict:
    pool = "gemini:acct:" + fp
    base = dict(
        provider="gemini",
        models=[model],
        billing="free_tier_verified",
        verified_free=True,
        overage_billing_disabled=True,
        billing_verified=True,
        verified_at=(NOW - timedelta(days=1)).isoformat(),
        account_label=label,
        account_fingerprint=fp,
        quota_window_start=ANCHOR.isoformat(),
        quota_reset_semantics="provider_documented",
        verification_evidence={
            "source": "provider_billing_console_export",
            "evidence_ref": "ev-2026-01-14",
            "collected_at": (NOW - timedelta(days=1)).isoformat(),
            "collected_by": "operator-1",
            "credential_bound": True,
            "billing_enforcement": "provider_hard_limit",
            "account_fingerprint": fp,
        },
        quotas=[{"pool": pool, "max_requests": max_requests, "max_tokens": max_tokens,
                 "period": "day"}],
    )
    base.update(overrides)
    return base


def registry(profiles, *, trusted=True, zero_cost=True) -> CloudProfileRegistry:
    return CloudProfileRegistry(
        profiles=[CloudProfile.model_validate(p) for p in profiles],
        zero_cost=zero_cost,
        max_spend_usd=0.0 if zero_cost else None,
        trusted_evidence_binding=trusted,
    )


def make_gate(tmp_path, profiles, *, trusted=True, name="gate.db", **gate_kw):
    store = CheckpointStore(str(tmp_path / name), utcnow=lambda: NOW)
    reg = registry(profiles, trusted=trusted)
    gate = ZeroCostGate(reg, store, adapter=LiveStubAdapter(), utcnow=lambda: NOW,
                        **gate_kw)
    return gate, store, reg


def fake_response(content="ok", model=M1):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model=model,
    )


def make_gateway(store, gate, *, max_attempts=1, sleeps=None, ctx_lookup=None):
    gw = ModelGateway(
        tracker=UsageTracker(),
        retry_policy=RetryPolicy(max_attempts=max_attempts, base_delay_seconds=0,
                                 max_delay_seconds=0, jitter_ratio=0),
        environ={"GEMINI_API_KEY": "sk-test"},
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )
    gw.set_durable_store(store)
    if ctx_lookup is not None:
        gw.set_context_lookup(ctx_lookup)
    gw.set_cloud_gate(gate)
    return gw


@pytest.fixture
def litellm_spy(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs.get("model", M1))

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs.get("model", M1))

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


@pytest.fixture
def no_network(monkeypatch):
    """Spy that forbids outbound connections (asyncio's event loop needs
    socketpair, so the constructor itself must stay intact)."""
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


# ---- wiring / fail-closed ----

def test_zero_cost_requires_durable_store(tmp_path):
    reg = registry([free_profile()])
    with pytest.raises(CloudConfigError, match="durable shared quota store"):
        ZeroCostGate(reg, None, adapter=LiveStubAdapter())


def test_gate_blocks_without_trusted_binding(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile()], trusted=False)
    gw = make_gateway(store, gate)
    with pytest.raises(CloudConfigError, match="zero-cost gate blocked"):
        gw.complete([Message("user", "hi")], model=M1, max_tokens=100)
    assert litellm_spy == []  # nothing dispatched
    assert store.quota_status(POOL_A) == []  # nothing reserved
    store.close()


def test_gate_blocks_unknown_billing_and_paid(tmp_path, litellm_spy, no_network):
    paid = dict(provider="gemini", models=["gemini/paid-model"], billing="paid",
                account_label="paid", account_fingerprint="c" * 64)
    gate, store, _ = make_gate(tmp_path, [free_profile(), paid])
    gw = make_gateway(store, gate)
    with pytest.raises(CloudConfigError, match="zero-cost gate blocked"):
        gw.complete([Message("user", "hi")], model="gemini/paid-model", max_tokens=10)
    with pytest.raises(CloudConfigError, match="zero-cost gate blocked"):
        gw.complete([Message("user", "hi")], model="gemini/not-registered", max_tokens=10)
    assert litellm_spy == []
    store.close()


def test_gate_rejects_mock_adapter_for_live_estimation(tmp_path, litellm_spy, no_network):
    from geosgan_hive.cloud_profiles import FakeCloudAdapter

    store = CheckpointStore(str(tmp_path / "g.db"), utcnow=lambda: NOW)
    reg = registry([free_profile()])
    gate = ZeroCostGate(reg, store, adapter=FakeCloudAdapter(), utcnow=lambda: NOW)
    gw = make_gateway(store, gate)
    with pytest.raises(CostUnknownError, match="not live-capable|test-only|live"):
        gw.complete([Message("user", "hi")], model=M1, max_tokens=100)
    assert litellm_spy == []
    store.close()


# ---- happy path: durable reservation + single-use authorization ----

def test_gate_authorizes_and_records_durable_dispatch(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile()])
    gw = make_gateway(store, gate)
    out = gw.complete([Message("user", "hello world")], model=M1, max_tokens=64)
    assert out == "ok"
    assert len(litellm_spy) == 1
    assert litellm_spy[0]["max_tokens"] == 64  # enforced on the actual request
    # exactly one durable reservation, dispatched exactly once
    rows = store.quota_status(POOL_A)
    kinds = {(r["kind"]): r["used"] for r in rows}
    assert kinds["requests"] == 1
    base = (len(json.dumps(litellm_spy[0]["messages"], default=str))
            + len(json.dumps(litellm_spy[0].get("tools") or [])) + len(M1))
    prompt_chars = base + max(256, base // 10)  # conservative protocol overhead
    assert kinds["tokens"] == prompt_chars + 64  # complete prompt + max output
    # credentials are never part of the prompt accounting
    assert "sk-test" not in json.dumps(litellm_spy[0].get("messages"))
    with store._lock:
        res = store._conn.execute("SELECT * FROM quota_reservations").fetchall()
    assert len(res) == 1 and res[0]["status"] == "dispatched"
    store.close()


def test_gate_requires_bounded_max_tokens(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile()])  # no default cap
    gw = make_gateway(store, gate)
    with pytest.raises(CostUnknownError, match="bounded max_tokens"):
        gw.complete([Message("user", "hi")], model=M1)  # max_tokens=None
    assert litellm_spy == []
    # a configured default cap makes it dispatchable
    gate2, store2, _ = make_gate(tmp_path, [free_profile()], name="g2.db",
                                 default_max_output_tokens=128)
    gw2 = make_gateway(store2, gate2)
    gw2.complete([Message("user", "hi")], model=M1)
    assert litellm_spy[0]["max_tokens"] == 128
    store.close()
    store2.close()


# ---- retry: 429 + Retry-After vs quota exhaustion ----

def test_gate_covers_retries_and_honors_retry_after(tmp_path, litellm_spy, no_network, monkeypatch):
    import litellm
    gate, store, _ = make_gate(tmp_path, [free_profile()])
    sleeps = []
    gw = ModelGateway(
        tracker=UsageTracker(),
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.1,
                                 max_delay_seconds=30, jitter_ratio=0),
        environ={"GEMINI_API_KEY": "sk-test"},
        sleep=sleeps.append,
    )
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            exc = litellm.exceptions.RateLimitError(
                "slow down", llm_provider="gemini", model=M1)
            exc.response = SimpleNamespace(headers={"retry-after": "7"})
            raise exc
        return fake_response()

    monkeypatch.setattr(litellm, "completion", flaky)
    out = gw.complete([Message("user", "hi")], model=M1, max_tokens=32)
    assert out == "ok" and len(attempts) == 2
    # every physical attempt got its own durable reservation + authorization
    with store._lock:
        res = store._conn.execute(
            "SELECT dispatch_id, status FROM quota_reservations").fetchall()
    assert len(res) == 2 and all(r["status"] == "dispatched" for r in res)
    assert len({r["dispatch_id"] for r in res}) == 2  # unique dispatch IDs
    rows = [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"]
    assert rows[0]["used"] == 2
    # Retry-After honored (never invented): sleep >= 7s, hint recorded
    assert sleeps and sleeps[0] >= 7.0
    assert gw.retry_after_hint("gemini") == 7.0
    store.close()


def test_quota_exhaustion_is_not_retried_and_distinct_from_429(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile(max_requests=1)])
    sleeps = []
    gw = make_gateway(store, gate, max_attempts=3, sleeps=sleeps)
    gw.complete([Message("user", "one")], model=M1, max_tokens=16)
    with pytest.raises(QuotaExhaustedError):
        gw.complete([Message("user", "two")], model=M1, max_tokens=16)
    assert len(litellm_spy) == 1  # exhausted quota never reaches the provider again
    assert sleeps == []  # and is never retried (unlike 429)
    store.close()


# ---- fallback: free-verified candidates only, no paid fallback ----

def _continuity_policy(chain):
    return FallbackPolicy(mode=FallbackMode.CONTINUITY, fallback_models=tuple(chain),
                          max_fallback_attempts=5)


def test_fallback_to_free_verified_candidate(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [
        free_profile(M1, FP_A, "acct-a", max_requests=1),
        free_profile(M2, FP_B, "acct-b"),
    ])
    gw = make_gateway(store, gate)
    policy = _continuity_policy([M1, M2])
    # first call consumes M1's only request
    gw.complete([Message("user", "a")], model=M1, max_tokens=16, fallback=policy)
    # second call: M1 quota-exhausted at the GATE (never dispatched) -> M2 serves it
    out = gw.complete([Message("user", "b")], model=M1, max_tokens=16, fallback=policy)
    assert out == "ok"
    models_called = [c["model"] for c in litellm_spy]
    assert models_called == [M1, M2]  # exhausted primary never reaches the provider
    assert store.quota_status(POOL_A)[0]["used"] == 1
    assert store.quota_status(POOL_B)[0]["used"] == 1
    store.close()


def test_no_automatic_paid_fallback(tmp_path, litellm_spy, no_network):
    paid = dict(provider="gemini", models=["gemini/paid-model"], billing="paid",
                account_label="paid", account_fingerprint="c" * 64)
    gate, store, _ = make_gate(tmp_path, [free_profile(M1, FP_A, "acct-a", max_requests=1),
                                         paid])
    gw = make_gateway(store, gate)
    policy = _continuity_policy([M1, "gemini/paid-model"])
    gw.complete([Message("user", "a")], model=M1, max_tokens=16, fallback=policy)
    with pytest.raises(FallbackExhaustedError):
        gw.complete([Message("user", "b")], model=M1, max_tokens=16, fallback=policy)
    # the paid candidate was never dispatched
    assert all(c["model"] == M1 for c in litellm_spy)
    store.close()


# ---- lease fencing on run-bound (resumed) dispatches ----

def test_run_bound_dispatch_fenced_when_lease_lost(tmp_path, litellm_spy, no_network):
    from geosgan_hive.run_context import CollaborationPolicy, RunContext

    db = str(tmp_path / "lease.db")
    gate, store, _ = make_gate(tmp_path, [free_profile()], name="lease.db")
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1,
                     root_task_id="root")
    ctx = RunContext("r", CollaborationPolicy())
    ctx.lease_owner = "w1"  # worker-owned identity from the execution context
    ctx.lease_generation = 0
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    gw.complete([Message("user", "x")], model=M1, run_id="r", max_tokens=16)
    assert len(litellm_spy) == 1
    with store._lock:
        res = store._conn.execute(
            "SELECT run_id, lease_owner, lease_generation, status FROM quota_reservations"
        ).fetchall()
    assert res[0]["run_id"] == "r" and res[0]["lease_owner"] == "w1"
    # lease expires (stale worker) -> the gate refuses to authorize dispatch
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET lease_expires_at='2020-01-01T00:00:00.000+00:00' "
                 "WHERE run_id='r'")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="fencing"):
        gw.complete([Message("user", "y")], model=M1, run_id="r", max_tokens=16)
    assert len(litellm_spy) == 1  # stale worker never dispatches
    # terminal run -> also blocked
    store.mark_failed("r", owner="w1", error="boom")
    with pytest.raises(CheckpointError):
        gw.complete([Message("user", "z")], model=M1, run_id="r", max_tokens=16)
    assert len(litellm_spy) == 1
    store.close()


# ---- async path ----

def test_async_dispatch_goes_through_gate(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile()])
    gw = make_gateway(store, gate)

    async def run():
        return await gw.acomplete([Message("user", "async")], model=M1, max_tokens=8)

    out = asyncio.run(run())
    assert out == "ok"
    assert len(litellm_spy) == 1 and litellm_spy[0]["max_tokens"] == 8
    rows = [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"]
    assert rows[0]["used"] == 1
    store.close()


def test_async_quota_exhaustion(tmp_path, litellm_spy, no_network):
    gate, store, _ = make_gate(tmp_path, [free_profile(max_requests=1)])
    gw = make_gateway(store, gate)

    async def run():
        await gw.acomplete([Message("user", "a")], model=M1, max_tokens=8)
        await gw.acomplete([Message("user", "b")], model=M1, max_tokens=8)

    with pytest.raises(QuotaExhaustedError):
        asyncio.run(run())
    assert len(litellm_spy) == 1
    store.close()


# ---- non-zero-cost mode: gate inactive, legacy behavior preserved ----

def test_gate_inactive_outside_zero_cost(tmp_path, litellm_spy, no_network):
    store = CheckpointStore(str(tmp_path / "legacy.db"), utcnow=lambda: NOW)
    reg = CloudProfileRegistry(profiles=[], zero_cost=False, max_spend_usd=None)
    gate = ZeroCostGate(reg, store, adapter=LiveStubAdapter(), utcnow=lambda: NOW)
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_cloud_gate(gate)
    out = gw.complete([Message("user", "hi")], model=M1)  # no max_tokens needed
    assert out == "ok" and len(litellm_spy) == 1
    assert store.quota_status(POOL_A) == []  # no quota accounting outside zero-cost
    store.close()
