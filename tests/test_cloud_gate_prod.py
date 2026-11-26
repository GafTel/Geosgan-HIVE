"""Phase 4 C4 hardening (audit of 87afc49): production integration.

Failing-first, fully offline (mocked LiteLLM + network spy). Covers:
- YAML zero_cost=true never silently runs legacy (config + factory + CLI);
- :memory: stores rejected in zero-cost mode;
- worker-owned lease identity taken from the execution context, never adopted
  from SQLite; missing run/context/binding fails closed;
- concurrency slot acquired before final authorization; lease/evidence/window/
  policy revalidated immediately before provider invocation (window rollover);
- adapter.estimate_cost enforced (unknown/nonfinite/negative/billable rejected);
- demonstrably conservative token bounds for the complete payload;
- verification evidence bound to provider/endpoint;
- injected gates/adapters cannot silently disable zero-cost enforcement.
"""

import json
import socket
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
import yaml
from pydantic import ValidationError

from geosgan_hive.agent_config import CheckpointConfig, CloudConfig, HiveConfig, load_hive_config
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
    BudgetExceededError,
    CheckpointError,
    CloudConfigError,
    CloudOfflineError,
    ConfigError,
    CostUnknownError,
)
from geosgan_hive.factory import build_team
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.usage import UsageTracker

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
ANCHOR = datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC)
FP_A = "a" * 64
POOL_A = "gemini:acct:" + FP_A
M1 = "gemini/gemini-2.0-flash"


def evidence(fp=FP_A, **kw):
    base = dict(source="provider_billing_console_export", evidence_ref="ev-1",
                collected_at=(NOW - timedelta(days=1)).isoformat(),
                collected_by="operator-1", credential_bound=True, account_fingerprint=fp,
                billing_enforcement="provider_hard_limit")
    base.update(kw)
    return base


def free_profile(model=M1, fp=FP_A, **overrides) -> dict:
    pool = "gemini:acct:" + fp
    base = dict(
        provider="gemini", models=[model], billing="free_tier_verified",
        verified_free=True, overage_billing_disabled=True, billing_verified=True,
        verified_at=(NOW - timedelta(days=1)).isoformat(), account_label="main",
        account_fingerprint=fp, quota_window_start=ANCHOR.isoformat(),
        quota_reset_semantics="provider_documented", verification_evidence=evidence(fp),
        quotas=[{"pool": pool, "max_requests": 100, "max_tokens": 10**9,
                 "period": "day"}],
    )
    base.update(overrides)
    return base


class ExactAdapter:
    name = "exact"
    allow_network = False
    is_test_only = False
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.EXACT)
    cost = 0.0

    def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
        return self.cost

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        return TokenEstimate(prompt_tokens=prompt_chars, max_output_tokens=max_output_tokens)

    def quota_pool(self, model):
        return POOL_A

    def dispatch(self, model, messages):
        raise CloudOfflineError("offline")


class HeuristicLiveAdapter(ExactAdapter):
    """Live-capable but only char/4 heuristic: NOT demonstrably conservative."""

    name = "heuristic-live"
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.BOUNDED_HEURISTIC)

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        return TokenEstimate(prompt_tokens=max(1, -(-prompt_chars // 4)),
                             max_output_tokens=max_output_tokens)


def registry(profiles, *, trusted=True, zero_cost=True):
    return CloudProfileRegistry(
        profiles=[CloudProfile.model_validate(p) for p in profiles],
        zero_cost=zero_cost, max_spend_usd=0.0 if zero_cost else None,
        trusted_evidence_binding=trusted)


def make_gate(tmp_path, profiles, *, adapter=None, clock=None, trusted=True, name="g.db"):
    clock = clock or (lambda: NOW)
    store = CheckpointStore(str(tmp_path / name), utcnow=clock)
    gate = ZeroCostGate(registry(profiles, trusted=trusted), store,
                        adapter=adapter or ExactAdapter(), utcnow=clock)
    return gate, store


def fake_response(content="ok", model=M1):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model=model)


@pytest.fixture
def litellm_spy(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs.get("model", M1))

    async def fake_a(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs.get("model", M1))

    monkeypatch.setattr(litellm, "completion", fake)
    monkeypatch.setattr(litellm, "acompletion", fake_a)
    return calls


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access attempted during offline test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def make_gateway(store, gate, ctx_lookup=None):
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_durable_store(store)
    if ctx_lookup is not None:
        gw.set_context_lookup(ctx_lookup)
    gw.set_cloud_gate(gate)
    return gw


# ---- point 1/2: zero_cost never silently legacy; :memory: rejected ----

REPO_ROOT = Path(__file__).resolve().parent.parent


def _with(cfg: HiveConfig, **updates) -> HiveConfig:
    """Re-validate through the model (model_copy would skip validators)."""
    data = cfg.model_dump()
    data.update(updates)
    return HiveConfig.model_validate(data)


def _cloud(zero_cost=True):
    return CloudConfig(zero_cost=zero_cost, max_spend_usd=0.0 if zero_cost else None,
                       profiles=[CloudProfile.model_validate(free_profile())]).model_dump()


def _config_with_cloud(tmp_path, *, zero_cost=True, checkpoint=True, mem=False):
    cfg = load_hive_config(REPO_ROOT / "config" / "agents.local.yaml", environ={})
    ckpt = None
    if checkpoint:
        path = ":memory:" if mem else str(tmp_path / "cp.db")
        ckpt = CheckpointConfig(path=path, persist_mode="resumable").model_dump()
    return _with(cfg, cloud=_cloud(zero_cost), checkpoint=ckpt)


def test_yaml_zero_cost_requires_durable_checkpoint_config(tmp_path):
    cfg = load_hive_config(REPO_ROOT / "config" / "agents.local.yaml", environ={})
    with pytest.raises(ValidationError, match="zero.cost|checkpoint"):
        _with(cfg, cloud=_cloud(True), checkpoint=None)
    with pytest.raises(ValidationError, match="memory"):
        _with(cfg, cloud=_cloud(True),
              checkpoint=CheckpointConfig(path=":memory:",
                                          persist_mode="resumable").model_dump())


def test_build_team_wires_gate_and_store(tmp_path):
    cfg = _config_with_cloud(tmp_path)
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    supervisor, agents = build_team(cfg, gw, checkpoint_store=None)
    gate = gw._get_cloud_gate()
    assert gate is not None and gate.enabled()
    assert gw._durable_store is not None and gw._durable_store.path != ":memory:"


def test_build_team_zero_cost_without_store_fails_closed(tmp_path):
    cfg = load_hive_config(REPO_ROOT / "config" / "agents.local.yaml", environ={})
    cloud = CloudConfig(zero_cost=True, max_spend_usd=0.0,
                        profiles=[CloudProfile.model_validate(free_profile())])
    # bypass the config validator to prove the FACTORY also fails closed
    object.__setattr__(cfg, "cloud", cloud)
    object.__setattr__(cfg, "checkpoint", None)
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    with pytest.raises(ConfigError, match="zero-cost|durable"):
        build_team(cfg, gw, checkpoint_store=None)


def test_build_team_zero_cost_with_memory_store_fails_closed(tmp_path):
    cfg = _config_with_cloud(tmp_path)
    # bypass config validation to prove the FACTORY also rejects :memory: stores
    object.__setattr__(cfg, "checkpoint",
                       CheckpointConfig(path=":memory:", persist_mode="resumable"))
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    with pytest.raises(ConfigError, match="memory|durable"):
        build_team(cfg, gw, checkpoint_store=CheckpointStore(":memory:"))


def test_gate_rejects_memory_store_directly(tmp_path):
    reg = registry([free_profile()])
    with pytest.raises(CloudConfigError, match="memory|:memory:"):
        ZeroCostGate(reg, CheckpointStore(":memory:"), adapter=ExactAdapter())
    with pytest.raises(CloudConfigError, match="durable|memory"):
        ZeroCostGate(reg, None, adapter=ExactAdapter())


def test_cli_zero_cost_run_fails_closed_without_live_dispatch(tmp_path, litellm_spy,
                                                              no_network, capsys):
    from geosgan_hive.cli import main

    cfg_path = tmp_path / "cloud.yaml"
    base = yaml.safe_load((REPO_ROOT / "config" / "agents.local.yaml").read_text())
    base["supervisor"]["model"] = M1
    for a in base["agents"]:
        a["model"] = M1
        a["permissions"] = {"allow_tools": False, "allowed_tools": []}
    base["cloud"] = {
        "zero_cost": True, "max_spend_usd": 0.0,
        "profiles": [free_profile()],
    }
    base["checkpoint"] = {"path": str(tmp_path / "cp.db"), "persist_mode": "resumable"}
    cfg_path.write_text(yaml.safe_dump(base))
    rc = main(["hello", "--agents-config", str(cfg_path)])
    assert rc != 0  # production live dispatch is blocked (no trusted verification)
    assert litellm_spy == []  # not a single provider request was made
    err = capsys.readouterr().err
    assert "zero-cost" in err or "blocked" in err or "Error" in err


# ---- point 3: worker-owned lease identity from the execution context ----

def test_gate_uses_context_identity_not_sqlite_owner(tmp_path, litellm_spy, no_network):
    gate, store = make_gate(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="root")
    # the execution context carries the WORKER-OWNED identity
    ctx = RunContext("r", CollaborationPolicy())
    ctx.lease_owner = "w1"
    ctx.lease_generation = 0
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    gw.complete([Message("user", "x")], model=M1, run_id="r", max_tokens=16)
    assert len(litellm_spy) == 1
    with store._lock:
        res = store._conn.execute(
            "SELECT lease_owner, lease_generation FROM quota_reservations").fetchall()
    assert res[0]["lease_owner"] == "w1" and res[0]["lease_generation"] == 0
    # another worker takes over the lease in SQLite; the stale context identity
    # (w1/gen0) must now be REJECTED (no adoption of the new owner)
    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE runs SET lease_owner='w2', lease_generation=7, "
                 "lease_expires_at='2999-01-01T00:00:00.000+00:00' WHERE run_id='r'")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="fencing"):
        gw.complete([Message("user", "y")], model=M1, run_id="r", max_tokens=16)
    assert len(litellm_spy) == 1
    store.close()


def test_run_bound_dispatch_without_context_binding_fails_closed(tmp_path, litellm_spy):
    gate, store = make_gate(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="root")
    ctx = RunContext("r", CollaborationPolicy())  # no lease identity set
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    with pytest.raises(CheckpointError, match="worker-owned|binding|context"):
        gw.complete([Message("user", "x")], model=M1, run_id="r", max_tokens=16)
    assert litellm_spy == []
    # no run context at all + run_id -> also fail closed
    gw2 = make_gateway(store, gate, ctx_lookup=lambda rid: None)
    with pytest.raises(CheckpointError):
        gw2.complete([Message("user", "x")], model=M1, run_id="r", max_tokens=16)
    assert litellm_spy == []
    store.close()


def test_claim_lease_returns_worker_owned_generation(tmp_path):
    store = CheckpointStore(str(tmp_path / "cl.db"), utcnow=lambda: NOW)
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="x")
    store.mark_waiting("r", owner="w1", error="e", counters_json="{}", steps_json="[]",
                       route_json="[]", transcript_json=None, step_count=0,
                       next_attempt_at=(NOW + timedelta(seconds=1)).isoformat(),
                       deadline_at=(NOW + timedelta(hours=1)).isoformat())
    gen = store.claim_lease_gen("r", "w2")
    assert gen == 1  # bumped, returned to the claiming worker
    assert store.claim_lease_gen("r", "w3") is None  # lease held
    store.close()


# ---- point 4: slot before final authorization; revalidation incl. rollover ----

def test_window_rollover_denies_authorization_and_releases_slot(tmp_path, litellm_spy, no_network):
    clock = {"now": NOW}
    gate, store = make_gate(tmp_path, [free_profile()], clock=lambda: clock["now"])
    ctx = RunContext("r", CollaborationPolicy())
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    kw = dict(model=M1, messages=[Message("user", "x").to_openai()], max_tokens=16,
              temperature=0.2, timeout=60.0, num_retries=0)
    dispatch_id = gate.prepare_dispatch(M1, dict(kw), run_id=None)
    assert dispatch_id
    # roll the clock into the next quota window BEFORE final authorization
    clock["now"] = NOW + timedelta(days=1)
    with pytest.raises(CheckpointError, match="authorization denied|window"):
        gw._dispatch_sync(dict(kw), model=M1, run_id=None) if False else gate.finalize_dispatch(
            M1, dispatch_id, run_id=None)
    # through the gateway: slot acquired, authorization denied, slot released
    with pytest.raises(CheckpointError):
        gw.complete([Message("user", "x")], model=M1, max_tokens=16, run_id="r")
    assert litellm_spy == []
    sem = ctx.sync_semaphore()
    assert sem.acquire(blocking=False)  # slot was released, not leaked
    sem.release()
    store.close()


def test_semaphore_acquired_before_final_authorization(tmp_path, litellm_spy, no_network):
    """Ordering proof: the request slot is held while final authorization runs
    (mark_dispatched) and released after; finalization happens before invoke."""
    gate, store = make_gate(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="x")
    # single-slot run: the model semaphore has exactly one slot
    ctx = RunContext("r", CollaborationPolicy(max_parallel_agents=1))
    ctx.lease_owner = "w1"  # worker-owned identity in the execution context
    ctx.lease_generation = 0
    order = []
    orig_mark = store.mark_dispatched

    def wrapped_mark(dispatch_id, **kw):
        sem = ctx.sync_semaphore()
        # if the slot were NOT held during authorization, a second acquire succeeds
        free = sem.acquire(blocking=False)
        if free:
            sem.release()
        order.append("slot-free" if free else "slot-held")
        return orig_mark(dispatch_id, **kw)

    store.mark_dispatched = wrapped_mark  # store is not frozen; gate/engine call it
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    gw.complete([Message("user", "x")], model=M1, max_tokens=16, run_id="r")
    assert order == ["slot-held"]  # slot held during authorization
    assert len(litellm_spy) == 1  # and invocation happened after
    store.close()


# ---- point 5: estimate_cost enforcement ----

@pytest.mark.parametrize("cost", [None, -1.0, float("inf"), float("nan"), "0", True])
def test_unknown_or_invalid_cost_rejected(tmp_path, litellm_spy, cost):
    class A(ExactAdapter):
        def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
            return cost
    gate, store = make_gate(tmp_path, [free_profile()], adapter=A())
    gw = make_gateway(store, gate)
    with pytest.raises(CostUnknownError):
        gw.complete([Message("user", "x")], model=M1, max_tokens=16)
    assert litellm_spy == []
    store.close()


def test_potentially_billable_request_rejected_in_zero_cost(tmp_path, litellm_spy):
    class A(ExactAdapter):
        cost = 0.01  # positive estimated cost: potentially billable
    gate, store = make_gate(tmp_path, [free_profile()], adapter=A())
    gw = make_gateway(store, gate)
    with pytest.raises((BudgetExceededError, CostUnknownError, CloudConfigError)):
        gw.complete([Message("user", "x")], model=M1, max_tokens=16)
    assert litellm_spy == []
    store.close()


# ---- point 6: demonstrably conservative token bounds ----

def test_non_conservative_heuristic_prompt_bound_rejected(tmp_path, litellm_spy):
    gate, store = make_gate(tmp_path, [free_profile()], adapter=HeuristicLiveAdapter())
    gw = make_gateway(store, gate)
    with pytest.raises(CostUnknownError, match="conservative|prompt"):
        gw.complete([Message("user", "x" * 100)], model=M1, max_tokens=16)
    assert litellm_spy == []
    store.close()


def test_prompt_accounting_covers_full_payload_with_overhead(tmp_path, litellm_spy):
    gate, store = make_gate(tmp_path, [free_profile()])
    gw = make_gateway(store, gate)
    msg = Message("user", "x" * 1000)
    gw.complete([msg], model=M1, max_tokens=32)
    payload = len(json.dumps([msg.to_openai()], default=str)) + len("[]") + len(M1)
    expected_prompt = payload + max(256, payload // 10)  # conservative protocol overhead
    rows = {r["kind"]: r["used"] for r in store.quota_status(POOL_A)}
    assert rows["tokens"] == expected_prompt + 32
    assert rows["requests"] == 1
    store.close()


# ---- point 7: evidence bound to provider / endpoint ----

def test_evidence_provider_mismatch_rejected(tmp_path):
    prof = free_profile()
    prof["verification_evidence"] = evidence(provider="openai")
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(prof)


def test_evidence_endpoint_mismatch_blocks_dispatch(tmp_path, litellm_spy):
    prof = free_profile()
    prof["verification_evidence"] = evidence(endpoint="https://wrong.example.com")
    gate, store = make_gate(tmp_path, [prof])
    kwargs = dict(model=M1, messages=[Message("user", "x").to_openai()], max_tokens=16,
                  api_base="https://api.google.com")
    with pytest.raises(CloudConfigError, match="endpoint"):
        gate.prepare_dispatch(M1, dict(kwargs))
    assert store.quota_status(POOL_A) == []  # nothing reserved
    # matching endpoint is fine
    kwargs["api_base"] = "https://wrong.example.com"
    assert gate.prepare_dispatch(M1, dict(kwargs))
    store.close()


# ---- point 9: injection cannot silently disable enforcement ----

def test_cannot_replace_zero_cost_gate_with_inactive_one(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    inactive = ZeroCostGate(registry([free_profile()], zero_cost=False), store,
                            adapter=ExactAdapter())
    with pytest.raises(ConfigError, match="zero-cost|disable"):
        gw.set_cloud_gate(inactive)
    with pytest.raises(ConfigError, match="zero-cost|store|durable"):
        gw.set_durable_store(None)
    store.close()


def test_gate_has_no_disable_switch(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    assert gate.enabled() is True
    with pytest.raises(AttributeError):
        gate.enabled = False
    store.close()


# ---- point 10: budgets/retries/cancellation preserved under the gate ----

def test_retry_and_quota_guarantees_preserved(tmp_path, litellm_spy, no_network, monkeypatch):
    gate, store = make_gate(tmp_path, [free_profile()])
    sleeps = []
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.1,
                                               max_delay_seconds=30, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"}, sleep=sleeps.append)
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    attempts = []

    def flaky(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            exc = litellm.exceptions.RateLimitError("slow", llm_provider="gemini", model=M1)
            exc.response = SimpleNamespace(headers={"retry-after": "5"})
            raise exc
        return fake_response()

    monkeypatch.setattr(litellm, "completion", flaky)
    out = gw.complete([Message("user", "x")], model=M1, max_tokens=16)
    assert out == "ok" and len(attempts) == 2 and sleeps and sleeps[0] >= 5.0
    rows = [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"]
    assert rows[0]["used"] == 2  # both physical attempts durably counted
    store.close()


def test_async_two_phase_gate(tmp_path, litellm_spy, no_network):
    import asyncio
    clock = {"now": NOW}
    gate, store = make_gate(tmp_path, [free_profile()], clock=lambda: clock["now"])
    gw = make_gateway(store, gate)

    async def run():
        return await gw.acomplete([Message("user", "a")], model=M1, max_tokens=8)

    assert asyncio.run(run()) == "ok"
    assert len(litellm_spy) == 1
    # rollover between prepare and finalize on the async path
    async def run2():
        kw = dict(model=M1, messages=[Message("user", "b").to_openai()], max_tokens=8,
                  temperature=0.2, timeout=60.0, num_retries=0)
        did = gate.prepare_dispatch(M1, dict(kw))
        clock["now"] = NOW + timedelta(days=1)
        with pytest.raises(CheckpointError):
            gate.finalize_dispatch(M1, did)
        return await gw.acomplete([Message("user", "b")], model=M1, max_tokens=8)

    # the gateway call itself succeeds in-window; direct rollover was denied above
    assert asyncio.run(run2()) == "ok"
    assert len(litellm_spy) == 2
    store.close()


# ---- C4 residual: exact reservation binding (audit point 1) ----

def test_reservation_bound_to_model_digest_tokens_cost_endpoint(tmp_path, litellm_spy, no_network):
    from geosgan_hive.cloud_gate import sanitized_request_digest

    gate, store = make_gate(tmp_path, [free_profile()])
    gw = make_gateway(store, gate)
    gw.complete([Message("user", "bind me")], model=M1, max_tokens=48)
    with store._lock:
        row = store._conn.execute("SELECT * FROM quota_reservations").fetchone()
    assert row["model"] == M1
    assert row["request_sha256"] == sanitized_request_digest(litellm_spy[0])
    assert row["max_output_tokens"] == 48 and row["prompt_tokens"] > 0
    assert row["cost_usd"] == 0.0 and row["endpoint"] is None  # gemini has no default base
    # credentials are never persisted anywhere in the database file
    raw = Path(store.path).read_bytes()
    assert b"sk-test" not in raw
    store.close()


def test_payload_change_before_finalization_rejected(tmp_path, litellm_spy, no_network):

    gate, store = make_gate(tmp_path, [free_profile()])
    kw = dict(model=M1, messages=[Message("user", "original").to_openai()],
              max_tokens=16, temperature=0.2, timeout=60.0, num_retries=0)
    did = gate.prepare_dispatch(M1, dict(kw))
    tampered = dict(kw, messages=[Message("user", "TAMPERED").to_openai()])
    with pytest.raises(CheckpointError, match="authorization denied"):
        gate.finalize_dispatch(M1, did, kwargs=tampered)
    with store._lock:
        row = store._conn.execute("SELECT status FROM quota_reservations").fetchone()
    assert row["status"] == "reserved"  # not dispatched, not authorized
    # the original payload still finalizes fine
    gate.finalize_dispatch(M1, did, kwargs=kw)
    store.close()


# ---- C4 residual: _reserve_physical uses worker-owned identity (point 2) ----

def test_reserve_physical_uses_context_identity_not_sqlite(tmp_path, litellm_spy):
    gate, store = make_gate(tmp_path, [free_profile()])
    store.create_run(run_id="r", owner="w1", task="t", agent="s", model=M1, root_task_id="x")
    ctx = RunContext("r", CollaborationPolicy())
    ctx.lease_owner = "w1"
    ctx.lease_generation = 0
    gw = make_gateway(store, gate, ctx_lookup=lambda rid: ctx if rid == "r" else None)
    gw.complete([Message("user", "x")], model=M1, run_id="r", max_tokens=16)
    assert store.get("r").physical_reserved == 1
    # takeover in SQLite: the stale context identity must be fenced
    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE runs SET lease_owner='w2', lease_generation=9, "
                 "lease_expires_at='2999-01-01T00:00:00.000+00:00' WHERE run_id='r'")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError):
        gw.complete([Message("user", "y")], model=M1, run_id="r", max_tokens=16)
    assert store.get("r").physical_reserved == 1  # no dispatch, no reservation
    assert len(litellm_spy) == 1
    store.close()


# ---- C4 residual: replacement prevention (point 3) ----

def test_gate_registry_store_replacement_rejected(tmp_path):
    gate, store = make_gate(tmp_path, [free_profile()])
    other_store = CheckpointStore(str(tmp_path / "other.db"))
    other_gate = ZeroCostGate(registry([free_profile()]), other_store,
                              adapter=ExactAdapter())
    gw = ModelGateway(tracker=UsageTracker(), environ={"GEMINI_API_KEY": "sk-test"})
    gw.set_durable_store(store)
    gw.set_cloud_gate(gate)
    with pytest.raises(ConfigError, match="replace"):
        gw.set_cloud_gate(other_gate)          # different gate object
    with pytest.raises(ConfigError, match="registry"):
        gw.set_cloud_registry(registry([free_profile()]))  # different registry
    with pytest.raises(ConfigError, match="swap|store"):
        gw.set_durable_store(other_store)      # different quota store
    assert gw.cloud_gate() is gate  # unchanged
    store.close()
    other_store.close()


def test_factory_fails_closed_on_unsupported_gateway(tmp_path):
    from types import SimpleNamespace

    cfg = _config_with_cloud(tmp_path)
    bogus = SimpleNamespace(tracker=UsageTracker())  # no set_cloud_registry
    with pytest.raises(ConfigError, match="zero-cost|failing closed"):
        build_team(cfg, bogus)


# ---- C4 residual: effective endpoint incl. provider default (point 4) ----

def test_effective_default_endpoint_verified(tmp_path, litellm_spy):
    ollama_model = "ollama/llama3.2"
    prof = dict(provider="ollama", models=[ollama_model], billing="free_tier_verified",
                verified_free=True, overage_billing_disabled=True, billing_verified=True,
                verified_at=(NOW - timedelta(days=1)).isoformat(), account_label="loc",
                account_fingerprint=FP_A, quota_window_start=ANCHOR.isoformat(),
                quota_reset_semantics="provider_documented",
                verification_evidence=evidence(endpoint="http://localhost:11434"),
                quotas=[{"pool": "ollama:acct:" + FP_A, "max_requests": 10,
                         "max_tokens": 10**6, "period": "day"}])
    gate, store = make_gate(tmp_path, [prof])
    # no api_base in kwargs -> provider DEFAULT endpoint must match the evidence
    kw = dict(model=ollama_model, messages=[Message("user", "x").to_openai()], max_tokens=8)
    assert gate.prepare_dispatch(ollama_model, dict(kw))  # default matches
    bad = dict(kw, api_base="https://evil.example")
    with pytest.raises(CloudConfigError, match="endpoint"):
        gate.prepare_dispatch(ollama_model, bad)
    store.close()


def test_zero_cost_cross_provider_fallback_needs_profile(tmp_path, litellm_spy, no_network):
    # fallback candidate on another provider without a zero-cost profile is rejected
    gate, store = make_gate(tmp_path, [free_profile()])
    ok, reason = gate.candidate_billing_ok("mistral/mistral-large-latest")
    assert not ok and "no cloud profile" in reason
    # legacy (gate inactive): legacy cost policy decides, unchanged
    reg_legacy = registry([free_profile()], zero_cost=False)
    gate_legacy = ZeroCostGate(reg_legacy, store, adapter=ExactAdapter())
    assert gate_legacy.candidate_billing_ok("mistral/mistral-large-latest")[0] is True
    store.close()
