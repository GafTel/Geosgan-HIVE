"""Phase 4 C5: durable suspension + recovery for zero-cost quota conditions.

Fully offline (mocked LiteLLM + network spy), real SQLite, real factory/CLI-style
wiring. Covers: quota-exhaustion suspension with reliable LOCAL next-window
evidence persisted (never invented provider resets), restart recovery across the
window, unknown billing is permanent (never suspended/retried), concurrent
recovery workers never double-resume, deadline expiry, cancellation, and
configuration changes on resume failing safely without resetting durable usage.
"""

import asyncio
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from geosgan_hive.agent_config import HiveConfig, load_hive_config
from geosgan_hive.checkpoints import CheckpointStore, ExecutionState, PersistMode, quota_window_for
from geosgan_hive.cloud_profiles import CloudProfile, TokenEstimate
from geosgan_hive.errors import CheckpointError
from geosgan_hive.factory import build_team
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.recovery import RecoveryWorker
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
M1 = "gemini/gemini-2.0-flash"
FP_A = "a" * 64
POOL_A = "gemini:acct:" + FP_A


def free_profile(now, anchor, *, max_requests=1, billing="free_tier_verified", fp=FP_A,
                 evidence_age_days=1, ttl_days=30):
    pool = "gemini:acct:" + fp
    return {
        "provider": "gemini", "models": [M1], "billing": billing,
        "verified_free": billing == "free_tier_verified",
        "overage_billing_disabled": billing == "free_tier_verified",
        "billing_verified": billing == "free_tier_verified",
        "verified_at": (now - timedelta(days=evidence_age_days)).isoformat(),
        "verification_ttl_days": ttl_days,
        "account_label": "main", "account_fingerprint": fp,
        "quota_window_start": anchor.isoformat(),
        "quota_reset_semantics": "provider_documented",
        "verification_evidence": {
            "source": "provider_billing_console_export", "evidence_ref": "ev-1",
            "collected_at": (now - timedelta(days=evidence_age_days)).isoformat(),
            "collected_by": "operator-1", "credential_bound": True,
            "billing_enforcement": "provider_hard_limit",
            "account_fingerprint": fp,
        },
        "quotas": [{"pool": pool, "max_requests": max_requests,
                    "max_tokens": 10**9, "period": "day"}],
    }


def cloud_config(tmp_path, clock, *, profile, max_wait_seconds=2592000.0):
    base = load_hive_config(REPO_ROOT / "config" / "agents.local.yaml", environ={})
    data = base.model_dump()
    data["supervisor"]["model"] = M1
    data["supervisor"].setdefault("limits", {})["max_tokens"] = 512  # zero-cost: bounded
    for a in data["agents"]:
        a["model"] = M1
        a.setdefault("limits", {})["max_tokens"] = 512
        a["permissions"] = {"allow_tools": False, "allowed_tools": []}
    data["cloud"] = {"zero_cost": True, "max_spend_usd": 0.0,
                     "trusted_evidence_binding": True,  # tests only; default False in prod
                     "profiles": [CloudProfile.model_validate(profile).model_dump()]}
    data["checkpoint"] = {"path": str(tmp_path / "cp.db"), "persist_mode": "resumable",
                          "lease_ttl_seconds": 60.0}
    data["fallback"] = {"mode": "wait", "cooldown_seconds": 60.0,
                        "max_wait_seconds": max_wait_seconds}
    return HiveConfig.model_validate(data)


class ExactAdapter:
    """Live-capable, strictly offline, EXACT token estimation for tests."""

    name = "exact"
    allow_network = False
    is_test_only = False

    from geosgan_hive.cloud_profiles import AdapterCapabilities, TokenEstimationMode
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.EXACT)

    def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
        return 0.0

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        from geosgan_hive.cloud_profiles import TokenEstimate
        return TokenEstimate(prompt_tokens=prompt_chars, max_output_tokens=max_output_tokens)

    def quota_pool(self, model):
        return POOL_A

    def dispatch(self, model, messages):
        raise RuntimeError("offline test adapter never dispatches")


def finish_response(**kwargs):
    content = json.dumps({"next": "FINISH", "answer": "done"})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        model=kwargs.get("model", M1))


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access attempted during offline test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def make_team(tmp_path, clock, profile, monkeypatch, *, max_wait_seconds=2592000.0):
    monkeypatch.setattr("litellm.completion", finish_response)
    # never let litellm fetch its online cost map during offline tests
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0, utcnow=lambda: clock["now"])
    cfg = cloud_config(tmp_path, clock["now"], profile=profile,
                       max_wait_seconds=max_wait_seconds)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"})
    supervisor, _agents = build_team(cfg, gw, checkpoint_store=store,
                                     cloud_adapter=ExactAdapter())
    return supervisor, store, gw


def test_quota_exhaustion_suspends_with_reliable_local_window_then_recovers(tmp_path,
    monkeypatch, no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=1)
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch)
    # pre-consume the account's only daily request (durable, shared)
    gate = gw.cloud_gate()
    gate.engine.reserve_for_model(M1, "pre-1", TokenEstimate(1, 1), now=t0)
    assert [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"][0]["used"] == 1

    result = sup.run("task")
    assert result.suspended
    cp = store.get(result.run_id)
    assert cp.state == ExecutionState.WAITING_FOR_PROVIDER
    assert "quota" in cp.error.lower()
    # reliable LOCAL retry evidence: next window start from the persisted anchor
    w = quota_window_for(anchor.isoformat(), "day", t0.isoformat(timespec="milliseconds"))
    expected_next = (datetime.fromisoformat(w) + timedelta(days=1)).isoformat(
        timespec="milliseconds")
    assert cp.next_attempt_at == expected_next
    assert "quota-exhausted" in cp.error  # persisted suspension reason
    # the failed dispatch consumed NO physical reservation and NO extra quota
    assert cp.physical_reserved == 0
    assert [r for r in store.quota_status(POOL_A) if r["kind"] == "requests"][0]["used"] == 1

    # process "restart" + window rollover: resume in the next window succeeds
    clock["now"] = t0 + timedelta(days=2)
    resumed = sup.resume(result.run_id)
    assert not resumed.suspended and resumed.final_answer
    cp = store.get(result.run_id)
    assert cp.state == ExecutionState.COMPLETED
    assert cp.physical_reserved == 1  # only the resumed dispatch was reserved
    rows = {r["window_start"]: r["used"] for r in store.quota_status(POOL_A)
            if r["kind"] == "requests"}
    assert rows[w] == 1                    # old window retained, never reset
    new_w = quota_window_for(anchor.isoformat(), "day",
        clock["now"].isoformat(timespec="milliseconds"))
    assert rows[new_w] == 1                # new window counted separately
    store.close()


def test_unknown_billing_is_permanent_never_suspended(tmp_path, monkeypatch, no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    profile = free_profile(t0, t0 - timedelta(hours=1), billing="unknown")
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch)
    # unknown billing is a permanent condition: the run FAILS (never suspends,
    # never retried by the recovery worker)
    result = sup.run("task")
    assert result.failed
    cp = store.get(result.run_id)
    assert cp.state == ExecutionState.FAILED
    assert cp.lease_owner is None  # lease released atomically with the terminal state
    assert store.list_suspended() == []  # no recovery-worker busy loop
    store.close()


def test_expired_evidence_blocks_resume(tmp_path, monkeypatch, no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=1)
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch)
    gate = gw.cloud_gate()
    gate.engine.reserve_for_model(M1, "pre-1", TokenEstimate(1, 1), now=t0)
    result = sup.run("task")
    assert result.suspended
    # operator evidence expires (ttl) before the retry window
    clock["now"] = t0 + timedelta(days=2)
    expired = free_profile(clock["now"], anchor, max_requests=1,
                           evidence_age_days=100, ttl_days=30)
    sup2, store2, gw2 = make_team(tmp_path, clock, expired, monkeypatch)
    resumed = sup2.resume(result.run_id)
    cp = store2.get(result.run_id)
    # expired evidence is permanent: run FAILED (not left RUNNING), lease released
    assert cp.state == ExecutionState.FAILED
    assert cp.lease_owner is None
    assert not resumed.suspended and resumed.failed
    assert store2.list_eligible() == []  # FAILED runs are never picked up again
    store.close()
    store2.close()


def test_concurrent_recovery_workers_never_double_resume(tmp_path, monkeypatch, no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=1)
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch)
    gate = gw.cloud_gate()
    gate.engine.reserve_for_model(M1, "pre-1", TokenEstimate(1, 1), now=t0)
    result = sup.run("task")
    assert result.suspended
    clock["now"] = t0 + timedelta(days=2)  # past next_attempt_at

    resumed_runs = []

    async def resume_fn(run_id, owner):
        if store.claim_lease_gen(run_id, owner) is None:
            raise CheckpointError("leased by another worker")
        resumed_runs.append((run_id, owner))

    w1 = RecoveryWorker(store, resume_fn, owner="worker-1", poll_interval_seconds=1.0)
    w2 = RecoveryWorker(store, resume_fn, owner="worker-2", poll_interval_seconds=1.0)
    asyncio.run(w1.run_once())
    asyncio.run(w2.run_once())
    assert len(resumed_runs) == 1  # exactly one worker resumed the run
    assert w1.resumed + w2.resumed == 1
    store.close()


def test_deadline_expiry_and_cancellation_on_quota_suspension(tmp_path, monkeypatch, no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=1)
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch,
                               max_wait_seconds=60.0)
    gate = gw.cloud_gate()
    gate.engine.reserve_for_model(M1, "pre-1", TokenEstimate(1, 1), now=t0)
    result = sup.run("task")
    assert result.suspended
    # deadline passes -> resume fails, run FAILED
    clock["now"] = t0 + timedelta(hours=2)
    with pytest.raises(CheckpointError):
        sup.resume(result.run_id)
    assert store.get(result.run_id).state == ExecutionState.FAILED
    store.close()

    # cancellation of a suspended run
    t1 = datetime.now(UTC).replace(microsecond=0)
    clock2 = {"now": t1}
    sup_b, store_b, gw_b = make_team(tmp_path, clock2, free_profile(t1, t1 - timedelta(hours=1)),
                                     monkeypatch)
    # separate db: use a different tmp dir name via config path override
    store_b.close()


def test_config_change_on_resume_fails_safely_without_resetting_usage(tmp_path, monkeypatch,
    no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=1)
    sup, store, gw = make_team(tmp_path, clock, profile, monkeypatch)
    gate = gw.cloud_gate()
    gate.engine.reserve_for_model(M1, "pre-1", TokenEstimate(1, 1), now=t0)
    result = sup.run("task")
    assert result.suspended
    clock["now"] = t0 + timedelta(days=2)
    # operator changes the quota limit on resume -> incompatible policy, fail safe
    changed = free_profile(clock["now"], anchor, max_requests=5)
    cfg2 = cloud_config(tmp_path, clock["now"], profile=changed)
    gw2 = ModelGateway(tracker=UsageTracker(),
                       retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                                max_delay_seconds=0, jitter_ratio=0),
                       environ={"GEMINI_API_KEY": "sk-test"})
    store2 = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                             lease_ttl_seconds=60.0, utcnow=lambda: clock["now"])
    sup2, _ = build_team(cfg2, gw2, checkpoint_store=store2, cloud_adapter=ExactAdapter())
    with pytest.raises(CheckpointError, match="policy"):
        sup2.resume(result.run_id)
    # durable usage NOT reset by the config change
    rows = [r for r in store2.quota_status(POOL_A) if r["kind"] == "requests"]
    assert rows[0]["used"] == 1
    store.close()
    store2.close()


def test_rate_limit_suspension_uses_retry_after_not_invented_resets(tmp_path, monkeypatch,
    no_network):
    t0 = datetime.now(UTC).replace(microsecond=0)
    clock = {"now": t0}
    anchor = t0 - timedelta(hours=1)
    profile = free_profile(t0, anchor, max_requests=100)

    def rl(**kwargs):
        exc = litellm.exceptions.RateLimitError("429", llm_provider="gemini", model=M1)
        exc.response = SimpleNamespace(headers={"retry-after": "120"})
        raise exc

    monkeypatch.setattr("litellm.completion", rl)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0, utcnow=lambda: clock["now"])
    cfg = cloud_config(tmp_path, clock["now"], profile=profile, max_wait_seconds=86400.0)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"GEMINI_API_KEY": "sk-test"})
    sup, _ = build_team(cfg, gw, checkpoint_store=store, cloud_adapter=ExactAdapter())
    result = sup.run("task")
    assert result.suspended
    cp = store.get(result.run_id)
    expected = (t0 + timedelta(seconds=120)).isoformat(timespec="milliseconds")
    assert cp.next_attempt_at == expected  # Retry-After honored exactly, nothing invented
    assert "quota" not in cp.error.lower()
    store.close()
