from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.errors import (
    AuthenticationError,
    BudgetExceededError,
    FallbackExhaustedError,
    ProviderUnavailableError,
    RateLimitError,
)
from geosgan_hive.fallback import FallbackEngine, FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.health import CircuitState, HealthRegistry
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, EventType, RunContext
from geosgan_hive.usage import UsageTracker

PRIMARY = "ollama/primary"
BACKUP = "ollama/backup"


def ok_response(model):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer from {model}"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_gateway(monkeypatch, *, health=None, default_fallback=None, environ=None,
                 max_attempts=1, sleeps=None):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts, base_delay_seconds=0, max_delay_seconds=0,
            jitter_ratio=0,
        ),
        environ=environ or {},
        sleep=(lambda s: sleeps.append(s)) if sleeps is not None else (lambda s: None),
        health=health,
        default_fallback=default_fallback,
    )
    return gateway, tracker


def attach_ctx(gateway, ctx):
    gateway.set_context_lookup(lambda run_id: ctx)


def policy(**kwargs):
    kwargs.setdefault("mode", FallbackMode.CONTINUITY)
    kwargs.setdefault("fallback_models", [BACKUP])
    return FallbackPolicy(**kwargs)


def fail_on(models, exc_factory):
    """Return a fake litellm.completion that raises for the given models."""
    dispatched = []

    def fake_completion(**kwargs):
        model = kwargs["model"]
        dispatched.append(model)
        if model in models:
            raise exc_factory()
        return ok_response(model)

    fake_completion.dispatched = dispatched
    return fake_completion


def rate_limit():
    return litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")


def test_primary_success_never_falls_back(monkeypatch):
    dispatched = []

    def fake_completion(**kwargs):
        dispatched.append(kwargs["model"])
        return ok_response(kwargs["model"])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    gateway, tracker = make_gateway(monkeypatch)
    ctx = RunContext("r1", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY, agent="a", run_id="r1",
        fallback=policy(),
    )
    assert resp.content == f"answer from {PRIMARY}"
    assert resp.model == PRIMARY
    assert dispatched == [PRIMARY]
    assert ctx.physical_reserved == 1
    assert ctx.events(EventType.FALLBACK_SELECTED) == []


def test_ordered_fallback_on_transient_failure(monkeypatch):
    fake = fail_on({PRIMARY}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, tracker = make_gateway(monkeypatch)
    ctx = RunContext("r2", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY, agent="a", run_id="r2",
        fallback=policy(mode=FallbackMode.HYBRID),
    )
    assert resp.model == BACKUP  # served by the fallback model
    assert fake.dispatched == [PRIMARY, BACKUP]  # ordered chain
    assert ctx.physical_reserved == 2  # one reservation per physical dispatch
    assert tracker.physical_requests("r2") == 2
    selected = ctx.events(EventType.FALLBACK_SELECTED)
    assert len(selected) == 1
    assert selected[0].metadata == {"from_model": PRIMARY, "to_model": BACKUP}
    failed = ctx.events(EventType.PROVIDER_FAILED)
    assert len(failed) == 1 and failed[0].metadata["model"] == PRIMARY
    assert failed[0].metadata["error_class"] == "rate_limit"
    assert failed[0].metadata["fallback_eligible"] is True


def test_chain_exhausted_continuity_raises_fallback_exhausted(monkeypatch):
    fake = fail_on({PRIMARY, BACKUP}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch)
    with pytest.raises(FallbackExhaustedError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY, fallback=policy(),
        )
    assert fake.dispatched == [PRIMARY, BACKUP]


def test_wait_mode_suspends_without_switching_models(monkeypatch):
    fake = fail_on({PRIMARY, BACKUP}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch)
    ctx = RunContext("r3", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(ProviderUnavailableError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY, run_id="r3",
            fallback=policy(mode=FallbackMode.WAIT),
        )
    assert fake.dispatched == [PRIMARY]  # WAIT never tries the fallback model
    suspended = ctx.events(EventType.TASK_SUSPENDED)
    assert len(suspended) == 1 and suspended[0].metadata["primary"] == PRIMARY


def test_hybrid_suspends_when_no_candidate_eligible(monkeypatch):
    fake = fail_on({PRIMARY}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch)
    with pytest.raises(ProviderUnavailableError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY,
            fallback=policy(mode=FallbackMode.HYBRID, fallback_models=[]),
        )


def test_auth_error_never_falls_back_and_never_opens_provider_circuit(monkeypatch):
    def auth():
        return litellm_exc.AuthenticationError("401", llm_provider="ollama", model="primary")

    fake = fail_on({PRIMARY, BACKUP}, auth)
    monkeypatch.setattr(litellm, "completion", fake)
    health = HealthRegistry(failure_threshold=1, cooldown_seconds=60)
    gateway, _ = make_gateway(monkeypatch, health=health)
    ctx = RunContext("r4", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(AuthenticationError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY, run_id="r4", fallback=policy(),
        )
    assert fake.dispatched == [PRIMARY]  # no retry, no fallback on auth
    assert ctx.physical_reserved == 1
    # credential-specific: the PROVIDER circuit must stay closed
    assert health.state(HealthRegistry.provider_key("ollama")) == CircuitState.CLOSED
    assert health.state(HealthRegistry.model_key(PRIMARY)) != CircuitState.OPEN


def test_retry_after_header_is_respected(monkeypatch):
    def limited():
        exc = litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")
        exc.response = SimpleNamespace(headers={"retry-after": "5"})
        return exc

    monkeypatch.setattr(litellm, "completion", fail_on({PRIMARY}, limited))
    sleeps = []
    gateway, _ = make_gateway(monkeypatch, max_attempts=2, sleeps=sleeps)
    gateway.retry_policy = RetryPolicy(
        max_attempts=2, base_delay_seconds=0.1, max_delay_seconds=30.0, jitter_ratio=0
    )
    with pytest.raises(RateLimitError):
        gateway.complete([Message("user", "hi")], model=PRIMARY)  # mode OFF: pure retry
    assert sleeps == [5.0]  # Retry-After wins over the computed backoff


def test_capability_gate_skips_model_with_small_context_window(monkeypatch):
    import geosgan_hive.fallback as fallback_mod
    from geosgan_hive.providers import ModelCapabilities, model_capabilities

    def fake_caps(model):
        if model == BACKUP:
            return ModelCapabilities(context_window=10)
        return model_capabilities(model)

    monkeypatch.setattr(fallback_mod, "model_capabilities", fake_caps)
    monkeypatch.setattr(litellm, "completion", fail_on({PRIMARY}, rate_limit))
    gateway, _ = make_gateway(monkeypatch)
    ctx = RunContext("r5", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    big = "x" * 1000  # estimate ~251 tokens > window of 10
    with pytest.raises(FallbackExhaustedError):
        gateway.complete([Message("user", big)], model=PRIMARY, run_id="r5",
                         fallback=policy())
    rejected = ctx.events(EventType.FALLBACK_REJECTED)
    assert len(rejected) == 1 and rejected[0].metadata["model"] == BACKUP
    assert "context window" in rejected[0].metadata["reason"]


def test_unknown_cost_candidate_rejected_unless_authorized(monkeypatch):
    import geosgan_hive.fallback as fallback_mod
    from geosgan_hive.providers import cost_tier

    def fake_tier(model):
        return "unknown" if model == BACKUP else cost_tier(model)

    monkeypatch.setattr(fallback_mod, "cost_tier", fake_tier)
    monkeypatch.setattr(litellm, "completion", fail_on({PRIMARY}, rate_limit))
    gateway, _ = make_gateway(monkeypatch)
    ctx = RunContext("r6", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(FallbackExhaustedError):
        gateway.complete([Message("user", "hi")], model=PRIMARY, run_id="r6",
                         fallback=policy())
    rejected = ctx.events(EventType.FALLBACK_REJECTED)
    assert any(r.metadata["model"] == BACKUP and "unknown-cost" in r.metadata["reason"]
               for r in rejected)
    # explicit authorization allows the unknown-cost candidate
    fake2 = fail_on({PRIMARY}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake2)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY,
        fallback=policy(allow_unknown_cost_fallback=True),
    )
    assert resp.model == BACKUP


def test_free_to_paid_fallback_requires_explicit_authorization(monkeypatch):
    paid = "openai/gpt-4o-mini"
    monkeypatch.setattr(litellm, "completion", fail_on({PRIMARY}, rate_limit))
    gateway, _ = make_gateway(monkeypatch, environ={"OPENAI_API_KEY": "test-key"})
    ctx = RunContext("r7", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    # cross-provider gate satisfied so ONLY the cost gate can reject
    base = {"fallback_models": [paid], "allowed_fallback_providers": ["openai"]}
    with pytest.raises(FallbackExhaustedError):
        gateway.complete([Message("user", "hi")], model=PRIMARY, run_id="r7",
                         fallback=policy(**base))
    rejected = ctx.events(EventType.FALLBACK_REJECTED)
    assert any(r.metadata["model"] == paid and "allow_paid_fallback" in r.metadata["reason"]
               for r in rejected)
    # with authorization the paid fallback is dispatched (fake ignores the key)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY,
        fallback=policy(**base, allow_paid_fallback=True),
    )
    assert resp.model == paid


def test_cross_provider_fallback_requires_explicit_allowlist(monkeypatch):
    other = "mistral/mistral-small-latest"
    fake = fail_on({PRIMARY}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch, environ={"MISTRAL_API_KEY": "test-key"})
    ctx = RunContext("r8", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(FallbackExhaustedError):
        gateway.complete([Message("user", "hi")], model=PRIMARY, run_id="r8",
                         fallback=policy(fallback_models=[other]))
    assert any(r.metadata["model"] == other for r in ctx.events(EventType.FALLBACK_REJECTED))
    assert fake.dispatched == [PRIMARY]  # never dispatched cross-provider
    # explicit eligibility + paid authorization (free ollama primary -> paid mistral)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY,
        fallback=policy(fallback_models=[other], allowed_fallback_providers=["mistral"],
                        allow_paid_fallback=True),
    )
    assert resp.model == other


def test_same_provider_fallback_needs_no_cross_provider_gate(monkeypatch):
    fake = fail_on({PRIMARY}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch)
    resp = gateway.complete_response(
        [Message("user", "hi")], model=PRIMARY, fallback=policy()  # ollama -> ollama
    )
    assert resp.model == BACKUP


def test_max_fallback_attempts_bounds_the_chain(monkeypatch):
    extra = "ollama/extra"
    fake = fail_on({PRIMARY, BACKUP, extra}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch)
    ctx = RunContext("r9", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(FallbackExhaustedError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY, run_id="r9",
            fallback=policy(fallback_models=[BACKUP, extra], max_fallback_attempts=1),
        )
    assert fake.dispatched == [PRIMARY, BACKUP]  # extra never dispatched
    assert ctx.physical_reserved == 2


def test_physical_budget_exhaustion_is_terminal_and_blocks_fallback(monkeypatch):
    fake = fail_on({PRIMARY, BACKUP}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch, max_attempts=3)
    ctx = RunContext("r10", CollaborationPolicy(max_provider_requests_per_run=1))
    attach_ctx(gateway, ctx)
    with pytest.raises(BudgetExceededError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY, run_id="r10", fallback=policy(),
        )
    assert fake.dispatched == [PRIMARY]  # the retry was blocked BEFORE dispatch
    assert ctx.physical_reserved == 1
    assert len(ctx.events(EventType.BUDGET_EXHAUSTED)) == 1


def test_circuit_opens_after_threshold_then_recovers_and_prefers_primary(monkeypatch):
    clock = {"now": 1000.0}
    health = HealthRegistry(failure_threshold=2, cooldown_seconds=60.0,
                            clock=lambda: clock["now"])
    state = {"primary_ok": False}

    def timeout():
        # TIMEOUT is model-scoped: only the primary model's circuit opens, the
        # same-provider fallback stays available
        return litellm_exc.Timeout("t", model="primary", llm_provider="ollama")

    def fake_completion(**kwargs):
        model = kwargs["model"]
        if model == PRIMARY and not state["primary_ok"]:
            raise timeout()
        return ok_response(model)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    gateway, _ = make_gateway(monkeypatch, health=health)
    ctx = RunContext("r11", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    pol = policy()
    # two failing calls open the primary model's circuit
    for _ in range(2):
        resp = gateway.complete_response([Message("user", "hi")], model=PRIMARY,
                                         run_id="r11", fallback=pol)
        assert resp.model == BACKUP
    mkey = HealthRegistry.model_key(PRIMARY)
    assert health.state(mkey) == CircuitState.OPEN
    assert health.state(HealthRegistry.provider_key("ollama")) == CircuitState.CLOSED
    assert any(e.metadata["key"] == mkey for e in ctx.events(EventType.CIRCUIT_OPENED))
    # while OPEN, the primary is skipped and the fallback serves
    resp = gateway.complete_response([Message("user", "hi")], model=PRIMARY,
                                     run_id="r11", fallback=pol)
    assert resp.model == BACKUP
    # after the cooldown the circuit flips to HALF_OPEN and a single probe runs;
    # probe success closes it and the NEXT call prefers the primary again
    clock["now"] += 61.0
    state["primary_ok"] = True
    resp = gateway.complete_response([Message("user", "hi")], model=PRIMARY,
                                     run_id="r11", fallback=pol)
    assert resp.model == PRIMARY  # probe succeeded -> primary preferred again
    assert health.state(mkey) == CircuitState.CLOSED
    assert len(ctx.events(EventType.RECOVERY_PROBE)) == 1
    resp = gateway.complete_response([Message("user", "hi")], model=PRIMARY,
                                     run_id="r11", fallback=pol)
    assert resp.model == PRIMARY
    assert ctx.events(EventType.FALLBACK_SELECTED)[-1].metadata["to_model"] == BACKUP


def test_failed_probe_reopens_circuit(monkeypatch):
    clock = {"now": 0.0}
    health = HealthRegistry(failure_threshold=1, cooldown_seconds=30.0,
                            clock=lambda: clock["now"])
    monkeypatch.setattr(
        litellm, "completion",
        fail_on({PRIMARY}, lambda: litellm_exc.Timeout(
            "t", model="primary", llm_provider="ollama")),
    )
    gateway, _ = make_gateway(monkeypatch, health=health)
    ctx = RunContext("r12", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    pol = policy()
    gateway.complete_response([Message("user", "hi")], model=PRIMARY, run_id="r12",
                              fallback=pol)  # opens the model circuit (backup serves)
    mkey = HealthRegistry.model_key(PRIMARY)
    assert health.state(mkey) == CircuitState.OPEN
    clock["now"] += 31.0  # cooldown elapsed -> HALF_OPEN
    assert health.state(mkey) == CircuitState.HALF_OPEN
    gateway.complete_response([Message("user", "hi")], model=PRIMARY, run_id="r12",
                              fallback=pol)  # probe fails -> reopened
    assert health.state(mkey) == CircuitState.OPEN


def test_mode_off_is_byte_identical_no_fallback(monkeypatch):
    fake = fail_on({PRIMARY, BACKUP}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    gateway, _ = make_gateway(monkeypatch, max_attempts=2)
    # even with fallback models configured, mode=off never switches
    with pytest.raises(RateLimitError):
        gateway.complete(
            [Message("user", "hi")], model=PRIMARY,
            fallback=FallbackPolicy(mode=FallbackMode.OFF, fallback_models=[BACKUP]),
        )
    assert fake.dispatched == [PRIMARY, PRIMARY]  # only retries on the primary


def test_fallback_candidate_without_credentials_is_skipped(monkeypatch):
    paid_primary = "openai/gpt-4o-mini"
    no_key = "mistral/mistral-small-latest"
    fake = fail_on({paid_primary}, rate_limit)
    monkeypatch.setattr(litellm, "completion", fake)
    # OPENAI key present, MISTRAL key missing
    gateway, _ = make_gateway(monkeypatch, environ={"OPENAI_API_KEY": "test-key"})
    ctx = RunContext("r13", CollaborationPolicy())
    attach_ctx(gateway, ctx)
    with pytest.raises(FallbackExhaustedError):
        gateway.complete([Message("user", "hi")], model=paid_primary, run_id="r13",
                         fallback=policy(fallback_models=[no_key],
                                         allowed_fallback_providers=["mistral"]))
    assert fake.dispatched == [paid_primary]  # the key-less candidate was never dispatched


def test_effective_policy_agent_overrides_global():
    from geosgan_hive.fallback import effective_policy

    global_pol = policy(mode=FallbackMode.CONTINUITY)
    assert effective_policy(None, global_pol) is global_pol
    agent_pol = FallbackPolicy(mode=FallbackMode.OFF)
    assert effective_policy(agent_pol, global_pol) is agent_pol  # explicit off wins


def test_policy_validation_rejects_bad_models_and_providers():
    with pytest.raises(ValueError):
        FallbackPolicy(fallback_models=["not-a-model"])
    with pytest.raises(ValueError):
        FallbackPolicy(allowed_providers=["nope"])
    with pytest.raises(ValueError):
        FallbackPolicy(mode=FallbackMode.WAIT, max_fallback_attempts=-1)


def test_fallback_engine_chain_wait_mode_primary_only():
    engine = FallbackEngine(HealthRegistry())
    assert engine.chain(primary=PRIMARY, policy=policy(mode=FallbackMode.WAIT)) == [PRIMARY]
    assert engine.chain(primary=PRIMARY, policy=policy()) == [PRIMARY, BACKUP]
    p = policy(allowed_providers=["openai"])
    assert engine.chain(primary=PRIMARY, policy=p) == []
