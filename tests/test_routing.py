"""A3 tests: deterministic policy-first intelligent model routing (offline).

Covers: eligibility filters BEFORE ranking (billing/status/evidence/identity/
privacy/capability/context/tools/health/quota/budgets/cross-provider), all
three routing modes, deterministic tie-breaking, trusted-evidence-only quality
ranking (simulated + operator-reported excluded, forged tiers rejected),
provenance recording (bounded, metadata-only, replay rejected), revalidation
before dispatch, quota integration without oversubscription, CLI, and the
no-network guarantee. Groq/OpenRouter stay blocked."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from geosgan_hive.catalog import catalog_from_dict
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry, TokenEstimate
from geosgan_hive.errors import CheckpointError, ConfigError, QuotaExhaustedError
from geosgan_hive.health import HealthRegistry
from geosgan_hive.quota_engine import QuotaEngine
from geosgan_hive.routing import (
    MAX_CANDIDATES,
    ModelRouter,
    PrivacyLevel,
    RoutingMode,
    RoutingPolicy,
    RoutingWeights,
)

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
ANCHOR = NOW - timedelta(days=1)
FP_A = "a" * 64
MODEL = "ollama/llama3.2"
PINNED = "ollama/llama3.2@sha256:deadbeef"


# ---- fixtures ---------------------------------------------------------------------

def _evidence(collected_days_ago=1, ttl_days=90):
    return {
        "source": "official_docs",
        "evidence_ref": "ev-demo",
        "collected_at": (NOW - timedelta(days=collected_days_ago)).isoformat(),
        "collected_by": "operator-1",
        "ttl_days": ttl_days,
    }


def _provider(pid, *, category="llm_local", billing="local_inference",
              status="active", endpoints=("http://localhost:11434",),
              models=((MODEL, ("chat", "instruction_following"), 128_000, True),),
              evidence=None, docs=("https://example.com/docs",)):
    return {
        "id": pid,
        "name": pid,
        "category": category,
        "base_endpoints": list(endpoints),
        "models": [
            {"model_id": m, "capabilities": list(caps), "context_window": cw,
             "tool_support": tools}
            for (m, caps, cw, tools) in models
        ],
        "capabilities": ["chat"],
        "docs": list(docs),
        "billing": billing,
        "evidence": evidence if evidence is not None else [_evidence()],
        "operational_status": status,
    }


def _catalog(*providers):
    return catalog_from_dict({
        "catalog_version": "1.0.0",
        "generated_at": NOW.isoformat(),
        "providers": list(providers),
    })


def _store(tmp_path, name="cp.db"):
    return CheckpointStore(str(tmp_path / name), utcnow=lambda: NOW)


def _router(tmp_path, policy=None, *, catalog=None, health=None, store=None,
            quota_engine=None, registry=None, name="cp.db"):
    store = store or _store(tmp_path, name)
    return ModelRouter(
        policy or RoutingPolicy(),
        catalog=catalog if catalog is not None else _catalog(_provider("ollama")),
        cloud_registry=registry,
        quota_engine=quota_engine,
        health=health,
        store=store,
        utcnow=lambda: NOW,
    ), store


# ---- policy validation --------------------------------------------------------------

def test_weights_must_sum_to_one_and_be_in_range():
    with pytest.raises(ValidationError):
        RoutingWeights(quality=0.5, reliability=0.25, capability=0.15,
                       quota=0.10, privacy=0.10, speed=0.10)  # sums to 1.1
    with pytest.raises(ValidationError):
        RoutingWeights(quality=1.5)
    w = RoutingWeights()  # defaults are valid
    assert abs(sum((w.quality, w.reliability, w.capability, w.quota,
                    w.privacy, w.speed)) - 1.0) < 1e-9


def test_policy_validates_models_and_providers():
    with pytest.raises(ValidationError):
        RoutingPolicy(allowed_models=("not-a-model",))
    with pytest.raises(ValidationError):
        RoutingPolicy(allowed_providers=("nosuchprovider",))
    with pytest.raises(ValidationError):
        RoutingPolicy(defaults={"researcher": "bad"})
    p = RoutingPolicy(allowed_models=(MODEL,), allowed_providers=("ollama",),
                      defaults={"researcher": MODEL})
    assert p.mode == RoutingMode.FIXED  # automatic routing OFF by default


# ---- eligibility filters before ranking ----------------------------------------------

def test_unknown_billing_paid_and_unverified_fail_closed(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_cloud", billing="unknown",
                  endpoints=("https://api.mistral.ai/v1",),
                  models=(("mistral/x", ("chat",), 128_000, True),)),
        _provider("anthropic", category="llm_cloud", billing="paid",
                  endpoints=("https://api.anthropic.com",),
                  models=(("anthropic/c", ("chat",), 200_000, True),)),
        _provider("gemini", category="llm_cloud", billing="free_tier_unverified",
                  endpoints=("https://generativelanguage.googleapis.com",),
                  models=(("gemini/g", ("chat",), 1_000_000, True),)),
    )
    router, _ = _router(tmp_path, catalog=cat)
    assert router.evaluate("mistral/x").reason == "BILLING_UNKNOWN"
    assert router.evaluate("anthropic/c").reason == "BILLING_PAID"
    assert router.evaluate("gemini/g").reason == "BILLING_UNVERIFIED"
    assert router.evaluate(MODEL).eligible


def test_blocked_deprecated_provider_rejected_and_expired_evidence(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_cloud", billing="local_inference",
                  status="blocked", endpoints=("https://api.mistral.ai/v1",),
                  models=(("mistral/b", ("chat",), 128_000, True),)),
        _provider("gemini", category="llm_local", billing="local_inference",
                  status="active", endpoints=("http://localhost:9998",),
                  models=(("gemini/old", ("chat",), 32_000, False),),
                  evidence=[_evidence(collected_days_ago=200, ttl_days=90)]),
    )
    router, _ = _router(tmp_path, catalog=cat)
    assert router.evaluate("mistral/b").reason == "CATALOG_PROVIDER_DISABLED"
    assert router.evaluate("gemini/old").reason == "EVIDENCE_EXPIRED"


def test_missing_evidence_fails_closed_for_non_local_billing(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_cloud", billing="public_api_no_charge",
                  endpoints=("https://api.mistral.ai/v1",), evidence=[],
                  models=(("mistral/n", ("chat",), 128_000, True),)),
    )
    router, _ = _router(tmp_path, catalog=cat)
    assert router.evaluate("mistral/n").reason == "EVIDENCE_MISSING"


def test_local_inference_must_stay_loopback(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_local", billing="local_inference",
                  endpoints=("https://remote.example.com",),
                  models=(("mistral/r", ("chat",), 128_000, True),)),
    )
    router, _ = _router(tmp_path, catalog=cat)
    assert router.evaluate("mistral/r").reason == "ENDPOINT_NOT_LOOPBACK"


def test_model_identity_substitution_rejected(tmp_path):
    router, _ = _router(tmp_path)  # catalog has only ollama/llama3.2
    assert router.evaluate("ollama/other-model").reason == "CATALOG_MODEL_UNKNOWN"
    assert router.evaluate("ollama/llama3.2@sha256:deadbeef").reason == (
        "CATALOG_MODEL_UNKNOWN")  # pinned id is not the catalog identity
    assert router.evaluate("nosuch/x").reason == "MODEL_INVALID"


def test_policy_allow_block_lists(tmp_path):
    router, _ = _router(tmp_path, RoutingPolicy(blocked_models=(MODEL,)))
    assert router.evaluate(MODEL).reason == "POLICY_MODEL_BLOCKED"
    router2, _ = _router(tmp_path, name="b.db",
                         policy=RoutingPolicy(allowed_models=("ollama/other",)))
    assert router2.evaluate(MODEL).reason == "POLICY_MODEL_NOT_ALLOWED"
    router3, _ = _router(tmp_path, name="c.db",
                         policy=RoutingPolicy(allowed_providers=("mistral",)))
    assert router3.evaluate(MODEL).reason == "POLICY_PROVIDER_NOT_ALLOWED"


def test_capability_context_tool_privacy_filters(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("gemini", category="llm_local", billing="local_inference",
                  endpoints=("http://localhost:9997",),
                  models=(("gemini/small", ("chat",), 8_000, False),)),
    )
    router, _ = _router(tmp_path, catalog=cat)
    assert router.evaluate("gemini/small", required_capabilities=("vision",)).reason \
        == "CAPABILITY_MISSING"
    assert router.evaluate("gemini/small", context_tokens=9_000).reason \
        == "CONTEXT_WINDOW_TOO_SMALL"
    assert router.evaluate("gemini/small", needs_tools=True).reason \
        == "TOOL_SUPPORT_MISSING"
    assert router.evaluate(MODEL, required_capabilities=("chat",)).eligible
    # privacy=local_only still allows ollama; a cloud-category entry is rejected
    cat2 = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_cloud", billing="public_api_no_charge",
                  endpoints=("https://api.mistral.ai/v1",),
                  models=(("mistral/l", ("chat",), 128_000, True),)),
    )
    r2, _ = _router(tmp_path, name="p.db",
                    policy=RoutingPolicy(privacy=PrivacyLevel.LOCAL_ONLY), catalog=cat2)
    assert r2.evaluate(MODEL).eligible
    assert r2.evaluate("mistral/l").reason == "PRIVACY_LOCAL_ONLY"


def test_cross_provider_fallback_requires_explicit_authorization(tmp_path):
    cat = _catalog(
        _provider("ollama"),
        _provider("mistral", category="llm_local", billing="local_inference",
                  endpoints=("http://localhost:9995",),
                  models=(("mistral/m", ("chat",), 128_000, True),)),
    )
    pol = RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT,
                        defaults={"*": "mistral/m"})
    router, _ = _router(tmp_path, pol, catalog=cat)
    d = router.select(agent="researcher", configured_model=MODEL)
    assert d.selected_model is None and "CROSS_PROVIDER_NOT_AUTHORIZED" in d.reason
    pol2 = RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT,
                         defaults={"*": "mistral/m"}, cross_provider_fallback=True)
    router2, _ = _router(tmp_path, pol2, name="x.db", catalog=cat)
    d2 = router2.select(agent="researcher", configured_model=MODEL)
    assert d2.selected_model == "mistral/m"


def test_health_cooldown_and_run_budget_filters(tmp_path):
    health = HealthRegistry(failure_threshold=1)
    router, _ = _router(tmp_path, health=health)
    assert router.evaluate(MODEL).eligible
    health.record_failure(f"m:{MODEL}")
    assert router.evaluate(MODEL).reason == "PROVIDER_HEALTH_COOLDOWN"
    health2 = HealthRegistry()
    router2, _ = _router(tmp_path, name="h.db", health=health2)
    assert router2.evaluate(MODEL, calls_used=5, max_calls=5).reason == (
        "RUN_BUDGET_EXHAUSTED")


# ---- quota integration ----------------------------------------------------------------

def _quota_setup(tmp_path, max_requests=2, name="q.db"):
    profile = CloudProfile.model_validate({
        "provider": "ollama", "models": [MODEL], "billing": "free_tier_verified",
        "verified_free": True, "overage_billing_disabled": True,
        "billing_verified": True, "verified_at": ANCHOR.isoformat(),
        "account_label": "main", "account_fingerprint": FP_A,
        "quota_window_start": ANCHOR.isoformat(),
        "quota_reset_semantics": "provider_documented",
        "verification_evidence": {
            "source": "provider_billing_console_export", "evidence_ref": "ev-1",
            "collected_at": ANCHOR.isoformat(), "collected_by": "operator-1",
            "credential_bound": True, "account_fingerprint": FP_A,
            "billing_enforcement": "provider_hard_limit",
        },
        "quotas": [{"pool": "ollama:acct:" + FP_A, "max_requests": max_requests,
                    "max_tokens": 1_000_000, "period": "day"}],
    })
    reg = CloudProfileRegistry(profiles=[profile], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=True)
    store = _store(tmp_path, name)
    eng = QuotaEngine(store, reg, utcnow=lambda: NOW)
    return eng, store, reg


def test_quota_exhaustion_blocks_routing_and_concurrent_reservations_are_atomic(tmp_path):
    eng, store, reg = _quota_setup(tmp_path, max_requests=2)
    router = ModelRouter(RoutingPolicy(), catalog=_catalog(_provider("ollama")),
                         cloud_registry=reg, quota_engine=eng, store=store,
                         utcnow=lambda: NOW)
    assert router.evaluate(MODEL).eligible
    # concurrent reservations: exactly max_requests succeed, atomically
    outcomes = []
    lock = threading.Lock()

    def worker(i):
        try:
            eng.reserve_for_model(MODEL, f"d{i}", TokenEstimate(1, 1), now=NOW)
            res = "ok"
        except QuotaExhaustedError:
            res = "exhausted"
        with lock:
            outcomes.append(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 2 and outcomes.count("exhausted") == 4
    # now the router's pre-filter sees the exhausted window (fail closed)
    ev = router.evaluate(MODEL)
    assert ev.reason == "QUOTA_EXHAUSTED" and not ev.eligible
    # and a decision made earlier no longer revalidates
    eng2, store2, reg2 = _quota_setup(tmp_path, max_requests=5, name="q2.db")
    router2 = ModelRouter(RoutingPolicy(), catalog=_catalog(_provider("ollama")),
                          cloud_registry=reg2, quota_engine=eng2, store=store2,
                          utcnow=lambda: NOW)
    d = router2.select(agent="researcher", configured_model=MODEL)
    assert d.selected_model == MODEL
    ok, _ = router2.revalidate(d, configured_model=MODEL)
    assert ok
    for i in range(5):
        eng2.reserve_for_model(MODEL, f"x{i}", TokenEstimate(1, 1), now=NOW)
    ok, reason = router2.revalidate(d, configured_model=MODEL)
    assert not ok and reason == "QUOTA_EXHAUSTED"


def test_live_dispatch_denied_by_registry_fails_closed(tmp_path):
    # zero-cost registry without a profile for the model -> live dispatch denied
    reg = CloudProfileRegistry(profiles=(), zero_cost=True, max_spend_usd=0.0)
    router = ModelRouter(RoutingPolicy(), catalog=_catalog(_provider("ollama")),
                         cloud_registry=reg, utcnow=lambda: NOW)
    ev = router.evaluate(MODEL)
    assert not ev.eligible and ev.reason.startswith("LIVE_DISPATCH_DENIED")


# ---- modes, ranking, tie-breaking ------------------------------------------------------

def test_fixed_mode_and_provenance_recorded(tmp_path):
    router, store = _router(tmp_path)
    d = router.select(agent="researcher", run_id="r1", configured_model=MODEL,
                      required_capabilities=("chat",))
    assert d.selected_model == MODEL and d.selection_mode == "fixed"
    rows = store.list_routing_decisions(run_id="r1")
    assert len(rows) == 1
    row = rows[0]
    assert row["selection_mode"] == "fixed" and row["selected_model"] == MODEL
    assert row["scoring_method"] == "ranking-1.0" and row["policy_version"]
    # bounded metadata only: candidates carry ids/reasons/scores, never prompts
    cands = json.loads(row["candidates_json"])
    assert all(set(c) <= {"model", "eligible", "reason", "score", "components",
                          "evidence_refs"} for c in cands)
    assert "prompt" not in row["candidates_json"].lower()
    store.close()


def test_fixed_mode_ineligible_configured_model_fails_closed(tmp_path):
    router, _ = _router(tmp_path, RoutingPolicy(blocked_models=(MODEL,)))
    d = router.select(agent="researcher", configured_model=MODEL)
    assert d.selected_model is None and d.selection_mode == "none"
    assert d.reason == "POLICY_MODEL_BLOCKED"


def test_eligible_default_mode_uses_configured_default_only(tmp_path):
    pol = RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT, defaults={"*": MODEL})
    router, _ = _router(tmp_path, pol)
    d = router.select(agent="analyst", configured_model=MODEL)
    assert d.selection_mode == "eligible_default" and d.selected_model == MODEL
    # no default configured -> fail closed
    router2, _ = _router(tmp_path, name="n.db",
                         policy=RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT))
    d2 = router2.select(agent="analyst", configured_model=MODEL)
    assert d2.selected_model is None and d2.reason == "NO_DEFAULT_CONFIGURED"
    # ineligible default -> fail closed with the underlying reason
    router3, _ = _router(tmp_path, name="i.db",
                         policy=RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT,
                                              defaults={"*": MODEL},
                                              blocked_models=(MODEL,)))
    d3 = router3.select(agent="analyst", configured_model=MODEL)
    assert d3.selected_model is None and "DEFAULT_NOT_ELIGIBLE" in d3.reason


def test_evidence_based_mode_unavailable_without_trusted_evidence(tmp_path):
    router, store = _router(
        tmp_path, RoutingPolicy(mode=RoutingMode.EVIDENCE_BASED))
    # simulated + operator-reported evidence exist but must NOT qualify
    from geosgan_hive.benchmarks import SimulatedCaller, run_benchmark
    run_benchmark(store, caller=SimulatedCaller(), model_id=PINNED,
                  model_version="sha256:deadbeef")
    run_benchmark(store, caller=SimulatedCaller(), model_id=PINNED,
                  model_version="sha256:deadbeef", execution_mode="measured",
                  runner_attestation="operator-1")
    d = router.select(agent="analyst", configured_model=PINNED)
    assert d.selected_model is None and d.reason == "NO_TRUSTED_EVIDENCE"
    assert d.selection_mode == "none"
    store.close()


def test_trusted_runtime_evidence_enables_evidence_based_selection(tmp_path):
    cat = _catalog(_provider("ollama", models=(
        (PINNED, ("chat",), 128_000, True),
        ("ollama/qwen3@sha256:cafe", ("chat",), 128_000, True),
    )))
    router, store = _router(
        tmp_path, RoutingPolicy(mode=RoutingMode.EVIDENCE_BASED), catalog=cat)
    # lease-fenced trusted_runtime evidence for PINNED only
    store.create_run(run_id="r-bench", owner="w1", task="t", agent="supervisor",
                     model=PINNED)
    cp = store.get("r-bench")
    n_cases = 60
    cases_json = json.dumps([{"case_id": f"c{i}", "passed": True, "detail": ""}
                             for i in range(n_cases)])
    store.record_runtime_benchmark_evidence(
        run_id="r-bench", lease_owner=cp.lease_owner,
        lease_generation=int(cp.lease_generation),
        evidence_id="ev-trusted-1", suite_version="2.0.0", model_id=PINNED,
        model_version="sha256:deadbeef", aggregate_score=0.97, cases_passed=58,
        cases_total=n_cases, cases_json=cases_json, transcript_sha256="ab" * 32)
    d = router.select(agent="analyst", configured_model=PINNED)
    assert d.selection_mode == "evidence_based"
    assert d.selected_model == PINNED
    assert d.evidence_refs == ("ev-trusted-1",)
    # the other candidate has no trusted evidence -> not rankable
    other = next(c for c in d.candidates if c.model == "ollama/qwen3@sha256:cafe")
    assert other.eligible and not other.evidence_refs
    store.close()


def test_deterministic_tie_break_and_no_fabricated_components(tmp_path):
    cat = _catalog(_provider("ollama", models=(
        ("ollama/b-model", ("chat",), 128_000, True),
        ("ollama/a-model", ("chat",), 128_000, True),
    )))
    pol = RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT,
                        defaults={"*": "ollama/a-model"},
                        allowed_models=("ollama/a-model", "ollama/b-model"))
    router, _ = _router(tmp_path, pol, catalog=cat)
    d1 = router.select(agent="researcher", configured_model="ollama/a-model")
    d2 = router.select(agent="researcher", configured_model="ollama/a-model")
    assert d1.selected_model == d2.selected_model == "ollama/a-model"
    # equal scores -> lexicographically smaller model id wins in evidence mode
    # check tie-break via scoring directly: both eligible, equal components
    e1 = router._score("ollama/a-model", router.evaluate("ollama/a-model"), ())
    e2 = router._score("ollama/b-model", router.evaluate("ollama/b-model"), ())
    assert e1.score == e2.score
    # no speed component is ever fabricated; no quality without trusted evidence
    assert all(k != "speed" for k, _ in e1.components)
    assert all(k != "quality" for k, _ in e1.components)
    assert dict(e1.components)["privacy"] == 1.0  # local inference


def test_max_candidates_bounded(tmp_path):
    models = tuple((f"ollama/m{i:03d}", ("chat",), 128_000, True)
                   for i in range(200))
    cat = _catalog(_provider("ollama", models=models))
    router, _ = _router(tmp_path, catalog=cat)
    assert len(router._candidate_models(None)) <= MAX_CANDIDATES


# ---- provenance replay + privacy --------------------------------------------------------

def test_routing_decision_replay_rejected(tmp_path):
    store = _store(tmp_path)
    kw = dict(decision_id="dup-1", run_id=None, agent="researcher",
              required_capabilities=["chat"], candidates_json="[]",
              selected_model=MODEL, selection_mode="fixed")
    store.record_routing_decision(**kw)
    with pytest.raises(CheckpointError, match="replay"):
        store.record_routing_decision(**kw)
    with pytest.raises(CheckpointError):
        store.record_routing_decision(**{**kw, "decision_id": "x",
                                         "selection_mode": "bogus"})
    store.close()


def test_revalidate_catches_policy_change_before_dispatch(tmp_path):
    health = HealthRegistry(failure_threshold=1)
    router, _ = _router(tmp_path, health=health)
    d = router.select(agent="researcher", configured_model=MODEL)
    assert d.selected_model == MODEL
    ok, _ = router.revalidate(d, configured_model=MODEL)
    assert ok
    health.record_failure(f"m:{MODEL}")  # provider went into cooldown after selection
    ok, reason = router.revalidate(d, configured_model=MODEL)
    assert not ok and reason == "PROVIDER_HEALTH_COOLDOWN"


# ---- gateway integration: router selects, never authorizes ------------------------------

def test_gateway_client_router_fails_closed_without_dispatch(tmp_path, monkeypatch,
                                                             no_network):
    import litellm

    from geosgan_hive.gateway import ModelGateway

    def _boom(*a, **k):
        raise AssertionError("dispatch must not happen when routing selects nothing")

    monkeypatch.setattr(litellm, "completion", _boom)
    router, store = _router(
        tmp_path, RoutingPolicy(blocked_models=(MODEL,)))
    gw = ModelGateway()
    client = gw.bind(MODEL, agent="researcher", router=router)
    with pytest.raises(ConfigError, match="no eligible model"):
        client.complete([{"role": "user", "content": "hi"}], run_id="r9")
    # eligible path: router selects the configured model and dispatch proceeds to
    # the gateway (which would call litellm) - use a stub gateway instead
    calls = []

    class StubGateway:
        def complete(self, messages, *, model, **kw):
            calls.append(model)
            return "ok"

    router2, _ = _router(tmp_path, name="ok.db")
    client2 = gw.bind(MODEL, agent="researcher", router=router2)
    client2.gateway = StubGateway()
    assert client2.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == [MODEL]
    store.close()


def test_router_never_dispatches_and_groq_openrouter_stay_blocked(tmp_path, no_network):
    from pathlib import Path

    from geosgan_hive.catalog import load_catalog

    repo = Path(__file__).resolve().parent.parent
    cat = load_catalog(repo / "config" / "catalog.yaml")
    router, _ = _router(tmp_path, catalog=cat)
    # groq/openrouter are not even valid provider prefixes -> fail closed at the
    # identity gate; the catalog marks them blocked as well
    for m in ("groq/llama-3.3-70b-versatile", "openrouter/auto"):
        ev = router.evaluate(m)
        assert not ev.eligible and ev.reason == "MODEL_INVALID"
    groq = cat.provider("groq")
    openrouter = cat.provider("openrouter")
    assert groq.operational_status.value == "blocked"
    assert openrouter.operational_status.value == "blocked"
    # no dispatch happened: router has no dispatch API at all
    assert not hasattr(router, "dispatch")


# ---- CLI --------------------------------------------------------------------------------

def test_cli_route_and_routing_decisions(tmp_path, capsys, no_network):
    from geosgan_hive.cli import main

    repo = Path(__file__).resolve().parent.parent
    cfg = repo / "config" / "agents.local.yaml"
    rc = main(["--route", "--route-agent", "researcher",
               "--agents-config", str(cfg),
               "--catalog-path", str(repo / "config" / "catalog.yaml"),
               "--checkpoint", str(tmp_path / "cli.db")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mode=fixed" in out and f"selected={MODEL}" in out
    assert "ELIGIBLE" in out
    rc = main(["--routing-decisions",
               "--agents-config", str(cfg),
               "--checkpoint", str(tmp_path / "cli.db")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Routing decisions: 1 record(s)" in out and "mode=fixed" in out


from pathlib import Path  # noqa: E402  (used by CLI test above)
