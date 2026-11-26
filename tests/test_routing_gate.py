"""A2/A3 residual acceptance gate: routing integration with the REAL sync/async
dispatch paths (litellm stubbed; no network), revalidation at dispatch boundaries,
permanent-vs-temporary failure distinction, and proof that disabling automatic
routing can never bypass an active zero-cost gate.

Also: ranking must not present policy-derived/binary health indicators as
empirical model-quality measurements (quality component comes ONLY from
trusted_runtime evidence)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import litellm
import pytest

from geosgan_hive.catalog import catalog_from_dict
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.errors import ConfigError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.health import HealthRegistry
from geosgan_hive.messages import Message
from geosgan_hive.routing import ModelRouter, RoutingMode, RoutingPolicy

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
MODEL = "ollama/llama3.2"


def fake_response(content="ok", model=MODEL):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model=model,
    )


def _evidence():
    return {
        "source": "official_docs", "evidence_ref": "ev-demo",
        "collected_at": NOW.isoformat(), "collected_by": "operator-1",
        "ttl_days": 90,
    }


def _catalog():
    return catalog_from_dict({
        "catalog_version": "1.0.0", "generated_at": NOW.isoformat(),
        "providers": [{
            "id": "ollama", "name": "ollama", "category": "llm_local",
            "base_endpoints": ["http://localhost:11434"],
            "models": [{"model_id": MODEL, "capabilities": ["chat"],
                        "context_window": 128_000, "tool_support": True}],
            "capabilities": ["chat"], "docs": ["https://example.com/docs"],
            "billing": "local_inference", "evidence": [_evidence()],
            "operational_status": "active",
        }],
    })


def _router(tmp_path, policy=None, store=None, health=None, name="cp.db"):
    store = store or CheckpointStore(str(tmp_path / name), utcnow=lambda: NOW)
    router = ModelRouter(policy or RoutingPolicy(), catalog=_catalog(),
                         health=health, store=store, utcnow=lambda: NOW)
    return router, store


# ---- sync + async dispatch with the router wired ------------------------------------

def test_sync_dispatch_resolves_and_revalidates_through_router(tmp_path, monkeypatch,
                                                                no_network):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda r: 0.0)
    router, store = _router(tmp_path)
    selects = []
    orig_select = router.select
    orig_revalidate = router.revalidate

    def select_spy(**kw):
        d = orig_select(**kw)
        selects.append(d)
        return d

    router.select = select_spy
    revalidated = []

    def revalidate_spy(d, **kw):
        res = orig_revalidate(d, **kw)
        revalidated.append(res)
        return res

    router.revalidate = revalidate_spy
    gw = ModelGateway()
    client = gw.bind(MODEL, agent="researcher", router=router)
    out = client.complete([Message(role="user", content="hi")], run_id="r-1")
    assert out == "ok"
    assert calls and calls[0]["model"] == MODEL  # routed model actually dispatched
    assert len(selects) == 1 and selects[0].run_id == "r-1"
    assert revalidated == [(True, "ok")]  # revalidation executed before dispatch
    rows = store.list_routing_decisions(run_id="r-1")
    assert len(rows) == 1 and rows[0]["selected_model"] == MODEL
    store.close()


def test_async_dispatch_uses_same_router_path(tmp_path, monkeypatch, no_network):
    import asyncio

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response()

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda r: 0.0)
    router, store = _router(tmp_path)
    gw = ModelGateway()
    client = gw.bind(MODEL, agent="analyst", router=router)

    async def go():
        return await client.acomplete([Message(role="user", content="hi")], run_id="r-a")

    out = asyncio.run(go())
    assert out == "ok" and calls[0]["model"] == MODEL
    assert len(store.list_routing_decisions(run_id="r-a")) == 1
    store.close()


def test_retries_revalidate_per_logical_call_and_never_bypass(tmp_path, monkeypatch,
                                                               no_network):
    """A transient failure is retried inside the gateway on the SAME routed
    model; the zero-cost gate revalidates per physical attempt. The router
    resolves+revalidates once per logical call (selection is not dispatch)."""
    attempts = []

    def fake_completion(**kwargs):
        attempts.append(kwargs["model"])
        if len(attempts) == 1:
            raise litellm.RateLimitError("rl", llm_provider="ollama", model=MODEL)
        return fake_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda r: 0.0)
    router, store = _router(tmp_path)
    gw = ModelGateway()
    client = gw.bind(MODEL, agent="researcher", router=router)
    out = client.complete([Message(role="user", content="hi")])
    assert out == "ok"
    assert attempts == [MODEL, MODEL]  # retry used the same routed model
    assert len(store.list_routing_decisions()) == 1  # one logical routing decision
    store.close()


def test_policy_change_between_selection_and_dispatch_fails_closed(tmp_path,
                                                                   monkeypatch,
                                                                   no_network):
    """Permanent policy rejection (model blocked after the decision) must fail
    closed at dispatch time, NOT silently fall back to another model."""
    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("no dispatch")))
    monkeypatch.setattr(litellm, "completion_cost", lambda r: 0.0)
    router, store = _router(tmp_path)
    gw = ModelGateway()
    gw.bind(MODEL, agent="researcher", router=router)
    d = router.select(agent="researcher", configured_model=MODEL)
    assert d.selected_model == MODEL
    # policy changes now: the model becomes blocked (new frozen policy state)
    router2, _ = _router(tmp_path, RoutingPolicy(blocked_models=(MODEL,)),
                         store=store, name="cp.db")
    ok, reason = router2.revalidate(d, configured_model=MODEL)
    assert not ok and reason == "POLICY_MODEL_BLOCKED"
    client2 = gw.bind(MODEL, agent="researcher", router=router2)
    with pytest.raises(ConfigError, match="no eligible model"):
        client2.complete([Message(role="user", content="hi")])
    store.close()


def test_temporary_quota_exhaustion_is_not_permanent_rejection(tmp_path):
    """Quota exhaustion / health cooldown are TEMPORARY (reason codes), unlike
    permanent policy rejection; the existing WAIT/HYBRID suspension semantics
    are untouched (handled by the orchestrator/gate, not the router)."""
    from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry, TokenEstimate
    from geosgan_hive.quota_engine import QuotaEngine

    anchor = NOW.isoformat()
    fp = "a" * 64
    profile = CloudProfile.model_validate({
        "provider": "ollama", "models": [MODEL], "billing": "free_tier_verified",
        "verified_free": True, "overage_billing_disabled": True,
        "billing_verified": True, "verified_at": anchor,
        "account_label": "main", "account_fingerprint": fp,
        "quota_window_start": anchor, "quota_reset_semantics": "provider_documented",
        "verification_evidence": {
            "source": "provider_billing_console_export", "evidence_ref": "ev-1",
            "collected_at": anchor, "collected_by": "operator-1",
            "credential_bound": True, "account_fingerprint": fp,
            "billing_enforcement": "provider_hard_limit",
        },
        "quotas": [{"pool": "ollama:acct:" + fp, "max_requests": 1,
                    "max_tokens": 1000, "period": "day"}],
    })
    reg = CloudProfileRegistry(profiles=[profile], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=True)
    store = CheckpointStore(str(tmp_path / "q.db"), utcnow=lambda: NOW)
    eng = QuotaEngine(store, reg, utcnow=lambda: NOW)
    router = ModelRouter(RoutingPolicy(), catalog=_catalog(), cloud_registry=reg,
                         quota_engine=eng, store=store, utcnow=lambda: NOW)
    assert router.evaluate(MODEL).eligible
    eng.reserve_for_model(MODEL, "d1", TokenEstimate(1, 1), now=NOW)
    ev = router.evaluate(MODEL)
    assert not ev.eligible and ev.reason == "QUOTA_EXHAUSTED"  # temporary, not policy
    # health cooldown is temporary too
    health = HealthRegistry(failure_threshold=1)
    router_h = ModelRouter(RoutingPolicy(), catalog=_catalog(), health=health,
                           store=store, utcnow=lambda: NOW)
    health.record_failure(f"m:{MODEL}")
    assert router_h.evaluate(MODEL).reason == "PROVIDER_HEALTH_COOLDOWN"
    # after cooldown expires, the candidate is eligible again (no invented reset
    # time is stored by the router - it just re-evaluates against the clock)
    health2 = HealthRegistry(failure_threshold=1, cooldown_seconds=0.0)
    router_h2 = ModelRouter(RoutingPolicy(), catalog=_catalog(), health=health2,
                            store=store, utcnow=lambda: NOW)
    health2.record_failure(f"m:{MODEL}")
    assert router_h2.evaluate(MODEL).eligible
    store.close()


def test_concurrent_routing_does_not_oversubscribe_quota(tmp_path):
    """Many threads may SELECT concurrently; only atomic durable reservations
    count, so account-wide quota is never oversubscribed."""
    import threading

    from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry, TokenEstimate
    from geosgan_hive.errors import QuotaExhaustedError
    from geosgan_hive.quota_engine import QuotaEngine

    anchor = NOW.isoformat()
    fp = "b" * 64
    profile = CloudProfile.model_validate({
        "provider": "ollama", "models": [MODEL], "billing": "free_tier_verified",
        "verified_free": True, "overage_billing_disabled": True,
        "billing_verified": True, "verified_at": anchor,
        "account_label": "main", "account_fingerprint": fp,
        "quota_window_start": anchor, "quota_reset_semantics": "provider_documented",
        "verification_evidence": {
            "source": "provider_billing_console_export", "evidence_ref": "ev-1",
            "collected_at": anchor, "collected_by": "operator-1",
            "credential_bound": True, "account_fingerprint": fp,
            "billing_enforcement": "provider_hard_limit",
        },
        "quotas": [{"pool": "ollama:acct:" + fp, "max_requests": 3,
                    "max_tokens": 100_000, "period": "day"}],
    })
    reg = CloudProfileRegistry(profiles=[profile], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=True)
    store = CheckpointStore(str(tmp_path / "c.db"), utcnow=lambda: NOW)
    eng = QuotaEngine(store, reg, utcnow=lambda: NOW)
    router = ModelRouter(RoutingPolicy(), catalog=_catalog(), cloud_registry=reg,
                         quota_engine=eng, store=store, utcnow=lambda: NOW)
    # 8 concurrent selections (all fine - selection is read-only) ...
    decisions = []
    lock = threading.Lock()

    def sel(i):
        d = router.select(agent=f"a{i}", configured_model=MODEL)
        with lock:
            decisions.append(d)

    threads = [threading.Thread(target=sel, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert all(d.selected_model == MODEL for d in decisions)
    # ... but physical reservations are atomic: exactly 3 succeed
    results = []

    def res(i):
        try:
            eng.reserve_for_model(MODEL, f"cc{i}", TokenEstimate(1, 1), now=NOW)
            r = "ok"
        except QuotaExhaustedError:
            r = "exhausted"
        with lock:
            results.append(r)

    threads = [threading.Thread(target=res, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count("ok") == 3 and results.count("exhausted") == 5
    store.close()


# ---- nested delegation / recovery / resume -------------------------------------------

def test_nested_delegation_records_per_agent_decisions(tmp_path, monkeypatch,
                                                       no_network):
    """Supervisor -> agent delegation: each agent's routed client records its own
    decision with its agent name."""
    import pathlib

    import yaml

    from geosgan_hive.agent_config import HiveConfig
    from geosgan_hive.factory import build_team

    n = {"calls": 0}

    def fake_completion(**kwargs):
        n["calls"] += 1
        if n["calls"] == 1:  # supervisor delegates
            return fake_response(content=json.dumps(
                {"next": "researcher", "reason": "go"}))
        if n["calls"] == 2:  # researcher answers
            return fake_response(content="[researcher] done")
        return fake_response(content=json.dumps(
            {"next": "FINISH", "answer": "team answer"}))

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda r: 0.0)
    pathlib.Path(tmp_path / "catalog.yaml").write_text(yaml.safe_dump({
        "catalog_version": "1.0.0", "generated_at": NOW.isoformat(),
        "providers": [{
            "id": "ollama", "name": "ollama", "category": "llm_local",
            "base_endpoints": ["http://localhost:11434"],
            "models": [{"model_id": MODEL, "capabilities": ["chat"],
                        "context_window": 128_000, "tool_support": True}],
            "capabilities": ["chat"], "docs": ["https://example.com/docs"],
            "billing": "local_inference", "evidence": [_evidence()],
            "operational_status": "active"}],
    }))
    config = HiveConfig.model_validate({
        "supervisor": {"name": "supervisor", "description": "s",
                       "system_prompt": "s", "model": MODEL},
        "agents": [{"name": "researcher", "description": "r", "system_prompt": "r",
                    "model": MODEL,
                    "permissions": {"allow_tools": False, "allowed_tools": []}}],
        "collaboration": {"allow_agent_to_agent": True, "max_delegation_depth": 2,
                          "max_handoffs_per_run": 5, "max_parallel_agents": 2},
        "checkpoint": {"path": str(tmp_path / "cp.db"), "persist_mode": "resumable"},
        "routing": {"mode": "fixed"},
        "catalog_path": str(tmp_path / "catalog.yaml"),
    })
    gw = ModelGateway()
    sup, agents = build_team(config, gw)
    result = sup.run("do research")
    assert result is not None
    store = CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: NOW)
    rows = store.list_routing_decisions()
    agents_seen = {r["agent"] for r in rows}
    assert "supervisor" in agents_seen and "researcher" in agents_seen
    assert all(r["selected_model"] == MODEL for r in rows)
    store.close()


def test_resume_run_records_routing_decision_with_run_id(tmp_path):
    """Recovery/resume path: a routing decision taken during a resumed run carries
    the run id in durable provenance (lease fencing itself is enforced at the
    dispatch boundary by the gate, covered by existing recovery tests)."""
    router, store = _router(tmp_path)
    store.create_run(run_id="run-9", owner="w1", task="t", agent="supervisor",
                     model=MODEL)
    d = router.select(agent="supervisor", run_id="run-9", configured_model=MODEL)
    assert d.selected_model == MODEL
    rows = store.list_routing_decisions(run_id="run-9")
    assert len(rows) == 1 and rows[0]["run_id"] == "run-9"
    store.close()


# ---- routing OFF must never bypass the zero-cost gate --------------------------------

def test_disabling_routing_keeps_zero_cost_gate_active(tmp_path, monkeypatch,
                                                        no_network):
    """A config WITHOUT a routing block (automatic routing off) must still run
    with the zero-cost gate fully active; adding a routing block must not weaken
    it either; and an unknown/unprofiled model is denied at the gate."""
    from geosgan_hive.agent_config import HiveConfig
    from geosgan_hive.factory import build_team

    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("live dispatch")))
    anchor = NOW.isoformat()
    fp = "c" * 64
    cloud = {
        "zero_cost": True, "max_spend_usd": 0.0,
        "trusted_evidence_binding": True,
        "profiles": [{
            "provider": "ollama", "models": [MODEL],
            "billing": "free_tier_verified", "enabled": True,
            "verified_free": True, "overage_billing_disabled": True,
            "billing_verified": True, "verified_at": anchor,
            "account_label": "main", "account_fingerprint": fp,
            "quota_window_start": anchor,
            "quota_reset_semantics": "provider_documented",
            "verification_evidence": {
                "source": "provider_billing_console_export", "evidence_ref": "ev-1",
                "collected_at": anchor, "collected_by": "operator-1",
                "credential_bound": True, "account_fingerprint": fp,
                "billing_enforcement": "local_inference",
                "endpoint": "http://localhost:11434",
            },
            "quotas": [{"pool": "ollama:acct:" + fp, "max_requests": 100,
                        "max_tokens": 1_000_000, "period": "day"}],
        }],
    }
    base = {
        "supervisor": {"name": "supervisor", "description": "s",
                       "system_prompt": "s", "model": MODEL},
        "agents": [{"name": "researcher", "description": "r", "system_prompt": "r",
                    "model": MODEL}],
        "checkpoint": {"path": str(tmp_path / "cp.db"), "persist_mode": "resumable"},
        "cloud": cloud,
    }
    # without routing block
    cfg1 = HiveConfig.model_validate(base)
    assert cfg1.routing is None
    gw1 = ModelGateway()
    build_team(cfg1, gw1)
    assert gw1.cloud_gate() is not None and gw1.cloud_gate().enabled()
    # with a routing block (fixed mode) the gate must STILL be active
    cfg2 = HiveConfig.model_validate({**base, "routing": {"mode": "fixed"}})
    gw2 = ModelGateway()
    build_team(cfg2, gw2)
    assert gw2.cloud_gate() is not None and gw2.cloud_gate().enabled()
    # a model with NO profile is denied live dispatch in zero-cost mode even with
    # the router wired (fail closed at the gate, not just at the router)
    from geosgan_hive.errors import CloudConfigError
    with pytest.raises(CloudConfigError):
        gw2.complete([Message(role="user", content="hi")],
                     model="ollama/not-profiled")  # no run binding: gate denies unprofiled model


# ---- ranking must not present health/privacy as empirical quality ---------------------

def test_quality_component_only_from_trusted_evidence(tmp_path):
    router, store = _router(tmp_path)
    ev = router._score(MODEL, router.evaluate(MODEL), ("chat",))
    comps = dict(ev.components)
    assert "quality" not in comps  # no trusted evidence -> no quality score
    assert comps.get("reliability") == 1.0 and comps.get("privacy") == 1.0
    # these are policy/state-derived, NOT measurements; the decision records the
    # component names so they can never be mistaken for benchmark evidence
    router.select(agent="researcher", configured_model=MODEL,
                  required_capabilities=("chat",))
    row = store.list_routing_decisions()[0]
    cands = json.loads(row["candidates_json"])
    c = next(c for c in cands if c["model"] == MODEL)
    assert "quality" not in c["components"] and c["evidence_refs"] == []
    # evidence_based stays unavailable without trusted evidence, even though
    # health/privacy would rank the candidate
    router_eb, _ = _router(tmp_path, RoutingPolicy(mode=RoutingMode.EVIDENCE_BASED),
                           name="eb.db")
    d2 = router_eb.select(agent="researcher", configured_model=MODEL)
    assert d2.selected_model is None and d2.reason == "NO_TRUSTED_EVIDENCE"
    store.close()


# ---- A6 residual: DOI suffixes containing slashes and punctuation ----------------------

def test_doi_suffix_with_slashes_and_punctuation():
    from pydantic import ValidationError

    from geosgan_hive.api_adapters import CrossrefGetDoiAdapter, CrossrefGetDoiInput
    from geosgan_hive.http_client import _validate_path_encoding

    # legal: suffix with its own slash segments and punctuation
    doi = "10.1234/a/b(c):d;e+f"
    inp = CrossrefGetDoiInput(doi=doi)
    adapter = CrossrefGetDoiAdapter.__new__(CrossrefGetDoiAdapter)
    path, _ = adapter.build_request({"doi": inp.doi})
    assert path.isascii()
    _validate_path_encoding(path)  # transport accepts the canonical form
    assert path.startswith("/works/10.1234/a/b%28c%29%3Ad%3Be%2Bf")
    # illegal: traversal / pre-encoded / controls / missing registrant
    for bad in ("10.1234/../x", "10.1234/a%2fb", "10.1234/a\\b", "10.1234/a\x00b",
                "1234/abc", "10.1234/"):
        with pytest.raises(ValidationError):
            CrossrefGetDoiInput(doi=bad)
