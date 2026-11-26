"""A7: v0.7.0 production release gate (failing-first).

MANDATORY: in zero-cost (FREE_ONLY) mode, ONLY local inference (ollama,
loopback) may receive live dispatch in v0.7.0. Even an INJECTED evidence vault
containing syntactically valid provider_enforced records must NOT activate
cloud inference through any supported production path (registry oracle, gate,
gateway dispatch, routing, config flags). Legacy behavior OUTSIDE zero-cost
mode is preserved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import litellm
import pytest

from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.cloud_evidence import (
    AUTHENTICATED_SOURCES,
    CloudEvidenceRecord,
    CloudEvidenceVault,
    EvidenceClass,
    OverageBehavior,
    SpendingRestriction,
)
from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry
from geosgan_hive.errors import CloudConfigError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.routing import ModelRouter, RoutingPolicy

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
ANCHOR = NOW - timedelta(days=1)
FP = "a" * 64
GEMINI = "gemini/gemini-2.0-flash"
OLLAMA = "ollama/llama3.2"


def _evidence(fp=FP, enforcement="provider_hard_limit", endpoint=None):
    ev = {"source": "provider_billing_console_export", "evidence_ref": "ev-1",
          "collected_at": ANCHOR.isoformat(), "collected_by": "operator-1",
          "credential_bound": True, "account_fingerprint": fp,
          "billing_enforcement": enforcement}
    if endpoint:
        ev["endpoint"] = endpoint
    return ev


def _profile(provider="gemini", models=(GEMINI,), fp=FP, enforcement="provider_hard_limit",
             endpoint=None):
    return CloudProfile.model_validate({
        "provider": provider, "models": list(models), "billing": "free_tier_verified",
        "enabled": True, "verified_free": True, "overage_billing_disabled": True,
        "billing_verified": True, "verified_at": ANCHOR.isoformat(),
        "account_label": "main", "account_fingerprint": fp,
        "quota_window_start": ANCHOR.isoformat(),
        "quota_reset_semantics": "provider_documented",
        "verification_evidence": _evidence(fp, enforcement, endpoint),
        "quotas": [{"pool": f"{provider}:acct:{fp}", "max_requests": 1000,
                    "max_tokens": 10**9, "period": "day"}],
    })


def _pe_record(**over):
    base = dict(
        evidence_id="pe-injected", provider_id="gemini",
        evidence_class=EvidenceClass.PROVIDER_ENFORCED,
        account_org_label="org-main", credential_fingerprint=FP, model_id=GEMINI,
        endpoint="https://generativelanguage.googleapis.com",
        billing_plan="free", quota_pool="gemini:acct:" + FP,
        spending_restriction=SpendingRestriction.HARD_LIMIT_USD0,
        overage_behavior=OverageBehavior.BLOCK,
        source=AUTHENTICATED_SOURCES[0], collection_method="signed_export_v1",
        docs_url="https://ai.google.dev/gemini-api/docs", policy_version="tos-2026-11",
        collected_at=ANCHOR, ttl_days=30, limitations="injected fixture",
    )
    base.update(over)
    return CloudEvidenceRecord.model_validate(base)


def _zero_cost_registry(tmp_path, *, vault=None, profile=None, trusted=True):
    store = CheckpointStore(str(tmp_path / "g.db"), utcnow=lambda: NOW)
    reg = CloudProfileRegistry(profiles=[profile or _profile()], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=trusted,
                               evidence_vault=vault)
    return reg, store


# ---- 1. injected vault with a syntactically valid provider_enforced record -------

def test_injected_vault_cannot_activate_cloud_inference_in_zero_cost(tmp_path):
    """FAILING-FIRST: previously this returned eligible=True (the injected vault
    record + trusted profile activated gemini). The v0.7.0 release gate must
    deny it regardless of vault contents."""
    reg, store = _zero_cost_registry(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_pe_record())  # syntactically valid, fully bound
    reg2 = CloudProfileRegistry(profiles=[_profile()], zero_cost=True,
                                max_spend_usd=0.0, trusted_evidence_binding=True,
                                evidence_vault=vault)
    ok, why = reg2.live_dispatch_eligible(GEMINI, now=NOW)
    assert not ok, f"release gate must deny non-local cloud inference in zero-cost mode: {why}"
    assert "release gate" in why.lower() or "non-local" in why.lower()
    store.close()


def test_release_gate_independent_of_vault_state(tmp_path):
    """vault wired-but-empty, vault with informational records only, and no
    vault at all: all deny non-local cloud inference in zero-cost mode."""
    for label, vault in (("none", None), ("empty", "empty"), ("docs-only", "docs")):
        reg, store = _zero_cost_registry(tmp_path)
        v = None
        if vault == "empty":
            v = CloudEvidenceVault(store)
        elif vault == "docs":
            v = CloudEvidenceVault(store)
            v.record(_pe_record(evidence_id="d1", evidence_class=EvidenceClass.PUBLIC_DOCS,
                                spending_restriction=SpendingRestriction.NONE,
                                overage_behavior=OverageBehavior.UNKNOWN,
                                credential_fingerprint=None))
        reg2 = CloudProfileRegistry(profiles=[_profile()], zero_cost=True,
                                    max_spend_usd=0.0, trusted_evidence_binding=True,
                                    evidence_vault=v)
        ok, why = reg2.live_dispatch_eligible(GEMINI, now=NOW)
        assert not ok, label
        store.close()


def test_release_gate_preserves_legacy_outside_zero_cost(tmp_path):
    """Outside zero-cost mode, legacy semantics are preserved (known provider
    prefix allowed; cost_tier semantics untouched) - documented behavior."""
    reg = CloudProfileRegistry(profiles=(), zero_cost=False)
    ok, _ = reg.live_dispatch_eligible(GEMINI, now=NOW)
    assert ok is True
    ok, _ = reg.live_dispatch_eligible(OLLAMA, now=NOW)
    assert ok is True


def test_release_gate_allows_local_inference_in_zero_cost(tmp_path):
    reg, store = _zero_cost_registry(
        tmp_path, profile=_profile(provider="ollama", models=(OLLAMA,),
                                    enforcement="local_inference",
                                    endpoint="http://localhost:11434"))
    ok, why = reg.live_dispatch_eligible(OLLAMA, now=NOW)
    assert ok is True, why
    store.close()


# ---- 2. gateway-level production path ------------------------------------------------

def test_gated_dispatch_denies_cloud_model_and_never_calls_provider(tmp_path, monkeypatch,
                                                                     no_network):
    from geosgan_hive.cloud_gate import ZeroCostGate
    from geosgan_hive.cloud_profiles import AdapterCapabilities, TokenEstimationMode
    from geosgan_hive.quota_engine import TokenEstimate

    class OfflineAdapter:
        name = "offline"
        allow_network = False
        is_test_only = False
        capabilities = AdapterCapabilities(live_supported=True,
                                           token_estimation=TokenEstimationMode.EXACT)
        cost = 0.0

        def estimate_cost(self, model, *, prompt_tokens, max_output_tokens):
            return 0.0

        def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
            return TokenEstimate(prompt_tokens=prompt_chars, max_output_tokens=max_output_tokens)

        def quota_pool(self, model):
            return "gemini:acct:" + FP

        def dispatch(self, model, messages):
            raise AssertionError("provider must never be called")

    # reach the gate, not the credential check
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-not-real")
    calls = []
    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: calls.append(kw) or (_ for _ in ()).throw(
                            AssertionError("litellm called")))
    reg, store = _zero_cost_registry(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_pe_record())
    reg = CloudProfileRegistry(profiles=[_profile()], zero_cost=True, max_spend_usd=0.0,
                               trusted_evidence_binding=True, evidence_vault=vault)
    gate = ZeroCostGate(reg, store, adapter=OfflineAdapter(), utcnow=lambda: NOW)
    gw = ModelGateway()
    gw.set_cloud_gate(gate)
    gw.set_durable_store(store)
    from geosgan_hive.messages import Message
    with pytest.raises(CloudConfigError):
        gw.complete([Message(role="user", content="hi")], model=GEMINI, run_id=None)
    assert calls == []  # no provider call happened
    store.close()


# ---- 3. routing / fallback / config vectors -------------------------------------------

def test_router_cannot_route_to_cloud_in_zero_cost(tmp_path, no_network):
    reg, store = _zero_cost_registry(tmp_path)
    router = ModelRouter(RoutingPolicy(mode="eligible_default",
                                        defaults={"*": GEMINI},
                                        cross_provider_fallback=True),
                         cloud_registry=reg, store=store, utcnow=lambda: NOW)
    d = router.select(agent="researcher", configured_model=GEMINI)
    assert d.selected_model is None
    ev = router.evaluate(GEMINI)
    assert not ev.eligible and ev.reason.startswith("LIVE_DISPATCH_DENIED")
    store.close()


def test_config_flag_cannot_bypass_release_gate(tmp_path):
    """cloud.evidence_vault true/false makes no difference: non-local is denied
    in zero-cost mode either way."""
    from geosgan_hive.agent_config import CloudConfig

    for flag in (False, True):
        cloud = CloudConfig.model_validate({
            "zero_cost": True, "max_spend_usd": 0.0, "trusted_evidence_binding": True,
            "evidence_vault": flag, "profiles": [_profile()],
        })
        store = CheckpointStore(str(tmp_path / f"c{flag}.db"), utcnow=lambda: NOW)
        vault = CloudEvidenceVault(store, accept_provider_enforced=True)
        if flag:
            vault.record(_pe_record())
        reg = CloudProfileRegistry.from_config(cloud, evidence_vault=vault if flag else None)
        ok, why = reg.live_dispatch_eligible(GEMINI, now=NOW)
        assert not ok, f"flag={flag}"
        store.close()
