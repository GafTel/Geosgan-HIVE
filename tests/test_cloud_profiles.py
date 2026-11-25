"""Phase 4 C2: cloud provider profiles, validated configuration, billing
eligibility and offline adapter contracts.

Failing-first: these tests were written against the spec before
cloud_profiles.py existed. They cover backward compatibility of cost_tier(),
malformed profiles, unknown billing, expired verification, unauthorized
providers, shared quota pool identifiers, credential exclusion, zero-cost
max_spend_usd==0, and strictly-offline fake adapters."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import (
    AgentSpec,
    CloudConfig,
    HiveConfig,
    SupervisorSpec,
    apply_env_overrides,
)
from geosgan_hive.cloud_profiles import (
    AdapterCapabilities,
    BillingModel,
    CloudProfile,
    CloudProfileRegistry,
    FakeCloudAdapter,
    QuotaPeriod,
    QuotaSpec,
    TokenEstimate,
    TokenEstimationMode,
    VerificationEvidence,
    bounded_token_estimate,
)
from geosgan_hive.errors import (
    CloudConfigError,
    CloudOfflineError,
    ConfigError,
    CostUnknownError,
    UnknownProviderError,
)
from geosgan_hive.providers import cost_tier

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def free_profile(**overrides) -> dict:
    base = dict(
        provider="gemini",
        models=["gemini/gemini-2.0-flash"],
        billing="free_tier_verified",
        verified_free=True,
        overage_billing_disabled=True,
        billing_verified=True,
        verified_at=(NOW - timedelta(days=1)).isoformat(),
        account_label="main",
        account_fingerprint="a" * 64,
        quota_window_start=(NOW - timedelta(days=1)).isoformat(),
        verification_evidence={
            "source": "provider_billing_console_export",
            "evidence_ref": "ev-2026-01-14",
            "collected_at": (NOW - timedelta(days=1)).isoformat(),
            "collected_by": "operator-1",
            "credential_bound": True,
            "account_fingerprint": "a" * 64,
        },
        quotas=[{"pool": "gemini:acct:" + "a" * 64, "max_requests": 1500,
                 "max_tokens": 1_000_000, "period": "day"}],
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


# ---- backward compatibility: cost_tier() semantics unchanged ----

def test_cost_tier_legacy_semantics_preserved():
    assert cost_tier("ollama/llama3.2") == "free"
    assert cost_tier("openai/gpt-4o-mini") == "paid"
    assert cost_tier("mistral/mistral-large-latest") == "paid"
    assert cost_tier("gemini/gemini-2.0-flash") == "paid"
    assert cost_tier("anthropic/claude-3-7-sonnet-latest") == "paid"
    with pytest.raises(UnknownProviderError):
        cost_tier("evil/whatever")


# ---- malformed profiles are rejected ----

@pytest.mark.parametrize("bad", [
    {**free_profile(), "api_key": "sk-secret"},                 # credential field
    {**free_profile(), "account_id": "acct-12345"},             # raw account id
    {**free_profile(), "models": ["openai/gpt-4o"]},            # wrong prefix
    {**free_profile(), "provider": "evil"},                     # unknown provider
    {**free_profile(), "quotas": [{"pool": "Bad Pool!", "max_requests": 5}]},
    {**free_profile(), "quotas": [{"pool": "gemini:acct:" + "a" * 64,
                                   "max_requests": -1}]},
    {**free_profile(), "billing": "free_tier_verified",
     "overage_billing_disabled": False},                        # attestation incomplete
    {**free_profile(), "billing": "free_tier_verified", "verified_at": None},
    {**free_profile(), "account_fingerprint": "not-hex"},
    {**free_profile(), "billing": "sometimes_free"},            # bad enum
])
def test_malformed_profiles_rejected(bad):
    with pytest.raises((ValidationError, CloudConfigError)):
        CloudProfile.model_validate(bad)


def test_free_tier_verified_requires_both_attestations_and_timestamp():
    p = CloudProfile.model_validate(free_profile())
    assert p.billing == BillingModel.FREE_TIER_VERIFIED
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(verified_free=False))
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(verified_at=None))


# ---- billing eligibility in zero-cost mode ----

def test_verified_free_profile_is_eligible_when_verification_current():
    reg = registry([free_profile()], zero_cost=True, trusted=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is True, reason
    # production default (trusted binding not implemented): blocked
    reg0 = registry([free_profile()], zero_cost=True)
    ok, reason = reg0.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "blocked" in reason.lower()


def test_unknown_billing_blocks_zero_cost_live_dispatch():
    p = free_profile(billing="unknown", verified_free=False,
                     overage_billing_disabled=False, verified_at=None)
    reg = registry([p], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "unknown" in reason.lower()


def test_paid_profile_blocks_zero_cost_live_dispatch():
    p = free_profile(billing="paid", verified_free=False,
                     overage_billing_disabled=False, verified_at=None)
    reg = registry([p], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "paid" in reason.lower()


def test_expired_verification_blocks_zero_cost():
    p = free_profile(verified_at=(NOW - timedelta(days=31)).isoformat(),
                     verification_ttl_days=30)
    reg = registry([p], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "expired" in reason.lower()


def test_unprofiled_model_blocked_in_zero_cost_but_allowed_in_legacy_mode():
    reg = registry([free_profile()], zero_cost=True)
    ok, _ = reg.live_dispatch_eligible("mistral/mistral-small-latest", now=NOW)
    assert ok is False  # no profile -> unknown billing -> blocked
    legacy = registry([], zero_cost=False)  # legacy mode: cost_tier semantics only
    ok, _ = legacy.live_dispatch_eligible("mistral/mistral-small-latest", now=NOW)
    assert ok is True
    ok, _ = legacy.live_dispatch_eligible("ollama/llama3.2", now=NOW)
    assert ok is True


def test_disabled_profile_is_not_eligible():
    reg = registry([free_profile(enabled=False)], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "disabled" in reason.lower()


# ---- unauthorized providers ----

def test_unknown_provider_prefix_rejected_in_profile():
    with pytest.raises((ValidationError, CloudConfigError)):
        CloudProfile.model_validate(free_profile(provider="notaprovider"))


def test_registry_allowed_providers_gate():
    with pytest.raises(CloudConfigError):
        registry([free_profile()], zero_cost=True, allowed_providers=["openai"])
    reg = registry([free_profile()], zero_cost=True, allowed_providers=["gemini"])
    assert reg.profile_for("gemini/gemini-2.0-flash") is not None


# ---- shared quota pool identifiers across models/accounts ----

def test_shared_quota_pool_identifier_across_models_of_one_account():
    p = free_profile(models=["gemini/gemini-2.0-flash", "gemini/gemini-2.0-flash-lite"])
    reg = registry([p], zero_cost=True)
    pool_a = reg.quota_pool("gemini/gemini-2.0-flash")
    pool_b = reg.quota_pool("gemini/gemini-2.0-flash-lite")
    assert pool_a is not None and pool_a == pool_b
    assert pool_a.startswith("gemini:acct:")
    assert "a" * 64 in pool_a  # fingerprint, not a raw account id


def test_different_accounts_get_different_pools():
    p1 = free_profile(account_label="main", account_fingerprint="a" * 64)
    p2 = free_profile(account_label="other", account_fingerprint="b" * 64,
                      models=["gemini/gemini-2.0-flash-alt"])
    p2["quotas"] = [{"pool": "gemini:acct:" + "b" * 64, "max_requests": 100}]
    p2["verification_evidence"] = dict(p2["verification_evidence"],
                                        account_fingerprint="b" * 64)
    reg = registry([p1, p2], zero_cost=True)
    assert reg.quota_pool("gemini/gemini-2.0-flash") != reg.quota_pool(
        "gemini/gemini-2.0-flash-alt")


def test_quota_spec_periods():
    for period in ("day", "week", "month"):
        q = QuotaSpec(pool="gemini:acct:" + "a" * 64, max_requests=10, period=period)
        assert q.period == QuotaPeriod(period)
    with pytest.raises(ValidationError):
        QuotaSpec(pool="gemini:acct:" + "a" * 64)  # quota without any limit


# ---- config schema + zero-cost max_spend_usd rule ----

def test_zero_cost_mode_requires_max_spend_usd_strictly_zero():
    with pytest.raises(ValidationError):
        CloudConfig(zero_cost=True, max_spend_usd=5.0,
                    profiles=[free_profile()])
    cfg = CloudConfig(zero_cost=True, max_spend_usd=0.0, profiles=[free_profile()])
    assert cfg.max_spend_usd == 0.0
    with pytest.raises(ValidationError):
        CloudConfig(zero_cost=True, max_spend_usd=None, profiles=[free_profile()])
    cfg2 = CloudConfig(zero_cost=False, max_spend_usd=10.0, profiles=[])
    assert cfg2.max_spend_usd == 10.0


def test_hive_config_accepts_optional_cloud_block():
    base = dict(
        supervisor=SupervisorSpec(name="supervisor", description="d",
                                  system_prompt="s", model="ollama/s"),
        agents=[AgentSpec(name="a", description="d", system_prompt="s",
                          model="ollama/a")],
    )
    assert HiveConfig(**base).cloud is None  # absent -> legacy, unchanged
    cfg = HiveConfig(**base, cloud={"zero_cost": True, "max_spend_usd": 0.0,
                                    "profiles": [free_profile()]})
    assert cfg.cloud is not None and cfg.cloud.zero_cost is True
    assert cfg.cloud.profiles[0].billing == BillingModel.FREE_TIER_VERIFIED


def test_hive_cloud_env_overrides():
    out = apply_env_overrides({}, {"HIVE_CLOUD_ZERO_COST": "true",
                                    "HIVE_CLOUD_MAX_SPEND_USD": "0"})
    assert out["cloud"]["zero_cost"] is True
    assert out["cloud"]["max_spend_usd"] == "0"


# ---- offline adapter contracts (strictly offline, no network) ----

def test_fake_adapter_is_strictly_offline(monkeypatch):
    import socket

    def no_network(*a, **kw):
        raise AssertionError("network access attempted from fake adapter")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    adapter = FakeCloudAdapter()
    assert adapter.allow_network is False
    assert adapter.estimate_cost(model="fake/mini", prompt_tokens=10,
                                 max_output_tokens=100) == 0.0
    assert adapter.estimate_cost(model="openai/gpt-4o-mini", prompt_tokens=10,
                                 max_output_tokens=100) is None  # unknown -> None
    assert adapter.quota_pool("fake/mini") == "fake:pool"
    est = adapter.estimate_tokens("fake/mini", prompt_chars=400, max_output_tokens=128)
    assert est is not None and est.max_output_tokens == 128 and est.prompt_tokens > 0
    with pytest.raises(CloudOfflineError):
        adapter.dispatch("fake/mini", [])  # offline-only: dispatch must refuse


def test_bounded_token_estimate_fails_closed_on_unknown_or_unbounded():
    adapter = FakeCloudAdapter()
    est = bounded_token_estimate(adapter, "fake/mini", prompt_chars=100,
                                 max_output_tokens=64)
    assert isinstance(est, TokenEstimate) and est.max_output_tokens == 64
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "unknown/model", prompt_chars=100,
                               max_output_tokens=64)
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=100,
                               max_output_tokens=0)  # unbounded output -> fail closed


def test_profile_serialization_never_contains_credentials():
    p = CloudProfile.model_validate(free_profile())
    blob = p.model_dump_json()
    for needle in ("api_key", "sk-", "secret", "acct-12345"):
        assert needle not in blob.lower()


# ================= C2 audit hardening (failing-first) =================

# --- issue 1: verification window ---

def test_naive_verified_at_is_rejected():
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(verified_at="2026-01-14T12:00:00"))


def test_future_verification_timestamp_is_not_current():
    p = CloudProfile.model_validate(
        free_profile(verified_at=(NOW + timedelta(days=1)).isoformat()))
    assert p.verification_current(NOW) is False
    reg = registry([free_profile(verified_at=(NOW + timedelta(days=1)).isoformat())])
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "future" in reason.lower()


def test_verification_window_boundaries_and_naive_now():
    p = CloudProfile.model_validate(free_profile(verified_at=NOW.isoformat()))
    assert p.verification_current(NOW) is True                       # lower bound inclusive
    assert p.verification_current(NOW + timedelta(days=30)) is True  # expiry inclusive
    assert p.verification_current(NOW + timedelta(days=31)) is False
    with pytest.raises(CloudConfigError):
        p.verification_current(datetime(2026, 1, 15, 12, 0, 0))  # naive now -> reject


# --- issue 2: zero-cost max_spend_usd must be exactly 0.0, never None ---

def test_zero_cost_never_allows_unlimited_budget():
    with pytest.raises(ValidationError):
        CloudConfig(zero_cost=True, max_spend_usd=None, profiles=[free_profile()])
    with pytest.raises(ValidationError):
        CloudConfig(zero_cost=True, max_spend_usd=0.01, profiles=[free_profile()])
    assert CloudConfig(zero_cost=True, max_spend_usd=0.0,
                       profiles=[free_profile()]).max_spend_usd == 0.0


# --- issue 3: duplicate models / conflicting profiles ---

def test_duplicate_model_registration_rejected():
    p2 = free_profile(models=["gemini/gemini-2.0-flash", "gemini/other"])
    with pytest.raises(CloudConfigError):
        registry([free_profile(), p2])  # gemini-2.0-flash registered twice


def test_conflicting_profiles_for_same_account_rejected():
    p2 = free_profile(account_fingerprint="b" * 64)  # same provider+label, other fp
    p2["quotas"] = [{"pool": "gemini:acct:" + "b" * 64, "max_requests": 5}]
    p2["verification_evidence"] = dict(p2["verification_evidence"],
                                        account_fingerprint="b" * 64)
    with pytest.raises(CloudConfigError):
        registry([free_profile(), p2])


# --- issue 4: fingerprint + bounded quotas required; pool bound to fingerprint ---

def test_free_tier_requires_fingerprint_and_fingerprint_bound_pools():
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(account_fingerprint=None))
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(
            quotas=[{"pool": "gemini:acct:" + "c" * 64, "max_requests": 10}]))
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(free_profile(
            quotas=[{"pool": "gemini:acct:" + "a" * 64}]))  # no limit at all
    p = CloudProfile.model_validate(free_profile())
    assert all(q.pool == f"gemini:acct:{'a' * 64}" for q in p.quotas)


# --- issue 5: all quota constraints exposed; conflicting shared pools rejected ---

def test_all_quota_constraints_exposed_across_periods():
    p = free_profile(quotas=[
        {"pool": "gemini:acct:" + "a" * 64, "max_requests": 1500, "period": "day"},
        {"pool": "gemini:acct:" + "a" * 64, "max_requests": 10000, "period": "month"},
        {"pool": "gemini:acct:" + "a" * 64, "max_tokens": 1_000_000, "period": "day"},
    ])
    reg = registry([p])
    specs = reg.quotas_for("gemini/gemini-2.0-flash")
    assert len(specs) == 3
    assert {s.period for s in specs} == {QuotaPeriod.DAY, QuotaPeriod.MONTH}
    assert reg.quota_pool("gemini/gemini-2.0-flash") == "gemini:acct:" + "a" * 64


def test_conflicting_shared_pool_definitions_rejected():
    p1 = free_profile()
    p2 = free_profile(models=["gemini/gemini-alt"],
                      quotas=[{"pool": "gemini:acct:" + "a" * 64,
                               "max_requests": 999, "period": "day"}])  # same pool, other limit
    with pytest.raises(CloudConfigError):
        registry([p1, p2])


# --- issue 6: strict adapter estimate validation ---

class UnderReservingAdapter(FakeCloudAdapter):
    is_test_only = True

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        return TokenEstimate(prompt_tokens=1, max_output_tokens=max_output_tokens - 1)


def test_bounded_estimate_rejects_bad_types_and_bounds():
    adapter = FakeCloudAdapter()
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=10, max_output_tokens=0)
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=10, max_output_tokens=True)
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=-5, max_output_tokens=10)
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=10.5, max_output_tokens=10)
    # under-reservation vs the requested output cap must fail closed
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(UnderReservingAdapter(), "fake/mini", prompt_chars=10,
                               max_output_tokens=100)


def test_fake_estimates_never_used_for_live_enforcement():
    adapter = FakeCloudAdapter()
    assert adapter.is_test_only is True
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(adapter, "fake/mini", prompt_chars=100,
                               max_output_tokens=64, live=True)
    # non-live (test) use is fine
    est = bounded_token_estimate(adapter, "fake/mini", prompt_chars=100,
                                 max_output_tokens=64, live=False)
    assert est.max_output_tokens == 64


# --- issue 7: strict HIVE_CLOUD_ZERO_COST parsing ---

def test_hive_cloud_zero_cost_strict_parse():
    for good in ("true", "1", "yes", "on", "FALSE", "0", "no", "off"):
        apply_env_overrides({}, {"HIVE_CLOUD_ZERO_COST": good})
    for bad in ("maybe", "2", "", "enabled"):
        with pytest.raises(ConfigError):
            apply_env_overrides({}, {"HIVE_CLOUD_ZERO_COST": bad})


# --- issue 8: no bypass via direct construction or post-validation mutation ---

def test_registry_validates_even_when_constructed_directly():
    p1, p2 = free_profile(), free_profile(models=["gemini/gemini-2.0-flash"])
    with pytest.raises(CloudConfigError):
        CloudProfileRegistry(profiles=[p1, p2], zero_cost=True)


def test_profiles_and_registry_are_immutable_after_validation():
    from pydantic import ValidationError as PydValidationError

    p = CloudProfile.model_validate(free_profile())
    with pytest.raises(PydValidationError):
        p.billing = BillingModel.PAID  # frozen model
    reg = registry([free_profile()])
    with pytest.raises(AttributeError):
        reg.zero_cost = False
    with pytest.raises(AttributeError):
        reg.profiles = ()  # tuple is immutable anyway; assignment is blocked


# --- issue 9: absent cloud config preserves legacy behavior ---

def test_absent_cloud_config_preserves_legacy():
    out = apply_env_overrides({}, {})
    assert "cloud" not in out
    base = dict(
        supervisor=SupervisorSpec(name="supervisor", description="d",
                                  system_prompt="s", model="ollama/s"),
        agents=[AgentSpec(name="a", description="d", system_prompt="s",
                          model="ollama/a")],
    )
    cfg = HiveConfig(**base)
    assert cfg.cloud is None
    reg = CloudProfileRegistry.from_config(None)
    assert reg.zero_cost is False
    ok, _ = reg.live_dispatch_eligible("openai/gpt-4o-mini", now=NOW)
    assert ok is True  # legacy: known provider allowed, cost_tier semantics untouched
    assert cost_tier("openai/gpt-4o-mini") == "paid"


# --- issue 10: attestation != independently verified billing guarantee ---

def test_operator_attestation_alone_does_not_enable_live_dispatch():
    p = free_profile(billing_verified=False)
    p["verification_evidence"] = None  # attestation only, no trusted evidence
    reg = registry([p], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "evidence" in reason.lower()
    # evidence contract satisfied as well -> allowed
    reg2 = registry([free_profile(billing_verified=True)], zero_cost=True, trusted=True)
    ok, _ = reg2.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is True
    # billing_verified=True WITHOUT evidence is still just an assertion -> blocked
    no_ev = free_profile(billing_verified=True)
    no_ev["verification_evidence"] = None
    reg3 = registry([no_ev], zero_cost=True)
    ok, reason = reg3.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "evidence" in reason.lower()


# ================= C2 final hardening (failing-first) =================

# --- issue 1: deep immutability ---

def test_deep_immutability_nested_mutation_blocked():
    from types import MappingProxyType

    p = CloudProfile.model_validate(free_profile())
    assert isinstance(p.models, tuple) and isinstance(p.quotas, tuple)
    with pytest.raises(AttributeError):
        p.models.append("gemini/evil")  # tuple: no append
    with pytest.raises(AttributeError):
        p.quotas.append(p.quotas[0])
    with pytest.raises(ValidationError):
        p.quotas[0].max_requests = 1  # frozen QuotaSpec
    reg = registry([free_profile()])
    assert isinstance(reg._by_model, MappingProxyType)
    with pytest.raises(TypeError):
        reg._by_model["gemini/evil"] = p  # read-only mapping
    assert isinstance(reg.profiles, tuple)
    # defensive input handling: plain lists/dicts are copied+validated on construction
    reg2 = CloudProfileRegistry(profiles=[free_profile()], zero_cost=True,
                                max_spend_usd=0.0)
    assert isinstance(reg2.profiles, tuple) and isinstance(reg2.profiles[0].models, tuple)


# --- issue 2: zero-cost spend rule enforced inside the registry itself ---

def test_registry_rejects_nonzero_spend_in_zero_cost_direct_construction():
    with pytest.raises(CloudConfigError):
        CloudProfileRegistry(profiles=[], zero_cost=True, max_spend_usd=5.0)
    with pytest.raises(CloudConfigError):
        CloudProfileRegistry(profiles=[], zero_cost=True, max_spend_usd=None)
    reg = CloudProfileRegistry(profiles=[], zero_cost=True, max_spend_usd=0.0)
    assert reg.max_spend_usd == 0.0


# --- issue 3: account-wide quota aggregation by canonical fingerprint ---

def test_account_wide_quota_aggregation_across_models():
    fp = "a" * 64
    p1 = free_profile(models=["gemini/gemini-2.0-flash"],
                      quotas=[{"pool": f"gemini:acct:{fp}", "max_requests": 1500,
                               "period": "day"}])
    p2 = free_profile(models=["gemini/gemini-2.0-flash-lite"],
                      quotas=[{"pool": f"gemini:acct:{fp}", "max_tokens": 1_000_000,
                               "period": "day"},
                              {"pool": f"gemini:acct:{fp}", "max_requests": 1500,
                               "period": "day"}])  # duplicate of p1's constraint
    reg = registry([p1, p2], zero_cost=True)
    agg = reg.account_quotas("gemini", fp)
    # deduplicated union of ALL account constraints (day requests + day tokens)
    assert len(agg) == 2
    # every model of the account sees the account-wide limits (no bypass)
    for m in ("gemini/gemini-2.0-flash", "gemini/gemini-2.0-flash-lite"):
        assert {q.key() for q in reg.quotas_for(m)} == {q.key() for q in agg}
    assert reg.account_key_for("gemini/gemini-2.0-flash-lite") == ("gemini", fp)


# --- issue 4: canonical identity conflicts by fingerprint ---

def test_conflicting_account_identity_by_fingerprint_rejected():
    # same fingerprint, different label -> conflicting identity
    p2 = free_profile(account_label="other")  # same fingerprint "a"*64
    with pytest.raises(CloudConfigError):
        registry([free_profile(), p2])
    # same fingerprint, different billing -> conflict
    p3 = free_profile(billing="paid", verified_free=False,
                      overage_billing_disabled=False, verified_at=None,
                      verification_evidence=None,
                      models=["gemini/gemini-paid"])
    with pytest.raises(CloudConfigError):
        registry([free_profile(), p3])


# --- issue 5: explicit verification evidence contract ---

def test_evidence_contract_required_and_bound_to_account():
    base = free_profile(billing_verified=True)
    # missing evidence -> blocked
    no_ev = dict(base)
    no_ev["verification_evidence"] = None
    reg = registry([no_ev], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "evidence" in reason.lower()
    # evidence bound to a DIFFERENT account fingerprint -> construction error
    bad = dict(base)
    bad["verification_evidence"] = dict(base["verification_evidence"],
                                        account_fingerprint="b" * 64)
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(bad)
    # naive collected_at -> validation error
    bad2 = dict(base)
    bad2["verification_evidence"] = dict(base["verification_evidence"],
                                         collected_at="2026-01-14T12:00:00")
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(bad2)
    # unbound credential -> construction ok but eligibility blocks (live blocked)
    unb = dict(base)
    unb["verification_evidence"] = dict(base["verification_evidence"], credential_bound=False)
    reg2 = registry([unb], zero_cost=True)
    ok, reason = reg2.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "credential" in reason.lower()
    # stale evidence (beyond ttl) -> blocked
    old = dict(base)
    old["verification_evidence"] = dict(base["verification_evidence"],
                                        collected_at=(NOW - timedelta(days=40)).isoformat())
    reg3 = registry([old], zero_cost=True)
    ok, _ = reg3.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False
    # proper evidence + trusted binding hook -> allowed
    reg4 = registry([base], zero_cost=True, trusted=True)
    ok, _ = reg4.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is True
    # proper evidence but NO trusted binding (production C2 state) -> still blocked
    reg5 = registry([base], zero_cost=True)
    ok, reason = reg5.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "blocked" in reason.lower()


# --- issue 6: live token estimation capability flags ---

class LiveStubAdapter(FakeCloudAdapter):
    is_test_only = False
    capabilities = AdapterCapabilities(live_supported=True,
                                       token_estimation=TokenEstimationMode.BOUNDED_HEURISTIC)

    def estimate_tokens(self, model, *, prompt_chars, max_output_tokens):
        import math
        return TokenEstimate(prompt_tokens=max(1, math.ceil(prompt_chars / 4)),
                             max_output_tokens=max_output_tokens)


class NoCapsAdapter(FakeCloudAdapter):
    pass  # no explicit capabilities


def test_live_estimation_requires_explicit_capability_flags():
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(NoCapsAdapter(), "fake/mini", prompt_chars=10,
                               max_output_tokens=64, live=True)  # missing flags
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(FakeCloudAdapter(), "fake/mini", prompt_chars=10,
                               max_output_tokens=64, live=True)  # live_supported=False
    stub = LiveStubAdapter()
    est = bounded_token_estimate(stub, "fake/mini", prompt_chars=100,
                                 max_output_tokens=64, live=True)
    assert est.max_output_tokens == 64 and est.prompt_tokens >= 25  # conservative ceil


def test_verification_evidence_model_valid():
    ev = VerificationEvidence.model_validate({
        "source": "provider_billing_console_export",
        "evidence_ref": "ev-1",
        "collected_at": NOW.isoformat(),
        "collected_by": "operator-1",
        "credential_bound": True,
        "account_fingerprint": "a" * 64,
    })
    assert ev.credential_bound is True


# ================= C2 residual fixes (failing-first) =================

def test_account_quotas_keyed_by_provider_and_fingerprint():
    from types import MappingProxyType

    fp = "a" * 64
    registry([free_profile()], zero_cost=True)
    # same fingerprint under a DIFFERENT provider must not collide
    m = free_profile(provider="mistral", models=["mistral/mistral-small-latest"])
    m["quotas"] = [{"pool": f"mistral:acct:{fp}", "max_requests": 7, "period": "day"}]
    m["verification_evidence"] = dict(m["verification_evidence"], account_fingerprint=fp)
    both = registry([free_profile(), m], zero_cost=True)
    assert isinstance(both._account_quotas, MappingProxyType)
    assert len(both.account_quotas("gemini", fp)) == 1
    assert len(both.account_quotas("mistral", fp)) == 1
    assert both.account_quotas("gemini", fp) != both.account_quotas("mistral", fp)
    with pytest.raises(TypeError):
        both._account_quotas[("gemini", fp)] = ()


def test_defensive_revalidation_of_existing_instances():
    p = CloudProfile.model_validate(free_profile())
    reg = CloudProfileRegistry(profiles=[p], zero_cost=True, max_spend_usd=0.0)
    # instance passed in was re-validated from its dump (still a CloudProfile, tuple fields)
    assert isinstance(reg.profiles[0], CloudProfile)
    assert isinstance(reg.profiles[0].quotas, tuple)


def test_live_dispatch_blocked_until_trusted_binding_implemented():
    reg = registry([free_profile()], zero_cost=True, trusted=False)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is False and "not proof of authenticity" in reason
    reg_t = registry([free_profile()], zero_cost=True, trusted=True)
    ok, _ = reg_t.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is True


def test_adapter_capabilities_strictly_typed():
    with pytest.raises(ValidationError):
        AdapterCapabilities(live_supported="yes", token_estimation="exact")
    with pytest.raises(ValidationError):
        AdapterCapabilities(live_supported=True, token_estimation="maybe")
    with pytest.raises(ValidationError):
        AdapterCapabilities(live_supported=True, token_estimation="exact", extra=1)
    caps = AdapterCapabilities(live_supported=True, token_estimation=TokenEstimationMode.EXACT)
    assert caps.live_supported is True


def test_free_tier_requires_quota_window_start():
    p = free_profile()
    del p["quota_window_start"]
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(p)
    p2 = free_profile(quota_window_start="2026-01-14T00:00:00")  # naive
    with pytest.raises(ValidationError):
        CloudProfile.model_validate(p2)
