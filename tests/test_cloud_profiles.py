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
    BillingModel,
    CloudProfile,
    CloudProfileRegistry,
    FakeCloudAdapter,
    QuotaPeriod,
    QuotaSpec,
    TokenEstimate,
    bounded_token_estimate,
)
from geosgan_hive.errors import (
    CloudConfigError,
    CloudOfflineError,
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
        verified_at=(NOW - timedelta(days=1)).isoformat(),
        account_label="main",
        account_fingerprint="a" * 64,
        quotas=[{"pool": "gemini:acct:" + "a" * 64, "max_requests": 1500,
                 "max_tokens": 1_000_000, "period": "day"}],
    )
    base.update(overrides)
    return base


def registry(profiles, *, zero_cost=True, **kw) -> CloudProfileRegistry:
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
    reg = registry([free_profile()], zero_cost=True)
    ok, reason = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok is True, reason


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
    reg = registry([p1, p2], zero_cost=True)
    assert reg.quota_pool("gemini/gemini-2.0-flash") != reg.quota_pool(
        "gemini/gemini-2.0-flash-alt")


def test_quota_spec_periods():
    for period in ("day", "week", "month"):
        q = QuotaSpec(pool="gemini:acct:" + "a" * 64, period=period)
        assert q.period == QuotaPeriod(period)


# ---- config schema + zero-cost max_spend_usd rule ----

def test_zero_cost_mode_requires_max_spend_usd_strictly_zero():
    with pytest.raises(ValidationError):
        CloudConfig(zero_cost=True, max_spend_usd=5.0,
                    profiles=[free_profile()])
    cfg = CloudConfig(zero_cost=True, max_spend_usd=0.0, profiles=[free_profile()])
    assert cfg.max_spend_usd == 0.0
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
