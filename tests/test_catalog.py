"""A1: informational provider & capability catalog (v0.7.0).

Covers: strict validation, immutability, evidence freshness, the
informational-only guarantee (catalog data NEVER grants runtime permissions),
no hardcoded volatile limits, and loading config/catalog.yaml.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive.catalog import (
    BillingClassification,
    CatalogProvider,
    OperationalStatus,
    ProviderCategory,
    QuotaDimension,
    QuotaDimensionName,
    QuotaScope,
    catalog_from_dict,
    load_catalog,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_YAML = REPO_ROOT / "config" / "catalog.yaml"
NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)


def _provider(**overrides):
    base = {
        "id": "example",
        "name": "Example",
        "category": "public_api",
        "base_endpoints": ["https://api.example.com"],
        "docs": ["https://docs.example.com/"],
        "billing": "public_api_no_charge",
        "operational_status": "evaluation_candidate",
    }
    base.update(overrides)
    return base


def test_shipped_catalog_loads_and_is_wellformed():
    cat = load_catalog(CATALOG_YAML)
    assert cat.catalog_version == "1.0.0"
    ids = [p.id for p in cat.providers]
    assert len(ids) == len(set(ids))
    assert {"ollama", "groq", "openrouter", "wikimedia", "crossref", "github",
            "worldbank", "openmeteo", "searxng"} <= set(ids)
    ollama = cat.provider("ollama")
    assert ollama.category == ProviderCategory.LLM_LOCAL
    assert ollama.operational_status == OperationalStatus.ACTIVE
    assert ollama.billing == BillingClassification.LOCAL_INFERENCE
    # live cloud candidates are recorded as blocked, never enabled
    assert cat.provider("groq").operational_status == OperationalStatus.BLOCKED
    assert cat.provider("openrouter").operational_status == OperationalStatus.BLOCKED
    assert cat.provider("openrouter").billing == BillingClassification.FREE_TIER_UNVERIFIED
    # searxng has no endpoint until an instance is explicitly authorized
    assert cat.provider("searxng").base_endpoints == ()


def test_catalog_is_informational_only_never_grants_permissions():
    # even with an ACTIVE entry and current evidence, the runtime authorization
    # layer (registry/gate) is untouched: no profile -> live dispatch blocked
    from geosgan_hive.agent_config import CloudConfig
    from geosgan_hive.cloud_profiles import CloudProfileRegistry
    cat = catalog_from_dict({
        "catalog_version": "1.0.0",
        "generated_at": NOW.isoformat(),
        "providers": [{
            **_provider(id="groq", category="llm_cloud",
                        base_endpoints=["https://api.groq.com/openai/v1"],
                        models=[{"model_id": "groq/llama-3.3-70b"}],
                        billing="free_tier_unverified",
                        operational_status="active",
                        evidence=[{
                            "source": "official_docs", "evidence_ref": "x",
                            "collected_at": NOW.isoformat(), "collected_by": "operator",
                            "ttl_days": 30}]),
        }],
    })
    assert cat.provider("groq").evidence_current(NOW)
    reg = CloudProfileRegistry.from_config(CloudConfig(zero_cost=True, max_spend_usd=0.0))
    ok, reason = reg.live_dispatch_eligible("groq/llama-3.3-70b", now=NOW)
    assert not ok and "no cloud profile" in reason  # catalog granted nothing


def test_catalog_models_frozen_and_strict():
    cat = load_catalog(CATALOG_YAML)
    with pytest.raises(ValidationError):
        cat.providers = ()  # frozen
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(extra_field=1))  # extra=forbid
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(category="not_a_category"))
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(docs=["http://insecure.example.com"]))


def test_endpoints_https_except_loopback_local():
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(base_endpoints=["http://api.example.com"]))
    # loopback http allowed only for llm_local
    CatalogProvider.model_validate(_provider(
        category="llm_local", base_endpoints=["http://localhost:11434"]))
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(
            category="public_api", base_endpoints=["http://localhost:8080"]))


def test_llm_model_ids_must_match_provider_prefix():
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(
            id="groq", category="llm_cloud",
            base_endpoints=["https://api.groq.com/openai/v1"],
            docs=["https://docs.groq.com/"],
            models=[{"model_id": "ollama/llama3.2"}]))


def test_evidence_freshness_and_expiry():
    p = CatalogProvider.model_validate(_provider(evidence=[{
        "source": "official_docs", "evidence_ref": "e1",
        "collected_at": (NOW - timedelta(days=10)).isoformat(),
        "collected_by": "operator", "ttl_days": 30}]))
    assert p.evidence_current(NOW)
    assert not p.evidence_current(NOW + timedelta(days=25))  # expired
    assert not p.evidence_current(NOW - timedelta(days=11))  # future-dated now
    p2 = CatalogProvider.model_validate(_provider())  # no evidence
    assert not p2.evidence_current(NOW)
    with pytest.raises(ValidationError):
        CatalogProvider.model_validate(_provider(evidence=[{
            "source": "official_docs", "evidence_ref": "e1",
            "collected_at": "2026-11-26T00:00:00",  # naive rejected
            "collected_by": "operator"}]))


def test_quota_dimensions_carry_no_numeric_limits():
    # volatile free-tier numbers must not be hardcoded as guarantees
    assert set(QuotaDimension.model_fields) == {"name", "scope", "documented_in"}
    with pytest.raises(ValidationError):
        QuotaDimension.model_validate({
            "name": "calls_per_day", "scope": "ip",
            "documented_in": "https://open-meteo.com/en/terms",
            "limit": 10000})  # extra=forbid: no numeric limit field
    assert QuotaDimensionName.REQUESTS_PER_DAY.value == "requests_per_day"
    assert QuotaScope.ORGANIZATION.value == "organization"


def test_catalog_duplicate_provider_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        catalog_from_dict({
            "catalog_version": "1.0.0", "generated_at": NOW.isoformat(),
            "providers": [_provider(id="x"), _provider(id="x")],
        })


def test_catalog_version_and_generated_at_validated():
    with pytest.raises(ValidationError):
        catalog_from_dict({"catalog_version": "1.0", "generated_at": NOW.isoformat(),
                           "providers": [_provider()]})
    with pytest.raises(ValidationError):
        catalog_from_dict({"catalog_version": "1.0.0", "generated_at": "2026-11-26T00:00:00",
                           "providers": [_provider()]})  # naive rejected


def test_with_current_evidence_filter():
    cat = load_catalog(CATALOG_YAML)
    fresh = cat.with_current_evidence(NOW)
    assert all(p.evidence_current(NOW) for p in fresh)
    assert cat.provider("ollama") in fresh
    later = cat.with_current_evidence(NOW + timedelta(days=400))
    assert later == ()  # all evidence expired
