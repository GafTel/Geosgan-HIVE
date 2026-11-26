"""A1: typed, versioned, INFORMATIONAL provider & capability catalog (v0.7.0).

The catalog records what providers/models/APIs exist, what their official
documentation says, licensing, billing classification, evidence freshness,
quota dimensions, privacy notes and operational status. It is a data product:

- It NEVER grants runtime permissions. Authorization stays with
  CloudProfileRegistry / ZeroCostGate (cloud_profiles.py / cloud_gate.py);
  this module deliberately exposes no authorize()/enable() API.
- Provider metadata and model-generated recommendations are untrusted input.
- Unknown, expired or unverifiable evidence is recorded as such and must not
  authorize execution at runtime.
- Volatile free-tier limits are NOT hardcoded as guarantees: quota dimensions
  carry only name/scope/documentation reference, never numeric promises.

All models are frozen and extra='forbid'; timestamps must be tz-aware.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderCategory(StrEnum):
    LLM_LOCAL = "llm_local"
    LLM_CLOUD = "llm_cloud"
    PUBLIC_API = "public_api"
    SEARCH_ENGINE = "search_engine"


class BillingClassification(StrEnum):
    """Informational billing class. NEVER sufficient for runtime authorization."""

    LOCAL_INFERENCE = "local_inference"          # local compute, no API charge
    PUBLIC_API_NO_CHARGE = "public_api_no_charge"  # free public API, terms apply
    FREE_TIER_UNVERIFIED = "free_tier_unverified"  # advertised free tier, NOT verified
    PAID = "paid"
    UNKNOWN = "unknown"                          # fail closed at runtime


class OperationalStatus(StrEnum):
    ACTIVE = "active"
    EVALUATION_CANDIDATE = "evaluation_candidate"
    BLOCKED = "blocked"          # e.g. live cloud inference in v0.7.0
    DEPRECATED = "deprecated"


class QuotaDimensionName(StrEnum):
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_DAY = "requests_per_day"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_DAY = "tokens_per_day"
    CALLS_PER_MINUTE = "calls_per_minute"
    CALLS_PER_HOUR = "calls_per_hour"
    CALLS_PER_DAY = "calls_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    NONE_PUBLISHED = "none_published"


class QuotaScope(StrEnum):
    ORGANIZATION = "organization"
    ACCOUNT = "account"
    API_KEY = "api_key"
    IP = "ip"
    INSTANCE = "instance"
    NONE = "none"


_HTTPS_URL = re.compile(
    r"^https://[a-zA-Z0-9.-]{1,253}(:[0-9]{1,5})?(/[a-zA-Z0-9_./:?#&=%~+-]*)?$")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _is_loopback_url(url: str) -> bool:
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    if parts.scheme != "http":
        return False
    return (parts.hostname or "").lower() in _LOOPBACK_HOSTS


class EvidenceRecord(BaseModel):
    """One piece of evidence about a provider (e.g. an official doc snapshot).
    Opaque reference only - never contains secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(pattern=r"^[a-z0-9][a-z0-9_ -]{0,63}$")
    evidence_ref: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    collected_at: datetime
    collected_by: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    ttl_days: int = Field(default=30, ge=1, le=3650)

    @field_validator("collected_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware (naive rejected)")
        return v

    def current(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("evidence.current() requires a tz-aware 'now'")
        return self.collected_at <= now <= self.collected_at + timedelta(days=self.ttl_days)


class QuotaDimension(BaseModel):
    """A quota dimension with its scope and where it is documented. Deliberately
    NO numeric limit field: volatile free-tier numbers are not permanent
    guarantees and must be re-verified at enablement time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: QuotaDimensionName
    scope: QuotaScope
    documented_in: str = Field(pattern=_HTTPS_URL.pattern)


class CatalogModel(BaseModel):
    """One exact model identifier/version as offered by a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    context_window: int | None = Field(default=None, ge=1)
    tool_support: bool = False
    hardware_requirements: str | None = Field(default=None, max_length=200)
    license: str = Field(default="unknown", max_length=200)


class CatalogProvider(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID.pattern)
    name: str = Field(max_length=100)
    category: ProviderCategory
    # empty = no endpoint authorized yet (e.g. searxng until an instance is approved)
    base_endpoints: tuple[str, ...] = Field(default_factory=tuple)
    models: tuple[CatalogModel, ...] = Field(default_factory=tuple)
    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    docs: tuple[str, ...] = Field(min_length=1)
    license: str = Field(default="unknown", max_length=200)
    attribution: str | None = Field(default=None, max_length=300)
    billing: BillingClassification = BillingClassification.UNKNOWN
    evidence: tuple[EvidenceRecord, ...] = Field(default_factory=tuple)
    quota_dimensions: tuple[QuotaDimension, ...] = Field(default_factory=tuple)
    privacy: tuple[str, ...] = Field(default_factory=tuple)
    operational_status: OperationalStatus = OperationalStatus.EVALUATION_CANDIDATE
    benchmark_refs: tuple[str, ...] = Field(default_factory=tuple)
    notes: str = Field(default="", max_length=500)

    @field_validator("base_endpoints")
    @classmethod
    def _endpoints_ok(cls, v: tuple[str, ...], info) -> tuple[str, ...]:
        category = info.data.get("category")
        for url in v:
            if _HTTPS_URL.match(url):
                continue
            if category == ProviderCategory.LLM_LOCAL and _is_loopback_url(url):
                continue
            raise ValueError(
                f"endpoint {url!r}: external endpoints must be https; plain http is "
                "only allowed for loopback local-inference endpoints"
            )
        return v

    @field_validator("docs")
    @classmethod
    def _docs_https(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for url in v:
            if not _HTTPS_URL.match(url):
                raise ValueError(f"documentation reference {url!r} must be an https URL")
        return v

    @model_validator(mode="after")
    def _model_prefix(self) -> CatalogProvider:
        if self.category in (ProviderCategory.LLM_LOCAL, ProviderCategory.LLM_CLOUD):
            for m in self.models:
                if not m.model_id.startswith(f"{self.id}/"):
                    raise ValueError(
                        f"model {m.model_id!r} must be prefixed with the provider id "
                        f"{self.id!r} for LLM categories"
                    )
        return self

    def evidence_current(self, now: datetime) -> bool:
        """True iff at least one evidence record exists and ALL are current."""
        return bool(self.evidence) and all(e.current(now) for e in self.evidence)


class ProviderCatalog(BaseModel):
    """Versioned, immutable, informational catalog. Grants NO permissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    generated_at: datetime
    providers: tuple[CatalogProvider, ...] = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware (naive rejected)")
        return v

    @model_validator(mode="after")
    def _unique_ids(self) -> ProviderCatalog:
        ids = [p.id for p in self.providers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate provider ids: {sorted(ids)}")
        return self

    def provider(self, provider_id: str) -> CatalogProvider | None:
        for p in self.providers:
            if p.id == provider_id:
                return p
        return None

    def by_category(self, category: ProviderCategory) -> tuple[CatalogProvider, ...]:
        return tuple(p for p in self.providers if p.category == category)

    def with_current_evidence(self, now: datetime) -> tuple[CatalogProvider, ...]:
        """Informational freshness filter: providers whose evidence is all current."""
        return tuple(p for p in self.providers if p.evidence_current(now))


def load_catalog(path: str | Path) -> ProviderCatalog:
    """Load and strictly validate a catalog YAML file. Informational only."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("catalog file must contain a mapping at the top level")
    return ProviderCatalog.model_validate(raw)


def catalog_from_dict(data: dict[str, Any]) -> ProviderCatalog:
    return ProviderCatalog.model_validate(data)
