"""Phase 4 C2: cloud provider profiles, billing eligibility, offline adapters.

Design rules (approved):
- `providers.cost_tier()` keeps its legacy semantics OUTSIDE zero-cost mode; this
  module adds ACCOUNT-SCOPED cloud eligibility on top, never replacing it.
- In zero-cost mode, live dispatch requires a profile with
  billing=free_tier_verified AND current operator attestation
  (verified_free + overage_billing_disabled + verified_at within TTL).
  Operator attestation alone is NOT a billing guarantee: unknown billing or
  unknown overage conditions block live dispatch (fail closed).
- Profiles never contain credentials or raw account identifiers: extra fields are
  forbidden; the account is referenced only by an opaque label + sha256
  fingerprint, and quota pools are identified as '<provider>:acct:<fingerprint>'
  so one account pool is SHARED across all models of that account.
- Fake adapters are strictly offline: allow_network=False and any dispatch
  attempt raises; no socket is ever touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import CloudConfigError, CloudOfflineError, CostUnknownError, UnknownProviderError
from .providers import parse_model

if TYPE_CHECKING:
    from .agent_config import CloudConfig


class BillingModel(StrEnum):
    FREE_TIER_VERIFIED = "free_tier_verified"
    PAID = "paid"
    UNKNOWN = "unknown"


class QuotaPeriod(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class QuotaSpec(BaseModel):
    """One quota limit attached to a SHARED pool identifier. All models of the
    same account/profile share the pool, so provider/account quotas are tracked
    across models, not only per run."""

    model_config = ConfigDict(extra="forbid")

    pool: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    max_requests: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    period: QuotaPeriod = QuotaPeriod.DAY


class CloudProfile(BaseModel):
    """Operator-verified profile for one provider account and its models.

    `billing=free_tier_verified` requires the full attestation set; the
    attestation is a config claim by the operator, NOT a billing guarantee —
    anything not explicitly verified stays UNKNOWN and blocks live dispatch in
    zero-cost mode."""

    model_config = ConfigDict(extra="forbid")  # rejects api_key/secret-like fields

    provider: str
    models: list[str] = Field(min_length=1)
    billing: BillingModel = BillingModel.UNKNOWN
    verified_free: bool = False
    overage_billing_disabled: bool = False
    verified_at: datetime | None = None
    verification_ttl_days: int = Field(default=30, ge=1, le=3650)
    account_label: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    account_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    quotas: list[QuotaSpec] = Field(default_factory=list)
    enabled: bool = True

    @field_validator("provider")
    @classmethod
    def _provider_known(cls, v: str) -> str:
        try:
            parse_model(f"{v.lower()}/probe")
        except UnknownProviderError as exc:
            raise ValueError(str(exc)) from exc
        return v.lower()

    @field_validator("models")
    @classmethod
    def _models_prefixed(cls, v: list[str], info) -> list[str]:
        provider = info.data.get("provider", "")
        for m in v:
            try:
                p, name = parse_model(m)
            except UnknownProviderError as exc:
                raise ValueError(str(exc)) from exc
            if p.value != provider or not name:
                raise ValueError(
                    f"model {m!r} must be '<{provider}/<name>' for this profile"
                )
        return v

    @field_validator("verified_at")
    @classmethod
    def _tz_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def _attestation_and_pools(self) -> CloudProfile:
        if self.billing == BillingModel.FREE_TIER_VERIFIED and not (
            self.verified_free
            and self.overage_billing_disabled
            and self.verified_at is not None
        ):
            raise ValueError(
                "billing=free_tier_verified requires verified_free=True, "
                "overage_billing_disabled=True and verified_at set (operator "
                "attestation — a claim, not a billing guarantee)"
            )
        prefix = f"{self.provider}:"
        for q in self.quotas:
            if not q.pool.startswith(prefix):
                raise ValueError(
                    f"quota pool {q.pool!r} must start with {prefix!r} "
                    "(account-scoped pool identifier)"
                )
        return self

    def verification_current(self, now: datetime) -> bool:
        if self.verified_at is None:
            return False
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return now <= self.verified_at + timedelta(days=self.verification_ttl_days)

    def pool_ids(self) -> list[str]:
        return [q.pool for q in self.quotas]


class CloudProfileRegistry:
    """Resolved view of the cloud profiles: eligibility + shared quota pools."""

    def __init__(
        self,
        *,
        profiles: list[CloudProfile],
        zero_cost: bool = False,
        allowed_providers: list[str] | None = None,
        max_spend_usd: float | None = None,
    ) -> None:
        if allowed_providers is not None:
            for p in allowed_providers:
                try:
                    parse_model(f"{p.lower()}/probe")
                except UnknownProviderError as exc:
                    raise CloudConfigError(str(exc)) from exc
            allowed = {p.lower() for p in allowed_providers}
            for prof in profiles:
                if prof.provider not in allowed:
                    raise CloudConfigError(
                        f"profile provider {prof.provider!r} is not in "
                        f"allowed_providers {sorted(allowed)} — unauthorized provider"
                    )
        self.zero_cost = bool(zero_cost)
        self.max_spend_usd = max_spend_usd
        self.profiles = list(profiles)
        self._by_model: dict[str, CloudProfile] = {
            m: p for p in self.profiles for m in p.models
        }

    @classmethod
    def from_config(cls, cloud: CloudConfig | None) -> CloudProfileRegistry:
        if cloud is None:
            return cls(profiles=[], zero_cost=False)
        return cls(
            profiles=list(cloud.profiles),
            zero_cost=cloud.zero_cost,
            allowed_providers=cloud.allowed_providers,
            max_spend_usd=cloud.max_spend_usd,
        )

    def profile_for(self, model: str) -> CloudProfile | None:
        return self._by_model.get(model)

    def quota_pool(self, model: str) -> str | None:
        """Shared quota pool id for the model's account (None = no quota configured)."""
        p = self._by_model.get(model)
        return p.pool_ids()[0] if p is not None and p.quotas else None

    def live_dispatch_eligible(self, model: str, *, now: datetime) -> tuple[bool, str]:
        """May this model receive a LIVE provider dispatch right now?

        Outside zero-cost mode: legacy semantics are preserved (any known provider
        prefix is allowed; cost_tier()/usage tracking behave as before).
        In zero-cost mode: fail closed — the model must have an enabled profile
        with billing=free_tier_verified and a CURRENT attestation; unknown billing
        or expired verification blocks the dispatch before any request is sent."""
        if not self.zero_cost:
            try:
                parse_model(model)
            except UnknownProviderError:
                return False, f"unknown provider prefix in model {model!r}"
            return True, "legacy mode (zero-cost gate not active)"
        p = self._by_model.get(model)
        if p is None:
            return False, (
                f"no cloud profile for {model!r}: unknown billing blocks live "
                "dispatch in zero-cost mode"
            )
        if not p.enabled:
            return False, f"cloud profile for {model!r} is disabled"
        if p.billing != BillingModel.FREE_TIER_VERIFIED:
            return False, (
                f"billing={p.billing.value} for {model!r} — only free_tier_verified "
                "is allowed in zero-cost mode"
            )
        if not p.verification_current(now):
            return False, (
                f"operator verification for {model!r} expired "
                f"(ttl {p.verification_ttl_days}d from {p.verified_at}); attestation "
                "alone is not a billing guarantee — re-verify"
            )
        return True, "free_tier_verified, attestation current, quota pool checked at dispatch"


@dataclass(frozen=True)
class TokenEstimate:
    """Bounded token estimate reserved BEFORE dispatch (includes max output tokens)."""

    prompt_tokens: int
    max_output_tokens: int


class CloudProviderAdapter(Protocol):
    """Contract for provider adapters (live or fake). Live adapters must never be
    reachable from the fake path; fakes are strictly offline."""

    name: str
    allow_network: bool

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None: ...
    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None: ...
    def quota_pool(self, model: str) -> str | None: ...
    def dispatch(self, model: str, messages: list[Any]) -> Any: ...


class FakeCloudAdapter:
    """Strictly offline fake provider for tests/demos: no network ever."""

    name = "fake"
    allow_network = False

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None:
        return 0.0 if model.startswith("fake/") else None  # unknown -> None, never 0

    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None:
        if not model.startswith("fake/") or max_output_tokens < 1:
            return None  # unknown model or unbounded output -> None (fail closed)
        return TokenEstimate(
            prompt_tokens=max(1, -(-prompt_chars // 4)), max_output_tokens=max_output_tokens
        )

    def quota_pool(self, model: str) -> str | None:
        return "fake:pool" if model.startswith("fake/") else None

    def dispatch(self, model: str, messages: list[Any]) -> Any:
        raise CloudOfflineError(
            f"fake adapter {self.name!r} is offline-only (allow_network=False); "
            "no network request was made"
        )


def bounded_token_estimate(
    adapter: CloudProviderAdapter, model: str, *, prompt_chars: int, max_output_tokens: int
) -> TokenEstimate:
    """Reserve a BOUNDED token estimate before dispatch, including max output
    tokens. Unknown or unbounded usage fails closed (CostUnknownError)."""
    if max_output_tokens < 1:
        raise CostUnknownError(
            f"unbounded max_output_tokens for {model!r} — refusing to dispatch "
            "without a bounded estimate (fail closed)"
        )
    est = adapter.estimate_tokens(
        model=model, prompt_chars=prompt_chars, max_output_tokens=max_output_tokens
    )
    if est is None:
        raise CostUnknownError(
            f"no bounded token estimate for {model!r} — unknown/unbounded usage "
            "fails closed"
        )
    return est
