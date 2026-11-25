"""Phase 4 C2 (hardened): cloud provider profiles, billing eligibility, offline adapters.

Design rules (approved + C2 audit):
- `providers.cost_tier()` keeps its legacy semantics OUTSIDE zero-cost mode; this
  module adds ACCOUNT-SCOPED cloud eligibility on top, never replacing it.
- In zero-cost mode, live dispatch requires a profile with
  billing=free_tier_verified AND independent billing verification
  (billing_verified=True) AND current operator attestation
  (verified_free + overage_billing_disabled + verified_at within
  [verified_at, verified_at+ttl], never in the future, never naive).
  Operator attestation alone is NOT a billing guarantee: unknown billing or
  missing independent verification blocks live dispatch (fail closed).
- Profiles are immutable and never contain credentials or raw account
  identifiers: extra fields are forbidden; the account is referenced only by an
  opaque label + sha256 fingerprint, and quota pools are BOUND to the actual
  fingerprint: '<provider>:acct:<fingerprint>', shared across all models of
  that account. Duplicate models, conflicting profiles and conflicting shared
  pool definitions are rejected — no silent overwrites.
- Fake adapters are strictly offline and test-only: their char/4 estimates must
  never be used for live quota enforcement (bounded_token_estimate(live=True)
  refuses test-only adapters).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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
    across models, not only per run. A quota without any limit is rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    max_requests: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    period: QuotaPeriod = QuotaPeriod.DAY

    @model_validator(mode="after")
    def _has_limit(self) -> QuotaSpec:
        if self.max_requests is None and self.max_tokens is None:
            raise ValueError(
                "quota without any limit is meaningless — set max_requests "
                "and/or max_tokens"
            )
        return self

    def key(self) -> tuple[str, str]:
        """Conflict key: same pool AND same period (multiple periods/limit kinds on
        one pool are allowed)."""
        return (self.pool, self.period.value)

    def conflicts(self, other: QuotaSpec) -> bool:
        """True if both specs constrain the SAME limit kind on the same pool+period
        with different values (e.g. two different max_requests for one day)."""
        if self.key() != other.key():
            return False
        if (self.max_requests is not None and other.max_requests is not None
                and self.max_requests != other.max_requests):
            return True
        if (self.max_tokens is not None and other.max_tokens is not None
                and self.max_tokens != other.max_tokens):
            return True
        return False


class CloudProfile(BaseModel):
    """Operator-verified profile for one provider account and its models (immutable).

    `billing=free_tier_verified` requires the full attestation set AND independent
    billing verification (billing_verified). Attestation is an operator claim,
    not a billing guarantee — anything not verified stays UNKNOWN and blocks
    live dispatch in zero-cost mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)  # rejects api_key-like fields

    provider: str
    models: list[str] = Field(min_length=1)
    billing: BillingModel = BillingModel.UNKNOWN
    verified_free: bool = False            # operator attestation: free tier exists
    overage_billing_disabled: bool = False  # operator attestation: no overage billing
    billing_verified: bool = False         # INDEPENDENT billing verification (not attestation)
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
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "verified_at must carry a timezone offset (ambiguous naive "
                "timestamps are rejected)"
            )
        return v

    @model_validator(mode="after")
    def _hardened_rules(self) -> CloudProfile:
        if self.billing == BillingModel.FREE_TIER_VERIFIED:
            if not (
                self.verified_free
                and self.overage_billing_disabled
                and self.verified_at is not None
            ):
                raise ValueError(
                    "billing=free_tier_verified requires verified_free=True, "
                    "overage_billing_disabled=True and verified_at (tz-aware); "
                    "billing_verified (independent verification) is additionally "
                    "required for live dispatch eligibility — attestation alone is "
                    "not a billing guarantee"
                )
            if self.account_fingerprint is None:
                raise ValueError(
                    "free_tier_verified requires account_fingerprint (sha256 of the "
                    "account id; the raw id is never stored)"
                )
            if not self.quotas:
                raise ValueError(
                    "free_tier_verified requires at least one bounded quota spec"
                )
            expected_pool = f"{self.provider}:acct:{self.account_fingerprint}"
            for q in self.quotas:
                if q.pool != expected_pool:
                    raise ValueError(
                        f"quota pool {q.pool!r} must be bound to the actual account "
                        f"fingerprint: expected {expected_pool!r}"
                    )
            for i, a in enumerate(self.quotas):
                for b in self.quotas[i + 1:]:
                    if a.conflicts(b):
                        raise ValueError(
                            f"conflicting definitions for shared pool {a.pool!r} "
                            f"period {a.period.value} in one profile"
                        )
        return self

    def verification_current(self, now: datetime) -> bool:
        """True iff verified_at <= now <= verified_at + ttl. Future or naive
        timestamps are never 'current'."""
        if now.tzinfo is None:
            raise CloudConfigError(
                "verification_current() requires a timezone-aware 'now' "
                "(ambiguous naive timestamps are rejected)"
            )
        if self.verified_at is None:
            return False
        if self.verified_at > now:
            return False  # future verification timestamp
        return now <= self.verified_at + timedelta(days=self.verification_ttl_days)

    def pool_ids(self) -> list[str]:
        return [q.pool for q in self.quotas]


class CloudProfileRegistry:
    """Resolved, validated, IMMUTABLE view of the cloud profiles: eligibility +
    shared quota pools. All invariants are enforced in __init__, so direct
    construction cannot bypass them; attributes are read-only afterwards."""

    def __init__(
        self,
        *,
        profiles: list[CloudProfile],
        zero_cost: bool = False,
        allowed_providers: list[str] | None = None,
        max_spend_usd: float | None = None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
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
        profiles = [
            p if isinstance(p, CloudProfile) else CloudProfile.model_validate(p)
            for p in profiles
        ]
        # duplicate model registrations: no silent overwrites
        by_model: dict[str, CloudProfile] = {}
        for p in profiles:
            for m in p.models:
                if m in by_model:
                    raise CloudConfigError(
                        f"duplicate model registration {m!r} in profiles "
                        f"{by_model[m].account_label!r} and {p.account_label!r}"
                    )
                by_model[m] = p
        # conflicting profiles for the same provider account
        seen_accounts: dict[tuple[str, str], CloudProfile] = {}
        for p in profiles:
            key = (p.provider, p.account_label)
            if key in seen_accounts:
                other = seen_accounts[key]
                if (other.account_fingerprint != p.account_fingerprint
                        or other.billing != p.billing):
                    raise CloudConfigError(
                        f"conflicting profiles for {key}: different fingerprint or billing"
                    )
            else:
                seen_accounts[key] = p
        # conflicting shared pool definitions across profiles (same pool+period+
        # same limit kind with different values); multiple periods are allowed
        all_quotas = [q for p in profiles for q in p.quotas]
        for i, a in enumerate(all_quotas):
            for b in all_quotas[i + 1:]:
                if a.conflicts(b):
                    raise CloudConfigError(
                        f"conflicting quota definition for shared pool {a.pool!r} "
                        f"period {a.period.value}"
                    )
        object.__setattr__(self, "zero_cost", bool(zero_cost))
        object.__setattr__(self, "max_spend_usd", max_spend_usd)
        object.__setattr__(self, "profiles", tuple(profiles))
        object.__setattr__(self, "_by_model", by_model)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"CloudProfileRegistry is immutable after construction ({name!r})"
            )
        object.__setattr__(self, name, value)

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

    def quotas_for(self, model: str) -> list[QuotaSpec]:
        """ALL quota constraints applicable to the model (multiple periods/limits)."""
        p = self._by_model.get(model)
        return list(p.quotas) if p is not None else []

    def quota_pool(self, model: str) -> str | None:
        """Shared quota pool id for the model's account (first pool; None if none)."""
        p = self._by_model.get(model)
        return p.pool_ids()[0] if p is not None and p.quotas else None

    def live_dispatch_eligible(self, model: str, *, now: datetime) -> tuple[bool, str]:
        """May this model receive a LIVE provider dispatch right now?

        Outside zero-cost mode: legacy semantics are preserved (any known provider
        prefix is allowed; cost_tier()/usage tracking behave as before).
        In zero-cost mode: fail closed — the model must have an enabled profile
        with billing=free_tier_verified, independent billing verification, a
        CURRENT attestation and bounded quotas; unknown billing, paid, disabled,
        unverified or expired profiles block the dispatch before any request."""
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
        if not p.billing_verified:
            return False, (
                f"billing for {model!r} is operator-attested but NOT independently "
                "verified — attestation is not a billing guarantee; live dispatch "
                "stays disabled"
            )
        if not p.verification_current(now):
            if p.verified_at is not None and p.verified_at > now:
                return False, f"verification timestamp for {model!r} is in the future"
            return False, (
                f"operator verification for {model!r} expired "
                f"(ttl {p.verification_ttl_days}d from {p.verified_at}); re-verify"
            )
        return True, "free_tier_verified + independently verified + attestation current"


@dataclass(frozen=True)
class TokenEstimate:
    """Bounded token estimate reserved BEFORE dispatch (includes max output tokens)."""

    prompt_tokens: int
    max_output_tokens: int


class CloudProviderAdapter(Protocol):
    """Contract for provider adapters (live or fake). Live adapters must never be
    reachable from the fake path; fakes are strictly offline and test-only."""

    name: str
    allow_network: bool
    is_test_only: bool

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None: ...
    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None: ...
    def quota_pool(self, model: str) -> str | None: ...
    def dispatch(self, model: str, messages: list[Any]) -> Any: ...


class FakeCloudAdapter:
    """Strictly offline, test-only fake provider: no network ever; its char/4
    token heuristic must never be used for live quota enforcement."""

    name = "fake"
    allow_network = False
    is_test_only = True

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


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:  # bool is not accepted as an int here
        raise CostUnknownError(f"{name} must be a strict int, got {type(value).__name__}")
    return value


def bounded_token_estimate(
    adapter: CloudProviderAdapter, model: str, *, prompt_chars: int,
    max_output_tokens: int, live: bool = False,
) -> TokenEstimate:
    """Reserve a BOUNDED token estimate before dispatch, including max output
    tokens. Strict validation: integer types only (no bools), nonnegative prompt
    count, positive output bound, and no under-reservation relative to the
    requested output cap. Unknown/unbounded usage fails closed (CostUnknownError).
    Test-only (fake) adapters are refused for live enforcement."""
    if live and getattr(adapter, "is_test_only", False):
        raise CostUnknownError(
            f"adapter {adapter.name!r} is test-only; its heuristic estimates must "
            "never be used for live quota enforcement"
        )
    pc = _strict_int(prompt_chars, "prompt_chars")
    mo = _strict_int(max_output_tokens, "max_output_tokens")
    if pc < 0:
        raise CostUnknownError("prompt_chars must be nonnegative")
    if mo < 1:
        raise CostUnknownError(
            f"unbounded max_output_tokens for {model!r} — refusing to dispatch "
            "without a bounded estimate (fail closed)"
        )
    est = adapter.estimate_tokens(
        model=model, prompt_chars=pc, max_output_tokens=mo
    )
    if est is None:
        raise CostUnknownError(
            f"no bounded token estimate for {model!r} — unknown/unbounded usage "
            "fails closed"
        )
    pt, eo = _strict_int(est.prompt_tokens, "estimate.prompt_tokens"), _strict_int(
        est.max_output_tokens, "estimate.max_output_tokens")
    if pt < 0:
        raise CostUnknownError("estimate.prompt_tokens must be nonnegative")
    if eo < mo:
        raise CostUnknownError(
            f"under-reservation: estimate.max_output_tokens={eo} < requested cap "
            f"{mo} for {model!r} — fail closed"
        )
    return TokenEstimate(prompt_tokens=pt, max_output_tokens=eo)
