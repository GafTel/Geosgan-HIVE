"""Phase 4 C2 (final hardening): cloud provider profiles, billing eligibility,
offline adapter contracts.

Design rules (approved + C2 audits):
- `providers.cost_tier()` keeps its legacy semantics OUTSIDE zero-cost mode; this
  module adds ACCOUNT-SCOPED cloud eligibility on top, never replacing it.
- In zero-cost mode, live dispatch requires a profile with
  billing=free_tier_verified AND a VerificationEvidence record (trusted evidence
  bound to the actual credential/account fingerprint) that is current. Operator
  attestation alone (verified_free / overage_billing_disabled / billing_verified)
  is a configurable CLAIM, not independent evidence — it never enables live
  dispatch by itself. Unknown billing blocks live dispatch (fail closed).
- Profiles are deeply immutable (frozen models, tuple fields); the registry
  validates everything in __init__ (direct construction included), is immutable
  afterwards, and uses a read-only model mapping. No silent overwrites: duplicate
  models, conflicting profiles (by canonical fingerprint identity) and
  conflicting shared-pool quota definitions are all rejected.
- Quota pools are bound to the account fingerprint ('<provider>:acct:<sha256>')
  and aggregated account-wide across models, deduplicated, conflict-checked —
  no model may bypass account-wide limits.
- Fake adapters are strictly offline, test-only, and their heuristic token
  estimates are never used for live enforcement: live estimation requires
  explicit adapter capability flags and conservative bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
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


class BillingEnforcement(StrEnum):
    """How zero billing exposure is ESTABLISHED provider-side (not inferred from
    model names or local accounting):

    - PROVIDER_HARD_LIMIT: the provider enforces a hard free cap server-side.
    - FREE_ALLOWANCE_NO_OVERAGE: free allowance with overage billing verifiably
      disabled (requires credential binding + overage attestation).
    - LOCAL_INFERENCE: local model (ollama) — no per-request provider billing.
    - UNKNOWN: no evidence — FREE_ONLY denies dispatch (fail closed).
    """

    PROVIDER_HARD_LIMIT = "provider_hard_limit"
    FREE_ALLOWANCE_NO_OVERAGE = "free_allowance_no_overage"
    LOCAL_INFERENCE = "local_inference"
    UNKNOWN = "unknown"


class QuotaResetSemantics(StrEnum):
    """Evidence level for provider quota-RESET behavior. UNKNOWN fails closed in
    the QuotaEngine: an arbitrary window anchor alone is NOT evidence that the
    provider resets quotas on that schedule — reset times are never invented."""

    PROVIDER_DOCUMENTED = "provider_documented"
    OPERATOR_ATTESTED = "operator_attested"
    UNKNOWN = "unknown"


class TokenEstimationMode(StrEnum):
    EXACT = "exact"
    BOUNDED_HEURISTIC = "bounded_heuristic"
    NONE = "none"


class AdapterCapabilities(BaseModel):
    """Explicit, strictly typed adapter capability flags (frozen; extra fields
    forbidden). Missing/ambiguous flags fail closed for live estimation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    live_supported: bool = Field(strict=True)  # no string coercion
    token_estimation: TokenEstimationMode


class VerificationEvidence(BaseModel):
    """Trusted evidence contract for billing verification. Distinct from operator
    attestation: evidence references an external artifact (never a secret) and is
    bound to the actual credential/account via its fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(pattern=r"^[a-z0-9][a-z0-9_ -]{0,63}$")
    evidence_ref: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")  # opaque, no secrets
    collected_at: datetime
    collected_by: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    credential_bound: bool
    account_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: optional explicit bindings: the provider the evidence was collected for and
    #: the endpoint (api_base host) it applies to. When set, they must match the
    #: profile's provider / the request's endpoint at dispatch time.
    provider: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")
    endpoint: str | None = Field(default=None, pattern=r"^https?://(\[[0-9a-fA-F:.]+\]|[a-zA-Z0-9.-]{1,253})(:[0-9]{1,5})?(/[a-zA-Z0-9_./-]*)?$")
    #: how zero billing exposure is established; UNKNOWN (default) fails closed
    billing_enforcement: BillingEnforcement = BillingEnforcement.UNKNOWN

    @field_validator("collected_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware (naive timestamps rejected)")
        return v

    @field_validator("endpoint")
    @classmethod
    def _endpoint_port_range(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from urllib.parse import urlsplit

        parts = urlsplit(v)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"endpoint must be an absolute http(s) URL: {v!r}")
        if parts.port is None:
            return v
        if not 1 <= parts.port <= 65535:
            raise ValueError(f"endpoint port out of range: {parts.port}")
        return v

    def current(self, now: datetime, ttl_days: int) -> bool:
        if now.tzinfo is None:
            raise CloudConfigError("evidence.current() requires a tz-aware 'now'")
        return self.collected_at <= now <= self.collected_at + timedelta(days=ttl_days)


class QuotaSpec(BaseModel):
    """One quota limit attached to a SHARED pool identifier. All models of the
    same account share the pool. A quota without any limit is rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    max_requests: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    period: QuotaPeriod = QuotaPeriod.DAY

    @model_validator(mode="after")
    def _has_limit(self) -> QuotaSpec:
        if self.max_requests is None and self.max_tokens is None:
            raise ValueError(
                "quota without any limit is meaningless — set max_requests and/or max_tokens"
            )
        return self

    def key(self) -> tuple[str, str]:
        """Conflict key: same pool AND same period (multiple limit kinds/periods on
        one pool are allowed)."""
        return (self.pool, self.period.value)

    def dedup_key(self) -> tuple:
        return (self.pool, self.period.value, self.max_requests, self.max_tokens)

    def conflicts(self, other: QuotaSpec) -> bool:
        """True if both constrain the SAME limit kind on the same pool+period with
        different values."""
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
    """Operator-verified profile for one provider account and its models.
    Deeply immutable (frozen model, tuple fields). Never contains credentials or
    raw account ids (extra='forbid')."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    models: tuple[str, ...] = Field(min_length=1)
    billing: BillingModel = BillingModel.UNKNOWN
    verified_free: bool = False             # operator attestation (a claim, not evidence)
    overage_billing_disabled: bool = False  # operator attestation (a claim)
    billing_verified: bool = False          # operator assertion — NOT evidence by itself
    verified_at: datetime | None = None
    verification_ttl_days: int = Field(default=30, ge=1, le=3650)
    verification_evidence: VerificationEvidence | None = None
    account_label: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    account_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: explicit UTC anchor for quota period windows; REQUIRED for live zero-cost
    #: profiles — unknown reset semantics fail closed (never invented)
    quota_window_start: datetime | None = None
    #: evidence level for the reset schedule itself; UNKNOWN (the default) makes
    #: the QuotaEngine fail closed even when an anchor is configured
    quota_reset_semantics: QuotaResetSemantics = QuotaResetSemantics.UNKNOWN
    quotas: tuple[QuotaSpec, ...] = Field(default_factory=tuple)
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
    def _models_prefixed(cls, v, info) -> tuple[str, ...]:
        provider = info.data.get("provider", "")
        out = []
        for m in v:
            try:
                p, name = parse_model(m)
            except UnknownProviderError as exc:
                raise ValueError(str(exc)) from exc
            if p.value != provider or not name:
                raise ValueError(f"model {m!r} must be '<{provider}/<name>' for this profile")
            out.append(m)
        return tuple(out)

    @field_validator("verified_at", "quota_window_start")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (naive timestamps rejected)")
        return v

    @model_validator(mode="after")
    def _hardened_rules(self) -> CloudProfile:
        if self.billing == BillingModel.FREE_TIER_VERIFIED:
            if not (self.verified_free and self.overage_billing_disabled
                    and self.verified_at is not None):
                raise ValueError(
                    "billing=free_tier_verified requires verified_free=True, "
                    "overage_billing_disabled=True and verified_at (tz-aware); a "
                    "VerificationEvidence record bound to the account fingerprint "
                    "is additionally required for live dispatch eligibility"
                )
            if self.account_fingerprint is None:
                raise ValueError(
                    "free_tier_verified requires account_fingerprint (sha256 of the "
                    "account id; the raw id is never stored)"
                )
            if not self.quotas:
                raise ValueError("free_tier_verified requires at least one bounded quota spec")
            if self.quota_window_start is None:
                raise ValueError(
                    "free_tier_verified requires quota_window_start (explicit UTC anchor "
                    "for period resets) — unknown reset semantics fail closed"
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
        if (self.verification_evidence is not None and self.account_fingerprint is not None
                and self.verification_evidence.account_fingerprint != self.account_fingerprint):
            raise ValueError(
                "verification_evidence is bound to a different account fingerprint "
                "than this profile"
            )
        ev = self.verification_evidence
        if ev is not None and ev.provider is not None and ev.provider != self.provider:
            raise ValueError(
                f"verification_evidence is bound to provider {ev.provider!r}, but this "
                f"profile is for provider {self.provider!r}"
            )
        return self

    def verification_current(self, now: datetime) -> bool:
        """True iff verified_at <= now <= verified_at + ttl. Future/naive rejected."""
        if now.tzinfo is None:
            raise CloudConfigError(
                "verification_current() requires a timezone-aware 'now' "
                "(ambiguous naive timestamps are rejected)"
            )
        if self.verified_at is None:
            return False
        if self.verified_at > now:
            return False
        return now <= self.verified_at + timedelta(days=self.verification_ttl_days)

    def pool_ids(self) -> tuple[str, ...]:
        return tuple(q.pool for q in self.quotas)


class CloudProfileRegistry:
    """Validated, immutable view of cloud profiles: eligibility + account-wide
    quota aggregation. All invariants enforced in __init__ (direct construction
    included); read-only afterwards."""

    def __init__(
        self,
        *,
        profiles: list[CloudProfile] | tuple[CloudProfile, ...],
        zero_cost: bool = False,
        allowed_providers: list[str] | None = None,
        max_spend_usd: float | None = None,
        trusted_evidence_binding: bool = False,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        if zero_cost and max_spend_usd != 0.0:
            # None means unlimited — never allowed in zero-cost mode
            raise CloudConfigError(
                "zero-cost mode requires max_spend_usd == 0.0 exactly "
                "(None/unlimited rejected)"
            )
        # defensive revalidation: even existing Pydantic instances are re-validated
        # from their dump, so a mutated/bypassed instance cannot slip through
        profiles = tuple(
            CloudProfile.model_validate(
                p.model_dump() if isinstance(p, CloudProfile) else p
            )
            for p in profiles
        )
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
        by_model: dict[str, CloudProfile] = {}
        for p in profiles:
            for m in p.models:
                if m in by_model:
                    raise CloudConfigError(
                        f"duplicate model registration {m!r} in profiles "
                        f"{by_model[m].account_label!r} and {p.account_label!r}"
                    )
                by_model[m] = p
        # canonical account identity = (provider, fingerprint); label must agree
        by_label: dict[tuple[str, str], str] = {}
        by_fp: dict[tuple[str, str], tuple[str, BillingModel]] = {}
        for p in profiles:
            if p.account_fingerprint is None:
                continue
            lkey, fkey = (p.provider, p.account_label), (p.provider, p.account_fingerprint)
            if lkey in by_label and by_label[lkey] != p.account_fingerprint:
                raise CloudConfigError(
                    f"conflicting account identity for {lkey}: label maps to two "
                    "different fingerprints"
                )
            by_label[lkey] = p.account_fingerprint
            if fkey in by_fp:
                label, billing = by_fp[fkey]
                if label != p.account_label or billing != p.billing:
                    raise CloudConfigError(
                        f"conflicting account profile for canonical identity {fkey}: "
                        "different label or billing"
                    )
            else:
                by_fp[fkey] = (p.account_label, p.billing)
        # conflicting shared-pool quota definitions across profiles
        all_quotas = [q for p in profiles for q in p.quotas]
        for i, a in enumerate(all_quotas):
            for b in all_quotas[i + 1:]:
                if a.conflicts(b):
                    raise CloudConfigError(
                        f"conflicting quota definition for shared pool {a.pool!r} "
                        f"period {a.period.value}"
                    )
        # account-wide aggregated quotas, deduplicated, keyed by CANONICAL
        # (provider, fingerprint) — never fingerprint alone
        acct: dict[tuple[str, str], dict[tuple, QuotaSpec]] = {}
        fp_of_model: dict[str, tuple[str, str]] = {}
        for p in profiles:
            if p.account_fingerprint is None:
                continue
            key = (p.provider, p.account_fingerprint)
            bucket = acct.setdefault(key, {})
            for q in p.quotas:
                bucket[q.dedup_key()] = q
            for m in p.models:
                fp_of_model[m] = key
        object.__setattr__(self, "zero_cost", bool(zero_cost))
        object.__setattr__(self, "max_spend_usd", max_spend_usd)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "_by_model", MappingProxyType(by_model))
        object.__setattr__(
            self, "_account_quotas",
            MappingProxyType({k: tuple(v.values()) for k, v in acct.items()}),
        )
        object.__setattr__(self, "_fp_of_model", MappingProxyType(fp_of_model))
        object.__setattr__(self, "trusted_evidence_binding", bool(trusted_evidence_binding))
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
            return cls(profiles=(), zero_cost=False)
        return cls(
            profiles=tuple(cloud.profiles),
            zero_cost=cloud.zero_cost,
            allowed_providers=cloud.allowed_providers,
            max_spend_usd=cloud.max_spend_usd,
            trusted_evidence_binding=getattr(cloud, "trusted_evidence_binding", False),
        )

    def profile_for(self, model: str) -> CloudProfile | None:
        return self._by_model.get(model)

    def account_key_for(self, model: str) -> tuple[str, str] | None:
        """Canonical (provider, fingerprint) account key for a model."""
        return self._fp_of_model.get(model)

    def account_quotas(self, provider: str, fingerprint: str) -> tuple[QuotaSpec, ...]:
        """ALL quota constraints of one account, keyed by (provider, fingerprint),
        deduplicated."""
        return self._account_quotas.get((provider.lower(), fingerprint), ())

    def quotas_for(self, model: str) -> tuple[QuotaSpec, ...]:
        """Account-wide constraints applying to this model (no model bypass)."""
        key = self._fp_of_model.get(model)
        return self.account_quotas(*key) if key else ()

    def quota_pool(self, model: str) -> str | None:
        p = self._by_model.get(model)
        return p.pool_ids()[0] if p is not None and p.quotas else None

    def live_dispatch_eligible(self, model: str, *, now: datetime) -> tuple[bool, str]:
        """May this model receive a LIVE provider dispatch right now?

        Outside zero-cost mode: legacy semantics preserved (any known provider
        prefix allowed; cost_tier()/usage unchanged). In zero-cost mode: fail
        closed — enabled profile with billing=free_tier_verified, current
        attestation, AND a current VerificationEvidence bound to the account
        fingerprint with the credential bound. Attestation alone never enables
        live dispatch."""
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
            if p.verified_at is not None and p.verified_at > now:
                return False, f"verification timestamp for {model!r} is in the future"
            return False, (
                f"operator verification for {model!r} expired "
                f"(ttl {p.verification_ttl_days}d from {p.verified_at}); re-verify"
            )
        ev = p.verification_evidence
        if ev is None:
            return False, (
                f"no verification EVIDENCE for {model!r} — attestation "
                "(verified_free/billing_verified) is a claim, not evidence; live "
                "dispatch stays disabled"
            )
        if not ev.credential_bound:
            return False, (
                f"verification evidence for {model!r} is not bound to the actual "
                "credential/account — live dispatch stays disabled"
            )
        if not ev.current(now, p.verification_ttl_days):
            return False, (
                f"verification evidence for {model!r} is stale (collected "
                f"{ev.collected_at}); re-collect evidence"
            )
        if not self.trusted_evidence_binding:
            return False, (
                "trusted evidence/credential-account binding is not implemented yet "
                "(VerificationEvidence is a data contract, not proof of authenticity; "
                "the manual trusted_evidence_binding flag alone is NOT sufficient "
                "proof of provider-side billing safety) — production live dispatch "
                "stays blocked"
            )
        # strict zero-cost (FREE_ONLY): zero billing exposure must be ESTABLISHED,
        # not inferred. Local accounting cannot observe out-of-HIVE consumption.
        enf = ev.billing_enforcement
        if enf == BillingEnforcement.UNKNOWN:
            return False, (
                f"billing enforcement UNKNOWN for {model!r} — zero billing exposure "
                "cannot be established; FREE_ONLY denies dispatch"
            )
        if enf == BillingEnforcement.LOCAL_INFERENCE and p.provider != "ollama":
            return False, (
                f"local-inference claim is only valid for local providers, not "
                f"{p.provider!r}"
            )
        if enf == BillingEnforcement.FREE_ALLOWANCE_NO_OVERAGE and not p.overage_billing_disabled:
            return False, (
                f"free allowance with automatic paid overage for {model!r} — not "
                "zero exposure; overage billing must be verifiably disabled"
            )
        if enf == BillingEnforcement.FREE_ALLOWANCE_NO_OVERAGE and not ev.credential_bound:
            return False, (
                f"free-allowance evidence for {model!r} is not credential-bound"
            )
        return True, (
            "free_tier_verified + evidence bound to account/provider/endpoint + "
            f"billing enforcement={enf.value} + attestation current"
        )


@dataclass(frozen=True)
class TokenEstimate:
    """Bounded token estimate reserved BEFORE dispatch (includes max output tokens)."""

    prompt_tokens: int
    max_output_tokens: int


class CloudProviderAdapter(Protocol):
    """Contract for provider adapters (live or fake). Fakes are strictly offline
    and test-only; live adapters declare explicit capability flags."""

    name: str
    allow_network: bool
    is_test_only: bool
    capabilities: AdapterCapabilities

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None: ...
    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None: ...
    def quota_pool(self, model: str) -> str | None: ...
    def dispatch(self, model: str, messages: list[Any]) -> Any: ...


class FakeCloudAdapter:
    """Strictly offline, test-only fake provider: no network ever; heuristic
    estimates never used for live enforcement."""

    name = "fake"
    allow_network = False
    is_test_only = True
    capabilities = AdapterCapabilities(live_supported=False,
                                       token_estimation=TokenEstimationMode.BOUNDED_HEURISTIC)

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None:
        return 0.0 if model.startswith("fake/") else None  # unknown -> None, never 0

    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None:
        if not model.startswith("fake/") or max_output_tokens < 1:
            return None
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


class LocalOllamaAdapter:
    """Adapter for local Ollama inference: the FIRST-CLASS zero-cost option.

    Local inference has no per-request provider billing, so estimated cost is
    exactly 0.0 for ollama/ models (None -> fail closed for anything else).
    The adapter itself never touches the network (allow_network=False): the
    gateway performs the loopback-only Ollama call through LiteLLM. Token
    estimates use the conservative universal upper bound of 1 token per payload
    character (BOUNDED_HEURISTIC, never chars/4), which satisfies live
    conservative-bound enforcement.
    """

    name = "ollama-local"
    allow_network = False
    is_test_only = False
    capabilities = AdapterCapabilities(
        live_supported=True, token_estimation=TokenEstimationMode.BOUNDED_HEURISTIC
    )

    def estimate_cost(self, model: str, *, prompt_tokens: int,
                      max_output_tokens: int) -> float | None:
        # local inference: no per-request provider billing; unknown for others
        return 0.0 if model.startswith("ollama/") else None

    def estimate_tokens(self, model: str, *, prompt_chars: int,
                        max_output_tokens: int) -> TokenEstimate | None:
        if not model.startswith("ollama/") or max_output_tokens < 1:
            return None
        return TokenEstimate(
            prompt_tokens=max(1, prompt_chars), max_output_tokens=max_output_tokens
        )

    def quota_pool(self, model: str) -> str | None:
        return None  # pools come from the profile registry, not the adapter

    def dispatch(self, model: str, messages: list[Any]) -> Any:
        raise CloudOfflineError(
            f"adapter {self.name!r} never dispatches itself (allow_network=False); "
            "the gateway calls the local Ollama endpoint through LiteLLM"
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
    tokens. Strict validation: int types only (no bools), nonnegative prompt,
    positive output bound, no under-reservation vs the requested cap. Unknown or
    unbounded usage fails closed (CostUnknownError). For LIVE enforcement the
    adapter must declare explicit capabilities (live_supported, a usable token
    estimation mode) and must not be test-only."""
    pc = _strict_int(prompt_chars, "prompt_chars")
    mo = _strict_int(max_output_tokens, "max_output_tokens")
    if pc < 0:
        raise CostUnknownError("prompt_chars must be nonnegative")
    if mo < 1:
        raise CostUnknownError(
            f"unbounded max_output_tokens for {model!r} — refusing to dispatch "
            "without a bounded estimate (fail closed)"
        )
    if live:
        caps = getattr(adapter, "capabilities", None)
        if not isinstance(caps, AdapterCapabilities):
            raise CostUnknownError(
                f"adapter {getattr(adapter, 'name', '?')!r} has no explicit capability "
                "flags — live estimation fails closed"
            )
        if getattr(adapter, "is_test_only", False) or not caps.live_supported:
            raise CostUnknownError(
                f"adapter {adapter.name!r} is not live-capable — live estimation fails closed"
            )
        if caps.token_estimation not in (TokenEstimationMode.EXACT,
                                         TokenEstimationMode.BOUNDED_HEURISTIC):
            raise CostUnknownError(
                f"adapter {adapter.name!r} cannot estimate tokens "
                f"({caps.token_estimation.value}) — live estimation fails closed"
            )
    est = adapter.estimate_tokens(model=model, prompt_chars=pc, max_output_tokens=mo)
    if est is None:
        raise CostUnknownError(
            f"no bounded token estimate for {model!r} — unknown/unbounded usage "
            "fails closed"
        )
    pt = _strict_int(est.prompt_tokens, "estimate.prompt_tokens")
    eo = _strict_int(est.max_output_tokens, "estimate.max_output_tokens")
    if pt < 0:
        raise CostUnknownError("estimate.prompt_tokens must be nonnegative")
    if eo < mo:
        raise CostUnknownError(
            f"under-reservation: estimate.max_output_tokens={eo} < requested cap "
            f"{mo} for {model!r} — fail closed"
        )
    if live and caps.token_estimation != TokenEstimationMode.EXACT and pt < pc:
        # heuristic estimates (e.g. chars/4) are NOT demonstrably conservative for
        # the complete provider payload: the universal lower bound is >= 1 token per
        # character. Only EXACT estimation may report fewer prompt tokens.
        raise CostUnknownError(
            f"non-conservative prompt token bound for {model!r}: estimate "
            f"prompt_tokens={pt} < payload chars={pc} (heuristic estimates are not "
            "proof of conservative coverage — fail closed)"
        )
    return TokenEstimate(prompt_tokens=pt, max_output_tokens=eo)
