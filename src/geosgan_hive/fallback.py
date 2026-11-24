from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .errors import HiveError
from .health import HealthRegistry
from .providers import Provider, cost_tier, model_capabilities, parse_model


class FallbackMode(StrEnum):
    OFF = "off"  # default: current behavior, no fallback (backward compatible)
    CONTINUITY = "continuity"  # switch to an eligible alternative model
    WAIT = "wait"  # never switch models; suspend until the primary is available again
    HYBRID = "hybrid"  # alternatives first, suspend if none, recover primary later


class FallbackPolicy(BaseModel):
    """Fallback configuration for one agent (or globally). Opt-in: mode=off."""

    mode: FallbackMode = FallbackMode.OFF
    fallback_models: list[str] = Field(default_factory=list)  # ordered chain
    max_fallback_attempts: int = Field(default=2, ge=0, le=10)
    max_wait_seconds: float = Field(default=300.0, gt=0)
    cooldown_seconds: float = Field(default=60.0, ge=0)
    failure_threshold: int = Field(default=3, ge=1)
    half_open_max_probes: int = Field(default=1, ge=1)
    allowed_providers: list[str] | None = None  # None = all registered providers
    allowed_fallback_providers: list[str] | None = None  # cross-provider privacy gate
    allow_paid_fallback: bool = False  # never auto-switch to a paid model from a free/lower tier
    allow_unknown_cost_fallback: bool = False  # unknown pricing is NOT zero

    @field_validator("fallback_models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        for model in value:
            try:
                parse_model(model)
            except HiveError as exc:
                raise ValueError(str(exc)) from exc
        return value

    @field_validator("allowed_providers", "allowed_fallback_providers")
    @classmethod
    def _validate_providers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        valid = {p.value for p in Provider}
        for name in value:
            if name not in valid:
                raise ValueError(f"unknown provider {name!r}; valid: {sorted(valid)}")
        return value


def effective_policy(
    agent_policy: FallbackPolicy | None, global_policy: FallbackPolicy
) -> FallbackPolicy:
    """An explicitly set agent-level policy overrides the global one entirely
    (including an explicit mode=off); None inherits the global policy."""
    return agent_policy if agent_policy is not None else global_policy


@dataclass
class CandidateDecision:
    model: str
    ok: bool
    reason: str = ""


class FallbackEngine:
    """Decides, per logical call, which model may serve it.

    Gates (all restrictive by default):
    - provider allowlist (`allowed_providers`)
    - cross-provider privacy gate: a non-primary model of ANOTHER provider is
      rejected unless its provider is explicitly in `allowed_fallback_providers`
      (same-provider fallbacks need no extra opt-in)
    - cost gate: unknown-cost candidates are rejected unless
      `allow_unknown_cost_fallback` (unknown pricing is not zero); switching to a
      paid model from a non-paid primary requires `allow_paid_fallback`
    - capability gate: tool support and a rough context-window token estimate
    - health gate: model/provider circuit must not be OPEN; HALF_OPEN is allowed
      only with a single-flight probe claimed by the gateway
    """

    def __init__(self, registry: HealthRegistry) -> None:
        self.registry = registry

    def chain(self, *, primary: str, policy: FallbackPolicy) -> list[str]:
        """Ordered candidate models. WAIT never switches models: primary only."""
        models = [primary]
        if policy.mode != FallbackMode.WAIT:
            for model in policy.fallback_models:
                if model not in models:
                    models.append(model)
        allowed = policy.allowed_providers
        if allowed is not None:
            models = [m for m in models if parse_model(m)[0].value in allowed]
        return models

    def check_candidate(
        self,
        *,
        model: str,
        is_primary: bool,
        primary: str,
        policy: FallbackPolicy,
        needs_tools: bool,
        messages: list[dict[str, Any]] | None = None,
    ) -> CandidateDecision:
        provider, _ = parse_model(model)
        primary_provider = parse_model(primary)[0]
        if not is_primary and provider != primary_provider:
            gate = policy.allowed_fallback_providers
            if gate is None or provider.value not in gate:
                return CandidateDecision(
                    model,
                    False,
                    "cross-provider fallback requires the provider to be explicitly "
                    "listed in allowed_fallback_providers",
                )
        tier = cost_tier(model)
        primary_tier = cost_tier(primary)
        if tier == "unknown" and not policy.allow_unknown_cost_fallback:
            return CandidateDecision(
                model, False, "unknown-cost candidate rejected (unknown pricing is not zero)"
            )
        if tier == "paid" and primary_tier != "paid" and not policy.allow_paid_fallback:
            return CandidateDecision(
                model,
                False,
                f"paid fallback from a {primary_tier}-cost primary requires allow_paid_fallback",
            )
        caps = model_capabilities(model)
        if needs_tools and caps.supports_tools is False:
            return CandidateDecision(model, False, "model does not support tool calling")
        if messages is not None and caps.context_window is not None:
            estimate = sum(len(str(m.get("content") or "")) // 4 + 1 for m in messages)
            if estimate > caps.context_window:
                return CandidateDecision(
                    model,
                    False,
                    f"estimated {estimate} tokens exceed context window {caps.context_window}",
                )
        if not self.registry.is_available(HealthRegistry.model_key(model)):
            return CandidateDecision(model, False, "model circuit open (cooldown)")
        if not self.registry.is_available(HealthRegistry.provider_key(provider)):
            return CandidateDecision(model, False, "provider circuit open (cooldown)")
        return CandidateDecision(model, True)
