"""A3: deterministic, policy-first intelligent model routing (v0.7.0).

Design guarantees:

- The router SELECTS/RECOMMENDS models; it NEVER authorizes dispatch. The
  zero-cost gate in ModelGateway still revalidates evidence, quota and lease
  immediately before every physical provider call. There is no parallel
  dispatch path: routed calls go through the exact same gateway.
- MANDATORY ELIGIBILITY FILTERS run before any scoring: runtime authorization
  (cloud registry live-dispatch eligibility), provider/model allowlists, exact
  catalog model identity, catalog operational status, billing classification
  (unknown/paid/unverified fail closed), evidence freshness, agent capability
  requirements, context window, privacy restrictions, endpoint restrictions,
  provider health/cooldown, local quota availability, run budgets and explicit
  cross-provider fallback authorization.
- Zero-cost eligibility is a FILTER, never a weighted score. A model whose name
  contains "free" is not free; the catalog entry is informational only.
- DETERMINISTIC RANKING: fixed weights (validated, sum to 1) over measured
  components only. Missing measurements are EXCLUDED and the weights are
  renormalized over the remaining components - unknown quality is never
  fabricated as 0 or 1. Tie-break: higher score, then lexicographically smaller
  model id.
- Only trusted_runtime, lease-fenced benchmark evidence (see checkpoints.py)
  may feed the quality component. Simulated and operator-reported evidence are
  never used. Evidence-based selection is UNAVAILABLE when no trusted evidence
  exists (selection_mode='none', reason NO_TRUSTED_EVIDENCE).
- ROUTING MODES: 'fixed' (default; the configured model must pass the same
  filters), 'eligible_default' (explicit operator-configured default among
  eligible candidates; automatic routing is OFF unless configured),
  'evidence_based' (rank by trusted evidence only).
- PROVENANCE: every decision is durably recorded (schema v11
  routing_decisions) with bounded metadata only - candidate ids, eligibility
  decisions, reason codes, scores, selected model, mode, evidence refs, method
  and policy versions. Never credentials, raw prompts or private content.
- Quota integration: routing decisions never reserve; reservations stay atomic
  and durable at the dispatch boundary (ZeroCostGate.prepare_dispatch), so
  concurrent routing cannot oversubscribe account-wide quotas. revalidate()
  re-runs all filters immediately before dispatch.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .benchmarks import SUITE_VERSION, trusted_quality_evidence
from .catalog import (
    BillingClassification,
    CatalogModel,
    OperationalStatus,
    ProviderCatalog,
    ProviderCategory,
)
from .errors import HiveError
from .health import HealthRegistry
from .providers import parse_model
from .quota_engine import QuotaEngine


def _parse_or_value_error(model: str) -> None:
    try:
        parse_model(model)
    except HiveError as exc:
        raise ValueError(str(exc)) from exc

#: maximum candidates considered per decision (bounded work)
MAX_CANDIDATES = 64
#: scoring methodology version recorded in every provenance row
SCORING_METHOD = "ranking-1.0"


class RoutingMode(StrEnum):
    FIXED = "fixed"                      # configured model, must pass all filters
    ELIGIBLE_DEFAULT = "eligible_default"  # explicit configured eligible default
    EVIDENCE_BASED = "evidence_based"    # trusted evidence ranking only


class PrivacyLevel(StrEnum):
    NONE = "none"
    LOCAL_ONLY = "local_only"            # only local-inference providers eligible


class RoutingWeights(BaseModel):
    """Deterministic ranking weights. Each in [0, 1]; the sum must be 1.0
    (within float epsilon). Zero-cost eligibility is NOT a weight - it is a
    mandatory filter applied before any scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quality: float = 0.35
    reliability: float = 0.25
    capability: float = 0.15
    quota: float = 0.10
    privacy: float = 0.10
    speed: float = 0.05

    @model_validator(mode="after")
    def _valid(self) -> RoutingWeights:
        vals = (self.quality, self.reliability, self.capability,
                self.quota, self.privacy, self.speed)
        for v in vals:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"routing weight out of [0, 1]: {v}")
        if abs(sum(vals) - 1.0) > 1e-9:
            raise ValueError(f"routing weights must sum to 1.0, got {sum(vals)}")
        return self


_KNOWN_PROVIDER_PREFIXES = {"openai", "mistral", "anthropic", "gemini", "google", "ollama"}


class RoutingPolicy(BaseModel):
    """Validated, immutable routing configuration (YAML 'routing:' block).

    Automatic routing is OFF by default: with no routing block (or mode=fixed)
    the configured agent model is used, subject to the same eligibility
    filters when a router is wired."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RoutingMode = RoutingMode.FIXED
    weights: RoutingWeights = Field(default_factory=RoutingWeights)
    #: empty = no provider allowlist; otherwise only these provider ids/prefixes
    allowed_providers: tuple[str, ...] = ()
    #: empty = no model allowlist; otherwise only these exact model ids
    allowed_models: tuple[str, ...] = ()
    blocked_models: tuple[str, ...] = ()
    privacy: PrivacyLevel = PrivacyLevel.NONE
    #: explicit authorization for selecting a candidate on a DIFFERENT provider
    #: than the agent's configured model (default: rejected)
    cross_provider_fallback: bool = False
    #: explicit eligible defaults per agent name (or "*" for all agents);
    #: exact model ids, validated
    defaults: dict[str, str] = Field(default_factory=dict)
    context_window_min: int = Field(default=1, ge=1)
    policy_version: str = Field(default="routing-1.0.0", max_length=64)

    @field_validator("allowed_providers")
    @classmethod
    def _providers_known(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for p in v:
            if p.lower() not in _KNOWN_PROVIDER_PREFIXES:
                raise ValueError(f"unknown provider in allowed_providers: {p!r}")
        return tuple(x.lower() for x in v)

    @field_validator("allowed_models", "blocked_models")
    @classmethod
    def _models_parse(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for m in v:
            _parse_or_value_error(m)  # malformed/unknown prefix -> ValueError
        return v

    @field_validator("defaults")
    @classmethod
    def _defaults_valid(cls, v: dict[str, str]) -> dict[str, str]:
        for agent, model in v.items():
            if not agent or len(agent) > 100:
                raise ValueError(f"invalid agent key in routing.defaults: {agent!r}")
            _parse_or_value_error(model)
        return v


# ---- eligibility reason codes ----------------------------------------------------

R_OK = "ok"
R_MODEL_INVALID = "MODEL_INVALID"
R_MODEL_BLOCKED = "POLICY_MODEL_BLOCKED"
R_MODEL_NOT_ALLOWED = "POLICY_MODEL_NOT_ALLOWED"
R_PROVIDER_NOT_ALLOWED = "POLICY_PROVIDER_NOT_ALLOWED"
R_CATALOG_PROVIDER_UNKNOWN = "CATALOG_PROVIDER_UNKNOWN"
R_CATALOG_PROVIDER_DISABLED = "CATALOG_PROVIDER_DISABLED"
R_BILLING_PAID = "BILLING_PAID"
R_BILLING_UNKNOWN = "BILLING_UNKNOWN"
R_BILLING_UNVERIFIED = "BILLING_UNVERIFIED"
R_EVIDENCE_MISSING = "EVIDENCE_MISSING"
R_EVIDENCE_EXPIRED = "EVIDENCE_EXPIRED"
R_ENDPOINT_NOT_LOOPBACK = "ENDPOINT_NOT_LOOPBACK"
R_CATALOG_MODEL_UNKNOWN = "CATALOG_MODEL_UNKNOWN"
R_LIVE_DISPATCH_DENIED = "LIVE_DISPATCH_DENIED"
R_HEALTH_COOLDOWN = "PROVIDER_HEALTH_COOLDOWN"
R_QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
R_CONTEXT_TOO_SMALL = "CONTEXT_WINDOW_TOO_SMALL"
R_CAPABILITY_MISSING = "CAPABILITY_MISSING"
R_TOOL_SUPPORT_MISSING = "TOOL_SUPPORT_MISSING"
R_PRIVACY_LOCAL_ONLY = "PRIVACY_LOCAL_ONLY"
R_CROSS_PROVIDER = "CROSS_PROVIDER_NOT_AUTHORIZED"
R_RUN_BUDGET = "RUN_BUDGET_EXHAUSTED"
R_NO_DEFAULT = "NO_DEFAULT_CONFIGURED"
R_DEFAULT_INELIGIBLE = "DEFAULT_NOT_ELIGIBLE"
R_NO_TRUSTED_EVIDENCE = "NO_TRUSTED_EVIDENCE"


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate's eligibility verdict + (when ranked) deterministic score.
    `components` contains ONLY measured components - unknown quality/speed are
    absent, never fabricated."""

    model: str
    eligible: bool
    reason: str
    score: float | None = None
    components: tuple[tuple[str, float], ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass
class RoutingDecision:
    decision_id: str
    run_id: str | None
    agent: str
    required_capabilities: tuple[str, ...]
    selected_model: str | None
    selection_mode: str          # 'fixed' | 'eligible_default' | 'evidence_based' | 'none'
    reason: str
    candidates: tuple[CandidateEvaluation, ...]
    evidence_refs: tuple[str, ...]
    scoring_method: str
    policy_version: str
    created_at: str


def _is_loopback_http(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "http" and (parts.hostname or "").lower() in (
        "localhost", "127.0.0.1", "::1")


class ModelRouter:
    """Policy-first, deterministic model router. Reads catalog/registry/
    quota/health/store; never dispatches and never grants authorization."""

    def __init__(
        self,
        policy: RoutingPolicy,
        *,
        catalog: ProviderCatalog | None = None,
        cloud_registry: Any = None,
        quota_engine: QuotaEngine | None = None,
        health: HealthRegistry | None = None,
        store: Any = None,
        utcnow: Any = None,
    ) -> None:
        self.policy = policy
        self.catalog = catalog
        self.cloud_registry = cloud_registry
        self.quota_engine = quota_engine
        self.health = health
        self.store = store
        self._utcnow = utcnow or (lambda: datetime.now(UTC))

    # ---- eligibility filters (mandatory, before any scoring) -------------------

    def _catalog_model(self, provider_id: str, model: str) -> CatalogModel | None:
        if self.catalog is None:
            return None
        prov = self.catalog.provider(provider_id)
        if prov is None:
            return None
        for m in prov.models:
            if m.model_id == model:  # exact identity; no substitution
                return m
        return None

    def evaluate(
        self,
        model: str,
        *,
        agent: str | None = None,
        required_capabilities: tuple[str, ...] = (),
        needs_tools: bool = False,
        context_tokens: int = 0,
        configured_model: str | None = None,
        calls_used: int | None = None,
        max_calls: int | None = None,
    ) -> CandidateEvaluation:
        now = self._utcnow()
        # 1) exact identity parses
        try:
            provider, _name = parse_model(model)
        except Exception:
            return CandidateEvaluation(model, False, R_MODEL_INVALID)
        pid = provider.value
        # 2-4) policy allow/block lists
        if model in self.policy.blocked_models:
            return CandidateEvaluation(model, False, R_MODEL_BLOCKED)
        if self.policy.allowed_models and model not in self.policy.allowed_models:
            return CandidateEvaluation(model, False, R_MODEL_NOT_ALLOWED)
        if self.policy.allowed_providers and pid not in self.policy.allowed_providers:
            return CandidateEvaluation(model, False, R_PROVIDER_NOT_ALLOWED)
        # 5) catalog: provider + model identity, status, billing, evidence
        cat_model = None
        if self.catalog is not None:
            prov = self.catalog.provider(pid)
            if prov is None:
                return CandidateEvaluation(model, False, R_CATALOG_PROVIDER_UNKNOWN)
            if prov.operational_status in (OperationalStatus.BLOCKED,
                                           OperationalStatus.DEPRECATED):
                return CandidateEvaluation(model, False, R_CATALOG_PROVIDER_DISABLED)
            b = prov.billing
            if b == BillingClassification.PAID:
                return CandidateEvaluation(model, False, R_BILLING_PAID)
            if b == BillingClassification.UNKNOWN:
                return CandidateEvaluation(model, False, R_BILLING_UNKNOWN)
            if b == BillingClassification.FREE_TIER_UNVERIFIED:
                # an advertised free tier is not evidence of zero billing exposure
                return CandidateEvaluation(model, False, R_BILLING_UNVERIFIED)
            if prov.evidence:
                if not all(e.current(now) for e in prov.evidence):
                    return CandidateEvaluation(model, False, R_EVIDENCE_EXPIRED)
            elif b != BillingClassification.LOCAL_INFERENCE:
                return CandidateEvaluation(model, False, R_EVIDENCE_MISSING)
            if b == BillingClassification.LOCAL_INFERENCE:
                # local inference must stay on loopback endpoints
                if prov.base_endpoints and not all(
                        _is_loopback_http(u) for u in prov.base_endpoints):
                    return CandidateEvaluation(model, False, R_ENDPOINT_NOT_LOOPBACK)
            cat_model = self._catalog_model(pid, model)
            if cat_model is None:
                return CandidateEvaluation(model, False, R_CATALOG_MODEL_UNKNOWN)
        # 6) runtime authorization via the cloud registry (zero-cost gate input;
        #    the gateway gate revalidates again at dispatch time)
        if self.cloud_registry is not None:
            ok, why = self.cloud_registry.live_dispatch_eligible(model, now=now)
            if not ok:
                return CandidateEvaluation(model, False, f"{R_LIVE_DISPATCH_DENIED}:{why}")
        # 7) provider health / cooldown
        if self.health is not None:
            if not self.health.is_available(f"m:{model}") \
                    or not self.health.is_available(f"p:{pid}"):
                return CandidateEvaluation(model, False, R_HEALTH_COOLDOWN)
        # 8) local quota availability (durable engine; the dispatch-time
        #    reservation stays atomic and is what actually prevents
        #    oversubscription - this is only a pre-filter)
        if self.quota_engine is not None:
            try:
                status = self.quota_engine.quota_status(model, now=now)
            except Exception:
                status = {}  # no local pool for this model -> gate enforces at dispatch
            for (_kind, _period), row in status.items():
                limit = row.get("limit")
                if limit is not None and int(row.get("used", 0)) >= int(limit):
                    return CandidateEvaluation(model, False, R_QUOTA_EXHAUSTED)
        # 9) context window
        if cat_model is not None and cat_model.context_window is not None:
            if context_tokens > cat_model.context_window \
                    or self.policy.context_window_min > cat_model.context_window:
                return CandidateEvaluation(model, False, R_CONTEXT_TOO_SMALL)
        # 10) capabilities
        if required_capabilities:
            caps = set(cat_model.capabilities) if cat_model is not None else set()
            missing = [c for c in required_capabilities if c not in caps]
            if missing:
                return CandidateEvaluation(model, False, R_CAPABILITY_MISSING)
        # 11) tool support
        if needs_tools and cat_model is not None and not cat_model.tool_support:
            return CandidateEvaluation(model, False, R_TOOL_SUPPORT_MISSING)
        # 12) privacy restrictions
        if self.policy.privacy == PrivacyLevel.LOCAL_ONLY and self.catalog is not None:
            prov = self.catalog.provider(pid)
            if prov is None or prov.category != ProviderCategory.LLM_LOCAL:
                return CandidateEvaluation(model, False, R_PRIVACY_LOCAL_ONLY)
        # 13) explicit cross-provider fallback authorization
        if configured_model and self.policy.mode != RoutingMode.FIXED:
            try:
                cfg_provider = parse_model(configured_model)[0].value
            except Exception:
                cfg_provider = None
            if cfg_provider is not None and cfg_provider != pid \
                    and not self.policy.cross_provider_fallback:
                return CandidateEvaluation(model, False, R_CROSS_PROVIDER)
        # 14) run budget
        if max_calls is not None and calls_used is not None and calls_used >= max_calls:
            return CandidateEvaluation(model, False, R_RUN_BUDGET)
        return CandidateEvaluation(model, True, R_OK)

    # ---- deterministic scoring (measured components only) -----------------------

    def _score(self, model: str, ev: CandidateEvaluation,
               required_capabilities: tuple[str, ...]) -> CandidateEvaluation:
        if not ev.eligible:
            return ev
        w = self.policy.weights
        components: dict[str, float] = {}
        evidence_refs: list[str] = []
        # quality: ONLY trusted_runtime lease-fenced evidence
        if self.store is not None and "@" in model:
            try:
                evs = trusted_quality_evidence(self.store, model, SUITE_VERSION)
            except Exception:
                evs = []
            if evs:
                best = max(evs, key=lambda r: (r["aggregate_score"], r["evidence_id"]))
                components["quality"] = float(best["aggregate_score"])
                evidence_refs.append(best["evidence_id"])
        # reliability: deterministic from health state (available => 1.0;
        # cooldown already made the candidate ineligible)
        components["reliability"] = 1.0 if (
            self.health is None or self.health.is_available(f"m:{model}")) else 0.0
        # capability fit
        if required_capabilities:
            cat_model = (self._catalog_model(self._pid(model)[0], model)
                         if self.catalog is not None else None)
            caps = set(cat_model.capabilities) if cat_model is not None else set()
            components["capability"] = (
                len(set(required_capabilities) & caps) / len(required_capabilities))
        else:
            components["capability"] = 1.0
        # quota availability fraction (requests, best window)
        if self.quota_engine is not None:
            try:
                status = self.quota_engine.quota_status(model, now=self._utcnow())
            except Exception:
                status = {}
            fracs = []
            for row in status.values():
                limit = row.get("limit")
                if limit:
                    fracs.append(max(0.0, min(1.0, 1.0 - int(row.get("used", 0)) / int(limit))))
            if fracs:
                components["quota"] = min(fracs)
        # privacy: local inference scores 1.0
        if self.catalog is not None:
            prov = self.catalog.provider(self._pid(model)[0])
            components["privacy"] = 1.0 if (
                prov is not None and prov.category == ProviderCategory.LLM_LOCAL) else 0.0
        # speed: NO measurements exist -> deliberately absent (never fabricated)
        # renormalize over measured components only
        num = sum(getattr(w, k) * v for k, v in components.items())
        den = sum(getattr(w, k) for k in components)
        score = num / den if den > 0 else None
        return CandidateEvaluation(
            model, True, R_OK, score, tuple(sorted(components.items())),
            tuple(evidence_refs))

    def _pid(self, model: str) -> tuple[str, str]:
        provider, name = parse_model(model)
        return provider.value, name

    # ---- selection ----------------------------------------------------------------

    def _candidate_models(self, configured_model: str | None) -> list[str]:
        out: set[str] = set()
        if configured_model:
            out.add(configured_model)
        out.update(self.policy.defaults.values())
        if self.policy.allowed_models:
            out.update(self.policy.allowed_models)
        elif self.catalog is not None:
            for prov in self.catalog.providers:
                if self.policy.allowed_providers and prov.id not in self.policy.allowed_providers:
                    continue
                if prov.category not in (ProviderCategory.LLM_LOCAL,
                                         ProviderCategory.LLM_CLOUD):
                    continue
                out.update(m.model_id for m in prov.models)
        return sorted(out)[:MAX_CANDIDATES]

    def select(
        self,
        *,
        agent: str,
        run_id: str | None = None,
        configured_model: str | None = None,
        required_capabilities: tuple[str, ...] = (),
        needs_tools: bool = False,
        context_tokens: int = 0,
        calls_used: int | None = None,
        max_calls: int | None = None,
    ) -> RoutingDecision:
        """Evaluate all candidates (filters first), then apply the configured
        mode. Records bounded provenance in the durable store when wired."""
        now = self._utcnow()
        candidates = self._candidate_models(configured_model)
        evals: list[CandidateEvaluation] = []
        for m in candidates:
            ev = self.evaluate(
                m, agent=agent, required_capabilities=required_capabilities,
                needs_tools=needs_tools, context_tokens=context_tokens,
                configured_model=configured_model, calls_used=calls_used,
                max_calls=max_calls)
            evals.append(self._score(m, ev, required_capabilities)
                         if ev.eligible else ev)

        selected: str | None = None
        mode = self.policy.mode.value
        reason = R_OK
        evidence_refs: tuple[str, ...] = ()

        if self.policy.mode == RoutingMode.FIXED:
            if not configured_model:
                mode, reason = "none", "NO_CONFIGURED_MODEL"
            else:
                ev = next((e for e in evals if e.model == configured_model), None)
                if ev is None:  # configured model not in candidate set: evaluate it
                    ev = self.evaluate(configured_model, agent=agent,
                                       required_capabilities=required_capabilities,
                                       needs_tools=needs_tools,
                                       context_tokens=context_tokens,
                                       configured_model=configured_model,
                                       calls_used=calls_used, max_calls=max_calls)
                if ev.eligible:
                    selected = configured_model
                else:
                    mode, reason = "none", ev.reason
        elif self.policy.mode == RoutingMode.ELIGIBLE_DEFAULT:
            default = self.policy.defaults.get(agent) or self.policy.defaults.get("*")
            if not default:
                mode, reason = "none", R_NO_DEFAULT
            else:
                ev = next((e for e in evals if e.model == default), None)
                if ev is None:
                    ev = self.evaluate(default, agent=agent,
                                       required_capabilities=required_capabilities,
                                       needs_tools=needs_tools,
                                       context_tokens=context_tokens,
                                       configured_model=configured_model,
                                       calls_used=calls_used, max_calls=max_calls)
                    if ev.eligible:
                        ev = self._score(default, ev, required_capabilities)
                        evals.append(ev)
                if ev.eligible:
                    selected = default
                    evidence_refs = ev.evidence_refs
                else:
                    mode, reason = "none", f"{R_DEFAULT_INELIGIBLE}:{ev.reason}"
        else:  # EVIDENCE_BASED
            ranked = [e for e in evals if e.eligible and e.evidence_refs]
            if not ranked:
                mode, reason = "none", R_NO_TRUSTED_EVIDENCE
            else:
                ranked.sort(key=lambda e: (-(e.score or 0.0), e.model))
                best = ranked[0]
                selected = best.model
                evidence_refs = best.evidence_refs

        decision = RoutingDecision(
            decision_id=uuid.uuid4().hex,
            run_id=run_id,
            agent=agent,
            required_capabilities=tuple(required_capabilities),
            selected_model=selected,
            selection_mode=mode if selected else "none",
            reason=reason if selected is None else R_OK,
            candidates=tuple(evals),
            evidence_refs=evidence_refs,
            scoring_method=SCORING_METHOD,
            policy_version=self.policy.policy_version,
            created_at=now.isoformat(timespec="milliseconds"),
        )
        if self.store is not None:
            self._record(decision)
        return decision

    def _record(self, d: RoutingDecision) -> None:
        candidates_json = json.dumps([
            {"model": e.model, "eligible": e.eligible, "reason": e.reason,
             "score": e.score, "components": dict(e.components),
             "evidence_refs": list(e.evidence_refs)}
            for e in d.candidates
        ], sort_keys=True)
        self.store.record_routing_decision(
            decision_id=d.decision_id,
            run_id=d.run_id,
            agent=d.agent,
            required_capabilities=list(d.required_capabilities),
            candidates_json=candidates_json,
            selected_model=d.selected_model,
            selection_mode=d.selection_mode,
            evidence_refs=list(d.evidence_refs),
            scoring_method=d.scoring_method,
            policy_version=d.policy_version,
            now_iso=d.created_at,
        )

    def revalidate(
        self,
        decision: RoutingDecision,
        *,
        configured_model: str | None = None,
        required_capabilities: tuple[str, ...] = (),
        needs_tools: bool = False,
        context_tokens: int = 0,
        calls_used: int | None = None,
        max_calls: int | None = None,
    ) -> tuple[bool, str]:
        """Re-run ALL eligibility filters for the selected model immediately
        before dispatch. Returns (eligible, reason). A decision that no longer
        revalidates must not be dispatched."""
        if decision.selected_model is None:
            return False, decision.reason or "NO_SELECTION"
        ev = self.evaluate(
            decision.selected_model,
            agent=decision.agent,
            required_capabilities=required_capabilities or decision.required_capabilities,
            needs_tools=needs_tools,
            context_tokens=context_tokens,
            configured_model=configured_model,
            calls_used=calls_used,
            max_calls=max_calls,
        )
        return ev.eligible, ev.reason


__all__ = [
    "MAX_CANDIDATES", "SCORING_METHOD", "CandidateEvaluation", "ModelRouter",
    "PrivacyLevel", "RoutingDecision", "RoutingMode", "RoutingPolicy",
    "RoutingWeights",
]
