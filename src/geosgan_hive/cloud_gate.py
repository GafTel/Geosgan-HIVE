"""Phase 4 C4 (hardened): central zero-cost gate in front of EVERY physical
provider dispatch. ModelGateway._dispatch_sync / _dispatch_async are the single
choke points, so sync, async, retry, fallback-candidate and resumed-run paths are
all covered.

Two-phase protocol per physical dispatch:
1. prepare_dispatch — BEFORE the concurrency slot is taken: live-eligibility
   (billing evidence + trusted credential/account binding; production stays
   blocked until trusted verification exists), endpoint binding, bounded
   max_tokens enforced on the request kwargs, estimate_cost enforcement, complete
   payload accounting with a conservative protocol overhead, demonstrably
   conservative live token bounds, and the DURABLE quota reservation under a
   unique dispatch_id (lease-fenced by the worker-owned identity).
2. finalize_dispatch — AFTER the request slot is acquired, IMMEDIATELY BEFORE the
   provider invocation: re-validates eligibility/evidence and atomically
   revalidates lease + quota window + current policy (mark_dispatched, single-use
   dispatch authorization). Any denial means the request is never sent.

Fail-closed rules:
- Zero-cost mode requires a durable, non-:memory: shared quota store.
- Run-bound dispatches require the worker-owned lease identity from the execution
  context (never adopted by reading SQLite); missing run/context/binding fails
  closed.
- Unknown/nonfinite/negative costs and costs above max_spend_usd (0 in zero-cost)
  are rejected; heuristic token estimates are not accepted for live requests.
- The gate has no disable switch and cannot be replaced by an inactive gate while
  a zero-cost registry is wired.

Local quota tracking cannot observe usage outside HIVE and cannot guarantee
provider billing or quotas.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from .checkpoints import CheckpointStore, _add_months, quota_window_for
from .cloud_profiles import (
    BillingModel,
    CloudProfileRegistry,
    TokenEstimate,
    bounded_token_estimate,
)
from .errors import (
    BudgetExceededError,
    CheckpointError,
    CloudConfigError,
    CostUnknownError,
)
from .providers import PROVIDER_SPECS, parse_model
from .quota_engine import QuotaEngine

#: request fields that must never be persisted or digested in plaintext
_SECRET_KEY_MARKERS = ("api_key", "apikey", "authorization", "secret", "credential",
                       "password", "cookie")


def sanitized_request_digest(kwargs: dict[str, Any]) -> str:
    """One-way sha256 over the request payload with credential fields stripped.
    Credentials are never persisted — only this digest of the sanitized payload."""
    clean = {
        k: v for k, v in kwargs.items()
        if not any(m in k.lower() for m in _SECRET_KEY_MARKERS)
    }
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def effective_endpoint(model: str, kwargs: dict[str, Any]) -> str | None:
    """The endpoint the request will actually use: explicit api_base wins, else
    the provider's configured default."""
    provider, _ = parse_model(model)
    return kwargs.get("api_base") or PROVIDER_SPECS[provider].default_api_base


class ZeroCostGate:
    """The single zero-cost enforcement point in front of every physical dispatch.
    Immutable after construction (enabled() derives from the registry and cannot
    be switched off)."""

    def __init__(
        self,
        registry: CloudProfileRegistry,
        store: CheckpointStore | None,
        *,
        adapter: Any,
        utcnow: Any = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        if registry.zero_cost and (
            store is None or getattr(store, "path", None) == ":memory:"
        ):
            raise CloudConfigError(
                "zero-cost mode requires a durable shared quota store (a "
                "CheckpointStore with a persistent path; :memory: is rejected) — "
                "live dispatch fails closed without it"
            )
        self._registry = registry
        self.store = store
        self.engine = QuotaEngine(store, registry, utcnow=utcnow) if store is not None else None
        self.adapter = adapter
        # default the gate clock to the STORE's clock so quota windows and lease
        # checks always use one consistent time source
        self._utcnow = utcnow or (store.utcnow if store is not None
                                  else (lambda: datetime.now(UTC)))
        self.default_max_output_tokens = default_max_output_tokens
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"ZeroCostGate is immutable ({name!r}) — no disable switch")
        object.__setattr__(self, name, value)

    @property
    def registry(self) -> CloudProfileRegistry:
        return self._registry

    def enabled(self) -> bool:
        return bool(self._registry.zero_cost)

    def _now(self) -> datetime:
        n = self._utcnow()
        if not isinstance(n, datetime) or n.tzinfo is None:
            raise CloudConfigError("zero-cost gate requires a tz-aware clock")
        return n

    def _check_evidence_binding(self, model: str, kwargs: dict[str, Any], n: datetime) -> None:
        """Live eligibility incl. evidence; plus endpoint binding when the
        evidence declares one (must match the request's api_base)."""
        ok, reason = self._registry.live_dispatch_eligible(model, now=n)
        if not ok:
            raise CloudConfigError(f"zero-cost gate blocked dispatch of {model!r}: {reason}")
        p = self._registry.profile_for(model)
        ev = p.verification_evidence if p is not None else None
        if ev is not None and ev.endpoint is not None:
            # verify the EFFECTIVE endpoint (explicit api_base or provider default)
            eff = effective_endpoint(model, kwargs)
            if eff is None:
                raise CloudConfigError(
                    f"zero-cost gate blocked dispatch of {model!r}: evidence is bound "
                    f"to endpoint {ev.endpoint!r} but no effective endpoint can be "
                    "verified for this request — fail closed"
                )
            if str(eff).rstrip("/") != ev.endpoint.rstrip("/"):
                raise CloudConfigError(
                    f"zero-cost gate blocked dispatch of {model!r}: effective endpoint "
                    f"{eff!r} does not match the verification evidence endpoint "
                    f"{ev.endpoint!r}"
                )

    def candidate_billing_ok(self, model: str) -> tuple[bool, str]:
        """Candidate-level check for fallback chains: the model needs an enabled
        free_tier_verified profile with current attestation. Full evidence and
        credential/account binding are enforced again at dispatch time. Paid or
        unknown-billing candidates are rejected — no automatic paid fallback."""
        if not self.enabled():
            return True, "zero-cost gate not active"
        n = self._now()
        p = self._registry.profile_for(model)
        if p is None:
            return False, f"no cloud profile for {model!r} (unknown billing fails closed)"
        if not p.enabled:
            return False, f"profile for {model!r} is disabled"
        if p.billing != BillingModel.FREE_TIER_VERIFIED:
            return False, f"billing={p.billing.value} for {model!r} — not zero-cost eligible"
        if not p.verification_current(n):
            return False, f"operator verification for {model!r} expired"
        return True, "ok"

    def _payload_chars(self, kwargs: dict[str, Any]) -> int:
        """Complete serialized provider payload (messages + tool schemas + model),
        plus a conservative protocol overhead: max(256 chars, 10% of payload).
        The api_key is deliberately never serialized."""
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools") or []
        payload = (
            len(json.dumps(messages, default=str))
            + len(json.dumps(tools, default=str))
            + len(str(kwargs.get("model") or ""))
        )
        return payload + max(256, payload // 10)

    def prepare_dispatch(
        self, model: str, kwargs: dict[str, Any], *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
    ) -> str:
        """Phase 1 (before the concurrency slot): validate + durably RESERVE the
        dispatch. Returns the unique dispatch_id; the physical request is NOT yet
        authorized (see finalize_dispatch)."""
        if not self.enabled():
            return ""
        n = self._now()
        self._check_evidence_binding(model, kwargs, n)
        if run_id is not None and (lease_owner is None or lease_generation is None):
            raise CheckpointError(
                f"zero-cost run-bound dispatch of {model!r} requires the worker-owned "
                "lease binding (lease_owner + lease_generation from the execution "
                "context) — missing binding fails closed"
            )
        mo = kwargs.get("max_tokens")
        if mo is None:
            mo = self.default_max_output_tokens
        if type(mo) is not int or mo < 1:
            raise CostUnknownError(
                f"zero-cost dispatch of {model!r} requires a bounded max_tokens "
                f"(got {mo!r}) — unknown/unbounded usage fails closed"
            )
        kwargs["max_tokens"] = mo  # enforced on the actual provider request
        prompt_chars = self._payload_chars(kwargs)
        est: TokenEstimate = bounded_token_estimate(
            self.adapter, model, prompt_chars=prompt_chars,
            max_output_tokens=mo, live=True,
        )
        # cost enforcement: unknown/nonfinite/negative rejected; anything above the
        # configured spend cap (strictly 0 in zero-cost mode) is blocked
        cost = self.adapter.estimate_cost(
            model, prompt_tokens=est.prompt_tokens, max_output_tokens=mo)
        if cost is None:
            raise CostUnknownError(
                f"unknown estimated cost for {model!r} — unknown is not zero; blocked")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) \
                or not math.isfinite(cost) or cost < 0:
            raise CostUnknownError(
                f"invalid estimated cost for {model!r}: {cost!r} (nonfinite/negative/"
                "non-numeric rejected)")
        cap = self._registry.max_spend_usd if self._registry.zero_cost else None
        if cap is not None and cost > cap:
            raise BudgetExceededError(
                f"estimated cost {cost} for {model!r} exceeds the spend cap "
                f"{cap} — potentially billable request blocked in zero-cost mode"
            )
        dispatch_id = uuid4().hex
        assert self.engine is not None
        self.engine.reserve_for_model(
            model, dispatch_id, est, run_id=run_id, lease_owner=lease_owner,
            lease_generation=lease_generation, now=n,
            request_sha256=sanitized_request_digest(kwargs),
            cost_usd=float(cost),
            endpoint=effective_endpoint(model, kwargs),
        )
        return dispatch_id

    def finalize_dispatch(
        self, model: str, dispatch_id: str, *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Phase 2 (after the request slot is held, immediately before the provider
        invocation): revalidate evidence/eligibility (incl. endpoint binding),
        then atomically revalidate lease + quota window + current policy via the
        single-use mark_dispatched. Raises — and the request is never sent — on
        any denial."""
        if not self.enabled():
            return
        n = self._now()
        self._check_evidence_binding(model, kwargs or {}, n)
        assert self.engine is not None
        if not self.engine.mark_dispatched(
            dispatch_id, run_id=run_id, lease_owner=lease_owner,
            lease_generation=lease_generation, now=n,
            request_sha256=sanitized_request_digest(kwargs) if kwargs else None,
        ):
            raise CheckpointError(
                f"zero-cost gate: dispatch authorization denied for {dispatch_id} "
                "(already used / quota window rolled / policy changed / lease fenced) "
                "— not dispatched"
            )

    def authorize_physical_dispatch(
        self, model: str, kwargs: dict[str, Any], *, run_id: str | None = None,
        lease_owner: str | None = None, lease_generation: int | None = None,
    ) -> str:
        """Convenience one-shot (prepare + finalize) for non-gateway callers."""
        dispatch_id = self.prepare_dispatch(
            model, kwargs, run_id=run_id, lease_owner=lease_owner,
            lease_generation=lease_generation)
        if dispatch_id:
            self.finalize_dispatch(
                model, dispatch_id, run_id=run_id, lease_owner=lease_owner,
                lease_generation=lease_generation, kwargs=kwargs)
        return dispatch_id

    def quota_retry_at(self, model: str, now: datetime | None = None) -> datetime | None:
        """Reliable LOCAL retry evidence for a quota-suspended run: the start of the
        next quota window derived from the PERSISTED policy anchor (own accounting,
        never an invented provider reset time). None when no policy/pool exists."""
        if not self.enabled() or self.store is None:
            return None
        n = now or self._now()
        pool = self._registry.quota_pool(model)
        if pool is None:
            return None
        pol = self.store.quota_policy(pool)
        if pol is None:
            return None
        anchor = pol["anchor_iso"]
        now_iso = n.isoformat(timespec="milliseconds")
        best: datetime | None = None
        for period in pol["periods"]:
            w = datetime.fromisoformat(quota_window_for(anchor, period, now_iso))
            if period == "day":
                nxt = w + timedelta(days=1)
            elif period == "week":
                nxt = w + timedelta(days=7)
            else:
                nxt = _add_months(w, 1)
            if best is None or nxt < best:
                best = nxt
        return best
