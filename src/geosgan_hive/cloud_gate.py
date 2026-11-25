"""Phase 4 C4: central zero-cost gate enforced before EVERY physical provider
dispatch (sync, async, retry, fallback candidate and resumed runs all funnel
through ModelGateway._dispatch_sync / _dispatch_async).

Fail-closed rules (approved C4 spec):
- In zero-cost mode a durable shared quota store is REQUIRED.
- Every dispatch needs a live-eligible profile: free_tier_verified billing,
  current attestation AND current VerificationEvidence bound to the account
  fingerprint with the trusted credential/account binding implemented. Operator
  attestation alone never enables live dispatch (production stays blocked until
  trusted billing verification exists — C2 gate).
- Conservative token/request bounds are reserved BEFORE dispatch via the durable
  QuotaEngine, under a unique dispatch_id; dispatch authorization
  (mark_dispatched) is single-use — a replay can never authorize another request.
- max_tokens is always bounded and is set on the actual provider request kwargs;
  unbounded output fails closed (CostUnknownError).
- Prompt accounting covers the COMPLETE serialized prompt (messages + tool
  schemas + model + protocol overhead); the api_key is never serialized.
- LIVE token estimation requires an explicit live-capable adapter
  (bounded_token_estimate(live=True)); test-only/mock adapters (e.g.
  FakeCloudAdapter) are rejected for live enforcement — mock estimates are never
  used for live requests.
- QuotaExhaustedError (local durable quota) is distinguished from transient
  RateLimitError (HTTP 429, retryable, Retry-After honored, never invented).
- No automatic paid fallback: fallback candidates must pass the zero-cost
  candidate check (free_tier_verified profile) or they are rejected.
- Lease fencing: run-bound dispatches are bound to the run's lease
  owner+generation; stale workers are blocked by the store.

Local quota tracking cannot observe usage outside HIVE and cannot guarantee
provider billing or quotas.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .checkpoints import CheckpointStore
from .cloud_profiles import (
    BillingModel,
    CloudProfileRegistry,
    TokenEstimate,
    bounded_token_estimate,
)
from .errors import CheckpointError, CloudConfigError, CostUnknownError
from .quota_engine import QuotaEngine


class ZeroCostGate:
    """The single zero-cost enforcement point in front of every physical dispatch."""

    def __init__(
        self,
        registry: CloudProfileRegistry,
        store: CheckpointStore | None,
        *,
        adapter: Any,
        utcnow: Any = lambda: datetime.now(UTC),
        default_max_output_tokens: int | None = None,
        protocol_overhead_chars: int = 512,
    ) -> None:
        if registry.zero_cost and store is None:
            raise CloudConfigError(
                "zero-cost mode requires durable shared quota storage "
                "(a CheckpointStore) — live dispatch fails closed without it"
            )
        self.registry = registry
        self.store = store
        self.engine = QuotaEngine(store, registry, utcnow=utcnow) if store is not None else None
        self.adapter = adapter
        self._utcnow = utcnow
        self.default_max_output_tokens = default_max_output_tokens
        self.protocol_overhead_chars = protocol_overhead_chars

    def enabled(self) -> bool:
        return bool(self.registry.zero_cost)

    def _now(self) -> datetime:
        n = self._utcnow()
        if not isinstance(n, datetime) or n.tzinfo is None:
            raise CloudConfigError("zero-cost gate requires a tz-aware clock")
        return n

    def candidate_billing_ok(self, model: str) -> tuple[bool, str]:
        """Candidate-level check for fallback chains: the model needs an enabled
        free_tier_verified profile with current attestation. Full evidence and
        credential/account binding are enforced again at dispatch time. Paid or
        unknown-billing candidates are rejected — no automatic paid fallback."""
        if not self.enabled():
            return True, "zero-cost gate not active"
        n = self._now()
        p = self.registry.profile_for(model)
        if p is None:
            return False, f"no cloud profile for {model!r} (unknown billing fails closed)"
        if not p.enabled:
            return False, f"profile for {model!r} is disabled"
        if p.billing != BillingModel.FREE_TIER_VERIFIED:
            return False, f"billing={p.billing.value} for {model!r} — not zero-cost eligible"
        if not p.verification_current(n):
            return False, f"operator verification for {model!r} expired"
        return True, "ok"

    def authorize_physical_dispatch(
        self, model: str, kwargs: dict[str, Any], *, run_id: str | None = None,
    ) -> str:
        """Enforce the zero-cost policy for ONE physical dispatch and return the
        dispatch_id ONLY when authorized. Raises (and the request is never sent)
        on: ineligible billing/evidence/credential binding, unknown cost, unbounded
        output, quota exhaustion, fencing failure or double authorization."""
        if not self.enabled():
            return ""
        n = self._now()
        ok, reason = self.registry.live_dispatch_eligible(model, now=n)
        if not ok:
            raise CloudConfigError(
                f"zero-cost gate blocked dispatch of {model!r}: {reason}"
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
        # complete serialized prompt accounting: messages + tool schemas + model +
        # protocol overhead; the api_key is deliberately never serialized
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools") or []
        prompt_chars = (
            len(json.dumps(messages, default=str))
            + len(json.dumps(tools, default=str))
            + len(str(kwargs.get("model") or ""))
            + self.protocol_overhead_chars
        )
        est: TokenEstimate = bounded_token_estimate(
            self.adapter, model, prompt_chars=prompt_chars,
            max_output_tokens=mo, live=True,
        )
        dispatch_id = uuid4().hex
        bound_run, owner, gen = None, None, None
        if run_id is not None and self.store is not None:
            row = self.store.get(run_id)
            if row is not None:
                bound_run, owner, gen = run_id, row.lease_owner, row.lease_generation
        assert self.engine is not None
        self.engine.reserve_for_model(
            model, dispatch_id, est, run_id=bound_run, lease_owner=owner,
            lease_generation=gen, now=n,
        )
        if not self.engine.mark_dispatched(
            dispatch_id, run_id=bound_run, lease_owner=owner,
            lease_generation=gen, now=n,
        ):
            raise CheckpointError(
                f"zero-cost gate: dispatch authorization denied for {dispatch_id} "
                "(already used / quota window rolled / lease fenced) — not dispatched"
            )
        return dispatch_id
