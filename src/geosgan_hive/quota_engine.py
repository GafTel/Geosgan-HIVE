"""Phase 4 C3: durable SQLite Quota Engine.

Atomic, persistent quota reservations shared across runs and processes, built on
top of CheckpointStore schema v6 (quota_reservations + quota_usage).

Design rules (approved C3 spec):
- Every applicable account/model quota is enforced in ONE atomic transaction
  (store.reserve_quota_batch, BEGIN IMMEDIATE), BEFORE dispatch.
- Constraints are normalized by provider/account pool, period and limit kind
  (requests/tokens); account-wide quotas apply to ALL models of the account — no
  model bypass. Multiple pools for one model are ambiguous and fail closed.
- Request count AND conservative token bounds (prompt + max output) are reserved
  per dispatch; unique dispatch_id per physical dispatch; idempotent
  reconciliation; conflicting duplicates rejected.
- Unknown quota reset semantics fail closed: a profile without an explicit
  tz-aware quota_window_start, or with a future anchor, is rejected — reset times
  are never invented.
- Integer accounting only: bool/float/str amounts, negative values and unbounded
  output caps are rejected (CostUnknownError / CheckpointError).
- Stale workers are fenced by the existing lease owner+generation mechanism when a
  run_id is bound to the reservation.
- Usage is never refunded automatically after crashes; there is no decrement path.
- Configuration changes never reset consumed quotas: usage rows are keyed by
  (pool, kind, period, window_start) and historical windows are retained.

This engine enforces LOCAL quota accounting only. It cannot observe usage outside
HIVE and cannot guarantee provider-side billing or quota resets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .checkpoints import CheckpointStore, quota_window_for
from .cloud_profiles import CloudProfileRegistry, TokenEstimate
from .errors import CheckpointError, CloudConfigError, CostUnknownError


@dataclass(frozen=True)
class QuotaReservation:
    """Result of a successful atomic quota reservation (before dispatch)."""

    dispatch_id: str
    pool: str
    items: tuple[dict[str, Any], ...]
    status: str


class QuotaEngine:
    """Durable quota engine bound to one CheckpointStore and one immutable
    CloudProfileRegistry. The clock is injectable for deterministic tests; naive
    datetimes are rejected."""

    def __init__(
        self,
        store: CheckpointStore,
        registry: CloudProfileRegistry,
        *,
        utcnow: Any = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._registry = registry
        self._utcnow = utcnow

    def _now(self, now: datetime | None) -> datetime:
        n = now if now is not None else self._utcnow()
        if not isinstance(n, datetime) or n.tzinfo is None:
            raise CloudConfigError(
                "QuotaEngine requires a timezone-aware clock (naive rejected)"
            )
        return n

    def _resolve(self, model: str, now: datetime) -> tuple[Any, str, datetime, tuple]:
        """Resolve profile, single pool, window anchor and all account quotas."""
        profile = self._registry.profile_for(model)
        if profile is None:
            raise CloudConfigError(
                f"no cloud profile for model {model!r} — unknown account/quota "
                "semantics fail closed"
            )
        quotas = self._registry.quotas_for(model)  # account-wide: no model bypass
        if not quotas:
            raise CloudConfigError(
                f"no bounded quota constraints for {model!r} — fail closed"
            )
        pools = {q.pool for q in quotas}
        if len(pools) > 1:
            raise CloudConfigError(
                f"ambiguous multi-pool quota constraints for {model!r}: "
                f"{sorted(pools)} — fail closed"
            )
        anchor = profile.quota_window_start
        if anchor is None:
            raise CloudConfigError(
                f"profile for {model!r} has no quota_window_start — unknown reset "
                "semantics fail closed (reset times are never invented)"
            )
        if anchor.tzinfo is None:
            raise CloudConfigError("quota_window_start must be timezone-aware")
        if anchor > now:
            raise CloudConfigError(
                f"quota_window_start {anchor.isoformat()} is in the future relative "
                f"to now {now.isoformat()} — fail closed"
            )
        return profile, next(iter(pools)), anchor, quotas

    def reserve_for_model(
        self,
        model: str,
        dispatch_id: str,
        token_estimate: TokenEstimate,
        *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now: datetime | None = None,
    ) -> QuotaReservation:
        """Reserve quota for ONE physical dispatch of `model`, before it is sent.

        Reserves 1 request + (prompt + max_output) tokens in every period window
        that carries the corresponding limit. Raises QuotaExhaustedError when any
        limit would be exceeded (nothing persisted), CheckpointError when lease
        fencing rejects the reservation (stale worker), CloudConfigError for
        unknown model/quotas/anchor, CostUnknownError for non-strict or unbounded
        token estimates."""
        n = self._now(now)
        if not isinstance(dispatch_id, str) or not dispatch_id:
            raise CloudConfigError("dispatch_id must be a non-empty unique string")
        _profile, pool, anchor, quotas = self._resolve(model, n)
        if not isinstance(token_estimate, TokenEstimate):
            raise CostUnknownError("token_estimate must be a TokenEstimate")
        pt, mo = token_estimate.prompt_tokens, token_estimate.max_output_tokens
        if type(pt) is not int or type(mo) is not int:  # bool/float/str rejected
            raise CostUnknownError(
                "token estimate values must be strict ints "
                f"(got {type(pt).__name__}/{type(mo).__name__})"
            )
        if pt < 0:
            raise CostUnknownError("prompt_tokens must be nonnegative")
        if mo < 1:
            raise CostUnknownError(
                f"unbounded max_output_tokens for {model!r} — fail closed"
            )
        total_tokens = pt + mo
        items: list[dict[str, Any]] = []
        limits: dict[tuple[str, str], int] = {}
        for q in quotas:
            if q.max_requests is not None:
                items.append({"kind": "requests", "period": q.period.value, "amount": 1})
                limits[("requests", q.period.value)] = q.max_requests
            if q.max_tokens is not None:
                items.append({"kind": "tokens", "period": q.period.value,
                              "amount": total_tokens})
                limits[("tokens", q.period.value)] = q.max_tokens
        ok = self._store.reserve_quota_batch(
            dispatch_id,
            pool=pool,
            items=items,
            anchor_iso=anchor.isoformat(),
            limits=limits,
            now_iso=n.isoformat(timespec="milliseconds"),
            run_id=run_id,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
        if not ok:
            raise CheckpointError(
                f"lease fencing rejected quota reservation {dispatch_id!r} "
                "(stale worker / lost lease)"
            )
        return QuotaReservation(
            dispatch_id=dispatch_id, pool=pool, items=tuple(items),
            status="reserved",
        )

    def mark_dispatched(
        self,
        dispatch_id: str,
        *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
    ) -> bool:
        """Mark a reserved dispatch as dispatched (fenced; idempotent; never
        refunds usage)."""
        return self._store.mark_dispatched(
            dispatch_id, run_id=run_id, lease_owner=lease_owner,
            lease_generation=lease_generation,
        )

    def reservation(self, dispatch_id: str) -> dict[str, Any] | None:
        return self._store.get_reservation(dispatch_id)

    def quota_status(
        self, model: str, *, now: datetime | None = None
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Current-window usage vs limit per (kind, period) for the model's
        account pool. Historical windows stay in the store and are never reset."""
        n = self._now(now)
        _profile, pool, anchor, quotas = self._resolve(model, n)
        now_iso = n.isoformat(timespec="milliseconds")
        rows = self._store.quota_status(pool)
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for q in quotas:
            for kind, limit in (("requests", q.max_requests), ("tokens", q.max_tokens)):
                if limit is None:
                    continue
                window = quota_window_for(anchor.isoformat(), q.period.value, now_iso)
                used = 0
                for r in rows:
                    if (r["kind"] == kind and r["period"] == q.period.value
                            and r["window_start"] == window):
                        used = int(r["used"])
                out[(kind, q.period.value)] = {
                    "used": used, "limit": limit, "window_start": window, "pool": pool,
                }
        return out
