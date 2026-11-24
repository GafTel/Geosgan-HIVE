from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .checkpoints import ApprovalState, CheckpointStore
from .errors import ApprovalError
from .run_context import EventType
from .tool_spec import ToolSpec, args_sha256, canonical_args

if TYPE_CHECKING:
    from .run_context import RunContext
    from .tool_gateway import ToolExecutionContext

#: deciders that can never approve (model/agent claims are structurally impossible:
#: approvals only enter through the trusted CLI path, but we still reject these ids)
FORBIDDEN_DECIDERS = ("", "model", "agent", "llm", "assistant")


@dataclass
class ApprovalRequest:
    approval_id: str
    run_id: str | None
    task_id: str | None
    agent: str
    tool: str
    tool_version: str
    operation: str
    target: str | None
    args_sha256: str
    preview: str
    expires_at: str


class ApprovalManager:
    """Durable human-approval workflow (CLI-only operator interface in Phase 3).

    An approval is bound to: run_id, task_id, agent identity, tool identity+version,
    canonical validated args (+sha256), exact operation/target, file preconditions,
    expiry, and a unique id. Consumption is a single atomic transaction (single-use,
    args-hash-bound, expiry-checked) paired with the ledger execution-start row, so
    replay and argument substitution are impossible. Only the trusted operator
    interface (CLI) may decide; model-generated approval claims never reach this API.

    Local CLI authorization assumes a trusted OS user (documented).
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        expiry_seconds: float = 600.0,
        preview_bytes: int = 2_000,
    ) -> None:
        self.store = store
        self.expiry_seconds = float(expiry_seconds)
        self.preview_bytes = int(preview_bytes)

    # ---- creation ----
    def create(
        self,
        *,
        ctx: ToolExecutionContext,
        spec: ToolSpec,
        args: dict[str, Any],
        operation: str,
        target: str | None = None,
        preconditions: dict[str, Any] | None = None,
        run_context: RunContext | None = None,
    ) -> ApprovalRequest:
        import json as _json

        approval_id = uuid4().hex
        canonical = canonical_args(args)
        digest = args_sha256(args)
        preview = self._build_preview(
            spec=spec, operation=operation, target=target, args=args
        )
        expires_at = (
            self.store.utcnow() + timedelta(seconds=self.expiry_seconds)
        ).isoformat(timespec="milliseconds")
        from .tool_gateway import redact_secrets

        self.store.create_approval(
            approval_id=approval_id,
            run_id=ctx.run_id or "",
            task_id=ctx.task_id,
            agent=ctx.agent,
            tool=spec.name,
            tool_version=spec.version,
            operation=operation,
            target=target,
            # bounded + secret-redacted payload; the hash is authoritative
            args_json=redact_secrets(canonical)[: 8_000],
            args_sha256=digest,
            preconditions_json=_json.dumps(preconditions or {}, default=str)[: 4_000],
            preview=preview,
            expires_at=expires_at,
        )
        if run_context is not None:
            run_context.record_event(
                EventType.APPROVAL_REQUESTED, task_id=ctx.task_id, source_agent=ctx.agent,
                metadata={
                    "approval_id": approval_id, "tool": spec.name, "operation": operation,
                    "target": target, "expires_at": expires_at,
                },
            )
        return ApprovalRequest(
            approval_id=approval_id, run_id=ctx.run_id, task_id=ctx.task_id,
            agent=ctx.agent, tool=spec.name, tool_version=spec.version,
            operation=operation, target=target, args_sha256=digest,
            preview=preview, expires_at=expires_at,
        )

    def _build_preview(self, *, spec: ToolSpec, operation: str,
                       target: str | None, args: dict[str, Any]) -> str:
        """Bounded, meaningful human preview (a hash alone is not sufficient)."""
        import json as _json

        from .tool_gateway import redact_secrets

        arg_text = redact_secrets(_json.dumps(args, ensure_ascii=False, default=str))
        if len(arg_text) > self.preview_bytes:
            arg_text = arg_text[: self.preview_bytes] + "...[truncated]"
        lines = [
            f"tool={spec.name} v{spec.version}  operation={operation}",
            f"target={target or '-'}",
            f"side_effect={spec.side_effect.value} idempotent={spec.idempotent}",
            f"args={arg_text}",
            f"args_sha256={args_sha256(args)[:16]}...",
        ]
        return "\n".join(lines)

    # ---- operator decisions (trusted CLI path only) ----
    def decide(self, approval_id: str, *, approve: bool, operator: str,
               cancel: bool = False) -> bool:
        decider = operator.strip().lower()
        if decider in FORBIDDEN_DECIDERS or decider.startswith("agent:"):
            raise ApprovalError(
                f"decider {operator!r} is not a trusted operator — only the CLI "
                "operator interface may decide approvals"
            )
        decision = (
            ApprovalState.CANCELLED.value if cancel
            else ApprovalState.APPROVED.value if approve
            else ApprovalState.DENIED.value
        )
        ok = self.store.decide_approval(approval_id, decision=decision, operator=operator)
        if not ok:
            raise ApprovalError(
                f"approval {approval_id!r} cannot be decided (unknown, already decided, "
                "or expired)"
            )
        return True

    def get(self, approval_id: str) -> dict[str, Any] | None:
        return self.store.get_approval(approval_id)

    def list_pending(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        self.store.sweep_expired_approvals()
        return self.store.list_approvals(
            run_id=run_id, status=ApprovalState.PENDING.value
        )

    def preview(self, approval_id: str) -> str:
        row = self.store.get_approval(approval_id)
        if row is None:
            raise ApprovalError(f"unknown approval {approval_id!r}")
        return str(row["preview"])

    def consume(self, approval_id: str, *, expected_args_sha256: str,
                expected_tool_version: str | None = None,
                expected_run_id: str | None = None,
                expected_agent: str | None = None,
                expected_tool: str | None = None) -> bool:
        """Atomically consume (single-use, hash-bound, expiry-checked, and bound to
        the expected tool version / run / agent / tool when given)."""
        return self.store.consume_approval(
            approval_id, expected_args_sha256=expected_args_sha256,
            expected_tool_version=expected_tool_version,
            expected_run_id=expected_run_id, expected_agent=expected_agent,
            expected_tool=expected_tool,
        )

    def sweep_expired(self) -> int:
        return self.store.sweep_expired_approvals()
