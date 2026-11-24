from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .errors import (
    ApprovalError,
    ApprovalRequiredError,
    ToolBudgetExceededError,
    ToolError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
)
from .run_context import EventType, utcnow
from .tool_policy import ToolPolicy
from .tool_registry import ToolAdapter, ToolCatalog
from .tool_spec import SideEffect, ToolSpec, args_sha256, validate_args

if TYPE_CHECKING:
    from .approvals import ApprovalManager
    from .run_context import RunContext

#: Known secret shapes redacted from tool outputs, logs and errors. Regex redaction
#: CANNOT detect every possible secret — it is a mitigation, not a boundary.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
]


def redact_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***REDACTED***", text)
    return text


@dataclass
class ToolExecutionContext:
    """TRUSTED runtime identity for one tool call. run_id/task_id/agent/request_id
    always originate from the runtime — never from model-generated arguments."""

    agent: str
    run_id: str | None = None
    task_id: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utcnow)


@dataclass
class ToolResult:
    """Structured execution result. `output` is the redacted, size-bounded,
    model-visible projection; raw payloads are not persisted by default."""

    tool: str
    agent: str
    ok: bool
    output: str = ""
    error: str | None = None
    status: str = "completed"  # completed | rejected | failed | timeout | waiting_approval
    execution_id: str = ""
    duration_ms: float = 0.0
    output_bytes: int = 0
    truncated: bool = False


class BuiltinToolAdapter:
    """Wraps a legacy `tools.Tool` (calculator, current_datetime, word_count, ...)
    as a ToolAdapter, preserving its behavior exactly."""

    def __init__(self, tool: Any, spec: ToolSpec) -> None:
        self._tool = tool
        self.spec = spec

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        return self._tool.run(args)

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        return await asyncio.to_thread(self.execute, args, ctx)


class ToolGateway:
    """The single execution boundary for ALL tools in production.

    Flow: catalog lookup (fail closed) -> policy check -> strict schema/size
    validation -> atomic budget reservation (+ durable SQLite reservation before
    dispatch when wired) -> approval (fail closed if required but unconfigured)
    -> execution-ledger start for mutating ops -> bounded execution under a
    SEPARATE tool semaphore (never the model semaphore, never held during approval
    waits) with timeout -> redact/truncate -> ToolResult + audit event.

    Missing security context fails closed when `strict` is set (production wiring).
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        policy: ToolPolicy,
        *,
        store: Any = None,
        approval_manager: ApprovalManager | None = None,
        strict: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.catalog = catalog
        self.policy = policy
        self._store = store
        self.approval_manager = approval_manager
        self.strict = strict
        self._sleep = sleep
        self._async_sleep = async_sleep

    def set_durable_store(self, store: Any) -> None:
        self._store = store

    def schemas_for(self, agent: str) -> list[dict[str, Any]]:
        return self.catalog.schemas_for(self.policy, agent)

    # ---- reservation ----
    def _check_context(self, ctx: ToolExecutionContext, run_context: RunContext | None) -> None:
        if self.strict and (run_context is None or not ctx.run_id):
            raise ToolPermissionError(
                "missing security context (run_id/run context) — failing closed"
            )

    def _reserve(self, ctx: ToolExecutionContext, spec: ToolSpec,
                 run_context: RunContext | None) -> None:
        if run_context is not None:
            ok, reason = run_context.reserve_tool_call(agent=ctx.agent, tool=spec.name)
            if not ok:
                raise ToolBudgetExceededError(reason or "tool budget exhausted")
            if self._store is not None and ctx.run_id:
                # durable reservation BEFORE execution (atomic SQLite increment)
                if not self._store.reserve_tool_durable(ctx.run_id):
                    run_context.release_tool_reservation(agent=ctx.agent, tool=spec.name)
                    raise ToolPermissionError(
                        f"run {ctx.run_id!r} is no longer active — dispatch blocked (fail closed)"
                    )

    def _approval_needed(self, spec: ToolSpec, ctx: ToolExecutionContext, tool: str) -> bool:
        return self.policy.requires_approval(spec, ctx.agent, tool)

    # ---- audit ----
    def _record(self, ctx: ToolExecutionContext, run_context: RunContext | None, *,
                tool: str, status: str, execution_id: str, error: str | None = None,
                duration_ms: float = 0.0, output_bytes: int = 0) -> None:
        if run_context is None:
            return
        event = EventType.TOOL_EXECUTED if status == "completed" else EventType.TOOL_REJECTED
        run_context.record_event(
            event, task_id=ctx.task_id, source_agent=ctx.agent,
            metadata={
                "tool": tool, "status": status, "execution_id": execution_id,
                "duration_ms": round(duration_ms, 3), "output_bytes": output_bytes,
                **({"error": error} if error else {}),
            },
        )

    # ---- output shaping ----
    def _shape_output(self, spec: ToolSpec, output: str) -> tuple[str, int, bool]:
        output = redact_secrets(output)
        data = output.encode("utf-8")
        truncated = len(data) > spec.max_output_bytes
        if truncated:
            output = output.encode("utf-8")[: spec.max_output_bytes].decode("utf-8", "ignore")
            output += "\n...[truncated]"
        return output, len(output.encode("utf-8")), truncated

    # ---- approval + ledger (Phase 3 T4) ----
    def _approval_context(self, adapter: ToolAdapter, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(adapter, "approval_context", None)
        if fn is not None:
            return fn(args)
        return {"operation": adapter.spec.name, "target": None, "preconditions": {}}

    def _run_lease(self, ctx: ToolExecutionContext) -> dict[str, Any] | None:
        """Current run row (for ownership fencing); None when no durable store."""
        if self._store is None or not ctx.run_id:
            return None
        return self._store.get(ctx.run_id)

    def _obtain_approval(self, spec: ToolSpec, ctx: ToolExecutionContext,
                         args: dict[str, Any], adapter: ToolAdapter,
                         run_context: RunContext | None, wait) -> tuple[str, Any]:
        """Create a PENDING approval, wait for the operator (NO semaphore held),
        revalidate policy/lease/expiry, then consume atomically. Returns
        (approval_id, run row)."""
        mgr = self.approval_manager
        assert mgr is not None
        actx = self._approval_context(adapter, args)
        req = mgr.create(
            ctx=ctx, spec=spec, args=args, operation=actx["operation"],
            target=actx.get("target"), preconditions=actx.get("preconditions") or {},
            run_context=run_context,
        )
        # wait for the operator decision (bounded by approval expiry)
        from datetime import UTC, datetime

        def _now():
            if self._store is not None:
                return self._store.utcnow()
            return datetime.now(UTC)

        deadline = datetime.fromisoformat(req.expires_at)
        while True:
            row = mgr.get(req.approval_id)
            status = row["status"] if row else "unknown"
            if status == "approved":
                break
            if status in ("denied", "cancelled", "expired", "unknown"):
                raise ApprovalError(
                    f"approval {req.approval_id} was {status}; action not authorized"
                )
            if _now() >= deadline:
                mgr.sweep_expired()
                raise ApprovalError(f"approval {req.approval_id} expired")
            wait(0.2)
        # revalidate EVERYTHING immediately before execution
        if not self.policy.allows(ctx.agent, spec.name):
            raise ToolPermissionError("policy changed since request — re-denied")
        digest = args_sha256(args)
        run_row = self._run_lease(ctx)
        if self.strict:
            if run_row is None or run_row.state.value != "running":
                raise ToolPermissionError("run not active — fail closed")
        if not mgr.consume(req.approval_id, expected_args_sha256=digest):
            raise ApprovalError(
                "approval consume failed: replay, argument substitution or expiry"
            )
        return req.approval_id, run_row

    def _ledger_start(self, spec: ToolSpec, ctx: ToolExecutionContext,
                      run_context: RunContext | None, execution_id: str,
                      args: dict[str, Any], approval_id: str,
                      run_row: Any) -> None:
        """Persist the execution-start record BEFORE the side effect. Mutating tools
        REQUIRE a durable ledger and valid ownership — otherwise fail closed."""
        mutating = spec.side_effect in (SideEffect.LOCAL_MUTATION, SideEffect.EXTERNAL_MUTATION)
        if self._store is None or not ctx.run_id:
            if mutating:
                raise ToolPermissionError(
                    f"tool '{spec.name}' mutates state but no durable ledger is "
                    "configured — failing closed"
                )
            return
        if mutating:
            if run_row is None or run_row.state.value != "running":
                raise ToolPermissionError(
                    f"tool '{spec.name}' requires a valid lease owner — failing closed"
                )
        from .tool_spec import canonical_args

        ok = self._store.ledger_start(
            execution_id=execution_id, run_id=ctx.run_id, task_id=ctx.task_id,
            agent=ctx.agent, tool=spec.name, tool_version=spec.version,
            args_sha256=args_sha256(args),
            # hash is computed on the ORIGINAL args (authoritative binding); the
            # persisted payload is bounded AND secret-redacted
            args_json=redact_secrets(canonical_args(args))[: 8_000],
            status="started" if mutating else "reserved",
            approval_id=approval_id or None,
            lease_owner=(run_row.lease_owner if run_row is not None else None),
            lease_generation=(run_row.lease_generation if run_row is not None else None),
        )
        if not ok:
            raise ToolPermissionError(
                f"duplicate execution id {execution_id} — refusing to double-dispatch"
            )

    def _ledger_finish(self, spec: ToolSpec, execution_id: str, status: str,
                       run_row: Any, *, error: str | None = None,
                       output: str = "") -> None:
        if self._store is None:
            return
        import hashlib as _hl

        self._store.ledger_update(
            execution_id, status=status, error=error,
            result_sha256=_hl.sha256(output.encode()).hexdigest() if output else None,
            lease_owner=(run_row.lease_owner if run_row is not None else None),
            lease_generation=(run_row.lease_generation if run_row is not None else None),
        )

    # ---- sync path ----
    def execute(self, tool: str, args: Any, *, ctx: ToolExecutionContext,
                run_context: RunContext | None = None) -> ToolResult:
        execution_id = uuid4().hex
        started = time.perf_counter()
        reg = None
        run_row = None
        approval_id = ""
        try:
            self._check_context(ctx, run_context)
            reg = self.catalog.get(tool)
            if reg is None:
                raise ToolNotFoundError(f"unknown tool '{tool}' — denied")
            spec = reg.spec
            if not self.policy.allows(ctx.agent, tool):
                raise ToolPermissionError(
                    f"agent '{ctx.agent}' is not allowed to use tool '{tool}'"
                )
            validated = validate_args(spec, args)
            self._reserve(ctx, spec, run_context)
            if self._approval_needed(spec, ctx, tool):
                if self.approval_manager is None:
                    raise ToolPermissionError(
                        f"tool '{tool}' requires approval but no ApprovalManager is "
                        "configured — failing closed"
                    )
                approval_id, run_row = self._obtain_approval(
                    spec, ctx, validated, reg.adapter, run_context, self._sleep
                )
            else:
                run_row = self._run_lease(ctx)
            self._ledger_start(spec, ctx, run_context, execution_id, validated,
                               approval_id, run_row)
            output = self._run_sync(reg.adapter, validated, ctx, spec, run_context)
            text, nbytes, truncated = self._shape_output(spec, output)
            self._ledger_finish(spec, execution_id, "completed", run_row, output=text)
            result = ToolResult(
                tool=tool, agent=ctx.agent, ok=True, output=text, status="completed",
                execution_id=execution_id, duration_ms=(time.perf_counter() - started) * 1000,
                output_bytes=nbytes, truncated=truncated,
            )
        except ToolError as exc:
            status = "rejected"
            if isinstance(exc, ApprovalRequiredError):
                status = "waiting_approval"
            elif isinstance(exc, ToolTimeoutError):
                status = "timeout"
            elif isinstance(exc, ApprovalError):
                status = "rejected"
            result = ToolResult(
                tool=tool, agent=ctx.agent, ok=False, error=str(exc), status=status,
                execution_id=execution_id, duration_ms=(time.perf_counter() - started) * 1000,
            )
            if reg is not None and self._store is not None and ctx.run_id:
                mutating = reg.spec.side_effect in (SideEffect.LOCAL_MUTATION,
                                                    SideEffect.EXTERNAL_MUTATION)
                final = "unknown" if isinstance(exc, ToolTimeoutError) and mutating else (
                    "failed" if status == "rejected" and execution_id else "rejected"
                )
                self._ledger_finish(reg.spec, execution_id, final, run_row, error=str(exc))
        self._record(ctx, run_context, tool=tool, status=result.status,
                     execution_id=execution_id, error=result.error,
                     duration_ms=result.duration_ms, output_bytes=result.output_bytes)
        return result

    def _run_sync(self, adapter: ToolAdapter, args: dict[str, Any],
                  ctx: ToolExecutionContext, spec: ToolSpec,
                  run_context: RunContext | None) -> str:
        sem = run_context.sync_tool_semaphore() if run_context is not None else None
        if sem is not None:
            sem.acquire()
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["out"] = adapter.execute(args, ctx)
            except Exception as exc:  # adapter failure -> structured error
                box["err"] = exc
            finally:
                box["done"] = True

        thread = threading.Thread(target=target, daemon=True, name=f"tool-{spec.name}")
        thread.start()
        thread.join(spec.timeout_seconds)
        if thread.is_alive():
            # The underlying operation may still be running: keep the concurrency
            # occupancy until it actually terminates (a thread cannot be killed).
            if sem is not None:
                def watcher() -> None:
                    thread.join()
                    sem.release()

                threading.Thread(
                    target=watcher, daemon=True, name=f"tool-watch-{spec.name}"
                ).start()
            raise ToolTimeoutError(
                f"tool '{spec.name}' exceeded timeout {spec.timeout_seconds}s; outcome "
                "uncertain for mutating tools"
            )
        if sem is not None:
            sem.release()
        if "err" in box:
            raise ToolError(f"tool '{spec.name}' failed: {box['err']}") from box["err"]
        return str(box.get("out", ""))

    # ---- async path ----
    async def aexecute(self, tool: str, args: Any, *, ctx: ToolExecutionContext,
                       run_context: RunContext | None = None) -> ToolResult:
        execution_id = uuid4().hex
        started = time.perf_counter()
        reg = None
        run_row = None
        approval_id = ""
        try:
            self._check_context(ctx, run_context)
            reg = self.catalog.get(tool)
            if reg is None:
                raise ToolNotFoundError(f"unknown tool '{tool}' — denied")
            spec = reg.spec
            if not self.policy.allows(ctx.agent, tool):
                raise ToolPermissionError(
                    f"agent '{ctx.agent}' is not allowed to use tool '{tool}'"
                )
            validated = validate_args(spec, args)
            self._reserve(ctx, spec, run_context)
            if self._approval_needed(spec, ctx, tool):
                if self.approval_manager is None:
                    raise ToolPermissionError(
                        f"tool '{tool}' requires approval but no ApprovalManager is "
                        "configured — failing closed"
                    )

                approval_id, run_row = await self._obtain_approval_async(
                    spec, ctx, validated, reg.adapter, run_context
                )
            else:
                run_row = self._run_lease(ctx)
            self._ledger_start(spec, ctx, run_context, execution_id, validated,
                               approval_id, run_row)
            output = await self._run_async(reg.adapter, validated, ctx, spec, run_context)
            text, nbytes, truncated = self._shape_output(spec, output)
            self._ledger_finish(spec, execution_id, "completed", run_row, output=text)
            result = ToolResult(
                tool=tool, agent=ctx.agent, ok=True, output=text, status="completed",
                execution_id=execution_id, duration_ms=(time.perf_counter() - started) * 1000,
                output_bytes=nbytes, truncated=truncated,
            )
        except ToolError as exc:
            status = "rejected"
            if isinstance(exc, ApprovalRequiredError):
                status = "waiting_approval"
            elif isinstance(exc, ToolTimeoutError):
                status = "timeout"
            result = ToolResult(
                tool=tool, agent=ctx.agent, ok=False, error=str(exc), status=status,
                execution_id=execution_id, duration_ms=(time.perf_counter() - started) * 1000,
            )
            if reg is not None and self._store is not None and ctx.run_id:
                mutating = reg.spec.side_effect in (SideEffect.LOCAL_MUTATION,
                                                    SideEffect.EXTERNAL_MUTATION)
                final = "unknown" if isinstance(exc, ToolTimeoutError) and mutating else "failed"
                self._ledger_finish(reg.spec, execution_id, final, run_row, error=str(exc))
        except asyncio.CancelledError:
            if reg is not None and self._store is not None and ctx.run_id:
                mutating = reg.spec.side_effect in (SideEffect.LOCAL_MUTATION,
                                                    SideEffect.EXTERNAL_MUTATION)
                self._ledger_finish(reg.spec, execution_id,
                                    "unknown" if mutating else "cancelled",
                                    run_row, error="cancelled")
            raise
        self._record(ctx, run_context, tool=tool, status=result.status,
                     execution_id=execution_id, error=result.error,
                     duration_ms=result.duration_ms, output_bytes=result.output_bytes)
        return result

    async def _obtain_approval_async(self, spec: ToolSpec, ctx: ToolExecutionContext,
                                     args: dict[str, Any], adapter: ToolAdapter,
                                     run_context: RunContext | None) -> tuple[str, Any]:
        mgr = self.approval_manager
        assert mgr is not None
        actx = self._approval_context(adapter, args)
        req = mgr.create(
            ctx=ctx, spec=spec, args=args, operation=actx["operation"],
            target=actx.get("target"), preconditions=actx.get("preconditions") or {},
            run_context=run_context,
        )
        from datetime import UTC, datetime

        def _now():
            if self._store is not None:
                return self._store.utcnow()
            return datetime.now(UTC)

        deadline = datetime.fromisoformat(req.expires_at)
        while True:
            row = mgr.get(req.approval_id)
            status = row["status"] if row else "unknown"
            if status == "approved":
                break
            if status in ("denied", "cancelled", "expired", "unknown"):
                raise ApprovalError(
                    f"approval {req.approval_id} was {status}; action not authorized"
                )
            if _now() >= deadline:
                mgr.sweep_expired()
                raise ApprovalError(f"approval {req.approval_id} expired")
            await self._async_sleep(0.2)
        if not self.policy.allows(ctx.agent, spec.name):
            raise ToolPermissionError("policy changed since request — re-denied")
        digest = args_sha256(args)
        run_row = self._run_lease(ctx)
        if self.strict and (run_row is None or run_row.state.value != "running"):
            raise ToolPermissionError("run not active — fail closed")
        if not mgr.consume(req.approval_id, expected_args_sha256=digest):
            raise ApprovalError(
                "approval consume failed: replay, argument substitution or expiry"
            )
        return req.approval_id, run_row

    async def _run_async(self, adapter: ToolAdapter, args: dict[str, Any],
                         ctx: ToolExecutionContext, spec: ToolSpec,
                         run_context: RunContext | None) -> str:
        sem = run_context.semaphore_tool() if run_context is not None else None
        if sem is not None:
            await sem.acquire()
        handed_off = False  # occupancy transferred to a watcher (thread still running)
        try:
            aexec = getattr(adapter, "aexecute", None)
            if aexec is not None:
                try:
                    return await asyncio.wait_for(aexec(args, ctx), timeout=spec.timeout_seconds)
                except TimeoutError:
                    raise ToolTimeoutError(
                        f"tool '{spec.name}' exceeded timeout {spec.timeout_seconds}s"
                    ) from None
            # legacy sync-only adapter: thread fallback (never replaces the native path)
            loop = asyncio.get_running_loop()
            done = threading.Event()
            box: dict[str, Any] = {}

            def target() -> None:
                try:
                    box["out"] = adapter.execute(args, ctx)
                except Exception as exc:
                    box["err"] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=target, daemon=True, name=f"tool-{spec.name}")
            thread.start()

            async def watcher() -> None:
                # keep the concurrency slot until the underlying thread terminates
                while thread.is_alive():
                    await asyncio.sleep(0.05)
                if sem is not None:
                    sem.release()

            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, done.wait), timeout=spec.timeout_seconds
                )
            except TimeoutError:
                handed_off = True
                asyncio.create_task(watcher(), name=f"tool-watch-{spec.name}")
                raise ToolTimeoutError(
                    f"tool '{spec.name}' exceeded timeout; sync thread still running, "
                    "occupancy held until it terminates"
                ) from None
            except asyncio.CancelledError:
                handed_off = True
                asyncio.create_task(watcher(), name=f"tool-watch-{spec.name}")
                raise
            if "err" in box:
                raise ToolError(f"tool '{spec.name}' failed: {box['err']}") from box["err"]
            return str(box.get("out", ""))
        finally:
            if sem is not None and not handed_off:
                sem.release()
