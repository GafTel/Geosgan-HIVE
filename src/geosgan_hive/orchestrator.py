from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .agent_config import Limits
from .agents import Agent, arun_agent, check_run_limits, parse_json_object
from .checkpoints import (
    CheckpointStore,
    ExecutionState,
    PersistMode,
    counters_to_json,
    steps_to_json,
)
from .errors import CheckpointError, GatewayError, ProviderUnavailableError
from .llm import LLMClient
from .messages import Message
from .run_context import EventType, RunContext, TaskStatus, utcnow
from .usage import UsageTracker

if TYPE_CHECKING:
    from .collaboration import CollaborationEngine

FINISH = "FINISH"

SUPERVISOR_INSTRUCTIONS = """\
You are the supervisor of a team of specialist agents.
You decide which agent handles the task at each step.
Reply ONLY with a JSON:
- To delegate: {"next": "<agent_name>", "reason": "<why>"}
- To finish: {"next": "FINISH", "answer": "<final answer to the user>"}
Do not add any text outside the JSON."""


@dataclass
class OrchestratorResult:
    task: str
    final_answer: str
    route: list[str] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)
    steps: int = 0
    run_id: str | None = None
    suspended: bool = False  # WAIT/HYBRID: no eligible model; checkpointed as WAITING
    failed: bool = False  # terminal gateway/budget error


def parse_decision(text: str) -> dict[str, Any] | None:
    data = parse_json_object(text)
    if data and isinstance(data.get("next"), str):
        return data
    return None


class Supervisor:
    """Orchestrator that routes the task between agents until a final answer.

    Runs on its own model (bound via the gateway). When a CollaborationEngine is
    attached, the supervisor creates the run's root task, records one task node
    per delegation (root + delegated tasks all live in the run history), reserves
    the shared run-wide model-call budget before every model request, and closes
    the run context on completion, error or cancellation.

    Routing stays sequential in Phase 2; parallelism happens inside agents via
    the collaboration engine.
    """

    def __init__(
        self,
        agents: Mapping[str, Agent],
        llm: LLMClient,
        *,
        max_steps: int = 8,
        temperature: float = 0.2,
        usage_tracker: UsageTracker | None = None,
        limits: Limits | None = None,
        collaboration: CollaborationEngine | None = None,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_config_json: str | None = None,
        suspend_cooldown_seconds: float = 60.0,
        max_wait_seconds: float = 300.0,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent is required")
        self.agents = dict(agents)
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature
        self.usage_tracker = usage_tracker
        self.limits = limits or Limits()
        self.collaboration = collaboration
        # Phase 2.5: durable suspension (WAIT/HYBRID). Suspension happens only at
        # safe step boundaries of the supervisor, never while child tasks are active.
        self.checkpoint_store = checkpoint_store
        self.checkpoint_config_json = checkpoint_config_json
        self.suspend_cooldown_seconds = suspend_cooldown_seconds
        self.max_wait_seconds = max_wait_seconds

    def _system_message(self) -> Message:
        roster = "\n".join(f"- {name}: {agent.description}" for name, agent in self.agents.items())
        content = f"Available agents:\n{roster}\n\n{SUPERVISOR_INSTRUCTIONS}"
        return Message(role="system", content=content, agent="supervisor")

    def _open_run(self, task: str, run_id: str) -> tuple[RunContext | None, str | None]:
        if self.collaboration is None:
            return None, None
        ctx = self.collaboration.context_for(run_id)
        root_task_id = uuid4().hex
        ctx.create_task(
            task_id=root_task_id, parent_task_id=None, agent="supervisor",
            description=task, depth=0,
        )
        ctx.record_event(EventType.TASK_STARTED, task_id=root_task_id, source_agent="supervisor")
        return ctx, root_task_id

    def _close_run(self, run_id: str, ctx: RunContext | None, root_task_id: str | None,
                   terminal: TaskStatus | None = None) -> None:
        if ctx is not None and root_task_id is not None:
            node = ctx.task(root_task_id)
            if node is not None and node.status not in (
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
            ):
                ctx.update_task(
                    root_task_id,
                    status=terminal or TaskStatus.COMPLETED,
                    finished_at=utcnow(),
                )
        if self.collaboration is not None:
            self.collaboration.close_run(run_id)

    def _budget_stop(self, ctx: RunContext | None, task_id: str | None = None) -> str | None:
        if ctx is None:
            return None
        cap = ctx.policy.max_provider_requests_per_run
        if cap is not None and ctx.physical_requests >= cap:
            return (
                f"Provider-request limit ({cap}) reached for this run "
                f"(physical requests incl. retries)."
            )
        ok, reason = ctx.reserve_model_call_for(
            "supervisor", task_id, self.limits.max_calls_per_run
        )
        return None if ok else reason

    def _start_agent_task(self, ctx: RunContext | None, root_task_id: str | None,
                          agent_name: str, task: str) -> tuple[str | None, int]:
        if ctx is None:
            return None, 0
        child_id = uuid4().hex
        ctx.create_task(
            task_id=child_id, parent_task_id=root_task_id, agent=agent_name,
            description=task, depth=1,
        )
        ctx.update_task(child_id, status=TaskStatus.RUNNING)
        ctx.record_event(
            EventType.TASK_STARTED, task_id=child_id, source_agent="supervisor",
            target_agent=agent_name,
        )
        return child_id, ctx.model_calls

    def _finish_agent_task(self, ctx: RunContext | None, child_id: str | None, *,
                           agent_name: str, before: int, status: TaskStatus,
                           result: str = "", error: str | None = None) -> None:
        if ctx is None or child_id is None:
            return
        ctx.update_task(
            child_id, status=status, result=result, error=error,
            finished_at=utcnow(), model_calls=ctx.model_calls - before,
        )
        event = (
            EventType.TASK_COMPLETED if status == TaskStatus.COMPLETED else EventType.TASK_FAILED
        )
        ctx.record_event(
            event, task_id=child_id, source_agent="supervisor", target_agent=agent_name,
            metadata={"error": error} if error else {},
        )
        if status == TaskStatus.COMPLETED:
            ctx.record_event(
                EventType.RESULT_RETURNED, task_id=child_id, source_agent="supervisor",
                target_agent=agent_name, metadata={"status": status.value},
            )

    # ---- checkpoint helpers (Phase 2.5) ----
    def _steps_payload(self, ctx: RunContext | None) -> list[dict[str, Any]]:
        """Completed/failed child steps, persisted so a resume never repeats them."""
        if ctx is None:
            return []
        return [
            {
                "task_id": t.task_id,
                "agent": t.agent,
                "description": t.description,
                "status": t.status.value,
                "result": t.result,
                "error": t.error,
            }
            for t in ctx.tasks()
            if t.parent_task_id is not None
            and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]

    def _transcript_json(self, transcript: list[Message]) -> str | None:
        store = self.checkpoint_store
        if store is None or store.persist_mode != PersistMode.RESUMABLE:
            return None  # metadata_only: never store the transcript
        return json.dumps([asdict(m) for m in transcript], ensure_ascii=False)

    def _finalize(self, run_id: str, owner: str | None, result: OrchestratorResult,
                  ctx: RunContext | None, transcript: list[Message], *,
                  failed: bool) -> OrchestratorResult:
        """Persist the terminal state (fenced by lease ownership). Suspended results
        were already checkpointed as WAITING by `_suspend`."""
        store = self.checkpoint_store
        if store is None or owner is None or result.suspended:
            return result
        counters = counters_to_json(ctx.counters_snapshot()) if ctx is not None else "{}"
        steps = steps_to_json(self._steps_payload(ctx))
        route_json = json.dumps(result.route)
        transcript_json = self._transcript_json(transcript)
        if failed:
            store.mark_failed(
                run_id, owner=owner, error=result.final_answer, counters_json=counters,
                steps_json=steps, route_json=route_json, transcript_json=transcript_json,
                step_count=result.steps,
            )
        else:
            store.mark_completed(
                run_id, owner=owner, result=result.final_answer, counters_json=counters,
                steps_json=steps, route_json=route_json, transcript_json=transcript_json,
                step_count=result.steps,
            )
        return result

    def _suspend(self, run_id: str, ctx: RunContext | None, root_task_id: str | None,
                 task: str, route: list[str], transcript: list[Message], step: int,
                 exc: ProviderUnavailableError, *, owner: str | None) -> OrchestratorResult:
        """Suspend at a SAFE supervisor step boundary (WAIT/HYBRID signal).

        Never checkpoints while child tasks are active: if any child task is still
        RUNNING, the run fails safely with an explicit diagnostic instead (no
        mid-flight checkpoint, no silent reconstruction)."""
        store = self.checkpoint_store
        if store is None or owner is None:
            return OrchestratorResult(
                task=task, final_answer=f"Model gateway error: {exc}", route=route,
                transcript=transcript, steps=step, run_id=run_id, failed=True,
            )
        active = []
        if ctx is not None:
            active = [
                t.task_id for t in ctx.tasks()
                if t.status == TaskStatus.RUNNING and t.task_id != root_task_id
            ]
        counters = counters_to_json(ctx.counters_snapshot()) if ctx is not None else "{}"
        steps_json = steps_to_json(self._steps_payload(ctx))
        route_json = json.dumps(route)
        transcript_json = self._transcript_json(transcript)
        if active:
            msg = (
                "Unsupported nested suspension: no eligible model while child tasks "
                f"{active} are still active. The run is failed safely (no mid-flight "
                f"checkpoint). Original error: {exc}"
            )
            store.mark_failed(
                run_id, owner=owner, error=msg, counters_json=counters, steps_json=steps_json,
                route_json=route_json, transcript_json=transcript_json, step_count=step,
            )
            if ctx is not None:
                ctx.record_event(
                    EventType.TASK_FAILED, task_id=root_task_id, source_agent="supervisor",
                    metadata={"reason": "unsupported nested suspension",
                              "active_children": active},
                )
            return OrchestratorResult(
                task=task, final_answer=msg, route=route, transcript=transcript,
                steps=step, run_id=run_id, failed=True,
            )
        now = store.utcnow()
        next_attempt = (now + timedelta(seconds=self.suspend_cooldown_seconds)).isoformat(
            timespec="milliseconds"
        )
        deadline = (now + timedelta(seconds=self.max_wait_seconds)).isoformat(
            timespec="milliseconds"
        )
        store.mark_waiting(
            run_id, owner=owner, error=str(exc), counters_json=counters, steps_json=steps_json,
            route_json=route_json, transcript_json=transcript_json, step_count=step,
            next_attempt_at=next_attempt, deadline_at=deadline,
        )
        if ctx is not None:
            ctx.record_event(
                EventType.TASK_SUSPENDED, task_id=root_task_id, source_agent="supervisor",
                metadata={"error": str(exc), "next_attempt_at": next_attempt,
                          "deadline_at": deadline},
            )
        return OrchestratorResult(
            task=task,
            final_answer=f"Run suspended (no eligible model available): {exc}",
            route=route, transcript=transcript, steps=step, run_id=run_id, suspended=True,
        )

    def run(self, task: str, *, run_id: str | None = None) -> OrchestratorResult:
        run_id = run_id or uuid4().hex
        owner = f"supervisor:{uuid4().hex}" if self.checkpoint_store is not None else None
        ctx, root_task_id = self._open_run(task, run_id)
        try:
            if self.checkpoint_store is not None and owner is not None:
                self.checkpoint_store.create_run(
                    run_id=run_id, owner=owner, task=task, agent="supervisor",
                    model=getattr(self.llm, "model", "") or "",
                    config_json=self.checkpoint_config_json or "",
                )
            return self._run_loop(task, run_id, ctx, root_task_id, resume_owner=owner)
        finally:
            self._close_run(run_id, ctx, root_task_id)

    def resume(self, run_id: str, *, owner: str | None = None) -> OrchestratorResult:
        """Resume a suspended run from its durable checkpoint (supervisor-level).

        Requires persist_mode=resumable (metadata_only checkpoints are never
        silently reconstructed) and a free lease; the same lease mechanism fences
        concurrent/stale workers. Completed steps are restored, never repeated;
        counters are restored without double counting."""
        store = self.checkpoint_store
        if store is None:
            raise CheckpointError("no checkpoint store configured on this supervisor")
        if self.collaboration is None:
            raise CheckpointError("supervisor has no collaboration engine; cannot resume")
        owner = owner or f"resumer:{uuid4().hex}"
        cp = store.get(run_id)
        if cp is None:
            raise CheckpointError(f"no checkpoint found for run {run_id!r}")
        if cp.state in (ExecutionState.COMPLETED, ExecutionState.FAILED,
                        ExecutionState.CANCELLED):
            raise CheckpointError(f"run {run_id!r} is {cp.state.value}; nothing to resume")
        if cp.persist_mode != PersistMode.RESUMABLE:
            raise CheckpointError(
                f"run {run_id!r} was checkpointed in {cp.persist_mode.value} mode; durable "
                "resume requires persist_mode=resumable (missing context is never "
                "silently reconstructed)"
            )
        if cp.deadline_at and store.now_iso() >= cp.deadline_at:
            store.mark_failed(run_id, owner=None, error="maximum wait deadline exceeded")
            raise CheckpointError(f"run {run_id!r} exceeded its maximum wait deadline")
        if not store.claim_lease(run_id, owner):
            raise CheckpointError(f"run {run_id!r} is leased by another worker")
        store.mark_running(run_id, owner=owner)
        ctx = self.collaboration.context_for(run_id)
        root_task_id = uuid4().hex
        ctx.create_task(
            task_id=root_task_id, parent_task_id=None, agent="supervisor",
            description=cp.task, depth=0,
        )
        ctx.update_task(root_task_id, status=TaskStatus.RUNNING)
        ctx.restore_counters(json.loads(cp.counters_json or "{}"))
        for step in json.loads(cp.steps_json or "[]"):
            ctx.create_task(
                task_id=step["task_id"], parent_task_id=root_task_id,
                agent=step["agent"], description=step.get("description", ""), depth=1,
            )
            ctx.update_task(
                step["task_id"], status=TaskStatus(step["status"]),
                result=step.get("result", ""), error=step.get("error"),
            )
        ctx.record_event(
            EventType.TASK_RESUMED, task_id=root_task_id, source_agent="supervisor",
            metadata={"attempts": cp.attempts, "owner": owner},
        )
        transcript = [Message(**m) for m in json.loads(cp.transcript_json or "[]")]
        route = json.loads(cp.route_json or "[]")
        try:
            return self._run_loop(
                cp.task, run_id, ctx, root_task_id,
                transcript=transcript, route=route, start_step=cp.step_count + 1,
                resume_owner=owner,
            )
        finally:
            self._close_run(run_id, ctx, root_task_id)

    def _run_loop(self, task: str, run_id: str, ctx: RunContext | None,
                  root_task_id: str | None, *, transcript: list[Message] | None = None,
                  route: list[str] | None = None, start_step: int = 1,
                  resume_owner: str | None = None) -> OrchestratorResult:
        transcript = transcript if transcript is not None else [
            Message(role="user", content=task, agent="user")
        ]
        route = route if route is not None else []
        calls = 0
        for step in range(start_step, self.max_steps + 1):
            if resume_owner is not None and self.checkpoint_store is not None:
                if not self.checkpoint_store.renew_lease(run_id, resume_owner):
                    raise CheckpointError(
                        f"run {run_id!r}: lease lost during execution (stale worker fencing)"
                    )
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is None:
                stop = self._budget_stop(ctx, root_task_id)
            if stop is not None:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=stop, route=route, transcript=transcript,
                        steps=step, run_id=run_id, failed=True,
                    ),
                    ctx, transcript, failed=True,
                )
            try:
                raw = self.llm.complete(
                    [self._system_message(), *transcript],
                    temperature=self.temperature,
                    run_id=run_id,
                )
            except ProviderUnavailableError as exc:
                return self._suspend(
                    run_id, ctx, root_task_id, task, route, transcript, step, exc,
                    owner=resume_owner,
                )
            except GatewayError as exc:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=f"Model gateway error: {exc}", route=route,
                        transcript=transcript, steps=step, run_id=run_id, failed=True,
                    ),
                    ctx, transcript, failed=True,
                )
            calls += 1
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=raw, route=route, transcript=transcript,
                        steps=step, run_id=run_id,
                    ),
                    ctx, transcript, failed=False,
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=answer, route=route, transcript=transcript,
                        steps=step, run_id=run_id,
                    ),
                    ctx, transcript, failed=False,
                )
            agent = self.agents.get(nxt)
            if agent is None:
                valid = ", ".join(self.agents)
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error: agent '{nxt}' does not exist. Valid: {valid}.",
                    )
                )
                continue
            route.append(agent.name)
            child_id, before = self._start_agent_task(ctx, root_task_id, agent.name, task)
            try:
                result = agent.run(
                    task, context=transcript, run_id=run_id, task_id=child_id, run_context=ctx
                )
            except GatewayError as exc:
                self._finish_agent_task(
                    ctx, child_id, agent_name=agent.name, before=before,
                    status=TaskStatus.FAILED, error=str(exc),
                )
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error from agent '{nxt}': {exc}",
                    )
                )
                if isinstance(exc, ProviderUnavailableError):
                    # child hit a WAIT/HYBRID suspension signal: the child task is
                    # FAILED with an explicit diagnostic (no mid-flight checkpoint of
                    # the nested task) and the run suspends at THIS safe boundary
                    return self._suspend(
                        run_id, ctx, root_task_id, task, route, transcript, step,
                        exc, owner=resume_owner,
                    )
                continue
            self._finish_agent_task(
                ctx, child_id, agent_name=agent.name, before=before,
                status=TaskStatus.COMPLETED, result=result.answer,
            )
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return self._finalize(
            run_id, resume_owner,
            OrchestratorResult(
                task=task,
                final_answer="Maximum number of steps reached without a final answer.",
                route=route,
                transcript=transcript,
                steps=self.max_steps,
                run_id=run_id,
            ),
            ctx, transcript, failed=False,
        )

    async def _acomplete(self, messages: list[Message], *, run_id: str | None,
                         run_context: RunContext | None = None) -> str:
        # request-level concurrency is bounded at the gateway dispatch boundary
        acomplete = getattr(self.llm, "acomplete", None)
        if acomplete is not None:
            return await acomplete(messages, temperature=self.temperature, run_id=run_id)
        # legacy sync-only adapter: thread fallback only
        return await asyncio.to_thread(
            self.llm.complete, messages, temperature=self.temperature, run_id=run_id
        )

    async def arun(self, task: str, *, run_id: str | None = None) -> OrchestratorResult:
        """Native async twin of `run` (sequential routing; agents run via arun)."""
        run_id = run_id or uuid4().hex
        owner = f"supervisor:{uuid4().hex}" if self.checkpoint_store is not None else None
        ctx, root_task_id = self._open_run(task, run_id)
        try:
            if self.checkpoint_store is not None and owner is not None:
                self.checkpoint_store.create_run(
                    run_id=run_id, owner=owner, task=task, agent="supervisor",
                    model=getattr(self.llm, "model", "") or "",
                    config_json=self.checkpoint_config_json or "",
                )
            return await self._arun_loop(task, run_id, ctx, root_task_id, resume_owner=owner)
        except asyncio.CancelledError:
            if ctx is not None and root_task_id is not None:
                ctx.update_task(root_task_id, status=TaskStatus.CANCELLED, finished_at=utcnow())
                ctx.record_event(
                    EventType.TASK_FAILED, task_id=root_task_id, source_agent="supervisor",
                    metadata={"reason": "cancelled"},
                )
            raise
        finally:
            self._close_run(run_id, ctx, root_task_id)

    async def aresume(self, run_id: str, *, owner: str | None = None) -> OrchestratorResult:
        """Async twin of `resume` (same lease + persist-mode guarantees)."""
        store = self.checkpoint_store
        if store is None:
            raise CheckpointError("no checkpoint store configured on this supervisor")
        if self.collaboration is None:
            raise CheckpointError("supervisor has no collaboration engine; cannot resume")
        owner = owner or f"resumer:{uuid4().hex}"
        cp = store.get(run_id)
        if cp is None:
            raise CheckpointError(f"no checkpoint found for run {run_id!r}")
        if cp.state in (ExecutionState.COMPLETED, ExecutionState.FAILED,
                        ExecutionState.CANCELLED):
            raise CheckpointError(f"run {run_id!r} is {cp.state.value}; nothing to resume")
        if cp.persist_mode != PersistMode.RESUMABLE:
            raise CheckpointError(
                f"run {run_id!r} was checkpointed in {cp.persist_mode.value} mode; durable "
                "resume requires persist_mode=resumable (missing context is never "
                "silently reconstructed)"
            )
        if cp.deadline_at and store.now_iso() >= cp.deadline_at:
            store.mark_failed(run_id, owner=None, error="maximum wait deadline exceeded")
            raise CheckpointError(f"run {run_id!r} exceeded its maximum wait deadline")
        if not store.claim_lease(run_id, owner):
            raise CheckpointError(f"run {run_id!r} is leased by another worker")
        store.mark_running(run_id, owner=owner)
        ctx = self.collaboration.context_for(run_id)
        root_task_id = uuid4().hex
        ctx.create_task(
            task_id=root_task_id, parent_task_id=None, agent="supervisor",
            description=cp.task, depth=0,
        )
        ctx.update_task(root_task_id, status=TaskStatus.RUNNING)
        ctx.restore_counters(json.loads(cp.counters_json or "{}"))
        for step in json.loads(cp.steps_json or "[]"):
            ctx.create_task(
                task_id=step["task_id"], parent_task_id=root_task_id,
                agent=step["agent"], description=step.get("description", ""), depth=1,
            )
            ctx.update_task(
                step["task_id"], status=TaskStatus(step["status"]),
                result=step.get("result", ""), error=step.get("error"),
            )
        ctx.record_event(
            EventType.TASK_RESUMED, task_id=root_task_id, source_agent="supervisor",
            metadata={"attempts": cp.attempts, "owner": owner},
        )
        transcript = [Message(**m) for m in json.loads(cp.transcript_json or "[]")]
        route = json.loads(cp.route_json or "[]")
        try:
            return await self._arun_loop(
                cp.task, run_id, ctx, root_task_id,
                transcript=transcript, route=route, start_step=cp.step_count + 1,
                resume_owner=owner,
            )
        except asyncio.CancelledError:
            if ctx is not None:
                ctx.update_task(root_task_id, status=TaskStatus.CANCELLED, finished_at=utcnow())
            raise
        finally:
            self._close_run(run_id, ctx, root_task_id)

    async def _arun_loop(self, task: str, run_id: str, ctx: RunContext | None,
                         root_task_id: str | None, *, transcript: list[Message] | None = None,
                         route: list[str] | None = None, start_step: int = 1,
                         resume_owner: str | None = None) -> OrchestratorResult:
        transcript = transcript if transcript is not None else [
            Message(role="user", content=task, agent="user")
        ]
        route = route if route is not None else []
        calls = 0
        for step in range(start_step, self.max_steps + 1):
            if resume_owner is not None and self.checkpoint_store is not None:
                if not self.checkpoint_store.renew_lease(run_id, resume_owner):
                    raise CheckpointError(
                        f"run {run_id!r}: lease lost during execution (stale worker fencing)"
                    )
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is None:
                stop = self._budget_stop(ctx, root_task_id)
            if stop is not None:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=stop, route=route, transcript=transcript,
                        steps=step, run_id=run_id, failed=True,
                    ),
                    ctx, transcript, failed=True,
                )
            try:
                raw = await self._acomplete(
                    [self._system_message(), *transcript], run_id=run_id
                )
            except ProviderUnavailableError as exc:
                return self._suspend(
                    run_id, ctx, root_task_id, task, route, transcript, step, exc,
                    owner=resume_owner,
                )
            except GatewayError as exc:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=f"Model gateway error: {exc}", route=route,
                        transcript=transcript, steps=step, run_id=run_id, failed=True,
                    ),
                    ctx, transcript, failed=True,
                )
            calls += 1
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=raw, route=route, transcript=transcript,
                        steps=step, run_id=run_id,
                    ),
                    ctx, transcript, failed=False,
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return self._finalize(
                    run_id, resume_owner,
                    OrchestratorResult(
                        task=task, final_answer=answer, route=route, transcript=transcript,
                        steps=step, run_id=run_id,
                    ),
                    ctx, transcript, failed=False,
                )
            agent = self.agents.get(nxt)
            if agent is None:
                valid = ", ".join(self.agents)
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error: agent '{nxt}' does not exist. Valid: {valid}.",
                    )
                )
                continue
            route.append(agent.name)
            child_id, before = self._start_agent_task(ctx, root_task_id, agent.name, task)
            try:
                result = await arun_agent(
                    agent, task, context=transcript, run_id=run_id, task_id=child_id,
                    run_context=ctx,
                )
            except GatewayError as exc:
                self._finish_agent_task(
                    ctx, child_id, agent_name=agent.name, before=before,
                    status=TaskStatus.FAILED, error=str(exc),
                )
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error from agent '{nxt}': {exc}",
                    )
                )
                if isinstance(exc, ProviderUnavailableError):
                    return self._suspend(
                        run_id, ctx, root_task_id, task, route, transcript, step,
                        exc, owner=resume_owner,
                    )
                continue
            self._finish_agent_task(
                ctx, child_id, agent_name=agent.name, before=before,
                status=TaskStatus.COMPLETED, result=result.answer,
            )
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return self._finalize(
            run_id, resume_owner,
            OrchestratorResult(
                task=task,
                final_answer="Maximum number of steps reached without a final answer.",
                route=route,
                transcript=transcript,
                steps=self.max_steps,
                run_id=run_id,
            ),
            ctx, transcript, failed=False,
        )
