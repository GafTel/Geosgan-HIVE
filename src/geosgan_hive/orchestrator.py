from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .agent_config import Limits
from .agents import Agent, arun_agent, check_run_limits, parse_json_object
from .errors import GatewayError
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

    def _budget_stop(self, ctx: RunContext | None) -> str | None:
        if ctx is not None and not ctx.reserve_model_call():
            return f"Run-wide model-call limit ({ctx.policy.max_model_calls_per_run}) reached."
        return None

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

    def run(self, task: str, *, run_id: str | None = None) -> OrchestratorResult:
        run_id = run_id or uuid4().hex
        ctx, root_task_id = self._open_run(task, run_id)
        try:
            return self._run_loop(task, run_id, ctx, root_task_id)
        finally:
            self._close_run(run_id, ctx, root_task_id)

    def _run_loop(self, task: str, run_id: str, ctx: RunContext | None,
                  root_task_id: str | None) -> OrchestratorResult:
        transcript = [Message(role="user", content=task, agent="user")]
        route: list[str] = []
        calls = 0
        for step in range(1, self.max_steps + 1):
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is None:
                stop = self._budget_stop(ctx)
            if stop is not None:
                return OrchestratorResult(
                    task=task, final_answer=stop, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            try:
                raw = self.llm.complete(
                    [self._system_message(), *transcript],
                    temperature=self.temperature,
                    run_id=run_id,
                )
            except GatewayError as exc:
                return OrchestratorResult(
                    task=task, final_answer=f"Model gateway error: {exc}", route=route,
                    transcript=transcript, steps=step, run_id=run_id,
                )
            calls += 1
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return OrchestratorResult(
                    task=task, final_answer=raw, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return OrchestratorResult(
                    task=task, final_answer=answer, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
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
                continue
            self._finish_agent_task(
                ctx, child_id, agent_name=agent.name, before=before,
                status=TaskStatus.COMPLETED, result=result.answer,
            )
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return OrchestratorResult(
            task=task,
            final_answer="Maximum number of steps reached without a final answer.",
            route=route,
            transcript=transcript,
            steps=self.max_steps,
            run_id=run_id,
        )

    async def _acomplete(self, messages: list[Message], *, run_id: str | None) -> str:
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
        ctx, root_task_id = self._open_run(task, run_id)
        try:
            return await self._arun_loop(task, run_id, ctx, root_task_id)
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

    async def _arun_loop(self, task: str, run_id: str, ctx: RunContext | None,
                         root_task_id: str | None) -> OrchestratorResult:
        transcript = [Message(role="user", content=task, agent="user")]
        route: list[str] = []
        calls = 0
        for step in range(1, self.max_steps + 1):
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is None:
                stop = self._budget_stop(ctx)
            if stop is not None:
                return OrchestratorResult(
                    task=task, final_answer=stop, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            try:
                raw = await self._acomplete(
                    [self._system_message(), *transcript], run_id=run_id
                )
            except GatewayError as exc:
                return OrchestratorResult(
                    task=task, final_answer=f"Model gateway error: {exc}", route=route,
                    transcript=transcript, steps=step, run_id=run_id,
                )
            calls += 1
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return OrchestratorResult(
                    task=task, final_answer=raw, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return OrchestratorResult(
                    task=task, final_answer=answer, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
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
                continue
            self._finish_agent_task(
                ctx, child_id, agent_name=agent.name, before=before,
                status=TaskStatus.COMPLETED, result=result.answer,
            )
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return OrchestratorResult(
            task=task,
            final_answer="Maximum number of steps reached without a final answer.",
            route=route,
            transcript=transcript,
            steps=self.max_steps,
            run_id=run_id,
        )
