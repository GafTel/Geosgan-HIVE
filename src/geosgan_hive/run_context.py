from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .messages import Message

if TYPE_CHECKING:
    from .usage import UsageTracker


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class CollaborationPolicy(BaseModel):
    """Run-wide collaboration policy. Conservative defaults, YAML-configurable.

    Values are validated: invalid values are rejected at config-load time.
    """

    max_delegation_depth: int = Field(default=2, ge=1, le=10)
    max_handoffs_per_run: int = Field(default=10, ge=0)
    max_parallel_agents: int = Field(default=3, ge=1, le=32)
    default_delegation_timeout_seconds: float = Field(default=120.0, gt=0)
    max_model_calls_per_run: int = Field(default=200, ge=1)
    allow_agent_to_agent: bool = False  # global kill switch; supervisor retains authority


class EventType(StrEnum):
    DELEGATION_REQUESTED = "delegation_requested"
    DELEGATION_APPROVED = "delegation_approved"
    DELEGATION_REJECTED = "delegation_rejected"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RESULT_RETURNED = "result_returned"


class CollaborationEvent(BaseModel):
    """Structured collaboration event. Carries identifiers and metadata only —
    never secrets, credentials or full transcripts."""

    event: EventType
    timestamp: str
    run_id: str
    task_id: str | None = None
    request_id: str | None = None
    source_agent: str | None = None
    target_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """One task in the run's task tree (root task + delegated child tasks)."""

    task_id: str
    parent_task_id: str | None
    agent: str
    description: str
    status: TaskStatus
    depth: int
    result: str = ""
    error: str | None = None
    model_calls: int = 0
    started_at: str = ""
    finished_at: str | None = None


@dataclass
class RunContext:
    """Per-run shared state: task tree, events, budgets, idempotency, semaphore.

    Isolated per run_id. All mutable state is guarded by a lock so concurrent
    (sync threads or async tasks) executions share it without races. Model-call
    reservations are atomic check+increment, so counts are exact, never estimated.
    """

    run_id: str
    policy: CollaborationPolicy
    tracker: UsageTracker | None = None
    closed: bool = field(default=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskNode] = {}
        self._events: list[CollaborationEvent] = []
        self._results: dict[str, Any] = {}
        self._model_calls = 0
        self._handoffs = 0
        self._agent_calls: dict[str, int] = {}
        self._slot_holders: set[str] = set()  # task_ids whose subtree holds a semaphore slot
        self._semaphore: asyncio.Semaphore | None = None

    # ---- model-call budget (run-wide, atomic) ----
    def reserve_model_call(self, task_id: str | None = None) -> bool:
        """Atomically reserve one model call before the actual LLM request.

        Returns False when the run-wide budget is exhausted. Every actual model
        request (supervisor, specialists, delegated and nested) must reserve first.
        When `task_id` is given, the reservation is also counted on that task node,
        so per-task counts stay exact even under parallel execution.
        """
        with self._lock:
            if self._model_calls >= self.policy.max_model_calls_per_run:
                return False
            self._model_calls += 1
            node = self._tasks.get(task_id) if task_id else None
            if node is not None:
                node.model_calls += 1
                self._agent_calls[node.agent] = self._agent_calls.get(node.agent, 0) + 1
            return True

    @property
    def model_calls(self) -> int:
        """Logical model calls reserved in this run (one per llm.complete/acomplete)."""
        with self._lock:
            return self._model_calls

    def agent_calls(self, agent: str) -> int:
        """Logical model calls made by one agent across the WHOLE run (all its tasks),
        so per-agent limits cannot be bypassed by repeated delegation."""
        with self._lock:
            return self._agent_calls.get(agent, 0)

    @property
    def physical_requests(self) -> int:
        """Physical provider requests in this run (sum of retry attempts from usage
        records). Distinct from `model_calls`: one logical call may produce several
        physical requests when the gateway retries transient errors."""
        if self.tracker is None:
            return 0
        return self.tracker.physical_requests(self.run_id)

    # ---- handoff budget ----
    def reserve_handoff(self) -> bool:
        with self._lock:
            if self._handoffs >= self.policy.max_handoffs_per_run:
                return False
            self._handoffs += 1
            return True

    @property
    def handoffs(self) -> int:
        with self._lock:
            return self._handoffs

    # ---- task tree ----
    def create_task(
        self,
        *,
        task_id: str,
        parent_task_id: str | None,
        agent: str,
        description: str,
        depth: int,
    ) -> TaskNode:
        node = TaskNode(
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent=agent,
            description=description,
            status=TaskStatus.PENDING,
            depth=depth,
            started_at=utcnow(),
        )
        with self._lock:
            self._tasks[task_id] = node
        return node

    def update_task(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            node = self._tasks.get(task_id)
            if node is None:
                return
            for key, value in fields.items():
                setattr(node, key, value)

    def task(self, task_id: str) -> TaskNode | None:
        with self._lock:
            return self._tasks.get(task_id)

    def tasks(self) -> list[TaskNode]:
        with self._lock:
            return list(self._tasks.values())

    def ancestors(self, task_id: str | None) -> list[TaskNode]:
        """Chain from the root down to `task_id` (inclusive); empty for unknown/None."""
        chain: list[TaskNode] = []
        current = task_id
        with self._lock:
            while current is not None:
                node = self._tasks.get(current)
                if node is None:
                    break
                chain.append(node)
                current = node.parent_task_id
        chain.reverse()
        return chain

    def scoped_context(self, task_id: str | None, max_result_chars: int = 500) -> list[Message]:
        """Scoped context for a subtask: only its ancestor task summaries/results.

        Never exposes the full conversation transcript or any secrets.
        """
        messages: list[Message] = []
        for anc in self.ancestors(task_id):
            content = (
                f"[context] Task {anc.task_id} (agent: {anc.agent}, depth: {anc.depth}): "
                f"{anc.description}"
            )
            if anc.status == TaskStatus.COMPLETED and anc.result:
                content += f"\nResult: {anc.result[:max_result_chars]}"
            elif anc.status in (TaskStatus.FAILED, TaskStatus.CANCELLED) and anc.error:
                content += f"\nError: {anc.error[:max_result_chars]}"
            messages.append(Message(role="user", agent="system", content=content))
        return messages

    # ---- events ----
    def record_event(
        self,
        event: EventType,
        *,
        task_id: str | None = None,
        request_id: str | None = None,
        source_agent: str | None = None,
        target_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollaborationEvent:
        ev = CollaborationEvent(
            event=event,
            timestamp=utcnow(),
            run_id=self.run_id,
            task_id=task_id,
            request_id=request_id,
            source_agent=source_agent,
            target_agent=target_agent,
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(ev)
        return ev

    def events(self, event: EventType | None = None) -> list[CollaborationEvent]:
        with self._lock:
            if event is None:
                return list(self._events)
            return [e for e in self._events if e.event == event]

    # ---- idempotency (avoid duplicate executions) ----
    def get_result(self, request_id: str) -> Any:
        with self._lock:
            return self._results.get(request_id)

    def set_result(self, request_id: str, result: Any) -> None:
        with self._lock:
            self._results[request_id] = result

    # ---- bounded concurrency (async runs; one event loop per run) ----
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.policy.max_parallel_agents)
        return self._semaphore

    def has_slot_ancestor(self, task_id: str | None) -> bool:
        """True if any task in `task_id`'s ancestor chain (inclusive) holds a
        semaphore slot. Nested delegations inherit the ancestor's slot instead of
        acquiring a new one, so a full semaphore can never deadlock on agents that
        are merely waiting for their own delegated subtasks."""
        with self._lock:
            current = task_id
            while current is not None:
                if current in self._slot_holders:
                    return True
                node = self._tasks.get(current)
                if node is None:
                    return False
                current = node.parent_task_id
            return False

    def mark_slot(self, task_id: str) -> None:
        with self._lock:
            self._slot_holders.add(task_id)

    def unmark_slot(self, task_id: str) -> None:
        with self._lock:
            self._slot_holders.discard(task_id)

    def close(self) -> None:
        with self._lock:
            self.closed = True
