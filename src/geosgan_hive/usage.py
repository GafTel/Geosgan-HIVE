from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class UsageRecord:
    """Usage and cost of one model call.

    `cost_usd` is None when the cost is unknown (local/unknown models); it is
    never 0 for an unknown cost. Token counts are always recorded when the
    provider reports them.
    """

    agent: str | None
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    latency_ms: float
    attempts: int
    timestamp: str
    run_id: str | None = None


@dataclass
class UsageSummary:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # None when any contributing record has an unknown cost: we never claim a
    # total (or zero) cost that we cannot actually know.
    cost_usd: float | None = None


class UsageTracker:
    """Thread-safe collector of UsageRecord, with optional JSONL persistence."""

    def __init__(self, sink: str | Path | None = None) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._sink = Path(sink) if sink is not None else None
        if self._sink is not None:
            self._sink.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self._sink is not None:
                line = json.dumps(asdict(record), ensure_ascii=False)
                with self._sink.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    @staticmethod
    def _summarize(records: list[UsageRecord]) -> UsageSummary:
        costs = [r.cost_usd for r in records]
        cost = sum(costs) if records and all(c is not None for c in costs) else None  # type: ignore[arg-type]
        return UsageSummary(
            calls=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            total_tokens=sum(r.total_tokens for r in records),
            cost_usd=cost,
        )

    def summary(self, agent: str | None = None) -> UsageSummary:
        records = self.records()
        if agent is not None:
            records = [r for r in records if r.agent == agent]
        return self._summarize(records)

    def summary_by_agent(self) -> dict[str, UsageSummary]:
        by_agent: dict[str, list[UsageRecord]] = {}
        for r in self.records():
            by_agent.setdefault(r.agent or "(unknown)", []).append(r)
        return {name: self._summarize(recs) for name, recs in sorted(by_agent.items())}

    def cost_for_run(self, run_id: str) -> float | None:
        """Known cost of a run, or None if any of its calls has an unknown cost."""
        records = [r for r in self.records() if r.run_id == run_id]
        return self._summarize(records).cost_usd if records else None

    def total_cost(self) -> float | None:
        return self._summarize(self.records()).cost_usd
