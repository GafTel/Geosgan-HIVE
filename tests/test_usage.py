import json
import threading
from datetime import UTC, datetime

from geosgan_hive.usage import UsageRecord, UsageTracker


def make_record(
    cost: float | None = 0.001, agent: str | None = "researcher", run_id: str | None = "r1"
):
    return UsageRecord(
        agent=agent,
        model="openai/gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=cost,
        latency_ms=12.5,
        attempts=1,
        timestamp=datetime.now(UTC).isoformat(),
        run_id=run_id,
    )


def test_record_and_summary_known_cost():
    tracker = UsageTracker()
    tracker.record(make_record(cost=0.001))
    tracker.record(make_record(cost=0.002, agent="writer"))
    summary = tracker.summary()
    assert summary.calls == 2
    assert summary.total_tokens == 30
    assert summary.cost_usd == 0.003
    assert tracker.summary(agent="writer").cost_usd == 0.002
    by_agent = tracker.summary_by_agent()
    assert set(by_agent) == {"researcher", "writer"}
    assert by_agent["researcher"].calls == 1


def test_unknown_cost_is_none_never_zero():
    tracker = UsageTracker()
    tracker.record(make_record(cost=None))  # e.g. local Ollama model
    summary = tracker.summary()
    assert summary.cost_usd is None
    assert summary.total_tokens == 15  # tokens are still tracked
    assert tracker.total_cost() is None


def test_mixed_known_and_unknown_costs_yield_none_total():
    tracker = UsageTracker()
    tracker.record(make_record(cost=0.001, run_id="r1"))
    tracker.record(make_record(cost=None, run_id="r1"))
    assert tracker.cost_for_run("r1") is None  # cannot claim a reliable total
    assert tracker.cost_for_run("missing-run") is None
    tracker.record(make_record(cost=0.004, run_id="r2"))
    assert tracker.cost_for_run("r2") == 0.004


def test_jsonl_sink_writes_one_record_per_line(tmp_path):
    sink = tmp_path / "logs" / "usage.jsonl"
    tracker = UsageTracker(sink=sink)
    tracker.record(make_record(cost=0.001, agent="researcher"))
    tracker.record(make_record(cost=None, agent="reviewer"))
    lines = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["agent"] == "researcher"
    assert parsed[0]["cost_usd"] == 0.001
    assert parsed[1]["cost_usd"] is None  # unknown cost serialized as null, not 0


def test_tracker_is_thread_safe():
    tracker = UsageTracker()

    def worker():
        for _ in range(100):
            tracker.record(make_record())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.summary().calls == 1000


def test_reset_clears_records():
    tracker = UsageTracker()
    tracker.record(make_record())
    tracker.reset()
    assert tracker.records() == []
