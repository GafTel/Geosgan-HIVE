"""Offline demo of the Phase 2.5 INTELLIGENT MODEL FALLBACK ENGINE (no API keys).

LiteLLM is replaced with a scripted fake provider, so the demo runs fully offline:

1. CONTINUITY: the primary model is rate-limited once, the ordered fallback
   model serves the call, and the fallback event is recorded.
2. Circuit breaker: after `failure_threshold` failures the model circuit OPENS,
   the next call skips the primary; after the cooldown a single HALF_OPEN probe
   runs and, on success, the circuit CLOSES and the primary is preferred again.
3. WAIT: a run whose only authorized model is down suspends into a durable
   (resumable) SQLite checkpoint, then is resumed manually once the provider
   "recovers". Counters are restored without double counting.

Run: python examples/demo_fallback.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc

from geosgan_hive.agents import Agent
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.health import HealthRegistry
from geosgan_hive.messages import Message
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy
from geosgan_hive.usage import UsageTracker

PRIMARY = "ollama/primary"
BACKUP = "ollama/backup"


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def ok(model, content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        model=model,
    )


def main() -> None:
    # ---- scripted fake provider ----
    state = {"primary_failures": 0, "fail_primary": True}

    def fake_completion(**kwargs):
        model = kwargs["model"]
        if model == PRIMARY and state["fail_primary"]:
            state["primary_failures"] += 1
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")
        return ok(model, f"answer from {model}")

    litellm.completion = fake_completion
    litellm.completion_cost = lambda completion_response: None  # local: unknown cost

    clock = FakeClock()
    health = HealthRegistry(failure_threshold=2, cooldown_seconds=30.0, clock=clock)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
        sleep=lambda s: None,
        health=health,
    )
    engine = CollaborationEngine(policy=CollaborationPolicy(allow_agent_to_agent=True),
                                 tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    policy = FallbackPolicy(
        mode=FallbackMode.CONTINUITY, fallback_models=[BACKUP], max_fallback_attempts=2
    )
    print("== 1. ordered fallback on a transient failure ==")
    resp = gateway.complete_response([Message("user", "hi")], model=PRIMARY,
                                     agent="demo", run_id="demo-1", fallback=policy)
    print(f"   served by: {resp.model}")
    ctx = engine._contexts["demo-1"]
    print(f"   events: {[e.event.value for e in ctx.events()]}")
    print(f"   physical requests reserved: {ctx.physical_reserved}")

    print("== 2. circuit opens after threshold, probe recovers primary ==")
    gateway.complete_response([Message("user", "again")], model=PRIMARY, agent="demo",
                              run_id="demo-2", fallback=policy)  # 2nd failure -> OPEN
    mkey = HealthRegistry.model_key(PRIMARY)
    print(f"   circuit after 2 failures: {health.state(mkey).value}")
    resp = gateway.complete_response([Message("user", "third")], model=PRIMARY,
                                     agent="demo", run_id="demo-3", fallback=policy)
    print(f"   while OPEN, served by: {resp.model} (primary skipped)")
    clock.advance(31)  # cooldown elapsed -> HALF_OPEN
    state["fail_primary"] = False  # provider recovered
    resp = gateway.complete_response([Message("user", "fourth")], model=PRIMARY,
                                     agent="demo", run_id="demo-4", fallback=policy)
    print(f"   after cooldown probe: {resp.model}, circuit={health.state(mkey).value}")
    resp = gateway.complete_response([Message("user", "fifth")], model=PRIMARY,
                                     agent="demo", run_id="demo-5", fallback=policy)
    print(f"   next call prefers primary again: {resp.model}")
    print(f"   health snapshot: {health.snapshot()}")

    print("== 3. WAIT mode: durable suspension + manual resume ==")
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "checkpoints.db")
        store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE,
                                lease_ttl_seconds=60.0)
        wait_policy = FallbackPolicy(mode=FallbackMode.WAIT, cooldown_seconds=30.0,
                                     max_wait_seconds=300.0)
        sup_llm = gateway.bind(PRIMARY, agent="supervisor", fallback=wait_policy)
        analyst_llm = gateway.bind(BACKUP, agent="analyst", fallback=FallbackPolicy())
        analyst_agent = Agent("analyst", "demo analyst", "You are a demo analyst.",
                              analyst_llm, collaboration=engine)
        engine.register({"analyst": analyst_agent})
        analyst = Supervisor(
            {"analyst": analyst_agent}, sup_llm, collaboration=engine,
            checkpoint_store=store, checkpoint_config_json="{}",
            suspend_cooldown_seconds=30.0, max_wait_seconds=300.0,
        )
        state["fail_primary"] = True

        def failing(**kwargs):
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

        litellm.completion = failing
        result = analyst.run("demo task", run_id="demo-wait")
        cp = store.get("demo-wait")
        print(f"   suspended={result.suspended}, state={cp.state.value}, "
              f"next_attempt_at={cp.next_attempt_at}")
        print(f"   transcript stored (resumable): {cp.transcript_json is not None}")

        def finish(**kwargs):
            return ok(kwargs["model"], json.dumps({"next": "FINISH", "answer": "resumed!"}))

        litellm.completion = finish
        resumed = analyst.resume("demo-wait", owner="demo-worker")
        print(f"   resumed answer: {resumed.final_answer!r}, "
              f"final state={store.get('demo-wait').state.value}")
        print(f"   counters after resume (no double counting): "
              f"{store.get('demo-wait').counters_json}")
        store.close()

    print("\nDemo finished. All provider calls were faked locally; no API keys used.")


if __name__ == "__main__":
    main()
