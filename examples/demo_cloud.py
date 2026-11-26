"""Offline demo of the Phase 4 ZERO-COST CLOUD engine (no API keys, no network).

Loads config/agents.cloud.yaml (zero_cost=true, local Ollama profile, durable
SQLite quota store), wires the real ModelGateway + ZeroCostGate through
build_team(), blocks all outbound network access, and scripts LiteLLM responses
so no Ollama server is needed either.

Every physical dispatch passes the zero-cost gate: evidence + endpoint binding,
conservative token bounds, durable quota reservation, then single-use dispatch
authorization. Total spend is strictly $0.

Run from the repo root:  python examples/demo_cloud.py [--checkpoint PATH]
"""

import argparse
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import litellm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geosgan_hive.agent_config import load_hive_config  # noqa: E402
from geosgan_hive.cloud_profiles import LocalOllamaAdapter  # noqa: E402
from geosgan_hive.factory import build_team  # noqa: E402
from geosgan_hive.gateway import ModelGateway  # noqa: E402
from geosgan_hive.retry import RetryPolicy  # noqa: E402
from geosgan_hive.usage import UsageTracker  # noqa: E402

MODEL = "ollama/llama3.2"


def block_network() -> None:
    """Fail loudly on any outbound network attempt (offline guarantee)."""

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted during the offline demo")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def make_handler():
    """Scripted LiteLLM replacement: the supervisor routes once, then finishes."""
    state = {"sup_calls": 0}

    def handler(**kwargs):
        messages = kwargs.get("messages") or []
        system = messages[0].get("content", "") if messages else ""
        if "supervisor" in system:
            state["sup_calls"] += 1
            if state["sup_calls"] == 1:
                content = json.dumps({"next": "researcher", "reason": "need the data first"})
            else:
                content = json.dumps(
                    {"next": "FINISH", "answer": "Zero-cost answer assembled by the HIVE team."}
                )
        else:
            content = "[agent] working on the task..."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=kwargs.get("model", MODEL),
        )

    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Override the checkpoint/quota store path from the config")
    args = parser.parse_args()

    block_network()
    litellm.completion = make_handler()
    litellm.acompletion = make_handler()
    litellm.completion_cost = lambda *a, **k: 0.0

    config = load_hive_config(REPO_ROOT / "config" / "agents.cloud.yaml", environ={})
    if args.checkpoint:
        cc = config.checkpoint.model_copy(update={"path": args.checkpoint})
        config = config.model_copy(update={"checkpoint": cc})

    from geosgan_hive.checkpoints import CheckpointStore

    store = CheckpointStore(
        config.checkpoint.path,
        persist_mode=config.checkpoint.persist_mode,
        lease_ttl_seconds=config.checkpoint.lease_ttl_seconds,
    )
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                 max_delay_seconds=0, jitter_ratio=0),
    )
    supervisor, _agents = build_team(config, gateway, tracker,
                                     checkpoint_store=store,
                                     cloud_adapter=LocalOllamaAdapter())

    gate = gateway.cloud_gate()
    print(f"Zero-cost gate active: {gate.enabled()}")
    print(f"Model: {MODEL}  (local inference, billing enforcement=local_inference)")

    result = supervisor.run("Calculate 12 * 8 + 4 and write a sentence with the result")
    print("\nRoute:", " -> ".join(result.route) or "(direct)")
    print("Supervisor steps:", result.steps)
    print("Suspended:", result.suspended)
    print("Final answer:", result.final_answer)

    status = gate.engine.quota_status(MODEL)
    print("\nDurable quota usage (shared account pool, current window):")
    for key, row in sorted(status.items()):
        print(f"  {key}: {row}")
    total = tracker.total_cost()
    if total is not None:
        cost_str = f"${total:.6f}"
    else:
        # None = no confirmed per-request provider cost (local inference); the
        # zero-cost guarantee is enforced by the gate BEFORE dispatch (cap $0).
        cost_str = "$0 (local inference; gate-enforced cap $0.0, estimate 0.0)"
    print(f"\nTotal cost: {cost_str}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
