"""Offline demo of the multi-model GEOSGAN HIVE (no API keys needed).

Uses a scripted FakeGateway that binds one fake LLM per agent, so each of the
five roles runs on its own configured model from config/agents.yaml.

Run from the repo root:  python examples/demo_offline.py
"""

import json
from pathlib import Path

from geosgan_hive.agent_config import load_hive_config
from geosgan_hive.factory import build_team
from geosgan_hive.usage import UsageTracker

ROLES = ("researcher", "developer", "analyst", "reviewer")


class FakeGateway:
    """Offline stand-in for ModelGateway: binds one scripted client per agent."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.tracker = UsageTracker()
        self.calls: list[tuple[str, str]] = []  # (model, agent)

    def bind(self, model, *, agent=None, temperature=0.2, max_tokens=None, timeout_seconds=None):
        gateway = self

        class _Client:
            def complete(self, messages, *, temperature=0.2, run_id=None):
                gateway.calls.append((model, agent))
                return gateway.handler(agent, list(messages))

        return _Client()


def routed_handler(agent, messages):
    if agent == "supervisor":
        delegations = sum(1 for m in messages if m.agent in ROLES)
        if delegations == 0:
            return json.dumps({"next": "researcher", "reason": "need the data first"})
        if delegations == 1:
            return json.dumps({"next": "developer", "reason": "now write it up"})
        return json.dumps({"next": "FINISH", "answer": "Final answer assembled by the team."})
    return f"[{agent}] working on the task..."


def main() -> None:
    config = load_hive_config(Path("config/agents.yaml"))
    gateway = FakeGateway(routed_handler)
    supervisor, _agents = build_team(config, gateway)

    print("Team (role -> model -> allowed tools):")
    for spec in [config.supervisor, *config.agents]:
        tools = spec.permissions.allowed_tools if spec.permissions.allow_tools else []
        print(f"  {spec.name}: {spec.model}  tools={tools or ['(none)']}")

    result = supervisor.run("Calculate 12 * 8 + 4 and write a sentence with the result")
    print("\nRoute:", " -> ".join(result.route) or "(direct)")
    print("Supervisor steps:", result.steps)
    print("Final answer:", result.final_answer)
    print("\nModel calls (model <- agent):")
    for model, agent in gateway.calls:
        print(f"  {model}  <- {agent}")


if __name__ == "__main__":
    main()
