import json
from pathlib import Path

from geosgan_hive.agent_config import load_hive_config
from geosgan_hive.factory import build_team
from geosgan_hive.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent

TEAM_YAML = """
supervisor:
  name: supervisor
  description: routes
  system_prompt: you supervise
  model: openai/gpt-4o-mini
agents:
  - name: researcher
    description: investigates
    system_prompt: you research
    model: gemini/gemini-2.0-flash
    permissions: {allow_tools: true, allowed_tools: [calculator]}
  - name: developer
    description: codes
    system_prompt: you code
    model: mistral/mistral-large-latest
    permissions: {allow_tools: true, allowed_tools: [calculator]}
  - name: analyst
    description: analyzes
    system_prompt: you analyze
    model: anthropic/claude-3-7-sonnet-latest
    permissions: {allow_tools: false, allowed_tools: []}
  - name: reviewer
    description: reviews
    system_prompt: you review
    model: ollama/llama3.2
    permissions: {allow_tools: true, allowed_tools: [word_count]}
"""


class RecordingClient:
    def __init__(self, gateway, model, agent):
        self.gateway = gateway
        self.model = model
        self.agent = agent

    def complete(self, messages, *, temperature=0.2, run_id=None):
        self.gateway.calls.append(
            {
                "model": self.model,
                "agent": self.agent,
                "run_id": run_id,
                "messages": list(messages),
            }
        )
        return self.gateway.handler(self.agent, list(messages))


class FakeGateway:
    """Offline gateway double: records which model each agent used."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.tracker = UsageTracker()

    def bind(self, model, *, agent=None, temperature=0.2, max_tokens=None, timeout_seconds=None):
        return RecordingClient(self, model, agent)


def make_handler():
    def handler(agent, messages):
        if agent == "supervisor":
            delegations = sum(1 for m in messages if m.agent in ("researcher", "developer"))
            if delegations == 0:
                return json.dumps({"next": "researcher", "reason": "data first"})
            if delegations == 1:
                return json.dumps({"next": "developer", "reason": "then code"})
            return json.dumps({"next": "FINISH", "answer": "team answer"})
        return f"[{agent}] done"

    return handler


def load_team(tmp_path):
    path = tmp_path / "team.yaml"
    path.write_text(TEAM_YAML, encoding="utf-8")
    return load_hive_config(path, environ={})


def test_build_team_gives_each_role_its_own_model(tmp_path):
    config = load_team(tmp_path)
    gateway = FakeGateway(make_handler())
    supervisor, agents = build_team(config, gateway)
    assert set(agents) == {"researcher", "developer", "analyst", "reviewer"}
    result = supervisor.run("do something")
    assert result.route == ["researcher", "developer"]
    assert result.final_answer == "team answer"
    by_agent = {c["agent"]: c["model"] for c in gateway.calls}
    assert by_agent["supervisor"] == "openai/gpt-4o-mini"
    assert by_agent["researcher"] == "gemini/gemini-2.0-flash"
    assert by_agent["developer"] == "mistral/mistral-large-latest"
    # all calls of one run share the same run_id
    assert len({c["run_id"] for c in gateway.calls}) == 1


def test_build_team_applies_least_privilege_per_agent(tmp_path):
    config = load_team(tmp_path)
    gateway = FakeGateway(make_handler())
    _supervisor, agents = build_team(config, gateway)
    assert agents["researcher"].tools.names() == ["calculator"]
    assert agents["analyst"].tools.names() == []  # allow_tools: false
    assert agents["reviewer"].tools.names() == ["word_count"]
    # the model only sees its allowed tools in the system prompt
    gateway.handler("researcher", [])
    supervisor = build_team(config, gateway)[0]
    supervisor.run("x")
    researcher_system = next(
        c["messages"][0].content for c in gateway.calls if c["agent"] == "researcher"
    )
    assert "calculator" in researcher_system
    assert "word_count" not in researcher_system


def test_default_repo_config_builds_the_five_roles():
    config = load_hive_config(REPO_ROOT / "config" / "agents.yaml", environ={})
    gateway = FakeGateway(make_handler())
    supervisor, agents = build_team(config, gateway)
    assert supervisor.max_steps == 8
    assert set(agents) == {"researcher", "developer", "analyst", "reviewer"}
    models = {name: agent.llm.model for name, agent in agents.items()}
    models["supervisor"] = supervisor.llm.model
    assert len(set(models.values())) == 5  # five independent models
