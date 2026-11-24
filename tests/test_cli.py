import json
from pathlib import Path
from types import SimpleNamespace

import litellm

from geosgan_hive.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO_ROOT / "config" / "agents.local.yaml"


def fake_finish_response(**kwargs):
    content = json.dumps({"next": "FINISH", "answer": "done by the team"})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        model=kwargs.get("model", "ollama/llama3.2"),
    )


def test_cli_runs_offline_with_local_config_and_writes_usage_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(litellm, "completion", fake_finish_response)

    def no_cost(completion_response):
        raise ValueError("local model has no known cost")

    monkeypatch.setattr(litellm, "completion_cost", no_cost)
    log = tmp_path / "usage.jsonl"
    rc = main(["do something", "--agents-config", str(LOCAL_CONFIG), "--usage-log", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Route:" in out
    assert "Final answer:" in out and "done by the team" in out
    assert "Usage summary:" in out
    assert "cost_usd=unknown" in out  # unknown costs shown as unknown, never 0
    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # only the supervisor call happened
    record = json.loads(lines[0])
    assert record["agent"] == "supervisor"
    assert record["provider"] == "ollama"
    assert record["cost_usd"] is None
    assert record["total_tokens"] == 5


def test_cli_missing_config_returns_2_with_error(tmp_path, capsys):
    rc = main(["x", "--agents-config", str(tmp_path / "nope.yaml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err


def test_cli_routes_through_agents_with_per_agent_models(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if "Available agents" in kwargs["messages"][0]["content"]:
            delegations = sum(1 for m in kwargs["messages"] if m["content"] == "researcher output")
            if delegations == 0:
                content = json.dumps({"next": "researcher", "reason": "research first"})
            else:
                content = json.dumps({"next": "FINISH", "answer": "routed answer"})
        else:
            content = "researcher output"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0001)
    rc = main(["task", "--agents-config", str(LOCAL_CONFIG)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Route: researcher" in out
    assert "routed answer" in out
    # supervisor and researcher both ran on their own (here: same local) model
    assert calls == ["ollama/llama3.2", "ollama/llama3.2", "ollama/llama3.2"]
    assert "cost_usd=$0.000300" in out


def test_cli_async_flag_runs_arun_offline(monkeypatch, capsys):
    async def fake_acompletion(**kwargs):
        content = json.dumps({"next": "FINISH", "answer": "async done"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            model=kwargs.get("model", "ollama/llama3.2"),
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    def no_cost(completion_response):
        raise ValueError("local model has no known cost")

    monkeypatch.setattr(litellm, "completion_cost", no_cost)
    rc = main(["task", "--agents-config", str(LOCAL_CONFIG), "--async"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "async done" in out
    assert "Usage summary:" in out
