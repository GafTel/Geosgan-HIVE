"""Phase 4 C6: operator packaging of the zero-cost cloud engine.

Fully offline (mocked LiteLLM + network spy). Covers: the shipped
config/agents.cloud.yaml (validation, fail-closed zero-cost rules), the
LocalOllamaAdapter contract (incl. that FREE_ONLY requests can never be
routed to a remote/billable endpoint), factory + CLI wiring of the zero-cost
gate (including the regression that the gate's QuotaEngine shares the store
clock and that the CLI passes a real adapter), an end-to-end offline
supervisor run through the gate with durable quota accounting, the offline
cloud demo in a fresh subprocess, and version consistency.
"""

import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive import __version__
from geosgan_hive.agent_config import HiveConfig, load_hive_config
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.cloud_gate import ZeroCostGate, effective_endpoint
from geosgan_hive.cloud_profiles import (
    CloudProfileRegistry,
    FakeCloudAdapter,
    LocalOllamaAdapter,
    TokenEstimate,
    bounded_token_estimate,
)
from geosgan_hive.errors import CloudConfigError, CloudOfflineError, CostUnknownError
from geosgan_hive.factory import build_team
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUD_YAML = REPO_ROOT / "config" / "agents.cloud.yaml"
MODEL = "ollama/llama3.2"


# no_network fixture lives in tests/conftest.py (loopback-tolerant)


@pytest.fixture
def litellm_spy(monkeypatch):
    def finish_response(**kwargs):
        import json
        from types import SimpleNamespace
        content = json.dumps({"next": "FINISH", "answer": "done"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=kwargs.get("model", MODEL))
    monkeypatch.setattr("litellm.completion", finish_response)
    monkeypatch.setattr("litellm.acompletion", finish_response)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)


def cloud_config(tmp_path) -> HiveConfig:
    cfg = load_hive_config(CLOUD_YAML, environ={})
    cc = cfg.checkpoint.model_copy(update={"path": str(tmp_path / "cp.db")})
    return cfg.model_copy(update={"checkpoint": cc})


def make_gate(tmp_path):
    cfg = cloud_config(tmp_path)
    reg = CloudProfileRegistry.from_config(cfg.cloud)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0,
                            utcnow=lambda: datetime(2026, 1, 15, tzinfo=UTC))
    gate = ZeroCostGate(reg, store, adapter=LocalOllamaAdapter())
    return cfg, reg, store, gate


# ---- regression: gate engine must share the store clock (was None -> TypeError) ----

def test_gate_engine_uses_store_clock(tmp_path):
    _cfg, _reg, store, gate = make_gate(tmp_path)  # no utcnow passed to the gate
    n = datetime(2026, 1, 15, tzinfo=UTC)
    status = gate.engine.quota_status(MODEL, now=n)  # must not raise TypeError
    assert isinstance(status, dict)
    gate.engine.reserve_for_model(
        MODEL, "d-1", TokenEstimate(prompt_tokens=10, max_output_tokens=5),
        now=n, cost_usd=0.0, endpoint="http://localhost:11434")
    retry_at = gate.quota_retry_at(MODEL, now=n)
    assert retry_at is not None and retry_at > n  # local next window, never invented
    store.close()


# ---- LocalOllamaAdapter contract + endpoint safety ----

def test_local_ollama_adapter_contract():
    ad = LocalOllamaAdapter()
    assert ad.allow_network is False and ad.is_test_only is False
    assert ad.capabilities.live_supported is True
    assert ad.estimate_cost(MODEL, prompt_tokens=10, max_output_tokens=5) == 0.0
    assert ad.estimate_cost("gemini/gemini-2.0-flash", prompt_tokens=10,
                            max_output_tokens=5) is None  # unknown != zero
    est = ad.estimate_tokens(MODEL, prompt_chars=1000, max_output_tokens=128)
    assert est == TokenEstimate(prompt_tokens=1000, max_output_tokens=128)
    assert est.prompt_tokens >= 1000  # conservative: >= 1 token per payload char
    out = bounded_token_estimate(ad, MODEL, prompt_chars=1000, max_output_tokens=128, live=True)
    assert out.prompt_tokens >= 1000  # passes LIVE conservative-bound enforcement
    assert ad.estimate_tokens("gemini/x", prompt_chars=10, max_output_tokens=5) is None
    with pytest.raises(CloudOfflineError):
        ad.dispatch(MODEL, [])  # the adapter itself NEVER routes any request
    fake = FakeCloudAdapter()
    assert fake.is_test_only and fake.allow_network is False
    with pytest.raises(CostUnknownError):
        bounded_token_estimate(fake, "fake/mini", prompt_chars=100,
                               max_output_tokens=10, live=True)


def test_free_only_requests_cannot_be_routed_to_remote_or_billable_endpoints(tmp_path):
    _cfg, reg, store, gate = make_gate(tmp_path)
    n = datetime(2026, 1, 15, tzinfo=UTC)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32}
    # default effective endpoint is loopback-only for ollama
    assert effective_endpoint(MODEL, kwargs) == "http://localhost:11434"
    # the shipped evidence binds the endpoint: any other api_base is rejected
    bad = dict(kwargs, api_base="https://remote-billable.example.com")
    with pytest.raises(CloudConfigError, match="endpoint"):
        gate._check_evidence_binding(MODEL, bad, n)
    # a non-ollama (potentially billable) model has no profile -> blocked
    with pytest.raises(CloudConfigError, match="no cloud profile"):
        gate._check_evidence_binding("openai/gpt-4o-mini", kwargs, n)
    # and the adapter reports unknown (not zero) cost for it
    assert LocalOllamaAdapter().estimate_cost(
        "openai/gpt-4o-mini", prompt_tokens=1, max_output_tokens=1) is None
    store.close()


# ---- shipped cloud YAML ----

def test_cloud_yaml_loads_and_is_fail_closed(tmp_path):
    cfg = load_hive_config(CLOUD_YAML, environ={})
    assert cfg.cloud is not None and cfg.cloud.zero_cost is True
    assert cfg.cloud.max_spend_usd == 0.0
    assert cfg.checkpoint is not None and cfg.checkpoint.path != ":memory:"
    assert all(s.model == MODEL for s in [cfg.supervisor, *cfg.agents])
    reg = CloudProfileRegistry.from_config(cfg.cloud)
    ok, reason = reg.live_dispatch_eligible(MODEL, now=datetime.now(UTC))
    assert ok, reason  # local_inference (ollama only) is the supported zero-cost path
    ok2, _ = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=datetime.now(UTC))
    assert not ok2  # live cloud providers stay blocked


def test_cloud_yaml_zero_cost_requires_durable_store(tmp_path):
    cfg = load_hive_config(CLOUD_YAML, environ={})
    data = cfg.model_dump()
    data["checkpoint"] = None
    with pytest.raises(ValidationError, match="checkpoint"):
        HiveConfig.model_validate(data)
    data = cfg.model_dump()
    data["checkpoint"]["path"] = ":memory:"
    with pytest.raises(ValidationError, match=":memory:"):
        HiveConfig.model_validate(data)
    data = cfg.model_dump()
    data["cloud"]["max_spend_usd"] = None
    with pytest.raises(ValidationError):
        HiveConfig.model_validate(data)  # unlimited spend rejected in zero-cost mode


# ---- factory + CLI wiring ----

def test_build_team_from_cloud_yaml_wires_gate(tmp_path):
    cfg = cloud_config(tmp_path)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0))
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    supervisor, agents = build_team(cfg, gw, checkpoint_store=store,
                                    cloud_adapter=LocalOllamaAdapter())
    gate = gw.cloud_gate()
    assert gate is not None and gate.enabled()
    assert gate.adapter.name == "ollama-local"
    assert set(agents) == {"researcher", "developer", "analyst", "reviewer"}
    store.close()


def test_supervisor_run_offline_through_gate(tmp_path, monkeypatch, no_network, litellm_spy):
    cfg = cloud_config(tmp_path)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0))
    supervisor, _ = build_team(cfg, gw, checkpoint_store=store,
                               cloud_adapter=LocalOllamaAdapter())
    result = supervisor.run("Say hi")
    assert not result.suspended
    assert result.final_answer
    gate = gw.cloud_gate()
    status = gate.engine.quota_status(MODEL)
    used = status[("requests", "day")]["used"]
    assert used >= 1  # durable reservation+dispatch recorded, never reset
    assert status[("tokens", "day")]["used"] > 0
    store.close()


def test_cli_run_with_cloud_config_offline(tmp_path, monkeypatch, no_network, litellm_spy,
                                           capsys):
    # regression: CLI must pass a real cloud adapter; with None the gate failed
    # closed on every dispatch (CostUnknownError) and the run FAILED
    from geosgan_hive.cli import main
    rc = main(["Say hi", "--agents-config", str(CLOUD_YAML),
               "--checkpoint", str(tmp_path / "cli.db")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "SUSPENDED" not in out
    assert "Final answer" in out


def test_cli_live_cloud_model_blocked_in_zero_cost(tmp_path, monkeypatch, no_network,
                                                   litellm_spy, capsys):
    import yaml

    from geosgan_hive.cli import main
    cfg = load_hive_config(CLOUD_YAML, environ={})
    data = cfg.model_dump(mode="json")
    data["supervisor"]["model"] = "gemini/gemini-2.0-flash"  # no profile -> blocked
    data["checkpoint"]["path"] = str(tmp_path / "cli2.db")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    rc = main(["Say hi", "--agents-config", str(bad)])
    captured = capsys.readouterr()
    text = captured.out + captured.err
    # permanent billing-policy failure: fail closed, never suspended/retried
    assert "no cloud profile" in text
    assert "SUSPENDED" not in captured.out
    assert rc in (0, 1)


# ---- offline cloud demo in a fresh subprocess (network blocked inside the demo) ----

def test_demo_cloud_runs_fully_offline(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "demo_cloud.py"),
         "--checkpoint", str(tmp_path / "demo.db")],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "Zero-cost gate active: True" in proc.stdout
    assert "Durable quota usage" in proc.stdout
    assert "Total cost: $0" in proc.stdout
    assert "network access attempted" not in proc.stderr


# ---- version consistency ----

def test_version_matches_pyproject():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"] == "0.6.0"
