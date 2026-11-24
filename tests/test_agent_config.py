from pathlib import Path

import pytest

from geosgan_hive.agent_config import (
    CORE_ROLES,
    AgentSpec,
    HiveConfig,
    Limits,
    Permissions,
    SupervisorSpec,
    apply_env_overrides,
    load_hive_config,
)
from geosgan_hive.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "agents.yaml"
LOCAL_CONFIG = REPO_ROOT / "config" / "agents.local.yaml"


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL_YAML = """
supervisor:
  name: supervisor
  description: routes
  system_prompt: you supervise
  model: openai/gpt-4o-mini
agents:
  - name: researcher
    description: investigates
    system_prompt: you research
    model: ollama/llama3.2
"""


def test_default_config_has_exactly_the_five_core_roles():
    config = load_hive_config(DEFAULT_CONFIG, environ={})
    assert config.supervisor.name == "supervisor"
    assert sorted(a.name for a in config.agents) == sorted(CORE_ROLES[1:])
    assert set(CORE_ROLES) == {"supervisor", "researcher", "developer", "analyst", "reviewer"}


def test_default_config_uses_independent_models_per_agent():
    config = load_hive_config(DEFAULT_CONFIG, environ={})
    models = {spec.name: spec.model for spec in [config.supervisor, *config.agents]}
    assert len(set(models.values())) == 5  # five independent model configurations
    assert models["supervisor"].startswith("openai/")
    assert models["researcher"].startswith("gemini/")
    assert models["developer"].startswith("mistral/")
    assert models["analyst"].startswith("anthropic/")
    assert models["reviewer"].startswith("ollama/")


def test_local_config_requires_no_api_keys():
    config = load_hive_config(LOCAL_CONFIG, environ={})
    assert all(spec.model.startswith("ollama/") for spec in [config.supervisor, *config.agents])


def test_least_privilege_is_the_default():
    assert Permissions().allow_tools is False
    assert Permissions().allowed_tools == []
    spec = AgentSpec(
        name="reviewer", description="d", system_prompt="s", model="ollama/llama3.2"
    )
    assert spec.permissions.allow_tools is False  # no tools unless explicit
    assert spec.limits.max_calls_per_run == 50
    assert Limits().max_cost_usd_per_run is None


def test_minimal_yaml_loads_with_defaults(tmp_path):
    path = write_yaml(tmp_path, MINIMAL_YAML)
    config = load_hive_config(path, environ={})
    assert config.supervisor.max_steps == 8
    assert config.agents[0].permissions.allow_tools is False
    assert config.agents[0].limits.timeout_seconds == 60.0


def test_invalid_model_prefix_is_rejected(tmp_path):
    bad = MINIMAL_YAML.replace("ollama/llama3.2", "cohere/command-r")
    path = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError) as exc_info:
        load_hive_config(path, environ={})
    assert "cohere" in str(exc_info.value)


def test_model_without_prefix_is_rejected(tmp_path):
    bad = MINIMAL_YAML.replace("ollama/llama3.2", "llama3.2")
    path = write_yaml(tmp_path, bad)
    with pytest.raises(ConfigError):
        load_hive_config(path, environ={})


def test_duplicate_agent_names_rejected(tmp_path):
    dup = MINIMAL_YAML + """  - name: researcher
    description: again
    system_prompt: s
    model: ollama/llama3.2
"""
    path = write_yaml(tmp_path, dup)
    with pytest.raises(ConfigError) as exc_info:
        load_hive_config(path, environ={})
    assert "duplicate" in str(exc_info.value)


def test_missing_file_and_invalid_yaml_raise_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_hive_config(tmp_path / "nope.yaml", environ={})
    path = write_yaml(tmp_path, "supervisor: [unclosed")
    with pytest.raises(ConfigError):
        load_hive_config(path, environ={})


def test_env_overrides_are_explicit_and_typed(tmp_path):
    path = write_yaml(tmp_path, MINIMAL_YAML)
    env = {
        "HIVE_AGENT_RESEARCHER_MODEL": "mistral/mistral-small-latest",
        "HIVE_AGENT_RESEARCHER_TEMPERATURE": "0.7",
        "HIVE_SUPERVISOR_MAX_STEPS": "12",
        "HIVE_SUPERVISOR_MODEL": "anthropic/claude-3-7-sonnet-latest",
    }
    config = load_hive_config(path, environ=env)
    assert config.agents[0].model == "mistral/mistral-small-latest"
    assert config.agents[0].temperature == 0.7  # coerced to float by pydantic
    assert config.supervisor.max_steps == 12  # coerced to int
    assert config.supervisor.model == "anthropic/claude-3-7-sonnet-latest"
    # untouched fields keep their YAML values
    assert config.supervisor.name == "supervisor"


def test_env_overrides_do_not_leak_into_other_agents(tmp_path):
    path = write_yaml(tmp_path, MINIMAL_YAML)
    config = load_hive_config(path, environ={"HIVE_AGENT_RESEARCHER_MODEL": "ollama/qwen3"})
    assert config.agents[0].model == "ollama/qwen3"
    assert config.supervisor.model == "openai/gpt-4o-mini"


def test_apply_env_overrides_normalizes_dashes_in_names():
    data = {
        "agents": [
            {"name": "data-scientist", "model": "openai/gpt-4o-mini"},
        ]
    }
    out = apply_env_overrides(data, {"HIVE_AGENT_DATA_SCIENTIST_MODEL": "ollama/llama3.2"})
    assert out["agents"][0]["model"] == "ollama/llama3.2"
    assert data["agents"][0]["model"] == "openai/gpt-4o-mini"  # input not mutated


def test_agent_lookup():
    config = load_hive_config(DEFAULT_CONFIG, environ={})
    assert isinstance(config.agent("researcher"), AgentSpec)
    with pytest.raises(ConfigError):
        config.agent("ghost")


def test_supervisor_spec_extends_agent_spec():
    spec = SupervisorSpec(
        name="supervisor", description="d", system_prompt="s", model="openai/gpt-4o-mini"
    )
    assert isinstance(spec, AgentSpec)
    assert spec.max_steps == 8
    assert isinstance(spec, HiveConfig.model_fields["supervisor"].annotation)
