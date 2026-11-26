"""Final pre-PR targeted verification of the zero-cost gate (C6 approval checks).

Fully offline (mocked LiteLLM + loopback-tolerant network spy). Covers:
- LocalOllamaAdapter routing through the gateway: only the evidence-bound
  loopback endpoint is ever used; the adapter itself never dispatches.
- FREE_ONLY endpoint safety: local-inference dispatches must target a
  LOOPBACK endpoint even when evidence has no endpoint binding and even when
  OLLAMA_API_BASE is overridden; proxy env vars must not be able to reroute a
  zero-cost dispatch; DNS/connects to external hosts are blocked by the spy.
- Conservative token bounds cover the COMPLETE serialized payload (system
  messages + tool schemas + framing + output limit).
- The shared no_network fixture blocks all external addresses and allows only
  loopback.
"""

import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from geosgan_hive.agent_config import load_hive_config
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.cloud_gate import ZeroCostGate
from geosgan_hive.cloud_profiles import (
    CloudProfile,
    CloudProfileRegistry,
    LocalOllamaAdapter,
)
from geosgan_hive.errors import CloudConfigError
from geosgan_hive.factory import build_team
from geosgan_hive.gateway import GatewayLLMClient, ModelGateway
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUD_YAML = REPO_ROOT / "config" / "agents.cloud.yaml"
MODEL = "ollama/llama3.2"
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
FP = "b" * 64
POOL = "ollama:acct:" + FP


def ollama_local_profile(*, endpoint=None):
    ev = {
        "source": "local_inference_runtime", "evidence_ref": "ev-local",
        "collected_at": (NOW - timedelta(days=1)).isoformat(),
        "collected_by": "operator-1", "credential_bound": True,
        "billing_enforcement": "local_inference", "account_fingerprint": FP,
    }
    if endpoint is not None:
        ev["endpoint"] = endpoint
    return {
        "provider": "ollama", "models": [MODEL], "billing": "free_tier_verified",
        "verified_free": True, "overage_billing_disabled": True, "billing_verified": True,
        "verified_at": (NOW - timedelta(days=1)).isoformat(), "verification_ttl_days": 30,
        "account_label": "local", "account_fingerprint": FP,
        "quota_window_start": "2026-01-01T00:00:00+00:00",
        "quota_reset_semantics": "operator_attested",
        "verification_evidence": ev,
        "quotas": [{"pool": POOL, "max_requests": 1000, "max_tokens": 10**9,
                    "period": "day"}],
    }


def registry_from(profile, *, trusted=True, zero_cost=True):
    from geosgan_hive.agent_config import CloudConfig
    return CloudProfileRegistry.from_config(CloudConfig(
        zero_cost=zero_cost, max_spend_usd=0.0 if zero_cost else None,
        trusted_evidence_binding=trusted,
        profiles=[CloudProfile.model_validate(profile)]))


def make_gate(tmp_path, profile):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0, utcnow=lambda: NOW)
    gate = ZeroCostGate(registry_from(profile), store,
                        adapter=LocalOllamaAdapter(), utcnow=lambda: NOW)
    return store, gate


# ---- item 1+2: routing through the gateway goes only to the bound loopback ----

def test_gateway_routes_only_to_bound_loopback_endpoint(tmp_path, monkeypatch, no_network):
    cfg = load_hive_config(CLOUD_YAML, environ={})
    cc = cfg.checkpoint.model_copy(update={"path": str(tmp_path / "cp.db")})
    cfg = cfg.model_copy(update={"checkpoint": cc})
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    captured = []

    def recorder(**kwargs):
        captured.append(kwargs)
        from types import SimpleNamespace
        content = json.dumps({"next": "FINISH", "answer": "done"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=kwargs.get("model", MODEL))

    monkeypatch.setattr("litellm.completion", recorder)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)

    def adapter_must_never_dispatch(self, model, messages):
        raise AssertionError("adapter.dispatch must never be called by the gateway")

    monkeypatch.setattr(LocalOllamaAdapter, "dispatch", adapter_must_never_dispatch)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"OLLAMA_API_BASE": "http://localhost:11434"})
    supervisor, _ = build_team(cfg, gw, checkpoint_store=store,
                               cloud_adapter=LocalOllamaAdapter())
    result = supervisor.run("Say hi")
    assert result.final_answer
    assert captured, "expected at least one physical dispatch"
    for kwargs in captured:
        assert kwargs["model"].startswith("ollama/")
        # every physical request targets the evidence-bound LOOPBACK endpoint
        assert kwargs.get("api_base") == "http://localhost:11434"
    store.close()


# ---- item 2: local inference must be loopback even without endpoint binding ----

def test_local_inference_requires_loopback_endpoint(tmp_path):
    # evidence WITHOUT endpoint binding: an api_base override to a remote host
    # must still be rejected for local_inference enforcement (fail closed)
    store, gate = make_gate(tmp_path, ollama_local_profile(endpoint=None))
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32, "api_base": "https://remote-billable.example.com"}
    with pytest.raises(CloudConfigError, match="loopback"):
        gate.prepare_dispatch(MODEL, kwargs)
    # loopback variants are accepted
    for base in ("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"):
        kw = dict(kwargs, api_base=base)
        did = gate.prepare_dispatch(MODEL, kw)
        assert did
    store.close()


# ---- item 2: proxy env vars must not reroute a zero-cost dispatch ----

def _gateway_with_env(tmp_path, monkeypatch, environ):
    cfg = load_hive_config(CLOUD_YAML, environ={})
    cc = cfg.checkpoint.model_copy(update={"path": str(tmp_path / "cp.db")})
    cfg = cfg.model_copy(update={"checkpoint": cc})
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        from types import SimpleNamespace
        content = json.dumps({"next": "FINISH", "answer": "done"})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=kwargs.get("model", MODEL))

    monkeypatch.setattr("litellm.completion", recorder)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ=environ)
    supervisor, _ = build_team(cfg, gw, checkpoint_store=store,
                               cloud_adapter=LocalOllamaAdapter(),
                               cloud_default_max_output_tokens=512)
    return gw, supervisor, calls, store


def test_proxy_env_without_no_proxy_fails_closed(tmp_path, monkeypatch, no_network):
    _gw, supervisor, calls, store = _gateway_with_env(tmp_path, monkeypatch, {
        "OLLAMA_API_BASE": "http://localhost:11434",
        "HTTP_PROXY": "http://proxy.internal:8080",
        "HTTPS_PROXY": "http://proxy.internal:8080",
        "ALL_PROXY": "http://proxy.internal:8080",
    })
    from geosgan_hive.messages import Message
    client = GatewayLLMClient(_gw, model=MODEL, agent="supervisor")
    with pytest.raises(CloudConfigError, match="proxy"):
        client.complete([Message(role="user", content="hi")])
    assert not calls  # nothing was dispatched
    store.close()


def test_proxy_env_with_no_proxy_loopback_allows_dispatch(tmp_path, monkeypatch, no_network):
    _gw, _sup, calls, store = _gateway_with_env(tmp_path, monkeypatch, {
        "OLLAMA_API_BASE": "http://localhost:11434",
        "HTTPS_PROXY": "http://proxy.internal:8080",
        "NO_PROXY": "localhost,127.0.0.1",
    })
    from geosgan_hive.messages import Message
    client = GatewayLLMClient(_gw, model=MODEL, agent="supervisor")
    resp = client.complete([Message(role="user", content="hi")])
    assert resp  # completed (returns the answer string)
    assert calls and calls[0]["api_base"] == "http://localhost:11434"
    store.close()


# ---- item 3: conservative bounds cover the complete serialized payload ----

def test_payload_covers_system_tools_and_framing(tmp_path):
    store, gate = make_gate(tmp_path, ollama_local_profile(
        endpoint="http://localhost:11434"))
    messages = [{"role": "system", "content": "x" * 500},
                {"role": "user", "content": "y" * 2000},
                {"role": "assistant", "content": "z" * 300}]
    tools = [{"type": "function",
              "function": {"name": "calculator",
                           "parameters": {"type": "object",
                                          "properties": {"a": {"type": "number"}}}}}]
    kwargs = {"model": MODEL, "messages": messages, "tools": tools, "max_tokens": 64,
              "api_base": "http://localhost:11434"}
    pc = gate._payload_chars(kwargs)
    raw = len(json.dumps(messages)) + len(json.dumps(tools)) + len(MODEL)
    assert pc >= raw + 256  # complete payload + conservative protocol overhead
    did = gate.prepare_dispatch(MODEL, kwargs)
    res = store.get_reservation(did)
    # the durable reservation covers the whole payload at >= 1 token per char
    assert res["prompt_tokens"] >= pc >= raw
    assert res["max_output_tokens"] >= 64
    store.close()


# ---- item 4: the shared spy blocks everything external, allows only loopback ----

def test_network_spy_blocks_external_allows_loopback(no_network):
    with pytest.raises(AssertionError, match="network access"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    with pytest.raises(AssertionError, match="DNS"):
        socket.getaddrinfo("api.openai.com", 443)
    s = socket.socket()
    with pytest.raises(AssertionError, match="network access"):
        s.connect(("10.1.2.3", 443))
    s.close()
    # loopback stays available (Windows asyncio socketpair needs it)
    a, b = socket.socketpair()
    a.close()
    b.close()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    c = socket.create_connection(("127.0.0.1", port), timeout=2)
    c.close()
    srv.close()
