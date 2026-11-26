"""Real-transport verification of the zero-cost gate (pre-merge blocker 1).

Unlike the mocked-LiteLLM tests, these spin up a DETERMINISTIC local HTTP
server on 127.0.0.1 and let the REAL LiteLLM -> httpx transport run against it
(loopback only; the no_network fixture acts as an oracle that records any
external connection attempt). No live cloud request is ever made.

Covers:
- end-to-end dispatch over the real transport to the authorized loopback endpoint
- a 3xx redirect from the local endpoint to an EXTERNAL host is never followed
  (LiteLLM's module-level clients are hardened to follow_redirects=False)
- an endpoint host that resolves to a NON-loopback address fails closed
"""

import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.cloud_gate import ZeroCostGate
from geosgan_hive.cloud_profiles import (
    CloudProfile,
    CloudProfileRegistry,
    LocalOllamaAdapter,
)
from geosgan_hive.errors import CloudConfigError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

MODEL = "ollama/llama3.2"
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
FP = "c" * 64
POOL = "ollama:acct:" + FP


class _OllamaHandler(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        type(self).seen.append({"path": self.path, "host": self.headers.get("Host")})
        if type(self).mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://192.0.2.1:9/evil")
            self.end_headers()
            return
        payload = {"model": "llama3.2", "created_at": "2026-01-01T00:00:00Z", "done": True,
                   "response": "hello from real local transport",
                   "prompt_eval_count": 3, "eval_count": 5}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def local_server():
    _OllamaHandler.mode = "ok"
    _OllamaHandler.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", _OllamaHandler
    srv.shutdown()
    t.join(timeout=5)


def _profile(endpoint):
    return {
        "provider": "ollama", "models": [MODEL], "billing": "free_tier_verified",
        "verified_free": True, "overage_billing_disabled": True, "billing_verified": True,
        "verified_at": (NOW - timedelta(days=1)).isoformat(), "verification_ttl_days": 30,
        "account_label": "local", "account_fingerprint": FP,
        "quota_window_start": "2026-01-01T00:00:00+00:00",
        "quota_reset_semantics": "operator_attested",
        "verification_evidence": {
            "source": "local_inference_runtime", "evidence_ref": "ev-local",
            "collected_at": (NOW - timedelta(days=1)).isoformat(),
            "collected_by": "operator-1", "credential_bound": True,
            "billing_enforcement": "local_inference", "account_fingerprint": FP,
            "endpoint": endpoint,
        },
        "quotas": [{"pool": POOL, "max_requests": 1000, "max_tokens": 10**9,
                    "period": "day"}],
    }


def _make(tmp_path, endpoint, monkeypatch):
    from geosgan_hive.agent_config import CloudConfig
    reg = CloudProfileRegistry.from_config(CloudConfig(
        zero_cost=True, max_spend_usd=0.0, trusted_evidence_binding=True,
        profiles=[CloudProfile.model_validate(_profile(endpoint))]))
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0, utcnow=lambda: NOW)
    gate = ZeroCostGate(reg, store, adapter=LocalOllamaAdapter(),
                        utcnow=lambda: NOW, default_max_output_tokens=512)
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"OLLAMA_API_BASE": endpoint},
                      cloud_adapter=LocalOllamaAdapter())
    gw.set_cloud_registry(reg)
    gw.set_cloud_gate(gate)
    gw.set_durable_store(store)
    # cost lookup stays offline; the completion call itself is REAL
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    return gw, gate, store


def test_real_loopback_transport_end_to_end(tmp_path, local_server, monkeypatch, no_network):
    endpoint, handler = local_server
    gw, gate, store = _make(tmp_path, endpoint, monkeypatch)
    answer = gw.complete([Message(role="user", content="hi")], model=MODEL,
                         agent="supervisor", max_tokens=64)
    assert answer == "hello from real local transport"
    assert handler.seen and handler.seen[0]["path"] == "/api/generate"
    assert handler.seen[0]["host"].startswith("127.0.0.1:")  # loopback only
    status = gate.engine.quota_status(MODEL, now=NOW)
    assert status[("requests", "day")]["used"] == 1  # durable reservation+dispatch
    store.close()


def test_redirect_to_external_host_is_never_followed(tmp_path, local_server, monkeypatch,
                                                      no_network):
    endpoint, handler = local_server
    handler.mode = "redirect"
    gw, _gate, store = _make(tmp_path, endpoint, monkeypatch)
    external = []
    real_connect = socket.socket.connect

    def oracle(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost"):
            external.append(address)
            raise OSError("external connect blocked by test oracle")
        return real_connect(self, address, *a, **k)

    def oracle_gai(host, *a, **k):
        # IP literals resolve locally without any DNS query; names are blocked
        import ipaddress
        if isinstance(host, str):
            try:
                ipaddress.ip_address(host)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, 0))]
            except ValueError:
                pass
        raise AssertionError("DNS resolution attempted during offline test")

    monkeypatch.setattr(socket, "getaddrinfo", oracle_gai)
    monkeypatch.setattr(socket.socket, "connect", oracle)
    from geosgan_hive.errors import HiveError
    with pytest.raises(HiveError):
        gw.complete([Message(role="user", content="hi")], model=MODEL,
                    agent="supervisor", max_tokens=64)
    # the gateway must have disabled redirect-following on the REAL LiteLLM
    # transport (production control, not a spy): 3xx surfaces as an error
    import litellm
    assert litellm.module_level_client._client.follow_redirects is False
    assert litellm.module_level_aclient._client.follow_redirects is False
    # the 302 to http://192.0.2.1:9/evil must NOT have been followed
    assert external == [], f"redirect followed to external addresses: {external}"
    store.close()


def test_endpoint_host_resolving_to_external_ip_fails_closed(tmp_path, local_server,
                                                              monkeypatch):
    endpoint, _handler = local_server
    from geosgan_hive.agent_config import CloudConfig
    reg = CloudProfileRegistry.from_config(CloudConfig(
        zero_cost=True, max_spend_usd=0.0, trusted_evidence_binding=True,
        profiles=[CloudProfile.model_validate(_profile(endpoint))]))
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0, utcnow=lambda: NOW)
    gate = ZeroCostGate(reg, store, adapter=LocalOllamaAdapter(), utcnow=lambda: NOW)

    def evil_dns(host, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", evil_dns)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32, "api_base": endpoint}
    with pytest.raises(CloudConfigError, match="non-loopback|resolves"):
        gate.prepare_dispatch(MODEL, kwargs)
    store.close()
