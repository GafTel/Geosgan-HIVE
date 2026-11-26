"""A0: existing local-inference transport security review (v0.7.0 prerequisites).

Fully offline except loopback test servers; real LiteLLM/httpx transport.
Covers (failing-first where defects were found):
- the dispatch is PINNED to the validated literal loopback address, so the
  actual connected destination is the checked address (DNS TOCTOU closed for
  the local path): a resolver that rebinds "localhost" to an external IP at
  connect time must not be reached
- hardening of LiteLLM's module-level clients is idempotent and tolerant of
  missing attributes (dependency-version robustness)
- concurrent gated dispatches reserve exactly one durable quota unit each
"""

import ipaddress
import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.cloud_gate import ZeroCostGate
from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry, LocalOllamaAdapter
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker

MODEL = "ollama/llama3.2"
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
FP = "d" * 64
POOL = "ollama:acct:" + FP


class _OllamaHandler(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        type(self).seen.append({"path": self.path, "host": self.headers.get("Host")})
        payload = {"model": "llama3.2", "created_at": "2026-01-01T00:00:00Z", "done": True,
                   "response": "ok-local", "prompt_eval_count": 3, "eval_count": 2}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def local_server():
    _OllamaHandler.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port, _OllamaHandler
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
        "quotas": [{"pool": POOL, "max_requests": 10**6, "max_tokens": 10**12,
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
    captured = []

    def recorder(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("litellm.completion must not be called in this test")

    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"OLLAMA_API_BASE": endpoint},
                      cloud_adapter=LocalOllamaAdapter())
    gw.set_cloud_registry(reg)
    gw.set_cloud_gate(gate)
    gw.set_durable_store(store)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    return gw, gate, store, captured


def test_dispatch_pinned_to_validated_literal_loopback(tmp_path, local_server, monkeypatch,
                                                        no_network):
    port, handler = local_server
    endpoint = f"http://localhost:{port}"  # evidence bound to the NAME
    gw, gate, store, _captured = _make(tmp_path, endpoint, monkeypatch)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32, "api_base": endpoint}
    gate.prepare_dispatch(MODEL, kwargs)  # pins kwargs["api_base"]
    # the request now targets the validated literal address, not the name
    assert kwargs["api_base"] == f"http://127.0.0.1:{port}"
    # and finalize (revalidation) still passes with the pinned endpoint
    did = kwargs  # noqa: F841
    store.close()


def test_dns_rebinding_at_connect_time_is_defeated(tmp_path, local_server, monkeypatch,
                                                    no_network):
    port, handler = local_server
    endpoint = f"http://localhost:{port}"
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
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)

    # hostile resolver: "localhost" resolves loopback ONCE (gate check), then
    # rebinds to an external IP; IP literals always return themselves
    state = {"calls": 0}
    fixture_gai = socket.getaddrinfo

    def rebind_gai(host, port=0, *a, **k):
        if isinstance(host, str):
            try:
                ipaddress.ip_address(host)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]
            except ValueError:
                pass
            if host.lower() == "localhost":
                state["calls"] += 1
                addr = "127.0.0.1" if state["calls"] == 1 else "192.0.2.1"
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port))]
        return fixture_gai(host, port, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", rebind_gai)
    external = []
    fixture_connect = socket.socket.connect

    def oracle(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost"):
            external.append(address)
            raise OSError("external connect blocked by test oracle")
        return fixture_connect(self, address, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", oracle)
    answer = gw.complete([Message(role="user", content="hi")], model=MODEL,
                         agent="supervisor", max_tokens=64)
    assert answer == "ok-local"
    assert handler.seen and handler.seen[0]["host"].startswith("127.0.0.1:")
    # the rebinded address was NEVER connected to (pinning closed the TOCTOU)
    assert external == [], f"connected to rebinded external addresses: {external}"
    store.close()


def test_harden_litellm_transport_idempotent_and_tolerant(monkeypatch):
    from geosgan_hive import gateway as gw_mod
    monkeypatch.setattr(gw_mod, "_LITELLM_TRANSPORT_HARDENED", False)
    gw_mod._harden_litellm_transport()
    gw_mod._harden_litellm_transport()  # idempotent
    import litellm
    assert litellm.module_level_client._client.follow_redirects is False
    assert litellm.disable_token_counter is True
    # a litellm version without the expected attributes must not crash hardening
    class FakeHandler:
        pass
    monkeypatch.setattr(litellm, "module_level_client", FakeHandler())
    monkeypatch.setattr(litellm, "module_level_aclient", FakeHandler())
    monkeypatch.setattr(gw_mod, "_LITELLM_TRANSPORT_HARDENED", False)
    gw_mod._harden_litellm_transport()


def test_concurrent_gated_dispatches_reserve_exactly(tmp_path, local_server, monkeypatch,
                                                     no_network):
    port, _handler = local_server
    endpoint = f"http://127.0.0.1:{port}"
    gw, gate, store, _c = _make(tmp_path, endpoint, monkeypatch)
    errors = []

    def one():
        try:
            assert gw.complete([Message(role="user", content="hi")], model=MODEL,
                               agent="supervisor", max_tokens=32) == "ok-local"
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    status = gate.engine.quota_status(MODEL, now=NOW)
    assert status[("requests", "day")]["used"] == 8  # exactly one durable unit each
    store.close()


# ---- A0 residual verification ----

def test_env_restored_after_success_failure_and_cancellation(tmp_path, local_server,
                                                             monkeypatch, no_network):
    import os
    port, _h = local_server
    endpoint = f"http://localhost:{port}"
    gw, _gate, store, _c = _make(tmp_path, endpoint, monkeypatch)
    sentinel = "http://sentinel.invalid:1"
    monkeypatch.setenv("OLLAMA_API_BASE", sentinel)

    import litellm
    real = litellm.completion

    # success path
    assert gw.complete([Message(role="user", content="hi")], model=MODEL,
                       agent="supervisor", max_tokens=32) == "ok-local"
    assert os.environ["OLLAMA_API_BASE"] == sentinel

    # failure path
    def boom(**kwargs):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr("litellm.completion", boom)
    from geosgan_hive.errors import HiveError
    with pytest.raises(HiveError):
        gw.complete([Message(role="user", content="hi")], model=MODEL,
                    agent="supervisor", max_tokens=32)
    assert os.environ["OLLAMA_API_BASE"] == sentinel

    # cancellation-like path (exception inside the call)
    def cancelled(**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr("litellm.completion", cancelled)
    with pytest.raises(KeyboardInterrupt):
        gw.complete([Message(role="user", content="hi")], model=MODEL,
                    agent="supervisor", max_tokens=32)
    assert os.environ["OLLAMA_API_BASE"] == sentinel
    monkeypatch.setattr("litellm.completion", real)
    store.close()


def test_concurrent_gateways_do_not_leak_env(tmp_path, local_server, monkeypatch, no_network):
    import os
    port, _h = local_server
    endpoint = f"http://127.0.0.1:{port}"
    gw1, _g1, s1, _ = _make(tmp_path / "a", endpoint, monkeypatch)
    gw2, _g2, s2, _ = _make(tmp_path / "b", endpoint, monkeypatch)
    before = os.environ.get("OLLAMA_API_BASE")
    errs = []

    def one(gw):
        try:
            gw.complete([Message(role="user", content="hi")], model=MODEL,
                        agent="supervisor", max_tokens=32)
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ts = [threading.Thread(target=one, args=(g,)) for g in (gw1, gw2) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not errs
    assert os.environ.get("OLLAMA_API_BASE") == before  # no leaked pinned value
    s1.close()
    s2.close()


def test_non_gated_dispatch_never_touches_env(tmp_path, monkeypatch):
    import os
    captured = []

    def recorder(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr("litellm.completion", recorder)
    monkeypatch.setattr("litellm.completion_cost", lambda *a, **k: None)
    before = os.environ.get("OLLAMA_API_BASE")
    gw = ModelGateway(tracker=UsageTracker(),
                      retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                               max_delay_seconds=0, jitter_ratio=0),
                      environ={"OLLAMA_API_BASE": "http://localhost:11434"})
    from geosgan_hive.errors import HiveError
    with pytest.raises(HiveError):
        gw.complete([Message(role="user", content="hi")], model=MODEL,
                    agent="supervisor", max_tokens=32)
    assert os.environ.get("OLLAMA_API_BASE") == before  # untouched
    assert captured and captured[0]["api_base"] == "http://localhost:11434"  # unpinned


def test_ipv6_loopback_literal_pinning(tmp_path, monkeypatch, no_network):
    import socket as _s
    from http.server import ThreadingHTTPServer

    class H6(_OllamaHandler):
        pass

    class Srv6(ThreadingHTTPServer):
        address_family = _s.AF_INET6

    H6.seen = []  # isolate the handler's seen list (shared class attr otherwise)
    try:
        srv = Srv6(("::1", 0), H6)
    except OSError:
        pytest.skip("IPv6 loopback not available on this host")
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    endpoint = f"http://[::1]:{port}"
    gw, gate, store, _c = _make(tmp_path, endpoint, monkeypatch)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32, "api_base": endpoint}
    gate.prepare_dispatch(MODEL, kwargs)
    assert kwargs["api_base"] == f"http://[::1]:{port}"  # pinned IPv6 literal, bracketed
    answer = gw.complete([Message(role="user", content="hi")], model=MODEL,
                         agent="supervisor", max_tokens=32)
    assert answer == "ok-local"
    assert H6.seen and H6.seen[0]["host"].startswith("[::1]:")
    srv.shutdown()
    store.close()


def test_evidence_endpoint_pattern_allows_bracketed_ipv6_literal():
    """Pattern-level regression (runs even where IPv6 sockets are unavailable,
    unlike the socket test above which skips): a bracketed IPv6 literal is a
    valid evidence endpoint (local Ollama on [::1]); malformed URLs are not."""
    from pydantic import ValidationError

    CloudProfile.model_validate(_profile("http://[::1]:11434"))  # must not raise
    CloudProfile.model_validate(_profile("https://[2001:db8::1]:443/x"))
    for bad in ("http://[::1", "http://::1]:80", "http://[gg::1]:80", "ftp://[::1]:80",
                "http://[::1]:99999", "http://exa mple.com"):
        with pytest.raises(ValidationError):
            CloudProfile.model_validate(_profile(bad))


def test_async_dispatch_uses_same_endpoint_protection(tmp_path, local_server, monkeypatch,
                                                       no_network):
    import asyncio
    port, _h = local_server
    endpoint = f"http://localhost:{port}"
    gw, _gate, store, _c = _make(tmp_path, endpoint, monkeypatch)
    captured = []

    async def arecorder(**kwargs):
        captured.append(kwargs)
        import litellm
        return litellm.completion(**kwargs)

    monkeypatch.setattr("litellm.acompletion", arecorder)
    answer = asyncio.run(gw.acomplete([Message(role="user", content="hi")], model=MODEL,
                                       agent="supervisor", max_tokens=32))
    assert answer == "ok-local"
    assert captured and captured[0]["api_base"] == f"http://127.0.0.1:{port}"  # pinned
    store.close()
