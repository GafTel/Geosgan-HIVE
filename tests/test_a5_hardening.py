"""A5 security-review hardening tests (offline; loopback servers + controlled
fixtures only). Covers the six review items:

1. response-header byte limit enforced DURING reception (not after parsing)
2. total request deadline: bounded DNS, connect/TLS, send, reads; cancellation
3. segment-aware path-prefix matching (no /w/apix lookalikes, no traversal)
4. HIVE accounting window recalculated before EVERY durable reservation
5. authorization config bound to the exact transport endpoint at construction
6. billing classification = operator attestation, unknown fails closed
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import ApiConfig, ApiProviderConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.errors import (
    ConfigError,
    DestinationRejectedError,
    EndpointNotAllowedError,
    ResponseLimitExceededError,
    TransportDeadlineExceededError,
    TransportError,
)
from geosgan_hive.http_client import (
    EndpointRule,
    HttpLimits,
    SecureHttpClient,
    TransportConfig,
)
from geosgan_hive.outbound import ApiSession, OutboundAuthorizer

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 26, 23, 59, 50, tzinfo=UTC)


def make_client(port, *, limits=None, resolver=None, prefixes=("/api",),
                host="127.0.0.1", scheme="http", methods=("GET", "POST"),
                endpoint_id="test-api"):
    rule = EndpointRule(endpoint_id=endpoint_id, scheme=scheme, host=host, port=port,
                        path_prefixes=tuple(prefixes), methods=tuple(methods),
                        headers={"User-Agent": "hive-test/1.0"})
    cfg = TransportConfig(endpoints=(rule,), limits=limits or HttpLimits(),
                          allow_loopback=True)
    return SecureHttpClient(cfg, resolver=resolver)


def ip_resolver(mapping):
    def resolve(host, port, *a, **k):
        addrs = mapping.get(host, [host])
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs]
    return resolve


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if H.mode == "header_drip":
            # never-terminating header block, sent in small flushed chunks
            self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Big: ")
            self.wfile.flush()
            chunk = b"y" * 512
            for _ in range(4000):  # ~2MB total, far beyond any header cap
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.001)
            return
        if H.mode == "stall_body":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                self.wfile.write(b"x" * 10)
                self.wfile.flush()
                time.sleep(5)
            except OSError:
                pass
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if H.mode == "noread":
            time.sleep(5)  # never reads the request body -> client send blocks
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    H.mode = "ok"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()


# ---- 1. header limit enforced DURING reception ------------------------------

def test_header_cap_raises_mid_reception_on_never_ending_headers(server):
    """The server drip-feeds a ~2MB unterminated header block. With a 4KB header
    cap the client must abort DURING reception (ResponseLimitExceededError),
    not after buffering the whole block."""
    H.mode = "header_drip"
    c = make_client(server, limits=HttpLimits(max_response_header_bytes=4096,
                                              max_wire_bytes=10 * 1024 * 1024,
                                              total_deadline_seconds=20))
    t0 = time.monotonic()
    with pytest.raises(ResponseLimitExceededError, match="DURING reception"):
        c.request("test-api", "GET", "/api/x")
    elapsed = time.monotonic() - t0
    assert elapsed < 10  # aborted long before the server could send 2MB


def test_header_cap_counts_status_line_and_all_headers(server):
    H.mode = "ok"
    c = make_client(server, limits=HttpLimits(max_response_header_bytes=16))
    with pytest.raises(ResponseLimitExceededError):
        c.request("test-api", "GET", "/api/x")  # normal headers > 16 bytes


# ---- 2. total deadline: DNS / connect / send / reads -------------------------

def test_slow_dns_bounded_by_dns_timeout(server):
    def slow_resolver(host, port, *a, **k):
        time.sleep(5)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    c = make_client(server, limits=HttpLimits(dns_timeout_seconds=0.2,
                                              total_deadline_seconds=30),
                    resolver=slow_resolver)
    t0 = time.monotonic()
    with pytest.raises(TransportDeadlineExceededError, match="DNS"):
        c.request("test-api", "GET", "/api/x")
    assert time.monotonic() - t0 < 2.0


def test_dns_consuming_the_deadline_fails_closed_before_connect(server, monkeypatch):
    """If DNS resolution consumes the whole total deadline, the request must
    fail closed before any connection attempt."""
    clock = {"t": 1000.0}
    connected = []

    def resolver(host, port, *a, **k):
        clock["t"] += 100.0  # DNS took longer than the whole deadline
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    def record_connect(addr, timeout=None, source_address=None):
        connected.append(addr)
        raise OSError("must not be called")

    monkeypatch.setattr(socket, "create_connection", record_connect)
    c = make_client(server, limits=HttpLimits(total_deadline_seconds=1.0,
                                              dns_timeout_seconds=50),
                    resolver=resolver)
    c._clock = lambda: clock["t"]  # type: ignore[assignment]
    with pytest.raises(TransportDeadlineExceededError, match="before connect"):
        c.request("test-api", "GET", "/api/x")
    assert connected == []  # fail closed before any connect


def test_connect_timeout_wrapped_as_deadline_error(server, monkeypatch):
    def boom(addr, timeout=None, source_address=None):
        raise TimeoutError("simulated connect timeout")

    monkeypatch.setattr(socket, "create_connection", boom)
    c = make_client(server, limits=HttpLimits(connect_timeout_seconds=0.2))
    with pytest.raises(TransportDeadlineExceededError, match="connect"):
        c.request("test-api", "GET", "/api/x")


def test_connect_oserror_wrapped_as_transport_error(server, monkeypatch):
    def boom(addr, timeout=None, source_address=None):
        raise OSError("simulated network unreachable")

    monkeypatch.setattr(socket, "create_connection", boom)
    c = make_client(server)
    with pytest.raises(TransportError, match="connect failed"):
        c.request("test-api", "GET", "/api/x")


def test_blocked_transmission_hits_total_deadline(server):
    """Server never reads; an 8MB body cannot be transmitted within a 0.5s
    total deadline -> sendall must abort with the deadline error."""
    H.mode = "noread"
    c = make_client(server, limits=HttpLimits(max_request_bytes=16 * 1024 * 1024,
                                              total_deadline_seconds=0.5,
                                              read_timeout_seconds=5))
    with pytest.raises(TransportDeadlineExceededError):
        c.request("test-api", "POST", "/api/x", body=b"z" * (8 * 1024 * 1024))


class _ChunkedBody:
    """File-like body: http.client transmits it in multiple send() calls, so the
    cancel/deadline checks between chunks are actually exercised."""

    def __init__(self, total=8 * 1024 * 1024, chunk=256 * 1024):
        self._total = total
        self._left = total
        self._chunk = chunk

    def __len__(self):
        return self._total

    def read(self, size=-1):
        time.sleep(0.05)  # slow producer -> several sendall calls
        n = min(self._chunk, self._left, size if size and size > 0 else self._chunk)
        if n <= 0:
            return b""
        self._left -= n
        return b"z" * n


def test_cancel_during_send(server):
    from geosgan_hive.errors import TransportCancelledError

    H.mode = "noread"
    c = make_client(server, limits=HttpLimits(max_request_bytes=64 * 1024 * 1024,
                                              total_deadline_seconds=30))
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    with pytest.raises(TransportCancelledError, match="during send"):
        c.request("test-api", "POST", "/api/x", body=_ChunkedBody(), cancel=cancel)


def test_stalled_body_hits_total_deadline(server):
    H.mode = "stall_body"
    c = make_client(server, limits=HttpLimits(total_deadline_seconds=0.5,
                                              read_timeout_seconds=5))
    with pytest.raises(TransportDeadlineExceededError):
        c.request("test-api", "GET", "/api/x")


# ---- 3. segment-aware path prefixes ------------------------------------------

def _rule(prefixes):
    return EndpointRule(endpoint_id="e", scheme="http", host="127.0.0.1", port=80,
                        path_prefixes=tuple(prefixes), methods=("GET",),
                        headers={})


def test_path_prefix_segment_matching():
    c = make_client(80, prefixes=("/w/api",))
    rule = c.rule_for("test-api")
    ok = lambda p: c._validate_request_line(rule, "GET", p, None)  # noqa: E731
    assert ok("/w/api") == "/w/api"
    assert ok("/w/api/") == "/w/api/"
    assert ok("/w/api/search") == "/w/api/search"
    for bad in ("/w/apix", "/w/ap", "/w/api2/x", "/other", "/w/api/../secret",
                "/w/api/.", "/w/api%2f..%2fetc", "//w/api", "w/api"):
        with pytest.raises(EndpointNotAllowedError):
            ok(bad)


def test_path_prefix_trailing_slash_normalized():
    c = make_client(80, prefixes=("/w/api/",))
    rule = c.rule_for("test-api")
    assert c._validate_request_line(rule, "GET", "/w/api/x", None) == "/w/api/x"
    with pytest.raises(EndpointNotAllowedError):
        c._validate_request_line(rule, "GET", "/w/apix", None)


def test_root_prefix_allows_any_absolute_path():
    c = make_client(80, prefixes=("/",))
    rule = c.rule_for("test-api")
    assert c._validate_request_line(rule, "GET", "/anything/here", None)
    with pytest.raises(EndpointNotAllowedError):
        c._validate_request_line(rule, "GET", "/../etc/passwd", None)


def test_rule_rejects_unsafe_prefixes_at_construction():
    for bad in ("w/api", "/w/../x", "/w/api%2f", "//"):
        with pytest.raises(ValidationError):
            _rule((bad,))


# ---- 4. accounting window recalculated before every reservation ---------------

def _apis():
    return ApiConfig.model_validate({
        "enabled": ["wikimedia"], "max_wire_bytes_per_request": 4096,
        "budgets": {"per_endpoint_requests": 10, "per_endpoint_bytes": 10**9,
                    "per_provider_requests": 100, "per_provider_bytes": 10**9,
                    "per_run_requests": 50, "per_run_bytes": 10**9},
        "providers": [{"provider": "wikimedia", "billing": "public_api_no_charge",
                       "outbound_policy": "authorized", "allowed_operations": ["search"],
                       "endpoints": [{"endpoint_id": "test-ep", "scheme": "https",
                                      "host": "api.wikimedia.org",
                                      "path_prefixes": ["/api"], "methods": ["GET"]}]}],
    })


class RateLimitOnce(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        if len(type(self).hits) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_retry_crossing_midnight_charged_to_correct_windows(tmp_path):
    RateLimitOnce.hits = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RateLimitOnce)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    clock = {"now": NOW}  # 2026-11-26 23:59:50
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: clock["now"])
    apis = _apis()
    auth = OutboundAuthorizer(store, apis, ApprovalManager(store))
    rule = EndpointRule(endpoint_id="test-ep", scheme="http", host="127.0.0.1",
                        port=port, path_prefixes=("/api",), methods=("GET",),
                        headers={"User-Agent": "hive-test/1.0"})
    client = SecureHttpClient(TransportConfig(endpoints=(rule,),
                                              limits=HttpLimits(max_retries=1),
                                              allow_loopback=True))

    def sleep_advance(_seconds):
        clock["now"] += timedelta(days=1)  # retry happens after midnight

    from geosgan_hive.outbound import build_offline_test_session
    session = build_offline_test_session(client, auth, store, sleep=sleep_advance)
    r = session.request(provider="wikimedia", endpoint_id="test-ep",
                        operation_id="search", method="GET", path="/api/search",
                        params={"q": "x"}, agent="researcher")
    assert r.status == 200 and len(RateLimitOnce.hits) == 2
    rows = store.api_usage_status("provider:wikimedia")
    by_window = {}
    for row in rows:
        if row["kind"] == "requests":
            by_window[row["window_start"]] = row["used"]
    # attempt 1 charged to 2026-11-26, the retry to 2026-11-27 — each physical
    # attempt is charged to the window in effect at its reservation
    assert by_window == {"2026-11-26": 1, "2026-11-27": 1}
    srv.shutdown()
    store.close()


# ---- 5. authorization bound to the exact transport endpoint --------------------

def _matching_client(**rule_overrides):
    """A production-like client whose rule MATCHES the _apis() config on every
    field except the ones overridden (for substitution tests)."""
    apis = _apis()
    base = dict(endpoint_id="test-ep", scheme="https", host="api.wikimedia.org",
                port=443, path_prefixes=("/api",), methods=("GET",),
                headers={"User-Agent": apis.user_agent}, extra_header_keys=(),
                auth_env=None)
    base.update(rule_overrides)
    rule = EndpointRule(**base)
    return SecureHttpClient(TransportConfig(endpoints=(rule,)))


def _loopback_client(**rule_overrides):
    """A test-only loopback client (same endpoint id, different destination)."""
    base = dict(endpoint_id="test-ep", scheme="http", host="127.0.0.1", port=8080,
                path_prefixes=("/api",), methods=("GET",),
                headers={"User-Agent": _apis().user_agent})
    base.update(rule_overrides)
    rule = EndpointRule(**base)
    return SecureHttpClient(TransportConfig(endpoints=(rule,), allow_loopback=True))


def _session_for(client, store):
    auth = OutboundAuthorizer(store, _apis(), ApprovalManager(store))
    return ApiSession(client, auth, store)


def test_matching_endpoint_binding_constructs(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    sess = _session_for(_matching_client(), store)  # must NOT raise
    assert sess.client.rule_for("test-ep").host == "api.wikimedia.org"
    store.close()


def test_public_constructor_has_no_bypass_option():
    """The normal public API must not expose any option that disables endpoint
    binding verification."""
    import inspect

    sig = inspect.signature(ApiSession.__init__)
    for name in sig.parameters:
        assert "override" not in name and "bypass" not in name and "skip" not in name
    assert "test_endpoint_override" not in sig.parameters


@pytest.mark.parametrize("override", [
    {"host": "evil.example.com"},                       # destination substitution
    {"port": 8443},                                     # port substitution
    {"path_prefixes": ("/",)},                          # broader prefixes
    {"methods": ("GET", "POST", "DELETE")},             # broader methods
    {"auth_env": "OTHER_TOKEN"},                        # credential substitution
    {"extra_header_keys": ("X-Custom",)},               # extra header allowance
    {"headers": {"User-Agent": "evil/1.0"}},            # fixed header substitution
    {"headers": {"User-Agent": _apis().user_agent, "X-Extra": "1"}},  # added header
])
def test_mismatched_transport_rule_rejected_at_construction(tmp_path, override):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    client = _matching_client(**override)  # matches config except `override`
    with pytest.raises(ConfigError, match="does not match its transport rule"):
        _session_for(client, store)
    store.close()


def test_authorized_endpoint_without_transport_rule_rejected(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    rule = EndpointRule(endpoint_id="other-ep", scheme="https", host="api.wikimedia.org",
                        port=443, path_prefixes=("/api",), methods=("GET",),
                        headers={"User-Agent": _apis().user_agent})
    client = SecureHttpClient(TransportConfig(endpoints=(rule,)))
    with pytest.raises(ConfigError, match="no transport rule"):
        _session_for(client, store)
    store.close()


def test_loopback_rule_rejected_through_normal_public_api(tmp_path):
    """Even the offline loopback rule cannot be constructed through the normal
    public constructor — only via build_offline_test_session."""
    store = CheckpointStore(str(tmp_path / "cp.db"))
    with pytest.raises(ConfigError, match="does not match its transport rule"):
        _session_for(_loopback_client(), store)
    store.close()


def test_test_only_factory_allows_loopback_fixture(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    client = _loopback_client()  # loopback, same endpoint_id, different destination
    from geosgan_hive.outbound import build_offline_test_session
    sess = build_offline_test_session(
        client, OutboundAuthorizer(store, _apis(), ApprovalManager(store)), store)
    assert sess.client.rule_for("test-ep").host == "127.0.0.1"
    store.close()


# ---- 6. billing classification is an operator attestation ----------------------

def test_unknown_billing_fails_closed():
    with pytest.raises(ValidationError, match="attested"):
        ApiProviderConfig.model_validate({
            "provider": "wikimedia",
            "endpoints": [{"endpoint_id": "e", "scheme": "https", "host": "api.wikimedia.org",
                           "path_prefixes": ["/"], "methods": ["GET"]}]})
    with pytest.raises(ValidationError, match="attested"):
        ApiProviderConfig.model_validate({
            "provider": "wikimedia", "billing": "paid",
            "endpoints": [{"endpoint_id": "e", "scheme": "https", "host": "api.wikimedia.org",
                           "path_prefixes": ["/"], "methods": ["GET"]}]})


def test_billing_attestation_values_accepted_and_documented():
    ep = [{"endpoint_id": "e", "scheme": "https", "host": "api.wikimedia.org",
           "path_prefixes": ["/"], "methods": ["GET"]}]
    for b in ("public_api_no_charge", "local_inference"):
        ApiProviderConfig.model_validate({"provider": "x", "billing": b, "endpoints": ep})
    doc = ApiProviderConfig.__doc__ or ""
    assert "OPERATOR ATTESTATION" in doc and "not independently verified" in doc
    assert "guarantee zero provider-side charges" in doc.lower()


# ---- 1. deadline-aware semaphore acquisition ---------------------------------

def _shared_slot_holder(server):
    """A holder client (long deadline) that blocks inside DNS on a gate while
    holding the only concurrency slot, plus a waiter client (short deadline)
    sharing the SAME semaphore."""
    gate = threading.Event()
    resolved = []

    def resolver(host, port, *a, **k):
        resolved.append(host)
        gate.wait(10)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    holder = make_client(server, limits=HttpLimits(max_concurrent_requests=1,
                                                   total_deadline_seconds=30,
                                                   dns_timeout_seconds=30),
                         resolver=resolver)
    waiter = make_client(server, limits=HttpLimits(max_concurrent_requests=1,
                                                   total_deadline_seconds=0.3,
                                                   dns_timeout_seconds=5))
    waiter._sem = holder._sem  # test-only: share the concurrency slot
    return holder, waiter, gate, resolved


def test_slot_wait_times_out_with_deadline_and_no_dns(server):
    """All slots occupied: a waiting request must fail with the typed deadline
    error at the deadline, and must NOT start DNS resolution."""
    H.mode = "ok"
    holder, waiter, gate, resolved = _shared_slot_holder(server)

    def run_holder():
        try:
            holder.request("test-api", "GET", "/api/x")
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=run_holder)
    t.start()
    for _ in range(200):  # holder acquired the slot and entered DNS
        if resolved:
            break
        time.sleep(0.01)
    assert resolved == ["127.0.0.1"] and holder.free_slots == 0
    t0 = time.monotonic()
    with pytest.raises(TransportDeadlineExceededError, match="concurrency slot"):
        waiter.request("test-api", "GET", "/api/x")
    assert time.monotonic() - t0 < 2.0
    assert resolved == ["127.0.0.1"]  # waiter never started its own DNS
    gate.set()
    t.join(timeout=15)
    assert holder.free_slots == 1  # holder released its slot exactly once


def test_cancel_while_waiting_for_slot(server):
    from geosgan_hive.errors import TransportCancelledError

    H.mode = "ok"
    holder, waiter, gate, resolved = _shared_slot_holder(server)
    waiter_limits = waiter.config.limits
    assert waiter_limits.total_deadline_seconds == 0.3

    def run_holder():
        try:
            holder.request("test-api", "GET", "/api/x")
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=run_holder)
    t.start()
    while holder.free_slots != 0:
        time.sleep(0.01)
    # give the waiter a long deadline so only CANCELLATION can stop it
    waiter._config = waiter.config.model_copy(update={"limits": waiter_limits.model_copy(
        update={"total_deadline_seconds": 30})})
    cancel = threading.Event()
    threading.Timer(0.15, cancel.set).start()
    with pytest.raises(TransportCancelledError, match="concurrency slot"):
        waiter.request("test-api", "GET", "/api/x", cancel=cancel)
    assert resolved == ["127.0.0.1"]
    gate.set()
    t.join(timeout=15)
    assert holder.free_slots == 1  # no slot leaked by the cancelled waiter


def test_slot_released_after_success_and_failure_paths(server):
    """Correct accounting after every failure path: validation error before
    acquisition, DNS failure, deadline during request — the slot count must
    return to the maximum each time."""
    H.mode = "ok"
    c = make_client(server, limits=HttpLimits(max_concurrent_requests=2,
                                              total_deadline_seconds=5))
    cap = c.free_slots
    # 1. request-line validation fails BEFORE acquisition
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/nope")
    assert c.free_slots == cap
    # 2. unknown endpoint fails before acquisition
    with pytest.raises(EndpointNotAllowedError):
        c.request("unknown", "GET", "/api/x")
    assert c.free_slots == cap
    # 3. DNS rejects all addresses -> fails after acquisition, slot released
    def resolver(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))]

    c2 = make_client(server, limits=HttpLimits(max_concurrent_requests=2), resolver=resolver)
    with pytest.raises(DestinationRejectedError):
        c2.request("test-api", "GET", "/api/x")
    assert c2.free_slots == 2
    # 4. successful request releases its slot
    r = c.request("test-api", "GET", "/api/x")
    assert r.status == 200
    assert c.free_slots == cap
    # 5. deadline during the request releases the slot
    H.mode = "stall_body"
    c3 = make_client(server, limits=HttpLimits(max_concurrent_requests=1,
                                               total_deadline_seconds=0.3,
                                               read_timeout_seconds=5))
    with pytest.raises(TransportDeadlineExceededError):
        c3.request("test-api", "GET", "/api/x")
    assert c3.free_slots == 1
    H.mode = "ok"


# ---- 2. bounded DNS resolver pool ---------------------------------------------

def test_slow_dns_storm_never_exceeds_pool_limits(server):
    """Repeated slow-DNS requests: outstanding executing workers and queued
    items never exceed the configured maxima; excess requests fail closed;
    normal requests remain usable afterwards."""
    def slow_resolver(host, port, *a, **k):
        time.sleep(1.0)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    limits = HttpLimits(max_concurrent_requests=16, dns_timeout_seconds=0.2,
                        max_dns_workers=2, max_dns_pending=3,
                        total_deadline_seconds=30)
    c = make_client(server, limits=limits, resolver=slow_resolver)
    outcomes = []
    max_busy = [0]
    max_queued = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            st = c.dns_stats
            max_busy[0] = max(max_busy[0], st["busy"])
            max_queued[0] = max(max_queued[0], st["queued"])
            time.sleep(0.005)

    def one():
        try:
            c.request("test-api", "GET", "/api/x")
            outcomes.append("ok")
        except TransportDeadlineExceededError:
            outcomes.append("dns-timeout")
        except TransportError as exc:
            outcomes.append("capacity:" + str("capacity exhausted" in str(exc)))
        except Exception as exc:  # noqa: BLE001
            outcomes.append(type(exc).__name__)

    st = threading.Thread(target=sampler, daemon=True)
    st.start()
    threads = [threading.Thread(target=one) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    stop.set()
    st.join(timeout=2)
    assert max_busy[0] <= 2, f"busy workers exceeded limit: {max_busy[0]}"
    assert max_queued[0] <= 3, f"queued items exceeded limit: {max_queued[0]}"
    # 2 workers execute, 3 queue, 5 must fail closed at submit
    assert outcomes.count("dns-timeout") == 5
    assert outcomes.count("capacity:True") == 5
    assert "ok" not in outcomes
    # after all slow resolver work drains, the pool is idle (poll, don't guess)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        st = c.dns_stats
        if st["busy"] == 0 and st["queued"] == 0:
            break
        time.sleep(0.05)
    assert c.dns_stats["busy"] == 0 and c.dns_stats["queued"] == 0

    def fast_resolver(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    c2 = make_client(server, resolver=fast_resolver)
    r = c2.request("test-api", "GET", "/api/x")
    assert r.status == 200  # normal requests remain usable after DNS timeouts


def test_dns_pool_submit_fails_closed_when_queue_full():
    from geosgan_hive.http_client import _ResolverPool

    pool = _ResolverPool(max_workers=1, max_pending=2)
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return "done"

    f1 = pool.submit(slow)          # occupies the only worker
    # wait until the worker actually picked it up (deterministic accounting)
    for _ in range(200):
        if pool.stats()["busy"] == 1:
            break
        time.sleep(0.01)
    assert pool.stats()["busy"] == 1 and pool.stats()["queued"] == 0
    f2 = pool.submit(slow)          # queued (1/2)
    f3 = pool.submit(slow)          # queued (2/2)
    with pytest.raises(TransportError, match="capacity exhausted"):
        pool.submit(slow)           # queue full -> fail closed, no slot consumed
    assert pool.stats()["busy"] == 1 and pool.stats()["queued"] == 2
    gate.set()
    assert f1.result(timeout=3) == "done"
    assert f2.result(timeout=3) == "done"
    assert f3.result(timeout=3) == "done"
    pool.close()
