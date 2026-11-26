"""A5 adversarial tests for the secure public API transport (offline, loopback
servers + controlled resolver fixtures; network spies are detection oracles
only). Covers destination pinning/DNS rebinding, address classes, TLS, allowlist
bypass attempts, limits, decompression bombs, slow-stream deadlines, cancellation
and concurrency accounting, proxy env, redirects."""

import gzip
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive.errors import (
    DestinationRejectedError,
    EndpointNotAllowedError,
    RedirectRejectedError,
    RequestLimitExceededError,
    ResponseLimitExceededError,
    TransportCancelledError,
    TransportDeadlineExceededError,
)
from geosgan_hive.http_client import (
    EndpointRule,
    HttpLimits,
    SecureHttpClient,
    TransportConfig,
    parse_retry_after,
    validate_destination,
)

TLS_DIR = Path(__file__).parent / "tls"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: no premature close before body read
    mode = "ok"
    max_concurrent_seen = 0
    _active = 0
    _lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _track(self):
        with Handler._lock:
            Handler._active += 1
            Handler.max_concurrent_seen = max(Handler.max_concurrent_seen, Handler._active)

    def _untrack(self):
        with Handler._lock:
            Handler._active -= 1

    def do_GET(self):
        self._track()
        try:
            if Handler.mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.1:9/evil")
                self.end_headers()
                return
            if Handler.mode == "bigheader":
                self.send_response(200)
                self.send_header("X-Big", "y" * 200_000)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if Handler.mode == "bigbody":
                self.send_response(200)
                self.send_header("Content-Length", str(5 * 1024 * 1024))
                self.end_headers()
                self.wfile.write(b"z" * (5 * 1024 * 1024))
                return
            if Handler.mode == "gzipbomb":
                bomb = gzip.compress(b"\x00" * (100 * 1024 * 1024))
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(bomb)))
                self.end_headers()
                self.wfile.write(bomb)
                return
            if Handler.mode == "delay":
                time.sleep(0.3)
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if Handler.mode == "slow":
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                for _ in range(1000):
                    try:
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    except OSError:
                        return
                    time.sleep(0.05)
                return
            if Handler.mode == "close_body":
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return
            if Handler.mode == "ratelimit":
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            self._untrack()

    do_POST = do_GET


@pytest.fixture
def server():
    Handler.mode = "ok"
    Handler.max_concurrent_seen = 0
    Handler._active = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    t.join(timeout=5)


@pytest.fixture
def https_server():
    Handler.mode = "ok"
    Handler.max_concurrent_seen = 0
    Handler._active = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(TLS_DIR / "server.pem", TLS_DIR / "server.key")
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    t.join(timeout=5)


def make_client(port, *, scheme="http", host="127.0.0.1", limits=None, resolver=None,
                ssl_context=None, extra_headers=None, methods=("GET", "POST"),
                allow_loopback=True):
    rule = EndpointRule(
        endpoint_id="test-api", scheme=scheme, host=host, port=port,
        path_prefixes=("/api",), methods=methods,
        headers={"User-Agent": "hive-test/1.0", **(extra_headers or {})},
        extra_header_keys=("Accept",))
    cfg = TransportConfig(endpoints=(rule,),
                          limits=limits or HttpLimits(),
                          allow_loopback=allow_loopback)
    return SecureHttpClient(cfg, resolver=resolver, ssl_context=ssl_context)


def ip_resolver(mapping):
    """Controlled resolver fixture: maps host -> list of addresses."""
    def resolve(host, port, *a, **k):
        addrs = mapping.get(host, [host])
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs]
    return resolve


# ---- allowlist ----

def test_unknown_endpoint_method_path_rejected(server):
    c = make_client(server)
    with pytest.raises(EndpointNotAllowedError):
        c.request("nope", "GET", "/api/x")
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "DELETE", "/api/x")
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/other/x")
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/api/../secret")
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/api/x", params={"bad key": "v"})
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/api/x", params={"q": "a\r\nb"})
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/api/x", headers={"X-Evil": "1"})
    with pytest.raises(EndpointNotAllowedError):
        c.request("test-api", "GET", "/api/x", headers={"Accept": "a\r\nX: y"})


def test_production_config_rejects_http_and_loopback():
    with pytest.raises(ValidationError):
        TransportConfig(endpoints=(EndpointRule(
            endpoint_id="x", scheme="http", host="127.0.0.1", port=80,
            path_prefixes=("/",), methods=("GET",)),))
    with pytest.raises(ValidationError):
        TransportConfig(endpoints=(EndpointRule(
            endpoint_id="x", scheme="https", host="localhost", port=443,
            path_prefixes=("/",), methods=("GET",)),))
    with pytest.raises(ValidationError):
        TransportConfig(endpoints=(EndpointRule(
            endpoint_id="x", scheme="https", host="10.0.0.5", port=443,
            path_prefixes=("/",), methods=("GET",)),))


def test_request_size_limit(server):
    c = make_client(server, limits=HttpLimits(max_request_bytes=128))
    with pytest.raises(RequestLimitExceededError):
        c.request("test-api", "POST", "/api/x", body=b"y" * 10_000)


def test_connection_close_response_body_still_read(server):
    """Regression (failing-first): http.client's getresponse() calls sock.close()
    when the response says `Connection: close`, and THEN reads the body from the
    response file object. The bounded makefile() must take a socket io-ref like
    socket.makefile() does, otherwise the fd is released early and resp.read()
    intermittently fails with OSError EBADF (bad file descriptor)."""
    Handler.mode = "close_body"
    c = make_client(server)
    for _ in range(25):  # the defect was racy; loop to make it deterministic-ish
        r = c.request("test-api", "GET", "/api/x")
        assert r.status == 200
        assert r.body == b'{"ok": true}'


def test_redirect_rejected(server):
    Handler.mode = "redirect"
    c = make_client(server)
    with pytest.raises(RedirectRejectedError):
        c.request("test-api", "GET", "/api/x")


def test_proxy_env_vars_are_ignored(server, monkeypatch, no_network):
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://192.0.2.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://192.0.2.1:9")
    external = []
    real_connect = socket.socket.connect

    def oracle(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost"):
            external.append(address)
            raise OSError("external connect blocked by oracle")
        return real_connect(self, address, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", oracle)
    c = make_client(server)
    r = c.request("test-api", "GET", "/api/x")
    assert r.status == 200
    assert external == []  # stdlib http.client never reads proxy env vars


# ---- destination validation / pinning ----

def test_dns_rebinding_pinned_to_validated_address(server, monkeypatch, no_network):
    state = {"n": 0}

    def rebind_resolver(host, port, *a, **k):
        if host == "localhost":
            state["n"] += 1
            addr = "127.0.0.1" if state["n"] == 1 else "192.0.2.1"  # rebind after check
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    external = []
    real_connect = socket.socket.connect

    def oracle(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost"):
            external.append(address)
            raise OSError("blocked")
        return real_connect(self, address, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", oracle)
    c = make_client(server, host="localhost", resolver=rebind_resolver)
    r = c.request("test-api", "GET", "/api/x")
    assert r.status == 200 and r.peer == "127.0.0.1"
    assert external == []  # the rebinded address was never connected to


def test_non_global_destinations_rejected():
    for bad in ("10.0.0.1", "192.168.1.1", "169.254.1.1", "224.0.0.1", "192.0.2.1",
                "0.0.0.0", "255.255.255.255", "fd00::1", "::1", "127.0.0.1",
                "::ffff:10.0.0.1"):
        with pytest.raises(DestinationRejectedError):
            validate_destination(bad, allow_loopback=False)
    assert str(validate_destination("::ffff:127.0.0.1", allow_loopback=True)) == "127.0.0.1"
    assert str(validate_destination("::ffff:8.8.8.8", allow_loopback=False)) == "8.8.8.8"
    assert str(validate_destination("8.8.8.8", allow_loopback=False)) == "8.8.8.8"
    assert str(validate_destination("127.0.0.1", allow_loopback=True)) == "127.0.0.1"


def test_resolver_returning_mixed_addresses_fails_closed(server):
    # one loopback + one external: the external one must not be usable; since a
    # valid one exists we pin to it, but if ALL are external we fail closed
    c = make_client(server, host="localhost",
                    resolver=ip_resolver({"localhost": ["192.0.2.1", "10.0.0.1"]}))
    with pytest.raises(DestinationRejectedError):
        c.request("test-api", "GET", "/api/x")
    c2 = make_client(server, host="localhost",
                     resolver=ip_resolver({"localhost": ["192.0.2.1", "127.0.0.1"]}))
    r = c2.request("test-api", "GET", "/api/x")
    assert r.peer == "127.0.0.1"


# ---- TLS ----

def test_https_verified_with_ca_and_original_hostname(https_server):
    ctx = ssl.create_default_context(cafile=str(TLS_DIR / "ca.pem"))
    c = make_client(https_server, scheme="https", host="localhost",
                    resolver=ip_resolver({"localhost": ["127.0.0.1"]}),
                    ssl_context=ctx)
    r = c.request("test-api", "GET", "/api/x")
    assert r.status == 200 and r.peer == "127.0.0.1"  # pinned IP, TLS name = localhost


def test_tls_hostname_mismatch_rejected(https_server):
    # server presents a cert for other.example while we ask for localhost
    ctx_srv = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx_srv.load_cert_chain(TLS_DIR / "wrong.pem", TLS_DIR / "wrong.key")
    # rebuild server with wrong cert
    Handler.mode = "ok"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.socket = ctx_srv.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ctx = ssl.create_default_context(cafile=str(TLS_DIR / "ca.pem"))
    c = make_client(srv.server_address[1], scheme="https", host="localhost",
                    resolver=ip_resolver({"localhost": ["127.0.0.1"]}),
                    ssl_context=ctx)
    with pytest.raises((ssl.SSLError, OSError)):
        c.request("test-api", "GET", "/api/x")
    srv.shutdown()


def test_tls_untrusted_ca_rejected(https_server):
    ctx = ssl.create_default_context()  # no test CA -> verification must fail
    c = make_client(https_server, scheme="https", host="localhost",
                    resolver=ip_resolver({"localhost": ["127.0.0.1"]}),
                    ssl_context=ctx)
    with pytest.raises((ssl.SSLError, OSError)):
        c.request("test-api", "GET", "/api/x")


# ---- limits ----

def test_oversized_response_headers_rejected(server):
    Handler.mode = "bigheader"
    c = make_client(server, limits=HttpLimits(max_response_header_bytes=1024))
    import http.client
    with pytest.raises((ResponseLimitExceededError, http.client.LineTooLong)):
        c.request("test-api", "GET", "/api/x")


def test_wire_body_limit(server):
    Handler.mode = "bigbody"
    c = make_client(server, limits=HttpLimits(max_wire_bytes=1024))
    with pytest.raises(ResponseLimitExceededError):
        c.request("test-api", "GET", "/api/x")


def test_decompression_bomb_rejected(server):
    Handler.mode = "gzipbomb"
    c = make_client(server, limits=HttpLimits(max_wire_bytes=5 * 1024 * 1024,
                                              max_decompressed_bytes=1024 * 1024))
    with pytest.raises(ResponseLimitExceededError):
        c.request("test-api", "GET", "/api/x")


def test_slow_stream_total_deadline(server):
    Handler.mode = "slow"
    c = make_client(server, limits=HttpLimits(total_deadline_seconds=1.0,
                                              read_timeout_seconds=5.0))
    t0 = time.monotonic()
    with pytest.raises(TransportDeadlineExceededError):
        c.request("test-api", "GET", "/api/x")
    assert time.monotonic() - t0 < 4  # total deadline, not just read timeout


# ---- cancellation / concurrency ----

def test_cancel_before_connect_and_capacity_release(server):
    c = make_client(server, limits=HttpLimits(max_concurrent_requests=1))
    ev = threading.Event()
    ev.set()
    with pytest.raises(TransportCancelledError):
        c.request("test-api", "GET", "/api/x", cancel=ev)
    # capacity released only after close: semaphore fully available again
    assert c._sem._value == 1


def test_concurrency_limit_enforced(server):
    Handler.mode = "delay"
    c = make_client(server, limits=HttpLimits(max_concurrent_requests=2,
                                              total_deadline_seconds=30.0))
    results = []

    def one():
        try:
            results.append(c.request("test-api", "GET", "/api/x").status)
        except Exception as exc:  # noqa: BLE001
            results.append(type(exc).__name__)

    ts = [threading.Thread(target=one) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)
    assert Handler.max_concurrent_seen <= 2
    assert c._sem._value == 2  # all capacity released after sockets closed


def test_parse_retry_after_safe():
    assert parse_retry_after(None) is None
    assert parse_retry_after("abc") is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("NaN") is None
    assert parse_retry_after("1e999") is None
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("999999999") == 86400.0  # capped
