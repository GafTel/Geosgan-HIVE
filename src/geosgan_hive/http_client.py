"""A5: single shared secure HTTP transport for public API adapters (v0.7.0).

Stdlib only (http.client + ssl + socket). Properties:

- Explicit scheme/host/port/path/method/header allowlists per endpoint id.
- HTTPS-only for production public endpoints; plain HTTP loopback is possible
  ONLY via `TransportConfig(allow_loopback=True)`, which the production config
  loader (agent_config) can never produce — loopback endpoints are rejected
  there. Test fixtures construct the config directly.
- DNS rebinding / TOCTOU resistance: the host is resolved once through an
  injectable resolver, EVERY address is validated (private/loopback [unless
  explicitly allowed]/link-local/multicast/reserved/non-global rejected,
  IPv4-mapped IPv6 unmapped and re-validated), and the TCP connection is made
  DIRECTLY to the validated numeric address while TLS SNI + hostname
  verification still use the original host. After connect, the actual peer
  (`getpeername`) must equal the pinned address. No second uncontrolled
  resolution happens (connecting to a numeric literal performs no DNS query).
- Strict certificate verification and original-hostname checks via
  ssl.create_default_context() (a custom context is a TEST-ONLY constructor
  injection; production never passes one).
- Redirects disabled (3xx + Location -> error). http.client never reads proxy
  environment variables, so there is no proxy inheritance by construction and
  no proxy configuration exists. No connection pooling (one connection per
  request, always closed).
- No model-controlled arbitrary URLs/methods/headers: the request line is
  built only from the allowlisted endpoint + validated path prefix + strictly
  validated query params; headers come from the rule plus an allowlisted extra
  set; credentials (if any) are attached only to the exact endpoint that owns
  them and never forwarded elsewhere.
- Enforceable limits: request bytes, response header bytes, wire bytes,
  decompressed bytes (gzip/deflate only, capped; other encodings rejected),
  connect/read timeouts, a TOTAL deadline (a read timeout alone is not a
  deadline — every recv checks the total deadline, so slow streams are
  bounded), max concurrent requests (semaphore released only AFTER the socket
  is closed, so cancellation never frees capacity while a request is active),
  and caller-side pagination/retry caps.
"""

from __future__ import annotations

import concurrent.futures as _futures
import http.client
import io
import ipaddress
import queue
import re
import socket
import ssl
import threading
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import (
    DestinationRejectedError,
    EndpointNotAllowedError,
    RedirectRejectedError,
    RequestLimitExceededError,
    ResponseLimitExceededError,
    TransportCancelledError,
    TransportDeadlineExceededError,
    TransportError,
)

_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
_PARAM_KEY = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
_PATH_SAFE = re.compile(r"^[a-zA-Z0-9_./-]*$")


class HttpLimits(BaseModel):
    """Enforceable per-request limits. A read timeout alone is NOT a total
    deadline. Coverage of total_deadline_seconds: it starts BEFORE DNS
    resolution and bounds TCP connect + TLS handshake + request transmission
    + response reception (every send/recv checks the remaining time).
    DNS resolution is bounded separately by dns_timeout_seconds (a safe total
    DNS deadline cannot be guaranteed because the OS resolver call is not
    interruptible; resolution runs on an abandoned daemon thread on timeout)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_request_bytes: int = Field(default=65_536, ge=1)
    max_response_header_bytes: int = Field(default=65_536, ge=1)
    max_wire_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_decompressed_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=10.0, gt=0)
    total_deadline_seconds: float = Field(default=30.0, gt=0)
    dns_timeout_seconds: float = Field(default=5.0, gt=0)
    max_dns_workers: int = Field(default=4, ge=1, le=16)
    max_dns_pending: int = Field(default=16, ge=0, le=256)
    max_concurrent_requests: int = Field(default=4, ge=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_pages: int = Field(default=10, ge=1, le=1000)


class EndpointRule(BaseModel):
    """One allowlisted destination. `auth_env` names the ONLY env var whose
    value may be sent to this endpoint (as Authorization: Bearer), and only
    here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    scheme: str = Field(pattern=r"^(https|http)$")
    host: str = Field(min_length=1, max_length=253)
    port: int | None = Field(default=None, ge=1, le=65535)
    path_prefixes: tuple[str, ...] = Field(min_length=1)
    methods: tuple[str, ...] = Field(min_length=1)
    headers: Mapping[str, str] = Field(default_factory=dict)
    extra_header_keys: tuple[str, ...] = Field(default_factory=tuple)
    auth_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")

    @field_validator("methods")
    @classmethod
    def _methods_upper(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        out = tuple(m.upper() for m in v)
        for m in out:
            if m not in ("GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"):
                raise ValueError(f"method {m!r} not allowed")
        return out

    @field_validator("path_prefixes")
    @classmethod
    def _prefixes_safe(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for p in v:
            if (not p.startswith("/") or not _PATH_SAFE.match(p) or "//" in p
                    or ".." in p.split("/") or "." in p.split("/")):
                raise ValueError(f"unsafe path prefix {p!r}")
        return v

    def default_port(self) -> int:
        return self.port or (443 if self.scheme == "https" else 80)


def _normalize_prefix(prefix: str) -> str:
    return prefix.rstrip("/") or "/"


def _path_allowed(path: str, prefixes: tuple[str, ...]) -> bool:
    """Segment-aware prefix match: '/w/api' authorizes '/w/api' and
    '/w/api/...' ONLY — a lookalike such as '/w/apix' is rejected."""
    for p in prefixes:
        pp = _normalize_prefix(p)
        if pp == "/":
            if path.startswith("/"):
                return True
        elif path == pp or path.startswith(pp + "/"):
            return True
    return False


def _is_loopback_host(host: str) -> bool:
    return host.lower() in _LOOPBACK_NAMES or (
        _looks_like_ip(host) and ipaddress.ip_address(_unbracket(host)).is_loopback)


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(_unbracket(host))
        return True
    except ValueError:
        return False


def _unbracket(host: str) -> str:
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


class TransportConfig(BaseModel):
    """Transport configuration. allow_loopback=True exists ONLY for offline test
    fixtures; the production config loader rejects loopback endpoints, and
    destination validation ALWAYS runs (the flag never bypasses validation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoints: tuple[EndpointRule, ...] = Field(min_length=1)
    limits: HttpLimits = Field(default_factory=HttpLimits)
    allow_loopback: bool = False

    @model_validator(mode="after")
    def _no_loopback_in_production(self) -> TransportConfig:
        if not self.allow_loopback:
            for ep in self.endpoints:
                if ep.scheme != "https":
                    raise ValueError(
                        f"endpoint {ep.endpoint_id!r}: production public APIs require "
                        "https (plain http is test-loopback only)"
                    )
                if _is_loopback_host(ep.host) or (
                        _looks_like_ip(ep.host)
                        and not ipaddress.ip_address(_unbracket(ep.host)).is_global):
                    raise ValueError(
                        f"endpoint {ep.endpoint_id!r}: loopback/private literal hosts are "
                        "rejected in production transport config"
                    )
        ids = [e.endpoint_id for e in self.endpoints]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate endpoint ids: {sorted(ids)}")
        return self


@dataclass
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes
    wire_bytes: int
    duration_ms: int
    peer: str
    endpoint_id: str


def validate_destination(addr: str, *, allow_loopback: bool) -> ipaddress._BaseAddress:
    """Validate ONE resolved address. Returns the normalized ipaddress object.
    IPv4-mapped IPv6 is unmapped and re-validated as IPv4."""
    ip = ipaddress.ip_address(_unbracket(addr))
    if ip.version == 6 and getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped  # never allow mapped forms to smuggle an IPv4 check
    if allow_loopback and ip.is_loopback:
        return ip
    if (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified or not ip.is_global):
        raise DestinationRejectedError(
            f"destination address {addr} is not an authorized global address "
            "(private/loopback/link-local/multicast/reserved/non-global rejected)"
        )
    return ip


class _LimitedRaw(io.RawIOBase):
    """Raw reader that enforces the total deadline, per-read timeout, wire byte
    cap and cancellation on EVERY recv — so slow streams and oversized headers
    cannot bypass the limits."""

    def __init__(self, sock: socket.socket, *, limits: HttpLimits, deadline: float,
                 clock: Callable[[], float], cancel: threading.Event | None,
                 state: dict):
        self._sock = sock
        self._limits = limits
        self._deadline = deadline
        self._clock = clock
        self._cancel = cancel
        self._state = state

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        if self._cancel is not None and self._cancel.is_set():
            raise TransportCancelledError("request cancelled")
        now = self._clock()
        if now >= self._deadline:
            raise TransportDeadlineExceededError(
                "total request deadline exceeded (slow stream or stall)")
        remaining_wire = self._limits.max_wire_bytes - self._state["wire"]
        if remaining_wire <= 0:
            raise ResponseLimitExceededError("wire response byte limit exceeded")
        view: memoryview = memoryview(b)
        if len(view) > remaining_wire:
            view = view[:remaining_wire]
        timeout = min(self._limits.read_timeout_seconds, max(0.05, self._deadline - now))
        try:
            self._sock.settimeout(timeout)
            n = self._sock.recv_into(view)
        except TimeoutError as exc:
            if self._clock() >= self._deadline:
                raise TransportDeadlineExceededError(
                    "total request deadline exceeded") from exc
            raise TransportDeadlineExceededError(
                f"read timeout within total deadline: {exc}") from exc
        except OSError as exc:
            # e.g. the peer vanished mid-response; surface a typed transport error
            raise TransportError(f"connection error during bounded read: {exc}") from exc
        self._state["wire"] += n
        if self._state["wire"] > self._limits.max_wire_bytes:
            raise ResponseLimitExceededError("wire response byte limit exceeded")
        return n


class _HeaderCapRaw(io.RawIOBase):
    """Counts response bytes until the end of the header block (the first blank
    line, i.e. \\r\\n\\r\\n or \\n\\n, possibly straddling reads) and raises
    ResponseLimitExceededError AS SOON AS max_response_header_bytes is exceeded
    DURING reception — before http.client's header parser can buffer an
    oversized header block. After the terminator it passes through untouched
    (the wire-byte cap in _LimitedRaw still bounds the body)."""

    def __init__(self, raw: _LimitedRaw, *, max_header_bytes: int):
        self._raw = raw
        self._cap = max_header_bytes
        self._seen = 0
        self._tail = b""
        self._done = False

    def readable(self) -> bool:
        return True

    def readinto(self, b):
        n = self._raw.readinto(b)
        if not n or self._done:
            return n
        buf = self._tail + bytes(memoryview(b)[:n])
        end = -1
        for term in (b"\r\n\r\n", b"\n\n"):
            i = buf.find(term)
            if i != -1 and (end == -1 or i < end):
                end = i + len(term)
        if end != -1:
            self._seen += end
            self._done = True
            self._tail = b""
        else:
            # keep a short tail so a terminator split across reads is detected
            self._tail = buf[-3:] if len(buf) >= 3 else buf
            self._seen += len(buf) - len(self._tail)
        if self._seen > self._cap:
            raise ResponseLimitExceededError(
                f"response headers exceeded {self._cap} bytes DURING reception "
                f"(seen {self._seen}); aborting before buffering more")
        return n


class _LimitedFile:
    """BufferedReader wrapper that releases the socket's io-reference on close,
    mirroring socket.makefile() semantics (see _LimitedSocket.makefile)."""

    def __init__(self, br: io.BufferedReader, sock: socket.socket):
        self._br = br
        self._sock = sock

    def close(self):
        try:
            self._br.close()
        finally:
            sock = self._sock
            if hasattr(sock, "_io_refs"):
                sock._io_refs -= 1
                if getattr(sock, "_closed", False) and sock._io_refs <= 0:
                    try:
                        sock._real_close()
                    except OSError:
                        pass

    def __getattr__(self, name: str):
        return getattr(self._br, name)


class _LimitedSocket:
    """Socket wrapper whose makefile() returns a BOUNDED reader, so http.client's
    header parsing (unbounded readline internally) is also capped.

    makefile() also takes an io-reference on the underlying socket exactly like
    socket.makefile() does: http.client's getresponse() calls sock.close() when
    the response says the connection will close, but the body still has to be
    read from the response's file object afterwards. Without the io-ref the fd
    would be released too early and resp.read() would fail with EBADF.

    The reader is additionally wrapped in _HeaderCapRaw so the response-header
    byte limit is enforced WHILE the headers are being received, not only after
    they were parsed. sendall() is overridden so request transmission respects
    the remaining total deadline and cancellation."""

    def __init__(self, sock: socket.socket, raw: _LimitedRaw, *,
                 max_header_bytes: int | None = None,
                 limits: HttpLimits | None = None,
                 deadline: float | None = None,
                 clock: Callable[[], float] | None = None,
                 cancel: threading.Event | None = None):
        self._sock = sock
        self._raw = raw
        self._max_header_bytes = max_header_bytes
        self._limits = limits
        self._deadline = deadline
        self._clock = clock or time.monotonic
        self._cancel = cancel

    def makefile(self, mode: str = "rb", *a: Any, **k: Any):
        assert "r" in mode
        sock = self._sock
        reader: io.RawIOBase = self._raw
        if self._max_header_bytes is not None:
            reader = _HeaderCapRaw(self._raw, max_header_bytes=self._max_header_bytes)
        br = io.BufferedReader(reader)
        if hasattr(sock, "_io_refs"):  # CPython socket io-ref semantics
            sock._io_refs += 1
            return _LimitedFile(br, sock)
        return br

    def sendall(self, data, flags: int = 0):
        """Request transmission is covered by the total deadline: before every
        send, check cancellation and the remaining time, and cap the socket
        timeout at the remaining deadline."""
        if self._cancel is not None and self._cancel.is_set():
            raise TransportCancelledError("request cancelled during send")
        if self._deadline is not None:
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                raise TransportDeadlineExceededError(
                    "total request deadline exceeded during request transmission")
            timeout = remaining
            if self._limits is not None:
                timeout = min(self._limits.read_timeout_seconds, remaining)
            self._sock.settimeout(timeout)
        try:
            return self._sock.sendall(data, flags)
        except TimeoutError as exc:
            raise TransportDeadlineExceededError(
                f"send timeout within total deadline: {exc}") from exc
        except OSError as exc:
            raise TransportError(f"connection error during send: {exc}") from exc

    def __getattr__(self, name: str):
        return getattr(self._sock, name)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects DIRECTLY to the validated numeric address (no second DNS
    resolution) while TLS SNI + hostname verification use the original host."""

    pinned_ip: str = ""

    def _create_connection(self, address, timeout, source_address):
        return socket.create_connection(
            (self.pinned_ip, self.port), timeout=timeout, source_address=source_address)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    pinned_ip: str = ""

    def _create_connection(self, address, timeout, source_address):
        return socket.create_connection(
            (self.pinned_ip, self.port), timeout=timeout, source_address=source_address)


class _ResolverPool:
    """A small, FIXED, bounded DNS resolver facility (not one thread per
    request). Exactly `max_workers` daemon workers consume a bounded queue of
    at most `max_pending` items:

    - submit() uses put_nowait: when the queue is full the request fails CLOSED
      immediately and consumes NO worker or queue slot.
    - a worker holds its slot for the whole execution of the resolver call, so
      a request that times out while waiting NEVER frees a DNS worker slot
      while getaddrinfo is still running.
    - the OS getaddrinfo call cannot be forcibly interrupted: on a wait timeout
      the worker thread is abandoned (it finishes whenever the OS resolver
      returns) — it is NOT terminated, and this is documented, not claimed
      otherwise.
    """

    def __init__(self, *, max_workers: int, max_pending: int) -> None:
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._busy = 0
        self._workers = [
            threading.Thread(target=self._run, daemon=True, name=f"hive-dns-{i}")
            for i in range(max_workers)
        ]
        for w in self._workers:
            w.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:  # shutdown sentinel
                    return
                fn, args, fut = item
                with self._lock:
                    self._busy += 1
                try:
                    fut.set_result(fn(*args))
                except BaseException as exc:  # noqa: BLE001 - delivered via future
                    fut.set_exception(exc)
                finally:
                    with self._lock:
                        self._busy -= 1
            finally:
                self._q.task_done()

    def submit(self, fn: Callable, *args: Any) -> _futures.Future:
        """Enqueue resolver work. Raises TransportError (fail closed) when the
        bounded pending queue is full — no slot is consumed in that case."""
        fut: _futures.Future = _futures.Future()
        try:
            self._q.put_nowait((fn, args, fut))
        except queue.Full as exc:
            raise TransportError(
                f"DNS resolver capacity exhausted ({self._max_workers} workers busy, "
                f"{self._max_pending} pending) — request rejected (fail closed)") from exc
        return fut

    def stats(self) -> dict[str, int]:
        """Observability for tests/audit: outstanding executing workers and
        queued items (never above the configured maxima)."""
        with self._lock:
            busy = self._busy
        return {"workers": self._max_workers, "busy": busy, "queued": self._q.qsize(),
                "max_pending": self._max_pending}

    def close(self) -> None:
        for _ in self._workers:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass


class SecureHttpClient:
    """The single shared secure HTTP transport. One connection per request,
    always closed; no pooling, no redirects, no env proxies, no arbitrary
    URLs/methods/headers."""

    def __init__(
        self,
        config: TransportConfig,
        *,
        resolver: Callable | None = None,       # TEST-ONLY injection (default: real DNS)
        ssl_context: ssl.SSLContext | None = None,  # TEST-ONLY injection
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._resolver = resolver or socket.getaddrinfo
        self._ssl_context = ssl_context  # None -> strict default context in production
        self._environ = environ
        self._clock = clock
        self._rules = {e.endpoint_id: e for e in config.endpoints}
        self._sem = threading.Semaphore(config.limits.max_concurrent_requests)
        self._resolver_pool = _ResolverPool(
            max_workers=config.limits.max_dns_workers,
            max_pending=config.limits.max_dns_pending)

    @property
    def config(self) -> TransportConfig:
        return self._config

    @property
    def free_slots(self) -> int:
        """Currently free concurrency slots (observability; never negative)."""
        return max(0, self._sem._value)

    @property
    def dns_stats(self) -> dict[str, int]:
        return self._resolver_pool.stats()

    # -- allowlist validation -------------------------------------------------

    def _rule(self, endpoint_id: str) -> EndpointRule:
        rule = self._rules.get(endpoint_id)
        if rule is None:
            raise EndpointNotAllowedError(f"endpoint {endpoint_id!r} is not allowlisted")
        return rule

    def _validate_request_line(self, rule: EndpointRule, method: str, path: str,
                               params: Mapping[str, Any] | None) -> str:
        if method.upper() not in rule.methods:
            raise EndpointNotAllowedError(
                f"method {method!r} not allowed on endpoint {rule.endpoint_id!r}")
        if not _PATH_SAFE.match(path) or not path.startswith("/"):
            raise EndpointNotAllowedError(f"unsafe path {path!r}")
        if any(seg in ("..", ".") for seg in path.split("/")):
            raise EndpointNotAllowedError(f"path traversal segment in {path!r}")
        if not _path_allowed(path, rule.path_prefixes):
            raise EndpointNotAllowedError(
                f"path {path!r} is not under an allowed prefix of "
                f"{rule.endpoint_id!r}: {rule.path_prefixes}")
        query = ""
        if params:
            parts = []
            for k in sorted(params):
                if not _PARAM_KEY.match(k):
                    raise EndpointNotAllowedError(f"unsafe query parameter name {k!r}")
                v = params[k]
                if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                    raise EndpointNotAllowedError(f"unsafe query value for {k!r}")
                vs = str(v)
                if any(c in vs for c in "\r\n\x00"):
                    raise EndpointNotAllowedError(f"control characters in {k!r}")
                parts.append(f"{k}={quote(vs, safe='')}")
            query = "?" + "&".join(parts)
        return path + query

    def _build_headers(self, rule: EndpointRule, extra: Mapping[str, str] | None,
                       body: bytes | None) -> dict[str, str]:
        headers: dict[str, str] = {k: str(v) for k, v in rule.headers.items()}
        for k, v in (extra or {}).items():
            if k.lower() not in {h.lower() for h in rule.extra_header_keys}:
                raise EndpointNotAllowedError(
                    f"header {k!r} is not allowlisted for endpoint {rule.endpoint_id!r}")
            if any(c in str(v) for c in "\r\n\x00"):
                raise EndpointNotAllowedError(f"control characters in header {k!r}")
            headers[k] = str(v)
        if rule.auth_env:
            import os
            env = self._environ if self._environ is not None else os.environ
            token = env.get(rule.auth_env)
            if token:
                # credential attached ONLY to the exact endpoint that owns it
                headers["Authorization"] = f"Bearer {token}"
        if body is not None and "Content-Length" not in {k.title() for k in headers}:
            headers["Content-Length"] = str(len(body))
        return headers

    def rule_for(self, endpoint_id: str) -> EndpointRule | None:
        """Public accessor for the allowlisted rule of an endpoint id (used to
        bind authorization config to transport config at construction)."""
        return self._rules.get(endpoint_id)

    # -- destination resolution + pinning --------------------------------------

    def _resolve_pinned(self, rule: EndpointRule, *, deadline: float) -> ipaddress._BaseAddress:
        """Resolve the allowlisted host through the bounded _ResolverPool and
        wait at most min(dns_timeout, remaining total deadline). A safe TOTAL
        DNS deadline cannot be guaranteed (the OS getaddrinfo call is not
        interruptible): on wait-timeout the pool worker keeps running until the
        OS resolver returns — it is abandoned, NOT terminated — and the request
        fails closed with TransportDeadlineExceededError. The worker slot stays
        occupied for the whole execution, so timed-out requests never free DNS
        capacity early. Queue exhaustion fails closed without consuming a slot."""
        port = rule.default_port()
        remaining = deadline - self._clock()
        dns_wait = min(self._config.limits.dns_timeout_seconds, max(0.0, remaining))
        if dns_wait <= 0:
            raise TransportDeadlineExceededError(
                "total request deadline expired before DNS resolution")
        fut = self._resolver_pool.submit(
            self._resolver, rule.host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        try:
            infos = fut.result(timeout=dns_wait)
        except TimeoutError as exc:
            if not fut.done():
                raise TransportDeadlineExceededError(
                    f"DNS resolution of {rule.host!r} exceeded the bounded DNS timeout "
                    f"({dns_wait:.3f}s); pool worker abandoned (not terminated) — "
                    "the OS resolver call cannot be forcibly interrupted") from exc
            raise DestinationRejectedError(
                f"cannot resolve allowlisted host {rule.host!r}: {exc}") from exc
        except OSError as exc:
            raise DestinationRejectedError(
                f"cannot resolve allowlisted host {rule.host!r}: {exc}") from exc
        valid = []
        errors = []
        for info in infos:
            addr = info[4][0]
            try:
                valid.append(validate_destination(addr, allow_loopback=self._config.allow_loopback))
            except DestinationRejectedError as exc:
                errors.append(str(exc))
        if not valid:
            raise DestinationRejectedError(
                f"host {rule.host!r} resolved only to unauthorized addresses: {errors}")
        return valid[0]

    # -- the request ------------------------------------------------------------

    def request(
        self,
        endpoint_id: str,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        cancel: threading.Event | None = None,
    ) -> HttpResult:
        rule = self._rule(endpoint_id)
        url = self._validate_request_line(rule, method, path, params)
        hdrs = self._build_headers(rule, headers, body)
        req_bytes = (len(method) + len(url)
                     + sum(len(k) + len(v) + 4 for k, v in hdrs.items())
                     + (len(body) if body else 0))
        if req_bytes > self._config.limits.max_request_bytes:
            raise RequestLimitExceededError(
                f"request size {req_bytes} exceeds limit "
                f"{self._config.limits.max_request_bytes}")
        limits = self._config.limits
        # the TOTAL deadline starts BEFORE DNS and covers connect(+TLS handshake)
        # + request transmission + response reception; DNS has its own bounded
        # timeout (dns_timeout_seconds) because the OS resolver is not
        # interruptible.
        deadline = self._clock() + limits.total_deadline_seconds
        state = {"wire": 0}
        # Deadline-aware, cancellation-aware concurrency-slot acquisition: poll
        # in short slices so neither the total deadline nor a cancel event can
        # be missed while waiting. If acquisition fails, NO DNS resolution or
        # network operation is started and NO slot is released later.
        acquired = False
        while True:
            if cancel is not None and cancel.is_set():
                raise TransportCancelledError(
                    "cancelled while waiting for a concurrency slot")
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TransportDeadlineExceededError(
                    "total request deadline expired while waiting for a "
                    "concurrency slot")
            if self._sem.acquire(timeout=min(0.05, remaining)):
                acquired = True
                break
        conn: http.client.HTTPConnection | None = None
        try:
            if cancel is not None and cancel.is_set():
                raise TransportCancelledError("cancelled before connect")
            pinned = self._resolve_pinned(rule, deadline=deadline)
            port = rule.default_port()
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TransportDeadlineExceededError("deadline expired before connect")
            # connect timeout bounds TCP connect AND the TLS handshake (the
            # wrap_socket call happens inside HTTPSConnection.connect)
            timeout = min(limits.connect_timeout_seconds, remaining)
            if rule.scheme == "https":
                ctx = self._ssl_context or ssl.create_default_context()
                conn = _PinnedHTTPSConnection(rule.host, port, timeout=timeout, context=ctx)
            else:
                if not self._config.allow_loopback:
                    raise EndpointNotAllowedError("plain http requires allow_loopback (test-only)")
                conn = _PinnedHTTPConnection(rule.host, port, timeout=timeout)
            conn.pinned_ip = str(pinned)
            try:
                conn.connect()
            except ssl.SSLError:
                raise  # TLS verification/handshake failures keep their precise type
            except TimeoutError as exc:
                raise TransportDeadlineExceededError(
                    f"connect/TLS handshake exceeded the remaining deadline: {exc}") from exc
            except OSError as exc:
                raise TransportError(f"connect failed: {exc}") from exc
            # verify the ACTUAL connected peer equals the validated pinned address
            peer = conn.sock.getpeername()[0]
            if _unbracket(peer) != str(pinned):
                raise DestinationRejectedError(
                    f"connected peer {peer} != validated destination {pinned}")
            # bounded reads from here on (headers AND body); the header byte cap
            # is enforced DURING header reception via _HeaderCapRaw
            raw = _LimitedRaw(conn.sock, limits=limits, deadline=deadline,
                              clock=self._clock, cancel=cancel, state=state)
            conn.sock = _LimitedSocket(  # type: ignore[assignment]
                conn.sock, raw, max_header_bytes=limits.max_response_header_bytes,
                limits=limits, deadline=deadline, clock=self._clock, cancel=cancel)
            conn.request(method.upper(), url, body=body, headers=hdrs)
            resp = conn.getresponse()
            header_bytes = sum(len(k) + len(v) + 4 for k, v in resp.getheaders())
            if header_bytes > limits.max_response_header_bytes:
                raise ResponseLimitExceededError(
                    f"response headers {header_bytes} exceed limit "
                    f"{limits.max_response_header_bytes}")
            status = resp.status
            if 300 <= status < 400 and resp.getheader("Location"):
                raise RedirectRejectedError(
                    f"redirects are disabled (status {status} -> "
                    f"{resp.getheader('Location')})")
            wire_before = state["wire"]
            payload = resp.read()
            wire_bytes = state["wire"] - wire_before
            enc = (resp.getheader("Content-Encoding") or "identity").lower()
            if enc in ("gzip", "x-gzip"):
                dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
            elif enc == "deflate":
                dec = zlib.decompressobj()
            elif enc == "identity":
                dec = None
            else:
                raise ResponseLimitExceededError(f"unsupported content encoding {enc!r}")
            if dec is None:
                out = payload
            else:
                out = dec.decompress(payload, limits.max_decompressed_bytes + 1)
                if dec.unconsumed_tail:
                    raise ResponseLimitExceededError("decompressed response too large")
            if len(out) > limits.max_decompressed_bytes:
                raise ResponseLimitExceededError("decompressed response byte limit exceeded")
            return HttpResult(
                status=status,
                headers={k: v for k, v in resp.getheaders()},
                body=out,
                wire_bytes=wire_bytes,
                duration_ms=int(
                    (self._clock() - (deadline - limits.total_deadline_seconds)) * 1000),
                peer=str(pinned),
                endpoint_id=endpoint_id,
            )
        finally:
            # capacity is released ONLY after the socket is fully closed, so a
            # cancelled request never frees concurrency while still active —
            # and ONLY if this request actually acquired a slot
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if acquired:
                self._sem.release()


def parse_retry_after(value: str | None, *, cap_seconds: float = 86400.0) -> float | None:
    """Safe Retry-After parsing: malformed/nonfinite/negative -> None; capped."""
    if not value:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not v > 0 or v != v or v == float("inf"):
        return None
    return min(v, cap_seconds)
