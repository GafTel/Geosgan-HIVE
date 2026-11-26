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

import http.client
import io
import ipaddress
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
)

_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
_PARAM_KEY = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
_PATH_SAFE = re.compile(r"^[a-zA-Z0-9_./-]*$")


class HttpLimits(BaseModel):
    """Enforceable per-request limits. A read timeout alone is NOT a total
    deadline: total_deadline_seconds bounds connect+send+all reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_request_bytes: int = Field(default=65_536, ge=1)
    max_response_header_bytes: int = Field(default=65_536, ge=1)
    max_wire_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_decompressed_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=10.0, gt=0)
    total_deadline_seconds: float = Field(default=30.0, gt=0)
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

    def default_port(self) -> int:
        return self.port or (443 if self.scheme == "https" else 80)


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
        self._sock.settimeout(timeout)
        try:
            n = self._sock.recv_into(view)
        except TimeoutError as exc:
            if self._clock() >= self._deadline:
                raise TransportDeadlineExceededError(
                    "total request deadline exceeded") from exc
            raise TransportDeadlineExceededError(
                f"read timeout within total deadline: {exc}") from exc
        self._state["wire"] += n
        if self._state["wire"] > self._limits.max_wire_bytes:
            raise ResponseLimitExceededError("wire response byte limit exceeded")
        return n


class _LimitedSocket:
    """Socket wrapper whose makefile() returns a BOUNDED reader, so http.client's
    header parsing (unbounded readline internally) is also capped."""

    def __init__(self, sock: socket.socket, raw: _LimitedRaw):
        self._sock = sock
        self._raw = raw

    def makefile(self, mode: str = "rb", *a: Any, **k: Any):
        assert "r" in mode
        return io.BufferedReader(self._raw)

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
        if not any(path.startswith(p) for p in rule.path_prefixes):
            raise EndpointNotAllowedError(
                f"path {path!r} is not under an allowed prefix of "
                f"{rule.endpoint_id!r}: {rule.path_prefixes}")
        if ".." in path or not _PATH_SAFE.match(path):
            raise EndpointNotAllowedError(f"unsafe path {path!r}")
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

    # -- destination resolution + pinning --------------------------------------

    def _resolve_pinned(self, rule: EndpointRule) -> ipaddress._BaseAddress:
        port = rule.default_port()
        try:
            infos = self._resolver(rule.host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
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
        pinned = self._resolve_pinned(rule)
        limits = self._config.limits
        deadline = self._clock() + limits.total_deadline_seconds
        state = {"wire": 0}
        self._sem.acquire()
        conn: http.client.HTTPConnection | None = None
        try:
            if cancel is not None and cancel.is_set():
                raise TransportCancelledError("cancelled before connect")
            port = rule.default_port()
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TransportDeadlineExceededError("deadline expired before connect")
            timeout = min(limits.connect_timeout_seconds, remaining)
            if rule.scheme == "https":
                ctx = self._ssl_context or ssl.create_default_context()
                conn = _PinnedHTTPSConnection(rule.host, port, timeout=timeout, context=ctx)
            else:
                if not self._config.allow_loopback:
                    raise EndpointNotAllowedError("plain http requires allow_loopback (test-only)")
                conn = _PinnedHTTPConnection(rule.host, port, timeout=timeout)
            conn.pinned_ip = str(pinned)
            conn.connect()
            # verify the ACTUAL connected peer equals the validated pinned address
            peer = conn.sock.getpeername()[0]
            if _unbracket(peer) != str(pinned):
                raise DestinationRejectedError(
                    f"connected peer {peer} != validated destination {pinned}")
            # bounded reads from here on (headers AND body)
            raw = _LimitedRaw(conn.sock, limits=limits, deadline=deadline,
                              clock=self._clock, cancel=cancel, state=state)
            conn.sock = _LimitedSocket(conn.sock, raw)  # type: ignore[assignment]
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
            # cancelled request never frees concurrency while still active
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
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
