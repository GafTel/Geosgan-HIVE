"""M2 (v0.9.0): secure Streamable-HTTP transport for MCP — an httpx transport
adapter that routes EVERY physical request through the existing approved path:

    httpx request -> HiveMcpHttpTransport -> ApiSession.request
        (outbound authorization check -> durable v9 reservation -> pre-dispatch
         revalidation -> single-use approval consumption -> lease-fenced dispatch
         mark -> bounded audit)
      -> SecureHttpClient.request (endpoint rule, pinned numeric destination,
         connected-peer verification, TLS hostname/SNI, SSRF classes, redirect
         and proxy rejection, wire/decompression caps, total deadline, cancel)

There is NO alternative unrestricted HTTP transport and NO SDK default transport
fallback: the SDK's `streamable_http_client(url, http_client=...)` (mcp >= 1.30
API; the deprecated `streamablehttp_client` factory form is NOT used) is given
a pre-built `httpx.AsyncClient` on THIS transport with `follow_redirects=False`
(the SDK default factory enables redirects — forbidden here; the SDK follows
redirects itself only within the endpoint origin, which can never trigger
because SecureHttpClient rejects every 3xx+Location before httpx sees it).

Session integrity: cumulative per-session request/byte/deadline caps are
enforced here (in addition to the durable per-request budgets); the server-
issued `mcp-session-id` is treated as sensitive — only its sha256 is recorded,
bound to (server identity, endpoint, run, authorization scope), and any
substitution on a later response is rejected. Response bodies are buffered
bounded by the transport's wire caps and served to the SDK as a capped byte
stream (SSE included — the SDK parses it from this stream; a long-lived GET
stream is cut at the transport total deadline).
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl

import httpx

from .errors import HiveError

#: header names the MCP layer may pass per-request (the endpoint rule's
#: extra_header_keys allowlist still applies inside SecureHttpClient)
MCP_EXTRA_HEADERS = ("mcp-protocol-version", "mcp-session-id")


class McpSessionError(HiveError):
    """MCP session/transport contract violation — fail closed."""


@dataclass
class McpSessionState:
    """Cumulative per-session accounting (shared with the transport)."""

    server_id: str
    identity_digest: str
    endpoint_id: str
    run_id: str | None
    max_requests: int
    max_wire_bytes: int
    deadline_monotonic: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: int = 0
    wire_bytes: int = 0
    session_id_hash: str | None = None  # sha256 only — the raw id is never stored/logged
    last_error: BaseException | None = None  # first HIVE error raised by the transport

    def record_error(self, exc: BaseException) -> None:
        """Remember the first transport error: the SDK runs each request in a
        task-group child, so the error reaches the caller as a cancellation —
        mcp_client._sdk_call re-raises this recorded error instead."""
        with self.lock:
            if self.last_error is None:
                self.last_error = exc

    def check_and_consume(self, wire_bytes: int, max_request_bytes: int) -> None:
        with self.lock:
            if time.monotonic() >= self.deadline_monotonic:
                raise McpSessionError("session wall-clock deadline exceeded — fail closed")
            if self.requests >= self.max_requests:
                raise McpSessionError(
                    f"session request budget exhausted ({self.max_requests}) — fail closed")
            if self.wire_bytes + wire_bytes > self.max_wire_bytes:
                raise McpSessionError(
                    "session cumulative wire-byte budget exhausted — fail closed")
            self.requests += 1
            self.wire_bytes += wire_bytes

    def bind_session_id(self, raw_session_id: str) -> None:
        """Record only the hash, bound to this session's identity context."""
        h = hashlib.sha256(raw_session_id.encode("utf-8")).hexdigest()
        with self.lock:
            if self.session_id_hash is None:
                self.session_id_hash = h
            elif self.session_id_hash != h:
                raise McpSessionError(
                    "mcp-session-id substitution detected (bound to server identity, "
                    "endpoint, run and authorization scope) — fail closed")


class _BodyStream(httpx.AsyncByteStream):
    """Serves the (already transport-capped) response body to the SDK."""

    def __init__(self, data: bytes, chunk: int = 65_536) -> None:
        self._data = data
        self._chunk = chunk

    async def __aiter__(self):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i:i + self._chunk]

    async def aclose(self) -> None:
        pass


class HiveMcpHttpTransport(httpx.AsyncBaseTransport):
    """httpx transport that performs every request via ApiSession (which uses
    SecureHttpClient). Sync engine calls run in a worker thread; cancellation
    propagates through the shared threading.Event."""

    def __init__(self, *, api_session: Any, provider: str, endpoint_id: str,
                 operation_id: str, agent: str, run_id: str | None,
                 task_id: str | None, lease_owner: str | None,
                 lease_generation: int | None, cancel: threading.Event,
                 state: McpSessionState, max_request_bytes: int) -> None:
        self._session = api_session
        self._provider = provider
        self._endpoint_id = endpoint_id
        self._operation_id = operation_id
        self._agent = agent
        self._run_id = run_id
        self._task_id = task_id
        self._lease_owner = lease_owner
        self._lease_generation = lease_generation
        self._cancel = cancel
        self._state = state
        self._max_request_bytes = max_request_bytes

    def _do_request(self, *, method: str, path: str, params: dict[str, str],
                    body: bytes | None, headers: dict[str, str]) -> Any:
        return self._session.request(
            provider=self._provider, endpoint_id=self._endpoint_id,
            operation_id=self._operation_id, method=method, path=path,
            params=params or None, body=body, headers=headers or None,
            agent=self._agent, run_id=self._run_id, task_id=self._task_id,
            lease_owner=self._lease_owner, lease_generation=self._lease_generation,
            cancel=self._cancel)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import anyio

        if self._cancel.is_set():
            err = McpSessionError("session cancelled — no further requests")
            self._state.record_error(err)
            raise err
        body = await request.aread()
        if len(body) > self._max_request_bytes:
            raise McpSessionError(
                f"request body {len(body)}B exceeds cap {self._max_request_bytes}B")
        params: dict[str, str] = {}
        for k, v in parse_qsl(request.url.query.decode("latin-1"),
                              keep_blank_values=True):
            if k in params:
                raise McpSessionError(f"duplicate query parameter {k!r} — rejected")
            params[k] = v
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() in MCP_EXTRA_HEADERS}
        # cumulative session caps are checked BEFORE any reservation/dispatch
        self._state.check_and_consume(wire_bytes=len(body) + 256,
                                      max_request_bytes=self._max_request_bytes)
        try:
            # run_sync takes positional args only -> wrap the kwargs in a lambda
            result = await anyio.to_thread.run_sync(lambda: self._do_request(
                method=request.method, path=request.url.path, params=params,
                body=body if body else None, headers=headers))
        except HiveError as exc:
            self._state.record_error(exc)
            raise
        except Exception as exc:
            err = McpSessionError(f"MCP transport request failed: {type(exc).__name__}")
            self._state.record_error(err)
            raise err from exc
        try:
            sid = (result.headers.get("mcp-session-id")
                   or result.headers.get("Mcp-Session-Id"))
            if sid:
                self._state.bind_session_id(sid)
            with self._state.lock:
                self._state.wire_bytes += result.wire_bytes
                over = self._state.wire_bytes > self._state.max_wire_bytes
            if over:
                raise McpSessionError(
                    "session cumulative wire-byte budget exceeded — fail closed")
        except HiveError as exc:
            self._state.record_error(exc)
            raise
        return httpx.Response(result.status, headers=dict(result.headers),
                              stream=_BodyStream(result.body), request=request)

    async def aclose(self) -> None:
        self._cancel.set()


def hive_httpx_client(transport: HiveMcpHttpTransport, *,
                      timeout_seconds: float = 60.0) -> httpx.AsyncClient:
    """Build the ONLY httpx client the MCP SDK may use (mcp >= 1.30
    `streamable_http_client(url, http_client=...)`): it runs on HIVE's secure
    transport (ApiSession -> SecureHttpClient) with redirects disabled and a
    HIVE-owned deadline. The SDK does not close a provided client, so the
    caller (McpSession.close) owns `aclose()`. No auth flow is ever attached
    (no OAuth in v0.9.0); per-request headers are allowlisted by the transport.
    """
    timeout = httpx.Timeout(timeout_seconds, read=timeout_seconds)
    return httpx.AsyncClient(transport=transport, follow_redirects=False,
                             timeout=timeout)
