"""M2 (v0.9.0): secure MCP client/session manager (EXPERIMENTAL infrastructure).

Scope guards for this milestone:
- Real external (streamable_http) MCP dispatch stays BLOCKED:
  `REMOTE_DISPATCH_ENABLED = False`; the manager refuses remote sessions unless
  the caller explicitly passes allow_remote_dispatch=True (offline loopback
  test fixtures only, mirroring the transport's allow_loopback pattern).
- No MCP tools are registered on the production agent catalog (tool_factory is
  untouched); sessions are opened programmatically in tests/tools-to-come.
- Negotiated protocol revision is pinned to 2025-06-18: the initialize request
  is sent MANUALLY with that version (the SDK's LATEST_PROTOCOL_VERSION constant
  is NOT used — in mcp 1.30 it is 2025-11-25), and any other server-negotiated
  version is rejected.
- The optional `mcp` SDK extra is lazy-loaded; core imports never require it.
- Every remote physical request (initialize, discovery, notifications, tool
  calls, SSE, session DELETE) goes through HiveMcpHttpTransport -> ApiSession ->
  SecureHttpClient: endpoint authorization, durable reservations, revalidation,
  single-use approvals, lease-fenced dispatch marks, pinning/TLS/no-redirect,
  caps, deadlines, audit. Cumulative per-session caps are enforced on top.
- All server-provided content (descriptions, annotations, results) is treated
  as UNTRUSTED: bounded, secret-redacted, never interpreted; server-initiated
  sampling/elicitation/roots stay at the SDK's default "not supported" answers.
- Capability changes (tools/list_changed or drift) invalidate bindings: the
  next call re-binds and fails closed on any mismatch.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .errors import HiveError
from .mcp_config import McpServerConfig
from .mcp_contracts import DispatchClass, McpUnavailableError, ToolCapability
from .mcp_registry import McpRegistry
from .mcp_transport import (
    HiveMcpHttpTransport,
    McpSessionError,
    McpSessionState,
    hive_httpx_client,
)
from .tool_gateway import redact_secrets
from .tool_spec import args_sha256

#: the only negotiated protocol revision v0.9.0 supports (enforced, not advertised)
MCP_PROTOCOL_REVISION = "2025-06-18"
#: M2 milestone kill switch: real external MCP dispatch stays blocked
REMOTE_DISPATCH_ENABLED = False
#: pinned optional-extra range (1.x security-maintenance line; see pyproject)
SDK_MIN_VERSION = (1, 30)
SDK_MAX_VERSION = (2, 0)


class McpClientError(HiveError):
    """MCP client failure — fail closed."""


class McpProtocolError(McpClientError):
    """Protocol negotiation/capability integrity failure — fail closed."""


class McpDispatchError(McpClientError):
    """Authorization/lease/cost dispatch denial — fail closed."""


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("geosgan-hive")
    except Exception:
        return "0.8.0"


def check_sdk() -> Any:
    """Lazy-import the optional SDK and enforce the pinned range. The SDK's own
    LATEST_PROTOCOL_VERSION constant is never used for negotiation."""
    try:
        mcp = importlib.import_module("mcp")
        from importlib.metadata import version

        v = version("mcp")
    except Exception as exc:
        raise McpUnavailableError(
            "the optional 'mcp' SDK is not installed or not importable "
            "(pip install -e '.[mcp]'); MCP stays disabled") from exc
    parts = tuple(int(p) for p in v.split(".")[:3] if p.isdigit()) or (0,)
    parts = parts + (0,) * (3 - len(parts))
    if not (SDK_MIN_VERSION <= parts < SDK_MAX_VERSION):
        raise McpUnavailableError(
            f"mcp SDK version {v} is outside the pinned supported range "
            f"{SDK_MIN_VERSION}..<{SDK_MAX_VERSION} — MCP stays disabled")
    return mcp


@dataclass
class McpSession:
    """One open MCP session with its bound capabilities."""

    server_id: str
    server: McpServerConfig
    state: McpSessionState
    session: Any                    # SDK ClientSession
    bindings: dict[str, ToolCapability]
    protocol_revision: str
    stale: bool = False             # tools/list_changed seen -> re-bind required
    _cm: Any = None                 # transport CM (streamable_http_client / stdio_transport)
    _session_cm: Any = None         # SDK ClientSession CM (starts the receive loop)
    _client: Any = None             # httpx client (remote)
    _cancel: threading.Event = field(default_factory=threading.Event)

    async def close(self) -> None:
        """Order matters: stop the SDK session (receive loop) first, then the
        transport CM (sends the session DELETE), then the httpx client. All in
        the task that opened the session (anyio cancel-scope rule). The whole
        cleanup runs SHIELDED: after a transport failure the SDK cancels the
        enclosing task-group scope, and an unshielded await here would re-raise
        CancelledError and leak the session instead of closing it."""
        import anyio

        self._cancel.set()
        with anyio.CancelScope(shield=True):
            try:
                if self._session_cm is not None:
                    await self._session_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            try:
                if self._cm is not None:
                    await self._cm.__aexit__(None, None, None)
            except BaseException:
                pass
            if self._client is not None:
                try:
                    await self._client.aclose()
                except BaseException:
                    pass


class McpSessionManager:
    """Opens and supervises MCP sessions against the M1 registry.

    `lease_lookup(run_id) -> row with lease_owner/lease_generation` revalidates
    lease fencing at dispatch time (stale workers are rejected).
    `provider_for_endpoint(endpoint_id) -> provider name` maps a remote MCP
    endpoint to its enabled API provider for ApiSession authorization.
    """

    def __init__(self, registry: McpRegistry, *, api_session: Any = None,
                 provider_for_endpoint: Callable[[str], str] | None = None,
                 lease_lookup: Callable[[str], Any] | None = None,
                 agent: str = "mcp-client",
                 allow_remote_dispatch: bool = REMOTE_DISPATCH_ENABLED,
                 environ: dict[str, str] | None = None,
                 work_root: str | None = None,
                 platform: str | None = None,
                 op_timeout_seconds: float = 60.0) -> None:
        self._registry = registry
        self._api_session = api_session
        self._provider_for_endpoint = provider_for_endpoint
        self._lease_lookup = lease_lookup
        self._agent = agent
        self._allow_remote_dispatch = allow_remote_dispatch
        self._environ = environ
        self._work_root = work_root
        self._platform = platform
        self._op_timeout = op_timeout_seconds

    # ---- dispatch authorization (pure rules + lease revalidation) ----

    def _check_lease(self, run_id: str | None, lease_owner: str | None,
                     lease_generation: int | None) -> None:
        if run_id is None or lease_owner is None or lease_generation is None:
            raise McpDispatchError("MCP dispatch requires a run-bound lease identity")
        if self._lease_lookup is None:
            raise McpDispatchError("no lease lookup wired — fail closed")
        row = self._lease_lookup(run_id)
        owner = getattr(row, "lease_owner", None) if row is not None else None
        gen = getattr(row, "lease_generation", None) if row is not None else None
        if owner != lease_owner or gen != lease_generation:
            raise McpDispatchError(
                "stale worker: lease fencing failed at dispatch (owner/generation "
                "mismatch) — fail closed")

    def _tool_decision(self, server_id: str, tool_name: str):
        d = self._registry.dispatch_decision(server_id, tool_name)
        if d.decision is DispatchClass.DENIED:
            raise McpDispatchError(f"denied: {d.reason}")
        if d.decision is DispatchClass.ALLOWED_WITH_APPROVAL:
            # M2: human approval is not wired to MCP dispatch yet — never auto-approve
            raise McpDispatchError(
                "mutation requires human approval, which is not auto-granted — "
                "blocked in M2")
        return d

    # ---- session lifecycle ----

    async def open(self, server_id: str, *, run_id: str | None = None,
                   task_id: str | None = None, lease_owner: str | None = None,
                   lease_generation: int | None = None) -> McpSession:
        check_sdk()
        if not self._registry.enabled:
            raise McpDispatchError("MCP is globally disabled")
        entry = self._registry.server(server_id)
        server = entry.server
        if not server.enabled:
            raise McpDispatchError(f"server {server_id!r} is disabled")
        self._check_lease(run_id, lease_owner, lease_generation)

        if server.transport.kind == "streamable_http":
            sess = await self._open_remote(server_id, entry, server, run_id=run_id,
                                           task_id=task_id, lease_owner=lease_owner,
                                           lease_generation=lease_generation)
        else:
            sess = await self._open_stdio(server_id, entry, server, run_id=run_id,
                                          lease_owner=lease_owner,
                                          lease_generation=lease_generation)
        try:
            # capability discovery -> operator-bound capabilities (fail closed)
            tools_result = await self._sdk_call(sess, sess.session.list_tools())
            reported = [t.model_dump(by_alias=True, mode="json", exclude_none=True)
                        for t in tools_result.tools]
            bindings = self._registry.bind_capabilities(server_id, reported)
            sess.bindings = {b.tool_name: b for b in bindings}
            sess.protocol_revision = MCP_PROTOCOL_REVISION
            return sess
        except BaseException:
            await sess.close()  # never leak a half-open session — fail closed
            raise

    async def _sdk_call(self, sess: McpSession, coro):
        """Await an SDK call. The SDK runs each POST in a task-group child, so a
        transport failure surfaces to the awaiter as a cancellation; translate
        it back to the HIVE error the transport recorded (state.last_error).
        A genuine operator cancellation (cancel event set) always wins."""
        import anyio

        cancel_exc = anyio.get_cancelled_exc_class()
        try:
            return await coro
        except cancel_exc:
            if sess._cancel.is_set():
                raise McpSessionError("session cancelled — no further requests") from None
            err = sess.state.last_error
            if isinstance(err, HiveError):
                raise err from None
            raise McpClientError(
                "MCP session transport failed (task group cancelled) — fail closed") from None
        except HiveError:
            raise
        except Exception as exc:
            raise McpClientError(f"MCP session error: {type(exc).__name__}") from exc

    async def _open_remote(self, server_id: str, entry: Any, server: McpServerConfig, *,
                           run_id: str | None, task_id: str | None,
                           lease_owner: str | None,
                           lease_generation: int | None) -> McpSession:
        if not self._allow_remote_dispatch:
            raise McpDispatchError(
                "real external MCP dispatch is BLOCKED in M2 "
                f"(REMOTE_DISPATCH_ENABLED={REMOTE_DISPATCH_ENABLED}); only "
                "explicitly authorized offline fixtures may pass "
                "allow_remote_dispatch=True")
        if self._api_session is None or self._provider_for_endpoint is None:
            raise McpDispatchError("remote MCP requires an ApiSession + provider map")
        t = server.transport
        provider = self._provider_for_endpoint(t.endpoint_id)
        rule = self._api_session.client.rule_for(t.endpoint_id)
        if rule is None:
            raise McpDispatchError(f"endpoint {t.endpoint_id!r} is not allowlisted")
        port = rule.default_port()
        url = f"{rule.scheme}://{rule.host}:{port}{t.url_path}"
        state = McpSessionState(
            server_id=server_id, identity_digest=entry.identity_digest,
            endpoint_id=t.endpoint_id, run_id=run_id,
            max_requests=server.max_session_requests,
            max_wire_bytes=server.max_session_wire_bytes,
            deadline_monotonic=time.monotonic() + server.session_deadline_seconds)
        cancel = threading.Event()
        max_req = self._api_session.client.config.limits.max_request_bytes
        transport = HiveMcpHttpTransport(
            api_session=self._api_session, provider=provider,
            endpoint_id=t.endpoint_id, operation_id="mcp", agent=self._agent,
            run_id=run_id, task_id=task_id, lease_owner=lease_owner,
            lease_generation=lease_generation, cancel=cancel, state=state,
            max_request_bytes=max_req)
        # SDK >= 1.30 API: streamable_http_client(url, http_client=<pre-built
        # client>). The deprecated streamablehttp_client(httpx_client_factory=...)
        # is NOT used (its timeout/sse_read_timeout params are ignored in 1.30).
        # A provided http_client is NOT closed by the SDK -> McpSession.close()
        # owns aclose(). No SDK default transport is ever constructed.
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        client = hive_httpx_client(transport, timeout_seconds=self._op_timeout)
        cm = streamable_http_client(url, http_client=client, terminate_on_close=True)
        sess = McpSession(server_id=server_id, server=server, state=state,
                          session=None, bindings={}, protocol_revision="",
                          _cm=cm, _client=client, _cancel=cancel)
        try:
            read_stream, write_stream, get_sid = await cm.__aenter__()
            # the SDK ClientSession starts its receive loop only when used as a
            # context manager (mcp >= 1.30) — bare construction would hang
            session_cm = ClientSession(
                read_stream, write_stream,
                read_timeout_seconds=timedelta(seconds=self._op_timeout),
                message_handler=self._make_handler(sess))
            session = await session_cm.__aenter__()
            sess._session_cm = session_cm
            sess.session = session
            await self._sdk_call(sess, self._initialize(session, get_sid, state))
            return sess
        except BaseException:
            await sess.close()
            raise

    async def _open_stdio(self, server_id: str, entry: Any, server: McpServerConfig, *,
                          run_id: str | None, lease_owner: str | None,
                          lease_generation: int | None) -> McpSession:
        from mcp.client.session import ClientSession

        from .mcp_stdio import stdio_transport

        state = McpSessionState(
            server_id=server_id, identity_digest=entry.identity_digest,
            endpoint_id="stdio", run_id=run_id,
            max_requests=server.max_session_requests,
            max_wire_bytes=server.max_session_wire_bytes,
            deadline_monotonic=time.monotonic() + server.session_deadline_seconds)
        cancel = threading.Event()
        cm = stdio_transport(server, work_root=self._work_root,
                             timeout_s=server.session_deadline_seconds,
                             environ=self._environ, platform=self._platform)
        sess = McpSession(server_id=server_id, server=server, state=state,
                          session=None, bindings={}, protocol_revision="",
                          _cm=cm, _cancel=cancel)
        try:
            read_stream, write_stream, _get_sid = await cm.__aenter__()
            session_cm = ClientSession(
                read_stream, write_stream,
                read_timeout_seconds=timedelta(seconds=self._op_timeout),
                message_handler=self._make_handler(sess))
            session = await session_cm.__aenter__()
            sess._session_cm = session_cm
            sess.session = session
            await self._sdk_call(sess, self._initialize(session, None, state))
            return sess
        except BaseException:
            await sess.close()
            raise

    def _make_handler(self, sess: McpSession):
        import mcp.types as types

        async def handler(message: Any) -> None:
            # server-initiated requests keep the SDK default "not supported"
            # answers (sampling/elicitation/roots never advertised); capability
            # change notifications invalidate the bindings (re-bind, fail closed)
            if isinstance(message, types.ToolListChangedNotification):
                sess.stale = True

        return handler

    async def _initialize(self, session: Any, get_sid: Any, state: McpSessionState) -> None:
        import mcp.types as types

        init = types.ClientRequest(types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion=MCP_PROTOCOL_REVISION,  # pinned — never the SDK constant
                capabilities=types.ClientCapabilities(),  # no sampling/elicitation/roots
                clientInfo=types.Implementation(name="geosgan-hive",
                                               version=_pkg_version()))))
        result = await session.send_request(init, types.InitializeResult)
        negotiated = str(result.protocolVersion)
        if negotiated != MCP_PROTOCOL_REVISION:
            raise McpProtocolError(
                f"server negotiated unsupported protocol revision {negotiated!r}; "
                f"only {MCP_PROTOCOL_REVISION!r} is supported — rejected")
        await session.send_notification(types.ClientNotification(
            types.InitializedNotification(method="notifications/initialized")))
        if get_sid is not None:
            sid = get_sid()
            if sid:
                state.bind_session_id(sid)  # stores sha256 only

    # ---- tool dispatch ----

    async def call_tool(self, sess: McpSession, tool_name: str,
                        args: dict[str, Any], *, run_id: str | None = None,
                        lease_owner: str | None = None,
                        lease_generation: int | None = None) -> dict[str, Any]:
        from .mcp_args import validate_tool_args

        binding = sess.bindings.get(tool_name)
        if binding is None:
            raise McpDispatchError(f"unknown tool {tool_name!r} — not bound (fail closed)")
        # revalidate authorization + capability identity + lease BEFORE dispatch
        self._check_lease(run_id or sess.state.run_id, lease_owner, lease_generation)
        self._tool_decision(sess.server_id, tool_name)
        if sess.stale:
            tools_result = await self._sdk_call(sess, sess.session.list_tools())
            reported = [t.model_dump(by_alias=True, mode="json", exclude_none=True)
                        for t in tools_result.tools]
            bindings = self._registry.bind_capabilities(sess.server_id, reported)
            sess.bindings = {b.tool_name: b for b in bindings}
            sess.stale = False
            binding = sess.bindings.get(tool_name)
            if binding is None:
                raise McpProtocolError("capability changed: tool no longer present")
        # actual arguments validated against the operator-authorized schema
        validated = validate_tool_args(binding.input_schema, args)
        result = await self._sdk_call(sess, sess.session.call_tool(tool_name, validated))
        # result content is UNTRUSTED: bound + redact + never interpreted
        text_parts: list[str] = []
        total = 0
        for block in result.content:
            txt = getattr(block, "text", None)
            if isinstance(txt, str):
                text_parts.append(txt)
                total += len(txt)
                if total > binding.max_output_bytes:
                    raise McpClientError(
                        f"tool result exceeds max_output_bytes={binding.max_output_bytes} "
                        "— fail closed")
        payload = redact_secrets("\n".join(text_parts))
        return {
            "ok": not result.isError,
            "isError": result.isError,
            "content_text": payload[: binding.max_output_bytes],
            "structured_sha256": (args_sha256(result.structuredContent)
                                  if result.structuredContent is not None else None),
            "binding_digest": binding.binding_digest(),
            "note": "untrusted server content — provenance only, never factual "
                    "verification; not interpreted",
        }
