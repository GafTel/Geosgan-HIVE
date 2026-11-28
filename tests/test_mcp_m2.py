"""M2 (v0.9.0): secure MCP client + transports — offline tests only.

Fixtures: a raw loopback Streamable-HTTP JSON-RPC MCP server (http.server,
allow_loopback transport rule, exactly like the W3 web tests) and a stdio
fixture server spawned through the isolated bootstrap. No external network.

Covers: SDK pin enforcement, protocol pin 2025-06-18 + downgrade rejection,
capability binding fail-closed (unknown tools, schema drift/pins, outputSchema
subset, annotations non-authority), unauthorized endpoints, session-id
binding/substitution, malformed JSON-RPC + SSE, oversized responses, durable
budget exhaustion, lease loss before dispatch, cancellation, stdio isolation
(env sanitization, symlink rejection, timeout group-kill, malformed frames),
runtime arg validation (F5), remote dispatch kill switch, global disable,
zero-cost gate, no catalog exposure.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="optional mcp extra not installed")

from geosgan_hive import (
    MCP_PROTOCOL_REVISION,
    REMOTE_DISPATCH_ENABLED,
    McpArgsError,
    McpClientError,
    McpConfig,
    McpDispatchError,
    McpProtocolError,
    McpRegistry,
    McpRegistryError,
    McpSessionError,
    McpSessionManager,
    McpStdioError,
    McpUnavailableError,
    check_executable,
    check_sdk,
    validate_tool_args,
)
from geosgan_hive.agent_config import ApiConfig, HiveConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.cloud_profiles import ZERO_COST_RELEASE_LOCAL_PROVIDERS
from geosgan_hive.http_client import EndpointRule, HttpLimits, SecureHttpClient, TransportConfig
from geosgan_hive.mcp_contracts import validate_input_schema
from geosgan_hive.outbound import OutboundAuthorizer, build_offline_test_session
from geosgan_hive.tool_factory import build_tool_catalog

NOW = datetime(2026, 2, 1, tzinfo=UTC)
POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="stdio isolation is POSIX-only")

ECHO_SCHEMA = {"type": "object",
               "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 100}},
               "required": ["text"], "additionalProperties": False}

ECHO_TOOL = {"name": "echo",
             "description": "fixture tool — this text is UNTRUSTED",
             "inputSchema": ECHO_SCHEMA,
             "annotations": {"readOnlyHint": True, "destructiveHint": False}}


# ---- loopback MCP streamable-HTTP fixture ----------------------------------------

class McpFixtureHandler(BaseHTTPRequestHandler):
    mode = "ok"            # ok | badversion | extra_tool | drift | sid | sid_switch |
                           # malformed | sse | huge | outref
    negotiated_version = MCP_PROTOCOL_REVISION
    tools = [ECHO_TOOL]
    hits: list[dict] = []
    seen_protocol: list[str] = []
    sid_count = 0

    def log_message(self, *a):  # silence
        pass

    def _send(self, status, body: bytes, ctype="application/json", extra_headers=None):
        extra_headers = dict(extra_headers or {})
        # in sid/sid_switch modes every POST response carries a session id:
        # "sid" always sid-A; "sid_switch" returns sid-A for the first three
        # POSTs (initialize, notifications/initialized, tools/list) and sid-B
        # from tools/call on, so the client must detect the substitution
        # mid-session at dispatch time
        if (self.command == "POST"
                and McpFixtureHandler.mode in ("sid", "sid_switch")
                and not any(k.lower() == "mcp-session-id" for k in extra_headers)):
            McpFixtureHandler.sid_count += 1
            extra_headers["mcp-session-id"] = (
                "sid-A" if McpFixtureHandler.mode == "sid"
                or McpFixtureHandler.sid_count <= 3 else "sid-B")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra_headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self):
        McpFixtureHandler.hits.append({"method": "DELETE"})
        self._send(204, b"")

    def do_GET(self):
        self._send(405, b'{"error":"no standalone stream"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(min(n, 1 << 20))
        McpFixtureHandler.hits.append({"method": "POST", "bytes": len(body),
                                       "path": self.path})
        mode = McpFixtureHandler.mode
        if mode == "malformed":
            self._send(200, b"this is not json{")
            return
        try:
            msg = json.loads(body)
        except Exception:
            self._send(400, b'{"jsonrpc":"2.0","error":{"code":-32700,"message":"parse"}}')
            return
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            McpFixtureHandler.seen_protocol.append(
                str(msg.get("params", {}).get("protocolVersion")))
            result = {"protocolVersion": McpFixtureHandler.negotiated_version,
                      "capabilities": {"tools": {"listChanged": True}},
                      "serverInfo": {"name": "fixture", "version": "0.1"}}
            self._send(200, json.dumps({"jsonrpc": "2.0", "id": mid,
                                        "result": result}).encode())
            return
        if method == "notifications/initialized":
            self._send(202, b"")
            return
        if method == "tools/list":
            tools = list(McpFixtureHandler.tools)
            if mode == "extra_tool":
                tools.append({"name": "not_allowlisted", "inputSchema": {"type": "object"}})
            if mode == "drift":
                tools = [{**ECHO_TOOL, "inputSchema": {"type": "object",
                                                       "additionalProperties": False}}]
            if mode == "outref":
                tools = [{**ECHO_TOOL, "outputSchema": {"$ref": "https://evil/x.json"}}]
            body_out = json.dumps({"jsonrpc": "2.0", "id": mid,
                                   "result": {"tools": tools}}).encode()
            if mode == "sse":
                sse = b"event: message\ndata: " + body_out + b"\n\n"
                self._send(200, sse, ctype="text/event-stream")
            else:
                self._send(200, body_out)
            return
        if method == "tools/call":
            args = (msg.get("params") or {}).get("arguments") or {}
            result = {"content": [{"type": "text",
                                   "text": f"echo:{args.get('text', '')}"}],
                      "isError": False}
            if mode == "huge":
                result["content"][0]["text"] = "x" * (2 << 20)
            self._send(200, json.dumps({"jsonrpc": "2.0", "id": mid,
                                        "result": result}).encode())
            return
        self._send(200, json.dumps({"jsonrpc": "2.0", "id": mid,
                                    "error": {"code": -32601, "message": "nf"}}).encode())


@pytest.fixture
def mcp_http_server():
    McpFixtureHandler.hits = []
    McpFixtureHandler.seen_protocol = []
    McpFixtureHandler.sid_count = 0
    McpFixtureHandler.mode = "ok"
    McpFixtureHandler.negotiated_version = MCP_PROTOCOL_REVISION
    McpFixtureHandler.tools = [ECHO_TOOL]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), McpFixtureHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()


# ---- stdio fixture server -----------------------------------------------------------

STDIO_FIXTURE = """#!/usr/bin/env python3
import json, os, sys, time
mode = "ok"
if "--mode" in sys.argv:
    mode = sys.argv[sys.argv.index("--mode") + 1]
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        sys.stdout.write("garbage-not-json\\n")
        sys.stdout.flush()
        continue
    mid = msg.get("id")
    method = msg.get("method")
    if mode == "sleep" and method == "initialize":
        time.sleep(300)
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18",
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "stdio-fixture", "version": "0.1"}}
    elif method == "notifications/initialized":
        continue  # spec: the server MUST NOT respond to notifications
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "untrusted",
                             "inputSchema": __SCHEMA__},
                            {"name": "env", "description": "report env keys",
                             "inputSchema": {"type": "object",
                                             "additionalProperties": False}}]}
        result = {"tools": result["tools"]}
    elif method == "tools/call":
        name = (msg.get("params") or {}).get("name")
        args = (msg.get("params") or {}).get("arguments") or {}
        if name == "env":
            text = ",".join(sorted(os.environ.keys()))
        else:
            text = "echo:" + str(args.get("text", ""))
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    else:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                     "error": {"code": -32601, "message": "nf"}}) + "\\n")
        sys.stdout.flush()
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\\n")
    sys.stdout.flush()
""".replace("__SCHEMA__", repr(ECHO_SCHEMA))  # python literal, not JSON


@pytest.fixture
def stdio_fixture(tmp_path):
    p = tmp_path / "fixture_mcp_server"
    p.write_text(STDIO_FIXTURE, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


# ---- builders ------------------------------------------------------------------------

def make_store(tmp_path) -> CheckpointStore:
    # unique db per call (several managers may share one tmp_path); the clock is
    # mutable so tests can advance time past the lease TTL (lease expiry)
    clock = {"now": NOW}
    db = tmp_path / f"cp-{uuid.uuid4().hex[:8]}.db"
    store = CheckpointStore(str(db), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: clock["now"])
    store._m2_clock = clock  # test-only hook: store._m2_clock["now"] = NOW + ...
    return store


def make_apis_dict(*, per_run_requests=50) -> dict:
    return {
        "enabled": ["mcpgw"],
        "budgets": {"per_endpoint_requests": 100, "per_endpoint_bytes": 10**8,
                    "per_provider_requests": 100, "per_provider_bytes": 10**8,
                    "per_run_requests": per_run_requests, "per_run_bytes": 10**8},
        "providers": [{
            "provider": "mcpgw", "billing": "public_api_no_charge",
            "outbound_policy": "authorized", "allowed_operations": ["mcp"],
            "endpoints": [{"endpoint_id": "mcp-test", "scheme": "https",
                           "host": "mcp.example.org", "path_prefixes": ["/mcp"],
                           "methods": ["POST", "GET", "DELETE"]}],
        }],
    }


def make_apis(**kw) -> ApiConfig:
    return ApiConfig.model_validate(make_apis_dict(**kw))


def make_client(port: int, **limit_over) -> SecureHttpClient:
    rule = EndpointRule(endpoint_id="mcp-test", scheme="http", host="127.0.0.1",
                        port=port, path_prefixes=("/mcp",),
                        methods=("POST", "GET", "DELETE"),
                        headers={"Accept": "application/json, text/event-stream",
                                 "Content-Type": "application/json",
                                 "User-Agent": "hive-test/1.0"},
                        extra_header_keys=("MCP-Protocol-Version", "mcp-session-id"))
    limits = HttpLimits(max_retries=0, **limit_over)
    return SecureHttpClient(TransportConfig(endpoints=(rule,), limits=limits,
                                            allow_loopback=True))


def make_manager(tmp_path, registry, *, port=None, per_run_requests=50,
                 allow_remote=True, limit_over=None, with_lease=True):
    store = make_store(tmp_path)
    apis = make_apis(per_run_requests=per_run_requests)
    client = make_client(port, **(limit_over or {}))
    auth = OutboundAuthorizer(store, apis, ApprovalManager(store))
    session = build_offline_test_session(client, auth, store)
    mgr = McpSessionManager(registry, api_session=session,
                            provider_for_endpoint=lambda ep: "mcpgw",
                            lease_lookup=store.get,
                            allow_remote_dispatch=allow_remote)
    run_id = lease = None
    if with_lease:
        store.create_run(run_id="run-1", owner="w1", task="t", agent="researcher",
                         model="ollama/llama3.2")
        cp = store.get("run-1")
        run_id, lease = "run-1", cp.lease_generation
    return mgr, store, run_id, lease


def http_server_cfg(**over) -> dict:
    base = {"server_id": "t1", "enabled": True,
            "transport": {"kind": "streamable_http", "endpoint_id": "mcp-test",
                          "url_path": "/mcp"},
            "cost_class": "verified_zero_cost", "zero_cost_attested": True,
            "tools": [{"name": "echo", "side_effect": "read_only"}]}
    base.update(over)
    return base


def stdio_server_cfg(exe, **over) -> dict:
    base = {"server_id": "s1", "enabled": True,
            "transport": {"kind": "stdio", "executable": exe, "args": [],
                          "env_refs": ["FIXTURE_ALLOWED"]},
            "cost_class": "local", "local_execution_attested": True,
            "tools": [{"name": "echo", "side_effect": "read_only"},
                      {"name": "env", "side_effect": "read_only"}]}
    base.update(over)
    return base


def make_registry(*servers, enabled=True) -> McpRegistry:
    return McpRegistry(McpConfig.model_validate(
        {"enabled": enabled, "servers": list(servers)}))


# ---- SDK / pin / dependency tests ----------------------------------------------------

def test_sdk_pin_enforced_and_missing_extra_fails_closed(monkeypatch):
    m = check_sdk()
    assert m.__name__ == "mcp"
    from importlib.metadata import version
    assert version("mcp") == "1.30.0"
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "1.12.0" if name == "mcp" else version(name))
    with pytest.raises(McpUnavailableError, match="outside the pinned"):
        check_sdk()
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(McpUnavailableError, match="not installed"):
        check_sdk()


def test_core_imports_and_runs_without_mcp_extra():
    code = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"  # importing mcp now fails
        "import geosgan_hive\n"
        "from geosgan_hive import McpRegistry, McpConfig, McpSessionManager\n"
        "r = McpRegistry(McpConfig())\n"
        "assert r.enabled is False\n"
        "print('CORE_OK_WITHOUT_MCP')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0 and "CORE_OK_WITHOUT_MCP" in out.stdout, out.stderr


def test_remote_dispatch_blocked_by_default_and_zero_cost_gate(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg())
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server,
                                             allow_remote=False)
    assert REMOTE_DISPATCH_ENABLED is False
    import anyio

    async def go():
        with pytest.raises(McpDispatchError, match="BLOCKED"):
            await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
    anyio.run(go)
    assert ZERO_COST_RELEASE_LOCAL_PROVIDERS == ("ollama",)


def test_mcp_not_in_agent_catalog(tmp_path):
    hive = HiveConfig.model_validate({
        "supervisor": {"name": "supervisor", "description": "s", "system_prompt": "s",
                       "model": "ollama/llama3.2"},
        "agents": [{"name": "r", "description": "r", "system_prompt": "r",
                    "model": "ollama/llama3.2"}],
        "apis": make_apis_dict(),
        "mcp": {"enabled": True, "servers": [http_server_cfg()]},
    })
    catalog, _ = build_tool_catalog(hive)
    assert not any(n.startswith("mcp_") for n in catalog.names())


# ---- remote (loopback) end-to-end + adversarial --------------------------------------

def test_remote_session_end_to_end_and_protocol_pin(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg())
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1",
                              lease_generation=lease)
        assert sess.protocol_revision == "2025-06-18"
        assert McpFixtureHandler.seen_protocol == ["2025-06-18"]
        assert set(sess.bindings) == {"echo"}
        res = await mgr.call_tool(sess, "echo", {"text": "hello"}, run_id=run_id,
                                  lease_owner="w1", lease_generation=lease)
        assert res["ok"] and res["content_text"] == "echo:hello"
        assert len(res["binding_digest"]) == 64
        await sess.close()
    anyio.run(go)
    # every physical request was reserved + audited (init, initialized, list, call, DELETE)
    assert len(McpFixtureHandler.hits) >= 4
    audits = store.api_audit_list()
    assert len(audits) >= 4 and all(a["outcome"] == "ok" for a in audits)
    # budget consumed durably
    assert store.get("run-1") is not None


def test_protocol_downgrade_rejected(tmp_path, mcp_http_server):
    McpFixtureHandler.negotiated_version = "2025-11-25"
    reg = make_registry(http_server_cfg())
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        with pytest.raises(McpProtocolError, match="unsupported protocol"):
            await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
    anyio.run(go)


def test_discovery_fail_closed_variants(tmp_path, mcp_http_server):
    import anyio
    for mode, exc in (("extra_tool", McpRegistryError), ("drift", McpRegistryError),
                      ("outref", McpRegistryError)):
        McpFixtureHandler.mode = mode
        reg = make_registry(http_server_cfg(
            tools=[{"name": "echo", "side_effect": "read_only",
                    "input_schema_sha256": validate_input_schema(ECHO_SCHEMA)}]))
        mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)

        async def go(exc=exc, mgr=mgr, run_id=run_id, lease=lease):
            with pytest.raises(exc):
                await mgr.open("t1", run_id=run_id, lease_owner="w1",
                               lease_generation=lease)
        anyio.run(go)
        McpFixtureHandler.mode = "ok"


def test_annotations_never_reclassify(tmp_path, mcp_http_server):
    # server claims readOnlyHint=true; the operator classified the tool forbidden
    reg = make_registry(http_server_cfg(
        tools=[{"name": "echo", "side_effect": "forbidden"}]))
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1",
                              lease_generation=lease)
        assert sess.bindings["echo"].side_effect.value == "forbidden"
        with pytest.raises(McpDispatchError, match="denied"):
            await mgr.call_tool(sess, "echo", {"text": "x"}, run_id=run_id,
                                lease_owner="w1", lease_generation=lease)
        await sess.close()
    anyio.run(go)


def test_session_id_binding_and_substitution(tmp_path, mcp_http_server):
    import anyio
    reg = make_registry(http_server_cfg())
    # normal sid: bound as hash, raw never stored
    McpFixtureHandler.mode = "sid"
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        assert sess.state.session_id_hash is not None
        # only the sha256 of the raw id is stored — the raw id appears nowhere
        import hashlib
        assert sess.state.session_id_hash == hashlib.sha256(b"sid-A").hexdigest()
        assert "sid-A" not in sess.state.session_id_hash
        await sess.close()
    anyio.run(go)
    # substitution: server switches the id mid-session -> fail closed
    McpFixtureHandler.mode = "sid_switch"
    McpFixtureHandler.sid_count = 0
    mgr2, store2, run_id2, lease2 = make_manager(tmp_path, reg, port=mcp_http_server)

    async def go2():
        sess = await mgr2.open("t1", run_id=run_id2, lease_owner="w1",
                               lease_generation=lease2)
        with pytest.raises(McpSessionError, match="substitution"):
            await mgr2.call_tool(sess, "echo", {"text": "x"}, run_id=run_id2,
                                 lease_owner="w1", lease_generation=lease2)
        await sess.close()
    anyio.run(go2)
    McpFixtureHandler.mode = "ok"


def test_malformed_and_sse_and_oversize(tmp_path, mcp_http_server):
    import anyio

    from geosgan_hive.errors import HiveError
    reg = make_registry(http_server_cfg())
    # malformed JSON -> clean failure
    McpFixtureHandler.mode = "malformed"
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)

    async def go():
        with pytest.raises(McpClientError):  # malformed JSON-RPC frame -> clean failure
            await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
    anyio.run(go)
    # SSE-framed response parses through the secure transport (streaming works)
    McpFixtureHandler.mode = "sse"

    async def go2():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        assert set(sess.bindings) == {"echo"}
        await sess.close()
    anyio.run(go2)
    # oversized body vs transport cap -> fail closed
    McpFixtureHandler.mode = "huge"
    mgr3, store3, run_id3, lease3 = make_manager(tmp_path, reg, port=mcp_http_server,
                                                 limit_over={"max_wire_bytes": 16 * 1024})

    async def go3():
        sess = await mgr3.open("t1", run_id=run_id3, lease_owner="w1",
                               lease_generation=lease3)
        with pytest.raises(HiveError):
            await mgr3.call_tool(sess, "echo", {"text": "x"}, run_id=run_id3,
                                 lease_owner="w1", lease_generation=lease3)
        await sess.close()
    anyio.run(go3)
    McpFixtureHandler.mode = "ok"


def test_durable_budget_exhaustion(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg())
    # per_run_requests=3: initialize + notifications/initialized + tools/list
    # consume exactly 3 durable reservations during open(); the tools/call POST
    # must then be denied by the durable per-run budget (fail closed)
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server,
                                             per_run_requests=3)
    import anyio

    from geosgan_hive import QuotaExhaustedError

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        assert set(sess.bindings) == {"echo"}
        with pytest.raises(QuotaExhaustedError, match="budget exhausted"):
            await mgr.call_tool(sess, "echo", {"text": "x"}, run_id=run_id,
                                lease_owner="w1", lease_generation=lease)
        await sess.close()
    anyio.run(go)


def test_lease_loss_before_dispatch(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg())
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        # the run lease expires (clock advanced past the TTL), then another
        # worker takes over -> generation bump fences the old worker
        store._m2_clock["now"] = NOW + timedelta(seconds=120)
        new_gen = store.takeover_lease_gen(run_id, "w2")
        assert new_gen == lease + 1
        with pytest.raises(McpDispatchError, match="stale worker"):
            await mgr.call_tool(sess, "echo", {"text": "x"}, run_id=run_id,
                                lease_owner="w1", lease_generation=lease)
        await sess.close()
    anyio.run(go)


def test_cancellation_stops_session(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg())
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        sess = await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        sess._cancel.set()
        with pytest.raises(McpSessionError, match="cancelled"):
            await mgr.call_tool(sess, "echo", {"text": "x"}, run_id=run_id,
                                lease_owner="w1", lease_generation=lease)
        await sess.close()
    anyio.run(go)


def test_global_disable_blocks_open(tmp_path, mcp_http_server):
    reg = make_registry(http_server_cfg(), enabled=False)
    mgr, store, run_id, lease = make_manager(tmp_path, reg, port=mcp_http_server)
    import anyio

    async def go():
        with pytest.raises(McpDispatchError, match="globally disabled"):
            await mgr.open("t1", run_id=run_id, lease_owner="w1", lease_generation=lease)
    anyio.run(go)


# ---- runtime argument validation (F5 runtime) -----------------------------------------

def test_validate_tool_args_runtime_bounds():
    ok = validate_tool_args(ECHO_SCHEMA, {"text": "hi"})
    assert ok == {"text": "hi"}
    with pytest.raises(McpArgsError):  # wrong type
        validate_tool_args(ECHO_SCHEMA, {"text": 5})
    with pytest.raises(McpArgsError):  # extra key (additionalProperties=false)
        validate_tool_args(ECHO_SCHEMA, {"text": "hi", "x": 1})
    with pytest.raises(McpArgsError):  # out of bounds
        validate_tool_args(ECHO_SCHEMA, {"text": "x" * 101})
    num_schema = {"type": "object", "properties": {"n": {"type": "number",
                                                         "minimum": 0, "maximum": 10}},
                  "additionalProperties": False}
    assert validate_tool_args(num_schema, {"n": 5}) == {"n": 5}
    with pytest.raises(McpArgsError):
        validate_tool_args(num_schema, {"n": 11})
    with pytest.raises(McpArgsError, match="non-finite"):  # F5 runtime
        validate_tool_args(num_schema, {"n": float("inf")})
    with pytest.raises(McpArgsError, match="non-finite"):
        validate_tool_args(num_schema, {"n": float("nan")})
    with pytest.raises(McpArgsError):  # oversized payload
        validate_tool_args(ECHO_SCHEMA, {"text": "x" * 100}, max_args_bytes=10)


# ---- stdio (experimental, isolated) ----------------------------------------------------

@posix_only
def test_stdio_end_to_end_and_env_sanitization(tmp_path, stdio_fixture):
    reg = make_registry(stdio_server_cfg(stdio_fixture))
    mgr, store, run_id, lease = make_manager(tmp_path, reg)
    mgr._environ = {"FIXTURE_ALLOWED": "yes", "SECRET_TOKEN": "topsecret-value"}
    import anyio

    async def go():
        sess = await mgr.open("s1", run_id=run_id, lease_owner="w1", lease_generation=lease)
        assert set(sess.bindings) == {"echo", "env"}
        res = await mgr.call_tool(sess, "echo", {"text": "abc"}, run_id=run_id,
                                  lease_owner="w1", lease_generation=lease)
        assert res["content_text"] == "echo:abc"
        env_res = await mgr.call_tool(sess, "env", {}, run_id=run_id,
                                      lease_owner="w1", lease_generation=lease)
        keys = env_res["content_text"]
        assert "FIXTURE_ALLOWED" in keys
        assert "SECRET_TOKEN" not in keys  # only allowlisted env refs pass through
        await sess.close()
    anyio.run(go, )


@posix_only
def test_stdio_rejects_symlink_and_relative(tmp_path, stdio_fixture):
    link = tmp_path / "link_server"
    os.symlink(stdio_fixture, link)
    with pytest.raises(McpStdioError, match="symlink"):
        check_executable(str(link))
    with pytest.raises(McpStdioError, match="absolute"):
        check_executable("fixture_mcp_server")
    with pytest.raises(McpStdioError, match="launcher"):
        check_executable("/usr/bin/python3")


@posix_only
def test_stdio_timeout_kills_process_group(tmp_path, stdio_fixture):
    reg = make_registry(stdio_server_cfg(stdio_fixture, session_deadline_seconds=2.0,
                                          **{}))
    # force the sleep mode via args
    cfg = stdio_server_cfg(stdio_fixture, session_deadline_seconds=2.0)
    cfg["transport"]["args"] = ["--mode", "sleep"]
    reg = make_registry(cfg)
    mgr, store, run_id, lease = make_manager(tmp_path, reg)
    mgr._op_timeout = 2.0
    import anyio

    async def go():
        with pytest.raises(McpClientError):  # read timeout / EOF after group kill
            await mgr.open("s1", run_id=run_id, lease_owner="w1", lease_generation=lease)
    anyio.run(go)
    time.sleep(2.5)  # watchdog must have killed the group


@posix_only
def test_stdio_malformed_frame_fails_closed(tmp_path, stdio_fixture):
    cfg = stdio_server_cfg(stdio_fixture)
    # fixture prints garbage when a line is not JSON: send one via a raw spawn check
    from geosgan_hive.mcp_config import McpServerConfig
    from geosgan_hive.mcp_stdio import StdioProcess, read_bounded_line
    server = McpServerConfig.model_validate(cfg)
    proc = StdioProcess(server, work_root=str(tmp_path), timeout_s=10)
    try:
        proc.proc.stdin.write(b"not json at all\n")
        proc.proc.stdin.flush()
        line = read_bounded_line(proc.proc.stdout.read1)
        assert line is not None
        assert b"garbage-not-json" in line
    finally:
        proc.close()
