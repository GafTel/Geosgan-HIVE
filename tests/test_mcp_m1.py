"""M1 (v0.9.0): MCP registry, configuration contracts, trust policies and
capability identity primitives. Pure offline tests — no network, no subprocess,
no MCP SDK required, no MCP tools exposed to agents in M1.

Covers the M1 acceptance list: disabled-by-default layers, unknown fields /
servers / tools, duplicate + ambiguous identities, namespace collisions,
annotation non-authority, operator-owned side-effect and cost classification,
approval-cannot-override-forbidden, deterministic digests, capability-change
invalidation, snapshot immutability, secret-reference privacy, no
model-controlled registration, size bounds, protocol revision pinning, existing
config compatibility, no-I/O construction, optional-deps independence, the
unchanged zero-cost gate, and no agent-facing MCP tools in M1.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive import (
    DispatchClass,
    McpConfig,
    McpCostClass,
    McpRegistry,
    McpRegistryError,
    McpSchemaError,
    McpServerConfig,
    McpSideEffect,
    McpToolPolicy,
    dispatch_decision,
    namespaced_tool_name,
    server_identity_digest,
    validate_input_schema,
)
from geosgan_hive.agent_config import HiveConfig, load_hive_config
from geosgan_hive.cloud_profiles import ZERO_COST_RELEASE_LOCAL_PROVIDERS
from geosgan_hive.errors import ConfigError
from geosgan_hive.mcp_config import load_mcp_config_file
from geosgan_hive.tool_factory import build_tool_catalog

REPO = Path(__file__).resolve().parents[1]


# ---- helpers -------------------------------------------------------------------

def stdio_server(**over) -> dict:
    base = {
        "server_id": "fs",
        "transport": {"kind": "stdio", "executable": "/opt/mcp/fs",
                       "args": ["--ro"], "env_refs": ["DOCS_ROOT"]},
        "cost_class": "unknown",
        "tools": [{"name": "read_file", "side_effect": "read_only"}],
    }
    base.update(over)
    return base


def http_server(**over) -> dict:
    base = {
        "server_id": "remote",
        "transport": {"kind": "streamable_http", "endpoint_id": "mcp-gw",
                      "url_path": "/mcp"},
        "cost_class": "unknown",
        "tools": [{"name": "search", "side_effect": "read_only"}],
    }
    base.update(over)
    return base


def registry(*servers, enabled=True, **kw) -> McpRegistry:
    return McpRegistry(McpConfig.model_validate({"enabled": enabled,
                                                 "servers": list(servers)}), **kw)


def minimal_hive(mcp_block: dict | None) -> dict:
    h = {
        "supervisor": {"name": "supervisor", "description": "s",
                       "system_prompt": "s", "model": "ollama/llama3.2"},
        "agents": [{"name": "researcher", "description": "r",
                    "system_prompt": "r", "model": "ollama/llama3.2"}],
    }
    if mcp_block is not None:
        h["mcp"] = mcp_block
    return h


VALID_SCHEMA = {"type": "object",
                "properties": {"path": {"type": "string", "minLength": 1,
                                        "maxLength": 512, "pattern": "^[a-z0-9/]+$"}},
                "required": ["path"], "additionalProperties": False}


def reported(name="read_file", schema=None, annotations=None):
    tool = {"name": name, "inputSchema": schema if schema is not None else VALID_SCHEMA,
            "description": "untrusted server text"}
    if annotations is not None:
        tool["annotations"] = annotations
    return tool


# ---- 1. disabled by default ------------------------------------------------------

def test_mcp_disabled_by_default_every_layer():
    assert McpConfig().enabled is False
    assert McpConfig.model_validate({"servers": [stdio_server()]}).enabled is False
    assert registry(stdio_server(), enabled=False).enabled is False
    # every server disabled by default
    srv = McpServerConfig.model_validate(stdio_server())
    assert srv.enabled is False
    # HiveConfig without an mcp block: legacy configs stay valid, registry disabled
    hive = HiveConfig.model_validate(minimal_hive(None))
    assert hive.mcp is None
    assert McpRegistry.from_hive_config(hive).enabled is False
    # global switch off => every dispatch denied, even for an enabled,
    # attested, read-only server tool
    reg = registry(stdio_server(enabled=True, cost_class="local",
                                 local_execution_attested=True), enabled=False)
    assert reg.dispatch_decision("fs", "read_file").decision is DispatchClass.DENIED


# ---- 2. unknown fields rejected ---------------------------------------------------

def test_unknown_config_fields_rejected():
    with pytest.raises(ValidationError):
        McpConfig.model_validate({"enabled": True, "bogus": 1})
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate({**stdio_server(), "surprise": True})
    with pytest.raises(ValidationError):
        McpToolPolicy.model_validate({"name": "x", "side_effect": "read_only",
                                      "extra": 1})
    with pytest.raises(ValidationError):
        McpConfig.model_validate({"servers": [{"server_id": "a",
                                               "transport": {"kind": "stdio",
                                                             "executable": "/opt/x",
                                                             "headers": {"X": "y"}}}]})


# ---- 3/4/5. unknown + duplicate identities -----------------------------------------

def test_unknown_server_and_tool_fail_closed():
    reg = registry(stdio_server(enabled=True, cost_class="local",
                                 local_execution_attested=True))
    with pytest.raises(McpRegistryError, match="unknown MCP server"):
        reg.server("nope")
    with pytest.raises(McpRegistryError, match="unknown MCP server"):
        reg.dispatch_decision("nope", "read_file")
    with pytest.raises(McpRegistryError, match="unknown tool"):
        reg.tool("fs", "delete_everything")
    with pytest.raises(McpRegistryError, match="not in the operator allowlist"):
        reg.bind_capabilities("fs", [reported("delete_everything")])


def test_duplicate_server_identity_rejected():
    with pytest.raises(ValidationError, match="duplicate server ids"):
        McpConfig.model_validate({"enabled": True,
                                  "servers": [stdio_server(), stdio_server()]})
    with pytest.raises(ValidationError, match="duplicate tool names"):
        McpConfig.model_validate({"servers": [stdio_server(tools=[
            {"name": "read_file", "side_effect": "read_only"},
            {"name": "read_file", "side_effect": "read_only"}])]})


def test_duplicate_and_ambiguous_tool_identities_rejected():
    # crafted ambiguity: mcp_a__b__c arises from BOTH server a/tool b__c and
    # server a__b/tool c — deterministic canonicalization must reject it
    s1 = stdio_server(server_id="a", tools=[{"name": "b__c", "side_effect": "read_only"}])
    s2 = stdio_server(server_id="a__b", tools=[{"name": "c", "side_effect": "read_only"}])
    with pytest.raises(McpRegistryError, match="namespace collision"):
        registry(s1, s2)
    assert namespaced_tool_name("a", "b__c") == "mcp_a__b__c"
    assert namespaced_tool_name("a", "x") != namespaced_tool_name("b", "x")


def test_namespace_collision_with_reserved_builtin_names():
    srv = stdio_server(tools=[{"name": "read", "side_effect": "read_only"}])
    with pytest.raises(McpRegistryError, match="reserved/builtin"):
        registry(srv, reserved_names={"mcp_fs__read", "calculator"})
    # without the reserved name it is fine
    registry(srv, reserved_names={"calculator"})


# ---- 7. server annotations are never authorization --------------------------------

def test_server_annotations_cannot_authorize_operations():
    srv = stdio_server(enabled=True, cost_class="local", local_execution_attested=True,
                       tools=[{"name": "write_file", "side_effect": "external_mutation"}])
    reg = registry(srv)
    # the server claims readOnlyHint=true and destructiveHint=false — ignored
    caps = reg.bind_capabilities("fs", [reported("write_file",
                                                 annotations={"readOnlyHint": True,
                                                              "destructiveHint": False})])
    assert caps[0].side_effect is McpSideEffect.EXTERNAL_MUTATION  # operator's class wins
    d = reg.dispatch_decision("fs", "write_file")
    assert d.decision is DispatchClass.DENIED  # stdio external mutation can never be verified


# ---- 8. unknown/missing classification fails closed --------------------------------

def test_unknown_side_effect_classification_rejected():
    with pytest.raises(ValidationError):
        McpToolPolicy.model_validate({"name": "x", "side_effect": "maybe_read"})
    # forbidden classification -> denied, and approval cannot override (test 10)
    srv = stdio_server(enabled=True, cost_class="local", local_execution_attested=True,
                       tools=[{"name": "nuke", "side_effect": "forbidden"}])
    reg = registry(srv)
    d = reg.dispatch_decision("fs", "nuke")
    assert d.decision is DispatchClass.DENIED and not d.allowed
    assert "approval cannot override" in d.reason


# ---- 9/10. financial eligibility ----------------------------------------------------

def test_unknown_cost_operations_denied_and_paid_always_denied():
    reg = registry(stdio_server(enabled=True))  # cost_class unknown (default)
    d = reg.dispatch_decision("fs", "read_file")
    assert d.decision is DispatchClass.DENIED and "unknown cost" in d.reason
    paid = stdio_server(enabled=True, cost_class="paid",
                        tools=[{"name": "read_file", "side_effect": "read_only",
                                "cost_class": "paid"}])
    d2 = registry(paid).dispatch_decision("fs", "read_file")
    assert d2.decision is DispatchClass.DENIED and "paid" in d2.reason


def test_attestation_required_for_local_and_verified_zero_cost():
    with pytest.raises(ValidationError, match="local_execution_attested"):
        McpServerConfig.model_validate(stdio_server(cost_class="local"))
    with pytest.raises(ValidationError, match="zero_cost_attested"):
        McpServerConfig.model_validate(http_server(cost_class="verified_zero_cost"))
    # tool may not claim a LESS conservative cost class than its server
    with pytest.raises(ValidationError, match="LESS conservative"):
        McpServerConfig.model_validate(stdio_server(
            tools=[{"name": "read_file", "side_effect": "read_only",
                    "cost_class": "local"}]))


def test_human_approval_cannot_override_forbidden_capabilities():
    # contract-level: approval only gates ALLOWED_WITH_APPROVAL; DENIED stays denied
    srv = McpServerConfig.model_validate(stdio_server(
        enabled=True, cost_class="local", local_execution_attested=True,
        tools=[{"name": "forbidden_op", "side_effect": "forbidden",
                "requires_approval": True}]))
    d = dispatch_decision(srv, srv.tools[0], globally_enabled=True)
    assert d.decision is DispatchClass.DENIED
    # and a mutation that IS allowed still requires approval
    srv2 = McpServerConfig.model_validate(stdio_server(
        enabled=True, cost_class="local", local_execution_attested=True,
        tools=[{"name": "write", "side_effect": "local_mutation"}]))
    d2 = dispatch_decision(srv2, srv2.tools[0], globally_enabled=True)
    assert d2.decision is DispatchClass.ALLOWED_WITH_APPROVAL and d2.requires_approval


def test_external_mutation_requires_remote_transport_and_verified_zero_cost():
    http = http_server(enabled=True, cost_class="verified_zero_cost",
                       zero_cost_attested=True,
                       tools=[{"name": "post", "side_effect": "external_mutation"}])
    d = registry(http).dispatch_decision("remote", "post")
    assert d.decision is DispatchClass.ALLOWED_WITH_APPROVAL and d.requires_approval
    # same op over stdio is unverifiable -> denied even if "local" attested
    std = stdio_server(enabled=True, cost_class="local", local_execution_attested=True,
                       tools=[{"name": "post", "side_effect": "external_mutation"}])
    d2 = registry(std).dispatch_decision("fs", "post")
    assert d2.decision is DispatchClass.DENIED


# ---- 11. deterministic digests ------------------------------------------------------

def test_deterministic_config_and_identity_digests():
    a = McpServerConfig.model_validate(stdio_server())
    b = McpServerConfig.model_validate(stdio_server())
    assert server_identity_digest(a) == server_identity_digest(b)
    assert len(server_identity_digest(a)) == 64
    # volatile/display fields do not change identity
    c = stdio_server(enabled=True)
    c["description"] = "different untrusted text"
    assert server_identity_digest(a) == server_identity_digest(
        McpServerConfig.model_validate(c))
    # a real config change does (transport argv)
    d = stdio_server(transport={"kind": "stdio", "executable": "/opt/mcp/fs", "args": []})
    assert server_identity_digest(a) != server_identity_digest(
        McpServerConfig.model_validate(d))


# ---- 12. capability changes invalidate bindings -------------------------------------

def test_capability_changes_invalidate_bindings():
    srv = stdio_server(enabled=True, cost_class="local", local_execution_attested=True)
    reg = registry(srv)
    caps1 = reg.bind_capabilities("fs", [reported()])
    caps2 = reg.bind_capabilities("fs", [reported()])
    assert caps1[0].binding_digest() == caps2[0].binding_digest()  # stable
    # changed reported schema -> different binding digest (or pin mismatch error)
    changed = {"type": "object", "properties": {"path": {"type": "string"}},
               "additionalProperties": False}
    caps3 = reg.bind_capabilities("fs", [reported(schema=changed)])
    assert caps3[0].binding_digest() != caps1[0].binding_digest()
    # pinned schema mismatch -> hard error (invalidation)
    pinned = stdio_server(enabled=True, cost_class="local", local_execution_attested=True,
                          tools=[{"name": "read_file", "side_effect": "read_only",
                                  "input_schema_sha256": validate_input_schema(VALID_SCHEMA)}])
    reg2 = registry(pinned)
    with pytest.raises(McpRegistryError, match="capability CHANGE"):
        reg2.bind_capabilities("fs", [reported(schema=changed)])
    # missing allowlisted tool -> error
    with pytest.raises(McpRegistryError, match="did not report"):
        reg.bind_capabilities("fs", [])
    # duplicate reported tool -> error
    with pytest.raises(McpRegistryError, match="duplicate reported tool"):
        reg.bind_capabilities("fs", [reported(), reported()])
    # disabled server -> no bindings at all
    with pytest.raises(McpRegistryError, match="disabled"):
        registry(stdio_server(enabled=False)).bind_capabilities("fs", [reported()])


# ---- 13. snapshot immutability -------------------------------------------------------

def test_registry_snapshots_are_immutable():
    reg = registry(stdio_server())
    snap = reg.snapshot()
    assert snap is not reg
    assert snap.summary() == reg.summary()
    entry = snap.server("fs")
    with pytest.raises(ValidationError):  # frozen model
        entry.server.enabled = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        entry.server.tools[0].side_effect = McpSideEffect.FORBIDDEN  # type: ignore[misc]
    assert reg.server("fs").server.enabled is False  # original untouched


# ---- 14. secret references never exposed ---------------------------------------------

def test_secret_references_not_exposed_and_secret_values_rejected():
    reg = registry(stdio_server())
    s = json.dumps(reg.summary())
    assert "DOCS_ROOT" not in s  # env var NAME redacted from summaries
    assert "env_ref_count" in s
    # a secret-looking value anywhere in the config is rejected at validation
    with pytest.raises(ValidationError, match="secret-looking"):
        McpConfig.model_validate({"servers": [stdio_server(
            description="key sk-abcdefghijklmnopqrstuvwxyz1234567890")]})
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate(stdio_server(
            transport={"kind": "stdio", "executable": "/opt/mcp/fs",
                       "env_refs": ["sk-notanenvname12345678901234567890"]}))


# ---- 15/22. no model-controlled registration; no agent-facing MCP tools in M1 --------

def test_no_agent_facing_registration_path_and_no_mcp_tools_in_catalog():
    reg = registry(stdio_server(enabled=True, cost_class="local",
                                 local_execution_attested=True))
    for forbidden in ("register", "add_server", "enable", "add_tool", "connect"):
        assert not hasattr(reg, forbidden), f"registry must not expose {forbidden}()"
    # even with MCP fully configured, the M1 gateway catalog exposes NO mcp tools
    hive = HiveConfig.model_validate(minimal_hive(
        {"enabled": True, "servers": [stdio_server(enabled=True, cost_class="local",
                                                  local_execution_attested=True)]}))
    catalog, _sb = build_tool_catalog(hive)
    names = catalog.names()
    assert not any(n.startswith("mcp_") for n in names)
    assert not any("mcp" in n for n in names)


# ---- 16. oversized inputs rejected ----------------------------------------------------

def test_oversized_registry_and_config_inputs_rejected(tmp_path):
    with pytest.raises(ValidationError):  # > MAX_SERVERS
        McpConfig.model_validate({"servers": [stdio_server(server_id=f"s{i}")
                                              for i in range(17)]})
    with pytest.raises(ValidationError):  # tool name too long
        McpConfig.model_validate({"servers": [stdio_server(
            tools=[{"name": "x" * 65, "side_effect": "read_only"}])]})
    with pytest.raises(ValidationError):  # description too long
        McpConfig.model_validate({"servers": [stdio_server(description="d" * 3000)]})
    with pytest.raises(ValidationError):  # too many tools
        McpConfig.model_validate({"servers": [stdio_server(
            tools=[{"name": f"t{i}", "side_effect": "read_only"} for i in range(65)])]})
    big = tmp_path / "big.yaml"
    big.write_text("mcp:\n  enabled: true\n  servers:\n" + "    # pad\n" + "x" * (300 * 1024),
                   encoding="utf-8")
    with pytest.raises(ConfigError, match="cap"):
        load_mcp_config_file(str(big))
    with pytest.raises(McpSchemaError, match="exceeds"):  # oversized schema
        validate_input_schema({"type": "object",
                               "properties": {"blob": {"type": "string",
                                                       "maxLength": 10**9}},
                               "description": "x" * (70 * 1024)})


# ---- 17. protocol revision pinning -----------------------------------------------------

def test_unsupported_protocol_revisions_rejected():
    for rev in ("2024-11-05", "2025-03-26", "2025-11-25", "2026-07-28", "latest"):
        with pytest.raises(ValidationError):
            McpServerConfig.model_validate(stdio_server(protocol_revision=rev))
    assert McpServerConfig.model_validate(
        stdio_server()).protocol_revision == "2025-06-18"


# ---- 18. existing configurations remain valid -------------------------------------------

def test_existing_config_files_remain_valid():
    for name in ("agents.yaml", "agents.local.yaml", "agents.cloud.yaml",
                 "agents.routing.example.yaml", "agents.webdocs.example.yaml"):
        cfg = load_hive_config(REPO / "config" / name, environ={})
        assert cfg.mcp is None  # untouched by M1
    # the new example file loads standalone and is fully disabled
    mcp = load_mcp_config_file(str(REPO / "config" / "mcp.example.yaml"))
    assert mcp.enabled is False
    assert len(mcp.servers) == 2
    assert all(not s.enabled for s in mcp.servers)
    # and as an inline `mcp:` block on a team config
    cfg = HiveConfig.model_validate(minimal_hive(
        {"enabled": False, "servers": [stdio_server()]}))
    assert cfg.mcp is not None and cfg.mcp.enabled is False


def test_remote_transport_endpoint_must_be_enabled_api_endpoint():
    http = http_server(enabled=True)
    # no apis block at all -> rejected
    with pytest.raises(ValidationError, match="no 'apis:' block"):
        HiveConfig.model_validate({**minimal_hive({"enabled": True, "servers": [http]})})
    # apis present but provider not enabled -> rejected
    h = minimal_hive({"enabled": True, "servers": [http]})
    h["apis"] = {"enabled": [], "providers": [{
        "provider": "mcp-gw", "billing": "public_api_no_charge",
        "outbound_policy": "approval_required", "allowed_operations": ["mcp"],
        "endpoints": [{"endpoint_id": "mcp-gateway", "scheme": "https",
                       "host": "mcp.example.org", "path_prefixes": ["/mcp"],
                       "methods": ["POST"]}]}]}
    with pytest.raises(ValidationError, match="not an endpoint of an ENABLED"):
        HiveConfig.model_validate(h)
    # enabled provider -> accepted
    h["apis"]["enabled"] = ["mcp-gw"]
    h["mcp"]["servers"][0]["transport"]["endpoint_id"] = "mcp-gateway"
    assert HiveConfig.model_validate(h).mcp is not None


# ---- 19. construction performs no network or subprocess activity -----------------------

def test_registry_construction_does_no_network_or_subprocess(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("I/O attempted during registry construction")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "call", boom)
    reg = registry(stdio_server(enabled=True, cost_class="local",
                                 local_execution_attested=True),
                   http_server(enabled=True))
    reg.bind_capabilities("fs", [reported()])
    reg.summary()
    reg.dispatch_decision("fs", "read_file")
    f = tmp_path / "mcp.yaml"
    f.write_text(json.dumps({"enabled": True, "servers": [stdio_server()]}), encoding="utf-8")
    McpRegistry.from_file(str(f))


# ---- 20. optional MCP SDK not required for core imports ---------------------------------

def test_core_works_without_mcp_sdk(monkeypatch):
    import sys

    class _Block:
        def find_module(self, name, path=None):
            if name == "mcp" or name.startswith("mcp."):
                return self
            return None

        def load_module(self, name):
            raise ImportError(f"blocked (simulating missing optional mcp extra): {name}")

    monkeypatch.setitem(sys.modules, "mcp", None)  # import mcp -> fails
    import geosgan_hive
    assert geosgan_hive.McpRegistry is McpRegistry
    reg = registry(stdio_server())
    assert reg.server("fs").identity_digest
    # the SDK is never imported by core modules. Checked in a FRESH subprocess:
    # in-process sys.modules is polluted at collection time by test_mcp_m2's
    # module-level importorskip("mcp"), so an in-process assertion would be
    # order-dependent. The subprocess blocks `mcp` before importing the core.
    code = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"
        "import geosgan_hive\n"
        "leaked = [m for m in sys.modules\n"
        "          if (m == 'mcp' or m.startswith('mcp.')) and sys.modules[m] is not None]\n"
        "assert not leaked, f'SDK modules imported by core: {leaked}'\n"
        "print('NO_SDK_OK')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO))
    assert out.returncode == 0 and "NO_SDK_OK" in out.stdout, out.stderr


# ---- 21. zero-cost release gate unchanged -------------------------------------------------

def test_zero_cost_release_gate_unchanged():
    assert ZERO_COST_RELEASE_LOCAL_PROVIDERS == ("ollama",)
    src = (REPO / "src" / "geosgan_hive" / "cloud_profiles.py").read_text(encoding="utf-8")
    assert 'ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)' in src


# ---- amendment G: schema safety subset ---------------------------------------------------

def test_schema_subset_validation_and_deterministic_digest():
    d1 = validate_input_schema(VALID_SCHEMA)
    d2 = validate_input_schema(json.loads(json.dumps(VALID_SCHEMA)))
    assert d1 == d2 and len(d1) == 64


@pytest.mark.parametrize("schema", [
    {"$ref": "https://evil.example/schema.json"},                       # remote ref
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}}, "$defs": {"x": {}}},
    {"allOf": [{"type": "object"}]},                                      # combinator
    {"anyOf": [{"type": "string"}]},
    {"oneOf": [{"type": "string"}]},
    {"type": "object", "additionalProperties": True},                     # must be false
    {"type": "object", "additionalProperties": {"type": "string"}},
    {"type": "object", "patternProperties": {"^x": {}}},
    {"type": "object", "properties": {"a": {"type": "string", "pattern": r"(a+)\1"}}},
    {"type": "object", "properties": {"a": {"type": "string",
                                            "pattern": "(?=x)"}}},
    {"type": "object", "properties": {"a": {"type": "string", "pattern": "x" * 300}}},
    {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(65)}},
    {"type": "object", "properties": {"a": {"type": "array", "maxItems": 20_000}}},
    {"type": "object", "properties": {"a": {"enum": [f"v{i}" for i in range(65)]}}},
    {"type": "object", "properties": {"a": {"type": "string", "unknownKeyword": 1}}},
    {"type": "object", "if": {"type": "object"}, "then": {}},
    {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"},
    {"type": "object", "required": ["missing"], "properties": {"a": {"type": "string"}}},
])
def test_unsupported_schemas_fail_closed(schema):
    with pytest.raises(McpSchemaError):
        validate_input_schema(schema)


def test_deep_nesting_rejected():
    node = {"type": "string"}
    for _ in range(6):
        node = {"type": "object", "properties": {"a": node}, "additionalProperties": False}
    with pytest.raises(McpSchemaError, match="nesting"):
        validate_input_schema(node)
    node5 = {"type": "string"}
    for _ in range(4):
        node5 = {"type": "object", "properties": {"a": node5}, "additionalProperties": False}
    assert len(validate_input_schema(node5)) == 64  # depth 5 ok


def test_schema_validation_is_pure_and_safe_regex():
    # compiles patterns but rejects catastrophic constructs; no network/codegen
    assert len(validate_input_schema(
        {"type": "object",
         "properties": {"q": {"type": "string", "pattern": "^[a-z0-9 -]{1,64}$"}}})) == 64
    with pytest.raises(McpSchemaError):
        validate_input_schema({"type": "object", "properties": {
            "q": {"type": "string", "pattern": "(a|a)*$" + "b" * 300}}})


# ---- dispatch decision matrix ------------------------------------------------------------

def test_dispatch_matrix():
    local_ro = stdio_server(enabled=True, cost_class="local",
                            local_execution_attested=True)
    reg = registry(local_ro)
    d = reg.dispatch_decision("fs", "read_file")
    assert d.decision is DispatchClass.ALLOWED and not d.requires_approval
    assert d.cost_class is McpCostClass.LOCAL
    # server disabled overrides everything
    reg_off = registry(stdio_server(enabled=False, cost_class="local",
                                    local_execution_attested=True))
    assert reg_off.dispatch_decision("fs", "read_file").decision is DispatchClass.DENIED


# ---- M1 residual fixes F1-F6 (failing-first regression tests) --------------------

SECRET_VAL = "sk-abcdefghijklmnopqrstuvwxyz1234567890"


def test_f1_secret_scan_applies_to_every_model_and_never_leaks_value():
    # direct submodel construction must also reject secret-looking values
    with pytest.raises(ValidationError) as ei:
        McpServerConfig.model_validate({**stdio_server(), "description": f"key {SECRET_VAL}"})
    assert SECRET_VAL not in str(ei.value)  # value never exposed in the error
    with pytest.raises(ValidationError) as ei2:
        McpToolPolicy.model_validate({"name": "t", "side_effect": "read_only",
                                      "description": SECRET_VAL})
    assert SECRET_VAL not in str(ei2.value)
    with pytest.raises(ValidationError):
        McpConfig.model_validate({"servers": [stdio_server(
            transport={"kind": "stdio", "executable": "/opt/mcp/fs",
                       "args": [SECRET_VAL]})]})


def test_f2_registry_boundary_errors_are_registry_errors():
    reg = registry(stdio_server(enabled=True, cost_class="local",
                                 local_execution_attested=True))
    with pytest.raises(McpRegistryError):  # not a raw McpSchemaError
        reg.bind_capabilities("fs", [{"name": "read_file"}])  # missing inputSchema
    with pytest.raises(McpRegistryError):
        reg.bind_capabilities("fs", [reported(schema={"$ref": "https://x/y.json"})])


def test_f3_no_capability_binding_when_global_mcp_disabled():
    reg = McpRegistry(McpConfig.model_validate(
        {"enabled": False, "servers": [stdio_server(enabled=True, cost_class="local",
                                                    local_execution_attested=True)]}))
    with pytest.raises(McpRegistryError, match="globally disabled"):
        reg.bind_capabilities("fs", [reported()])


def test_f4_max_length_ids_still_namespace_without_weakening_collisions():
    from geosgan_hive.mcp_contracts import MAX_NAMESPACED_NAME_CHARS, McpContractError
    name = namespaced_tool_name("a" * 32, "b" * 64)
    assert name == "mcp_" + "a" * 32 + "__" + "b" * 64
    assert len(name) <= MAX_NAMESPACED_NAME_CHARS
    # collision protection unchanged
    s1 = stdio_server(server_id="a", tools=[{"name": "b__c", "side_effect": "read_only"}])
    s2 = stdio_server(server_id="a__b", tools=[{"name": "c", "side_effect": "read_only"}])
    with pytest.raises(McpRegistryError, match="namespace collision"):
        registry(s1, s2)
    # id patterns still enforced at config level, and the length cap still guards
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate(stdio_server(server_id="a" * 40))
    with pytest.raises(McpContractError):
        namespaced_tool_name("x" * 200, "b")  # over-cap -> still rejected


def test_f5_numeric_schema_keywords_must_be_finite_and_bounded():
    for bad in (1e308, -1e308, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(McpSchemaError):
            validate_input_schema({"type": "object", "properties": {
                "n": {"type": "number", "minimum": bad}}})
        with pytest.raises(McpSchemaError):
            validate_input_schema({"type": "object", "properties": {
                "n": {"type": "integer", "maximum": bad, "multipleOf": bad}}})
    # sane bounds still accepted
    assert len(validate_input_schema({"type": "object", "properties": {
        "n": {"type": "number", "minimum": -1e6, "maximum": 1e6}}})) == 64


def test_f6_tool_descriptions_excluded_from_identity_digest():
    a = McpServerConfig.model_validate({**stdio_server(), "tools": [
        {"name": "t", "side_effect": "read_only", "description": "one"}]})
    b = McpServerConfig.model_validate({**stdio_server(), "tools": [
        {"name": "t", "side_effect": "read_only", "description": "two"}]})
    assert server_identity_digest(a) == server_identity_digest(b)
    # but a classification change still changes identity
    c = McpServerConfig.model_validate({**stdio_server(), "tools": [
        {"name": "t", "side_effect": "forbidden"}]})
    assert server_identity_digest(a) != server_identity_digest(c)
