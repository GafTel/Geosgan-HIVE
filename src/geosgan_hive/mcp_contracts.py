"""M1 (v0.9.0): MCP capability identity, binding and JSON-Schema safety contracts.

Amendment B (server identity and capability integrity): every capability is
bound to a digest over ALL of: server identity, transport identity, executable
or endpoint identity, protocol revision, tool name, canonical input-schema
digest, operator-owned side-effect classification, effective cost class,
approval requirement, output cap and policy version. Any change produces a
different binding digest, which invalidates previous authorization (the
dispatch layer must bind approvals/executions to the digest and reject stale
ones — fail closed).

Tool namespaces are deterministic: ``mcp_<server_id>__<tool_name>`` with both
parts restricted to unambiguous ASCII charsets (no Unicode, no separator
ambiguity, no duplicate identifiers). Collisions with reserved/builtin names
are rejected at registry construction.

Amendment G (JSON Schema safety): remote tool input schemas are validated
against a small supported subset ONLY. Anything else — $ref/$defs, combinators,
conditional keywords, patternProperties, unknown keywords, oversized or
deeply nested schemas, unsafe regexes — fails closed. Validation is pure
structural (plus ``re.compile`` on bounded patterns): no network requests, no
code generation, no generated-code execution.

All functions here are pure (no I/O, no subprocess, no network).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import HiveError
from .mcp_config import (
    MAX_TOOL_DESCRIPTION_CHARS,
    SUPPORTED_PROTOCOL_REVISION,
    McpCostClass,
    McpServerConfig,
    McpSideEffect,
    McpToolPolicy,
)
from .tool_spec import args_sha256, canonical_args

#: policy version bound into every capability digest; bump on policy changes
POLICY_VERSION = "mcp-policy-1"
SUPPORTED_PROTOCOL_REVISIONS = (SUPPORTED_PROTOCOL_REVISION,)

NAMESPACE_PREFIX = "mcp"
#: 4 ("mcp_") + 32 (max server_id) + 2 ("__") + 64 (max tool name) — F4: max-length
#: valid ids must still namespace; collision protection is unchanged
MAX_NAMESPACED_NAME_CHARS = 102
#: numeric schema keywords must be finite and within this magnitude (F5)
MAX_NUMERIC_MAGNITUDE = 1e15

MAX_SCHEMA_BYTES = 64 * 1024
MAX_SCHEMA_DEPTH = 5
MAX_SCHEMA_PROPERTIES = 64
MAX_ENUM_ITEMS = 64
MAX_ENUM_ITEM_CHARS = 256
MAX_PATTERN_CHARS = 256
MAX_ARRAY_ITEMS = 10_000
MAX_STRING_CHARS = 1 << 20

#: JSON Schema 2020-12 keywords allowed inside the supported subset
_ROOT_KEYWORDS = {"type", "properties", "required", "additionalProperties",
                  "title", "description", "$schema"}
_COMMON_KEYWORDS = {"type", "enum", "const", "title", "description", "default"}
_STRING_KEYWORDS = _COMMON_KEYWORDS | {"minLength", "maxLength", "pattern", "format"}
_NUMBER_KEYWORDS = _COMMON_KEYWORDS | {"minimum", "maximum", "exclusiveMinimum",
                                        "exclusiveMaximum", "multipleOf"}
_ARRAY_KEYWORDS = _COMMON_KEYWORDS | {"items", "minItems", "maxItems", "uniqueItems"}
_OBJECT_KEYWORDS = _ROOT_KEYWORDS | {"minProperties", "maxProperties"}

#: unsafe regex constructs rejected outright (ReDoS / backtracking bombs)
_UNSAFE_PATTERN = re.compile(r"\\\d|\(\?[=!<]|\*\+|\+\*|\+\+|\*\*|\{\d+,\d*\}\s*[?+*]|&&")


class McpSchemaError(HiveError):
    """Remote input schema is outside the supported subset — fail closed."""


class McpContractError(HiveError):
    """Capability identity / namespace / binding contract violation."""


def namespaced_tool_name(server_id: str, tool_name: str) -> str:
    """Deterministic, collision-free namespace mapping (amendment B). Both
    parts are already charset-restricted by mcp_config patterns; Unicode and
    separator tricks are impossible by construction."""
    name = f"{NAMESPACE_PREFIX}_{server_id}__{tool_name}"
    if len(name) > MAX_NAMESPACED_NAME_CHARS:
        raise McpContractError(f"namespaced tool name too long: {name!r}")
    return name


def server_identity_digest(server: McpServerConfig) -> str:
    """Canonical digest of the server's identity-bearing configuration.
    `enabled` (a volatile switch) and `description` (untrusted display text)
    are excluded; everything else — transport, executable/args/env-ref names,
    attestations, cost class, tool classifications, protocol revision — is
    bound in. Changing any of them changes the identity."""
    data = server.model_dump(mode="json")
    data.pop("enabled", None)       # volatile switch — not identity
    data.pop("description", None)   # untrusted display text — not identity
    for tool in data.get("tools", []):  # F6: tool descriptions are display text too
        if isinstance(tool, dict):
            tool.pop("description", None)
    return args_sha256(data)


def transport_identity(server: McpServerConfig) -> str:
    """Transport identity string bound into capabilities: for stdio the
    executable path + argv + env-ref NAMES (never values); for remote the
    endpoint id + url path."""
    t = server.transport
    if t.kind == "stdio":
        return args_sha256({"kind": "stdio", "executable": t.executable,
                            "args": list(t.args), "env_refs": sorted(t.env_refs)})
    return args_sha256({"kind": "streamable_http", "endpoint_id": t.endpoint_id,
                        "url_path": t.url_path})


@dataclass(frozen=True)
class ToolCapability:
    """Immutable binding of one remote tool to the full identity context
    (amendment B). `binding_digest` changes if ANY bound field changes."""

    server_id: str
    server_identity: str          # server_identity_digest
    transport_kind: str
    transport_identity: str
    protocol_revision: str
    tool_name: str
    namespaced_name: str
    input_schema_sha256: str      # canonical digest of the validated schema
    side_effect: McpSideEffect
    cost_class: McpCostClass
    requires_approval: bool
    max_output_bytes: int
    policy_version: str

    def binding_digest(self) -> str:
        return args_sha256({
            "server_id": self.server_id,
            "server_identity": self.server_identity,
            "transport_kind": self.transport_kind,
            "transport_identity": self.transport_identity,
            "protocol_revision": self.protocol_revision,
            "tool_name": self.tool_name,
            "namespaced_name": self.namespaced_name,
            "input_schema_sha256": self.input_schema_sha256,
            "side_effect": self.side_effect.value,
            "cost_class": self.cost_class.value,
            "requires_approval": self.requires_approval,
            "max_output_bytes": self.max_output_bytes,
            "policy_version": self.policy_version,
        })


class DispatchClass(StrEnum):
    ALLOWED = "allowed"                    # may run (approval rules still apply)
    ALLOWED_WITH_APPROVAL = "allowed_with_approval"
    DENIED = "denied"


@dataclass(frozen=True)
class DispatchDecision:
    """Pure contract evaluation (amendment A/D): no I/O. `allowed` is False for
    every forbidden/unknown-cost case; human approval can NEVER turn a DENIED
    decision into an allowed one — approval only gates ALLOWED_WITH_APPROVAL."""

    decision: DispatchClass
    reason: str
    requires_approval: bool = False
    cost_class: McpCostClass = McpCostClass.UNKNOWN

    @property
    def allowed(self) -> bool:
        return self.decision is not DispatchClass.DENIED


def effective_cost_class(server: McpServerConfig, tool: McpToolPolicy) -> McpCostClass:
    return tool.cost_class if tool.cost_class is not None else server.cost_class


def effective_requires_approval(tool: McpToolPolicy) -> bool:
    if tool.requires_approval is not None:
        return tool.requires_approval
    return tool.side_effect in (McpSideEffect.LOCAL_MUTATION, McpSideEffect.EXTERNAL_MUTATION)


def dispatch_decision(server: McpServerConfig, tool: McpToolPolicy, *,
                      globally_enabled: bool = True) -> DispatchDecision:
    """Operator-owned rules, evaluated without any I/O:

    - MCP globally disabled or server disabled -> DENIED.
    - side_effect=FORBIDDEN -> DENIED (approval cannot override).
    - cost_class UNKNOWN -> DENIED (unknown financial consequences fail closed).
    - cost_class PAID -> DENIED (paid operations never run in the zero-cost
      release; ZERO_COST_RELEASE_LOCAL_PROVIDERS gate is untouched).
    - EXTERNAL_MUTATION additionally requires a remote (streamable_http)
      transport AND verified_zero_cost: external effects from an opaque local
      subprocess cannot be verified -> fail closed (amendment C).
    - Mutations require human approval (ALLOWED_WITH_APPROVAL); read-only
      attested operations are ALLOWED."""
    if not globally_enabled:
        return DispatchDecision(DispatchClass.DENIED, "MCP is globally disabled")
    if not server.enabled:
        return DispatchDecision(DispatchClass.DENIED, f"server {server.server_id!r} disabled")
    if tool.side_effect is McpSideEffect.FORBIDDEN:
        return DispatchDecision(
            DispatchClass.DENIED,
            f"tool {tool.name!r} is operator-classified forbidden — approval cannot override")
    cost = effective_cost_class(server, tool)
    if cost is McpCostClass.UNKNOWN:
        return DispatchDecision(
            DispatchClass.DENIED,
            f"tool {tool.name!r} has unknown cost — fails closed", cost_class=cost)
    if cost is McpCostClass.PAID:
        return DispatchDecision(
            DispatchClass.DENIED,
            f"tool {tool.name!r} is classified paid — denied in the zero-cost release",
            cost_class=cost)
    if tool.side_effect is McpSideEffect.EXTERNAL_MUTATION:
        if server.transport.kind != "streamable_http":
            return DispatchDecision(
                DispatchClass.DENIED,
                "external mutation via opaque local stdio subprocess cannot be verified — "
                "denied (amendment C)", cost_class=cost)
        if cost is not McpCostClass.VERIFIED_ZERO_COST:
            return DispatchDecision(
                DispatchClass.DENIED,
                "external mutation requires verified_zero_cost attestation", cost_class=cost)
    needs_approval = effective_requires_approval(tool)
    if needs_approval:
        return DispatchDecision(DispatchClass.ALLOWED_WITH_APPROVAL,
                                "operator-classified mutation requires human approval",
                                requires_approval=True, cost_class=cost)
    return DispatchDecision(DispatchClass.ALLOWED, "operator-classified read-only operation",
                            cost_class=cost)


# ---- JSON Schema safety (amendment G) -------------------------------------------

_ALLOWED_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


def _check_pattern(pattern: Any) -> None:
    if not isinstance(pattern, str):
        raise McpSchemaError("pattern must be a string")
    if len(pattern) > MAX_PATTERN_CHARS:
        raise McpSchemaError(f"pattern exceeds {MAX_PATTERN_CHARS} chars — rejected")
    if _UNSAFE_PATTERN.search(pattern):
        raise McpSchemaError("pattern contains unsafe regex constructs (backref/lookaround/"
                             "nested quantifier) — rejected")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise McpSchemaError(f"pattern does not compile: {exc}") from exc


def _validate_schema_node(node: Any, *, depth: int, path: str) -> None:
    if depth > MAX_SCHEMA_DEPTH:
        raise McpSchemaError(f"schema nesting deeper than {MAX_SCHEMA_DEPTH} at {path!r}")
    if not isinstance(node, dict):
        raise McpSchemaError(f"schema at {path!r} must be an object")
    stype = node.get("type")
    if stype is None:
        allowed = _ROOT_KEYWORDS  # untyped root treated as object
    elif stype == "object":
        allowed = _OBJECT_KEYWORDS
    elif stype == "array":
        allowed = _ARRAY_KEYWORDS
    elif stype == "string":
        allowed = _STRING_KEYWORDS
    elif stype in ("number", "integer"):
        allowed = _NUMBER_KEYWORDS
    elif stype in ("boolean", "null"):
        allowed = _COMMON_KEYWORDS
    else:
        raise McpSchemaError(f"unsupported type {stype!r} at {path!r}")
    unknown = set(node) - allowed
    if unknown:
        raise McpSchemaError(
            f"unsupported schema keywords {sorted(unknown)} at {path!r} — fail closed "
            "($ref/$defs/combinators/conditionals and unknown keywords are not supported)")
    if "$schema" in node and node["$schema"] != _ALLOWED_JSON_SCHEMA_URI:
        raise McpSchemaError("only JSON Schema 2020-12 is supported (annotation only)")
    for key in ("title", "description"):
        if key in node and (not isinstance(node[key], str)
                            or len(node[key]) > MAX_TOOL_DESCRIPTION_CHARS):
            raise McpSchemaError(f"{key} must be a bounded string at {path!r}")
    if "enum" in node:
        enum = node["enum"]
        if not isinstance(enum, list) or len(enum) > MAX_ENUM_ITEMS:
            raise McpSchemaError(f"enum at {path!r} exceeds {MAX_ENUM_ITEMS} items")
        for item in enum:
            if isinstance(item, str) and len(item) > MAX_ENUM_ITEM_CHARS:
                raise McpSchemaError(f"enum item too long at {path!r}")
    if stype == "string" or (stype is None and "pattern" in node):
        if "pattern" in node:
            _check_pattern(node["pattern"])
        for k in ("minLength", "maxLength"):
            if k in node:
                if not isinstance(node[k], int) or node[k] < 0 or node[k] > MAX_STRING_CHARS:
                    raise McpSchemaError(f"{k} out of bounds at {path!r}")
        if "format" in node and (not isinstance(node["format"], str)
                                 or len(node["format"]) > 64):
            raise McpSchemaError(f"format must be a short annotation string at {path!r}")
    if stype in ("number", "integer") or (stype is None and "minimum" in node):
        for k in ("minimum", "maximum", "multipleOf"):
            if k in node:
                if not isinstance(node[k], (int, float)) or isinstance(node[k], bool):
                    raise McpSchemaError(f"{k} must be numeric at {path!r}")
                # F5: finite and bounded — never inf/nan/huge
                if not math.isfinite(node[k]) or abs(node[k]) > MAX_NUMERIC_MAGNITUDE:
                    raise McpSchemaError(
                        f"{k} must be finite with |value| <= {MAX_NUMERIC_MAGNITUDE:g} "
                        f"at {path!r}")
        for k in ("exclusiveMinimum", "exclusiveMaximum"):
            if k in node:
                if isinstance(node[k], bool):
                    continue  # draft-07 boolean form: annotation only here
                if not isinstance(node[k], (int, float)):
                    raise McpSchemaError(f"{k} must be numeric/bool at {path!r}")
                if not math.isfinite(node[k]) or abs(node[k]) > MAX_NUMERIC_MAGNITUDE:
                    raise McpSchemaError(f"{k} must be finite and bounded at {path!r}")
        if stype == "integer" and "multipleOf" in node and node["multipleOf"] <= 0:
            raise McpSchemaError(f"multipleOf must be > 0 at {path!r}")
    if stype == "array" or (stype is None and "items" in node):
        if "items" in node:
            _validate_schema_node(node["items"], depth=depth + 1, path=f"{path}.items")
        for k in ("minItems", "maxItems"):
            if k in node:
                if not isinstance(node[k], int) or node[k] < 0 or node[k] > MAX_ARRAY_ITEMS:
                    raise McpSchemaError(f"{k} out of bounds at {path!r}")
        if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
            raise McpSchemaError(f"uniqueItems must be a boolean at {path!r}")
    if stype == "object" or stype is None:
        props = node.get("properties")
        if props is not None:
            if not isinstance(props, dict) or len(props) > MAX_SCHEMA_PROPERTIES:
                raise McpSchemaError(
                    f"properties at {path!r} must be an object with <= "
                    f"{MAX_SCHEMA_PROPERTIES} entries")
            for pname, pschema in props.items():
                if not isinstance(pname, str) or not pname or len(pname) > 128:
                    raise McpSchemaError(f"invalid property name at {path!r}")
                _validate_schema_node(pschema, depth=depth + 1,
                                      path=f"{path}.properties.{pname}")
        required = node.get("required")
        if required is not None:
            if (not isinstance(required, list) or len(required) > MAX_SCHEMA_PROPERTIES
                    or not all(isinstance(r, str) for r in required)):
                raise McpSchemaError(f"required must be a list of strings at {path!r}")
            if props is not None:
                missing = [r for r in required if r not in props]
                if missing:
                    raise McpSchemaError(
                        f"required names {missing} not present in properties at {path!r}")
        if "additionalProperties" in node and node["additionalProperties"] is not False:
            raise McpSchemaError(
                f"additionalProperties must be false at {path!r} (strict, extra=forbid "
                "semantics)")
        for k in ("minProperties", "maxProperties"):
            if k in node and (not isinstance(node[k], int) or node[k] < 0
                              or node[k] > MAX_SCHEMA_PROPERTIES):
                raise McpSchemaError(f"{k} out of bounds at {path!r}")
    if "const" in node and isinstance(node["const"], str) \
            and len(node["const"]) > MAX_ENUM_ITEM_CHARS:
        raise McpSchemaError(f"const string too long at {path!r}")


def validate_input_schema(schema: Any) -> str:
    """Validate a remote tool input schema against the supported subset and
    return its canonical sha256 digest. Pure structural validation: no network,
    no codegen, no execution. Anything unsupported raises McpSchemaError
    (fail closed)."""
    if not isinstance(schema, dict):
        raise McpSchemaError("inputSchema must be an object")
    raw = canonical_args(schema)
    if len(raw.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise McpSchemaError(f"inputSchema exceeds {MAX_SCHEMA_BYTES} bytes — rejected")
    _validate_schema_node(schema, depth=0, path="$")
    return args_sha256(schema)
