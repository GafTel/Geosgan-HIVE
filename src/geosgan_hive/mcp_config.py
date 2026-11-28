"""M1 (v0.9.0): MCP registry configuration contracts.

Configuration-only registry (no durable storage; schema v13 is preserved).
Everything here is OPERATOR-OWNED configuration, strictly validated with
pydantic (extra="forbid" everywhere, frozen models so snapshots are immutable):

- MCP is globally disabled by default; every server is disabled by default;
  remote (streamable_http) transport additionally requires an explicit enabled
  server AND an operator-allowlisted HTTPS endpoint rule.
- Server annotations reported by an MCP server at runtime are UNTRUSTED hints
  and are never read from this module: every permitted remote tool carries an
  explicit operator-owned `side_effect` classification. Unknown, missing or
  conflicting classifications fail closed at the registry (mcp_registry.py).
- Financial eligibility is explicit: `cost_class` is LOCAL only with a local
  execution attestation, VERIFIED_ZERO_COST only with a zero-cost attestation,
  UNKNOWN (the default) fails closed for any operation, PAID is always denied
  at dispatch in the zero-cost release. An HTTP 200, a free SDK, a free
  protocol or operator authorization never establish zero-cost eligibility.
- Secret VALUES never appear in configuration: stdio transports reference
  environment variables by NAME only (`env_refs`); a validator rejects any
  string field that looks like a pasted credential (fail closed).
- Bounded sizes: max servers, max tools per server, bounded strings/args/
  schemas. No network, no subprocess and no package installation happens
  anywhere in this module or at registry construction.

Supported protocol revision: 2025-06-18 only (see mcp_contracts). Unsupported
revisions are rejected, never approximated.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ConfigError

#: the only MCP protocol revision supported by v0.9.0 (verified against the
#: official specification; later revisions are rejected, not approximated)
SUPPORTED_PROTOCOL_REVISION = "2025-06-18"

SERVER_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,31}$"
TOOL_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
ENDPOINT_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
ENV_REF_PATTERN = r"^[A-Z_][A-Z0-9_]{0,63}$"
SHA256_HEX = r"^[0-9a-f]{64}$"

MAX_SERVERS = 16
MAX_TOOLS_PER_SERVER = 64
MAX_DESCRIPTION_CHARS = 2_048
MAX_TOOL_DESCRIPTION_CHARS = 4_096
MAX_TRANSPORT_ARGS = 16
MAX_ARG_CHARS = 512
MAX_ENV_REFS = 16
MAX_URL_PATH_CHARS = 256
#: hard cap for a standalone mcp config file (load_mcp_config_file)
MAX_MCP_CONFIG_BYTES = 256 * 1024

#: values that look like pasted credentials are rejected anywhere in the config
SECRET_LIKE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
)
#: characters that must never appear in an approved executable path or argv
#: (no shell is ever used, but reject metacharacters defensively)
_EXECUTABLE_FORBIDDEN = re.compile(r"[|&;$`><\"'\n\r\t\x00]")


class McpTransport(StrEnum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class McpSideEffect(StrEnum):
    """Operator-owned side-effect classification of a remote tool. Server-
    reported annotations are hints only and never set this value."""

    READ_ONLY = "read_only"
    LOCAL_MUTATION = "local_mutation"
    EXTERNAL_MUTATION = "external_mutation"
    FORBIDDEN = "forbidden"  # never executable; human approval cannot override


class McpCostClass(StrEnum):
    """Financial eligibility contract (amendment D). Severity order:
    LOCAL < VERIFIED_ZERO_COST < UNKNOWN < PAID."""

    LOCAL = "local"
    VERIFIED_ZERO_COST = "verified_zero_cost"
    UNKNOWN = "unknown"
    PAID = "paid"


_COST_SEVERITY = {
    McpCostClass.LOCAL: 0,
    McpCostClass.VERIFIED_ZERO_COST: 1,
    McpCostClass.UNKNOWN: 2,
    McpCostClass.PAID: 3,
}


def cost_severity(cost: McpCostClass) -> int:
    return _COST_SEVERITY[cost]


class _McpModel(BaseModel):
    """Base: unknown fields rejected, immutable (frozen) snapshots, and every
    model (not just the root) rejects secret-looking values (F1). The error
    never includes the offending value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _scan_secret_like(self) -> _McpModel:
        _reject_secret_like(self)
        return self


class McpStdioTransport(_McpModel):
    """Local stdio transport contract. The executable must be an explicit,
    absolute, operator-approved path; arguments are fixed; NO shell, NO npx /
    uvx / package launchers, NO downloads. `env_refs` are environment-variable
    NAMES (references) whose values are resolved only by the M2+ client inside
    a sanitized environment — secret values never appear in configuration.

    Trust boundary (amendment C): an approved executable is NOT automatically a
    sandboxed executable — a same-user subprocess can otherwise reach host
    files, credentials, local services and paid APIs. Production stdio
    activation stays blocked until the M2 isolation guarantees are enforceable;
    this module only carries the contract."""

    kind: Literal["stdio"] = "stdio"
    executable: str = Field(min_length=1, max_length=512)
    args: list[str] = Field(default_factory=list, max_length=MAX_TRANSPORT_ARGS)
    env_refs: list[str] = Field(default_factory=list, max_length=MAX_ENV_REFS)
    cwd: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _check_stdio(self) -> McpStdioTransport:
        if not os.path.isabs(self.executable):
            raise ValueError(
                f"stdio executable must be an absolute path (no PATH lookup): "
                f"{self.executable!r}"
            )
        if _EXECUTABLE_FORBIDDEN.search(self.executable):
            raise ValueError("stdio executable path contains forbidden characters")
        base = os.path.basename(self.executable).lower()
        if base in ("sh", "bash", "zsh", "cmd", "powershell", "pwsh", "npx", "uvx",
                    "uv", "pip", "pip3", "python", "python3"):
            raise ValueError(
                f"stdio executable {base!r} is a shell/launcher/interpreter — rejected; "
                "approved MCP servers must be standalone executables"
            )
        for a in self.args:
            if len(a) > MAX_ARG_CHARS:
                raise ValueError("stdio argument too long")
            if "\x00" in a or "\n" in a or "\r" in a:
                raise ValueError("stdio argument contains control characters")
        if len(set(self.env_refs)) != len(self.env_refs):
            raise ValueError(f"duplicate env_refs: {sorted(self.env_refs)}")
        for ref in self.env_refs:
            if not re.fullmatch(ENV_REF_PATTERN, ref):
                raise ValueError(f"invalid env reference {ref!r} (names only, never values)")
        if self.cwd is not None and not os.path.isabs(self.cwd):
            raise ValueError("stdio cwd must be an absolute path")
        return self


class McpHttpTransport(_McpModel):
    """Remote Streamable HTTP transport contract. Disabled by default; M2 will
    require the endpoint to be an operator-allowlisted HTTPS endpoint rule and
    route ALL traffic through the existing SecureHttpClient/ApiSession (pinned
    destinations, TLS, no redirects/proxies, durable reservations, deadlines,
    response caps, lease fencing). No arbitrary headers are configurable."""

    kind: Literal["streamable_http"] = "streamable_http"
    endpoint_id: str = Field(pattern=ENDPOINT_ID_PATTERN)
    url_path: str = Field(default="/mcp", min_length=1, max_length=MAX_URL_PATH_CHARS)

    @model_validator(mode="after")
    def _check_http(self) -> McpHttpTransport:
        if not self.url_path.startswith("/"):
            raise ValueError("url_path must be absolute")
        if ".." in self.url_path.split("/") or "\\" in self.url_path or "%" in self.url_path:
            raise ValueError(f"invalid url_path {self.url_path!r}")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in self.url_path):
            raise ValueError("url_path contains control characters")
        return self


class McpToolPolicy(_McpModel):
    """Operator-owned classification of ONE permitted remote tool. Server-
    reported annotations never set any field here. `input_schema_sha256` may
    pin the canonical digest of the tool's input schema (computed by
    mcp_contracts.validate_input_schema); a mismatch at discovery time is a
    capability CHANGE and invalidates previous authorization (fail closed)."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    side_effect: McpSideEffect
    cost_class: McpCostClass | None = None  # None -> inherit the server's class
    input_schema_sha256: str | None = Field(default=None, pattern=SHA256_HEX)
    requires_approval: bool | None = None  # None -> derived from side_effect
    max_output_bytes: int = Field(default=100_000, ge=1, le=10 << 20)
    #: untrusted display text; NEVER used for authorization, identity or digests
    description: str = Field(default="", max_length=MAX_TOOL_DESCRIPTION_CHARS)


class McpServerConfig(_McpModel):
    """One registered MCP server. Identity = server_id + canonical config
    digest (mcp_contracts.server_identity_digest). `enabled` defaults to False;
    even an enabled stdio server is not executable until M2+ isolation lands."""

    server_id: str = Field(pattern=SERVER_ID_PATTERN)
    enabled: bool = False
    protocol_revision: Literal["2025-06-18"] = SUPPORTED_PROTOCOL_REVISION
    transport: Annotated[McpStdioTransport | McpHttpTransport, Field(discriminator="kind")]
    cost_class: McpCostClass = McpCostClass.UNKNOWN
    #: operator attests the server performs ONLY local execution (no network /
    #: paid calls from inside the subprocess). Required for cost_class=LOCAL.
    #: Residual risk documented (amendment C): attestation is not a sandbox.
    local_execution_attested: bool = False
    #: operator attests all of the server's external operations are verified
    #: non-chargeable. Required for cost_class=VERIFIED_ZERO_COST. An HTTP 200
    #: or a free SDK never satisfies this.
    zero_cost_attested: bool = False
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    tools: list[McpToolPolicy] = Field(default_factory=list, max_length=MAX_TOOLS_PER_SERVER)

    @model_validator(mode="after")
    def _check_server(self) -> McpServerConfig:
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names in server {self.server_id!r}: "
                             f"{sorted(names)}")
        if self.cost_class is McpCostClass.LOCAL and not self.local_execution_attested:
            raise ValueError(
                f"server {self.server_id!r}: cost_class=local requires "
                "local_execution_attested=true (explicit operator attestation)"
            )
        if self.cost_class is McpCostClass.VERIFIED_ZERO_COST and not self.zero_cost_attested:
            raise ValueError(
                f"server {self.server_id!r}: cost_class=verified_zero_cost requires "
                "zero_cost_attested=true (explicit operator attestation; HTTP 200 / free "
                "SDK do not qualify)"
            )
        server_sev = cost_severity(self.cost_class)
        for t in self.tools:
            if t.cost_class is not None and cost_severity(t.cost_class) < server_sev:
                raise ValueError(
                    f"tool {t.name!r} in server {self.server_id!r} claims a LESS "
                    f"conservative cost class ({t.cost_class.value}) than the server "
                    f"({self.cost_class.value}) — rejected"
                )
        return self


class McpConfig(_McpModel):
    """MCP registry configuration (configuration-only; no durable storage, schema
    v13 preserved). Globally disabled by default."""

    enabled: bool = False
    servers: list[McpServerConfig] = Field(default_factory=list, max_length=MAX_SERVERS)

    @model_validator(mode="after")
    def _check_config(self) -> McpConfig:
        ids = [s.server_id for s in self.servers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate server ids: {sorted(ids)}")
        return self


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)
    elif isinstance(obj, BaseModel):
        yield from _iter_strings(obj.model_dump())


def _reject_secret_like(model: BaseModel) -> None:
    """Fail closed if any string in the model looks like a pasted credential.
    The offending value is NEVER included in the error (F1)."""
    for s in _iter_strings(model):
        for pat in SECRET_LIKE_PATTERNS:
            if pat.search(s):
                raise ValueError(
                    "configuration contains a secret-looking value (redacted) — "
                    "credentials are referenced by environment-variable NAME only and "
                    "are never stored in config files"
                )


def load_mcp_config_file(path: str, *, max_bytes: int = MAX_MCP_CONFIG_BYTES) -> McpConfig:
    """Load a standalone MCP registry YAML file. Bounded size, safe YAML only,
    strict validation. Pure local file read — no network, no subprocess."""
    import pathlib

    import yaml

    p = pathlib.Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read MCP config file {p}: {exc}") from exc
    if len(raw) > max_bytes:
        raise ConfigError(
            f"MCP config file {p} is {len(raw)} bytes > cap {max_bytes} — rejected"
        )
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"MCP config file {p} must contain a YAML mapping")
    # accept either the bare registry block or a file that wraps it in a single
    # top-level `mcp:` key (the documented example-file shape)
    if set(data) == {"mcp"} and isinstance(data["mcp"], dict):
        data = data["mcp"]
    try:
        return McpConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid MCP configuration in {p}:\n{exc}") from exc
