# MCP & API Engine — M1: Registry, Contracts, Trust Policies (v0.9.0)

Branch: `feat/mcp-api-engine` (base: main @ `6b86c5d`, v0.8.0, schema v13).
M1 implements **only** the MCP registry, configuration contracts, trust policies
and capability-identity primitives. No MCP connection, subprocess, tool
registration, network activity or schema migration exists in M1. The optional
MCP SDK is **not** a dependency yet (M2 decision, subject to the M0 amendments).

## 1. Modules

| Module | Contents |
|---|---|
| `src/geosgan_hive/mcp_config.py` | Strict frozen pydantic config models (`McpConfig`, `McpServerConfig`, `McpToolPolicy`, `McpStdioTransport`, `McpHttpTransport`), enums (`McpTransport`, `McpSideEffect`, `McpCostClass`), size caps, secret-like-value rejection, `load_mcp_config_file()` (bounded YAML, unwraps a single top-level `mcp:` key) |
| `src/geosgan_hive/mcp_contracts.py` | `POLICY_VERSION`, `SUPPORTED_PROTOCOL_REVISIONS = ("2025-06-18",)`, namespace mapping `mcp_<server_id>__<tool_name>`, `server_identity_digest`, `transport_identity`, frozen `ToolCapability` + `binding_digest()`, `DispatchDecision` + `dispatch_decision()` rules, JSON-Schema safety subset `validate_input_schema()` |
| `src/geosgan_hive/mcp_registry.py` | `McpRegistry` (immutable; pure construction — no network/subprocess), `from_hive_config` / `from_file`, `summary()` (secret-free), `server()`/`tool()` fail-closed lookups, `bind_capabilities()` (capability-change invalidation), `dispatch_decision()` |
| `agent_config.py` | `HiveConfig.mcp: McpConfig | None = None` + cross-validation: every `streamable_http` server must reference an endpoint of an **enabled** `apis:` provider |
| `config/mcp.example.yaml` | Documented opt-in example (all disabled) |
| `tests/test_mcp_m1.py` | 46 tests covering the full M1 acceptance list + adversarial cases |

## 2. Defaults (all deny)

- `McpConfig.enabled = False` (global kill switch).
- Every server `enabled = False`.
- Remote transport additionally requires an enabled HTTPS API endpoint.
- `cost_class = unknown` (default) → every operation denied at dispatch.
- No MCP tools are registered on the ToolGateway catalog in M1 (tested).
- Schema **v13 unchanged** — the registry is configuration-only by design; if a
  durable-state requirement is demonstrated later, a migration request will be
  made explicitly (not implemented silently).

## 3. Amendment A — operator-owned tool classification

- MCP server annotations (`readOnlyHint`, `destructiveHint`, …) are recorded
  **nowhere** and never influence anything; only the operator's `side_effect`
  per tool counts (test: a server claiming `readOnlyHint=true` on an
  `external_mutation` tool stays classified as external mutation and is denied
  over stdio).
- Unknown tool names reported by a server are rejected — never auto-registered
  as executable (or any) tools.
- `forbidden` is never executable; **human approval cannot override** DENIED
  decisions — approval only gates `ALLOWED_WITH_APPROVAL` (mutations).
- A server can never grant itself read-only status, extra permissions or
  financial authorization: classification lives only in operator config.

## 4. Amendment B — identity and capability integrity

- Server identity = `server_id` + `server_identity_digest` (canonical sha256
  over the whole server config except the volatile `enabled` switch and the
  untrusted `description`).
- Transport identity = sha256 over executable+argv+env-ref **names** (stdio)
  or endpoint id + url path (streamable_http).
- Every tool binds to a `ToolCapability` digest over: server identity,
  transport identity, protocol revision, tool name, namespaced name,
  canonical input-schema digest, operator side-effect class, effective cost
  class, approval requirement, output cap, policy version.
- **Capability changes invalidate previous authorization**: unknown/missing/
  duplicate reported tools, invalid schemas, or a pinned-schema digest
  mismatch raise and the old bindings are void. Dispatch layers (M2+) must
  bind approvals and executions to the binding digest and reject stale ones.
- Namespaces are deterministic ASCII (`mcp_<server>__<tool>`, both parts
  charset-restricted); duplicate/ambiguous mappings (e.g. server `a` tool
  `b__c` vs server `a__b` tool `c`) and collisions with reserved/builtin tool
  names are rejected at construction.

## 5. Amendment C — local subprocess trust boundary (documented)

Resource limits, sanitized environments and process groups are **not**
complete filesystem or network isolation: a same-user MCP subprocess may
otherwise access host files, credentials, local services and paid APIs. An
operator-approved executable is **not** automatically a sandboxed executable.
Evaluated and rejected at config level: relative/PATH executables, shells and
launchers (`sh`, `bash`, `npx`, `uvx`, `uv`, `pip`, `python*`, …), shell
metacharacters, control characters in argv, non-absolute cwd. Still open by
construction (documented, not hidden): executable substitution, symlinks,
inherited descriptors, child processes, package-launcher risks. Therefore
**production stdio activation remains blocked** until M2 isolation guarantees
are enforceable; M1 carries only the contract. Windows/macOS fail-closed
behavior is unchanged (POSIX-only isolation, no weakening).

## 6. Amendment D — financial eligibility

`McpCostClass`: `local` (requires `local_execution_attested`), 
`verified_zero_cost` (requires `zero_cost_attested`), `unknown` (default →
denied), `paid` (always denied in the zero-cost release). Tool-level classes
may only be *more* conservative than the server's. External mutations
additionally require a remote transport + verified zero cost (opaque local
subprocess effects cannot be verified). Distinctions preserved: free software
≠ local execution ≠ verified zero-cost external operation ≠ unknown-cost ≠
paid. Operator authorization, HTTP 200, free SDK/protocol never establish
zero-cost eligibility. `ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)` is
untouched (regression-tested). MCP must not become an indirect path to paid
inference/APIs — if trustworthy financial enforcement cannot be demonstrated
in M2+, the capability stays disabled.

## 7. Amendment E — secure transport feasibility (M2 gate)

M2 must verify the selected SDK can operate **exclusively** through the
approved transport without bypassing destination pinning, TLS verification,
endpoint authorization, redirect/proxy restrictions, durable reservations,
request deadlines, response limits or lease fencing. If secure Streamable
HTTP integration is not feasible, remote MCP stays disabled. No insecure
protocol approximations.

## 8. Amendment G — JSON Schema safety

`validate_input_schema()` supports a small subset only: root object with
`properties`/`required`/`additionalProperties: false` (≤64 props, required
must exist in properties), string/number/integer/array/boolean/null types,
bounded `enum`/`const`/`minLength`/`maxLength`/`pattern`/`minimum`/`maximum`/
`minItems`/`maxItems`, annotation-only `title`/`description`/`format`/`default`,
`$schema` only as the 2020-12 URI annotation. Rejected (fail closed):
`$ref`/`$defs`/combinators/conditionals/`patternProperties`/unknown keywords,
depth > 5, > 64 KB schemas, patterns > 256 chars or with backrefs/lookarounds/
nested quantifiers, arrays > 10 000 items, strings > 1 MiB. Validation is
pure structural + `re.compile` — no network, no codegen, no execution.
Anything outside the subset is unsupported, not weakened.

## 9. Protocol and SDK evaluation (verified against official docs, M0)

- MCP revisions: 2024-11-05, 2025-03-26 (Streamable HTTP introduced), 2025-06-18
  (structured output, elicitation, no batching, `MCP-Protocol-Version`),
  2025-11-25 (JSON Schema 2020-12), 2026-07-28 (current; stateless core, no
  initialize handshake). **v0.9.0 targets 2025-06-18 only**; other revisions
  are rejected by config (`Literal["2025-06-18"]`), never advertised or
  approximated.
- Official Python SDK `mcp` (MIT, py≥3.10): v1.x line (`>=1.28,<2`) implements
  ≤2025-11-25 incl. 2025-06-18; v2.x targets 2026-07-28. Planned as an
  **optional extra**, lazy-loaded, with all HTTP carried by our
  `SecureHttpClient` (custom transport) — one production HTTP stack. SDK
  custom-transport support must be verified in M2 (amendment E); if it cannot
  guarantee the transport properties, remote MCP stays disabled and stdio
  gets a stdlib JSON-RPC fallback.
- Unsupported in v0.9.0 by design: OAuth client flows, server→client
  sampling/elicitation/roots, tasks/MCP Apps (2026-07-28), legacy HTTP+SSE,
  JSON-RPC batching.

## 10. Dependency reproducibility proposal (no changes made in M1)

- Supported: Python 3.11/3.12/3.13; Linux (full), Windows (fail-closed document/
  parser paths; stdio MCP blocked until Job-Object isolation lands), macOS
  (POSIX rlimits/process groups OK; no /proc fd re-verification → existing
  fail-closed rules apply).
- Current deps: core = litellm/pydantic/pydantic-settings/PyYAML (all MIT);
  extras dev (pytest, ruff — MIT) and docs (pypdf BSD-3-Clause, python-docx
  MIT, openpyxl MIT, reportlab BSD-style, defusedxml PSFL). No platform-specific
  dependencies today.
- Proposal (needs separate authorization before any change): add
  `constraints/` pinned files per Python version covering **direct +
  transitive** deps (generated from a known-good resolve, e.g. `pip freeze` of
  `.[dev,docs]`), optionally with `--require-hashes` (hash-pinned requirements
  are feasible for pure-Python wheels/sdists; evaluate platform wheels), used
  by CI for a reproducible job while keeping the default unpinned job; license
  allowlist check (MIT/BSD-3-Clause/Apache-2.0/PSFL) in CI; documented security
  update procedure (review upstream advisories → bump constraints → full suite
  + CI before merge). Until then, note that CI resolves from ranges and
  versions float (observed: pydantic 2.14.3 emits new third-party warnings —
  warnings are NOT suppressed per M0 decision).

## 11. What M1 deliberately does NOT do

No MCP client/transport, no subprocess spawn, no SDK dependency, no agent-
facing MCP tools, no remote API activation, no FastAPI server, no schema
migration, no lockfile, no PR/merge. Registry construction performs zero I/O
(monkeypatched socket/subprocess test).

## 12. Tests

`tests/test_mcp_m1.py`: 46 tests — the 22 M1 acceptance items (defaults,
unknown fields/servers/tools, duplicates, namespace collisions, annotation
non-authority, classification fail-closed, unknown/paid cost denial,
approval-cannot-override, deterministic digests, capability invalidation,
snapshot immutability, secret-reference privacy, no model-controlled
registration, size bounds, protocol pinning, existing-config compatibility,
no-I/O construction, SDK independence, zero-cost gate, no agent-facing tools)
plus adversarial cases (launcher executables, env-ref abuse, cost-class
downgrade attempts, endpoint cross-validation, schema-subset attacks, deep
nesting, unsafe regexes, dispatch matrix).

## 13. Remaining risks (honest)

- Config-only registry means no durable capability history; cross-session
  change detection needs either operator-pinned schema digests (supported) or a
  later v14 migration (would require explicit authorization).
- Attestations (`local_execution_attested`, `zero_cost_attested`) are operator
  claims, not proofs — documented residual risk (amendments C/D).
- Prompt injection through MCP content remains possible by nature; M2+ must
  label/bound/redact all server-returned content.
- The 2026-07-28 revision and OAuth flows are out of scope until the ecosystem
  and our secure-transport verification mature.
