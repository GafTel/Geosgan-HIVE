# GEOSGAN HIVE v0.9.0 — M2: Secure MCP Client and Transports (EXPERIMENTAL)

Milestone M2 of the MCP & API Engine. Builds on M1 (`docs/mcp-api-engine-m1.md`):
the registry/config/contracts layer is unchanged in semantics; M2 adds the
client, the secure transports and runtime argument validation.

**Status: experimental infrastructure only.** Real external MCP dispatch stays
BLOCKED (`REMOTE_DISPATCH_ENABLED = False`); no MCP tool is registered on the
production agent catalog; no version bump; schema v13 unchanged.

## 1. Scope and kill switches

- `REMOTE_DISPATCH_ENABLED = False` (module constant, `mcp_client.py`). The
  manager refuses any `streamable_http` session unless the caller explicitly
  passes `allow_remote_dispatch=True` — used **only** by offline loopback test
  fixtures, mirroring the transport's `allow_loopback` test-only pattern.
- stdio sessions are constructible in tests (trusted fixed fixtures), but
  production stdio activation remains blocked (see §6 trust boundary).
- `tool_factory` is untouched: `build_tool_catalog()` exposes no `mcp_*` tools
  (regression-tested).

## 2. Optional SDK dependency (amendment B verification)

- Extra: `pip install -e '.[mcp]'` → `mcp = ["mcp>=1.30,<2"]` (pyproject), MIT,
  `Requires-Python: >=3.10` (covers the supported 3.11–3.13).
- Pin rationale (advisories on the `mcp` PyPI package): CVE-2025-53366 (fixed
  1.9.4), CVE-2026-115563 OAuth credential leak (GHSA-qx49-fqc8-xw99, fixed
  1.30.0), CVE-2026-59951 unbounded session accumulation (GHSA-84m7-p3x7-pcfv,
  fixed 1.30.0), CVE-2025-66416 DNS rebinding (server-side; our client never
  uses the SDK server). `>=1.30,<2` selects the 1.x security-maintenance line.
- `check_sdk()` lazy-imports `mcp` and enforces the pin at runtime
  (`McpUnavailableError` outside the range or when missing). Core installs and
  imports work without the extra (subprocess-tested with `mcp` blocked).
- **Protocol pin**: `MCP_PROTOCOL_REVISION = "2025-06-18"`. The SDK's own
  `LATEST_PROTOCOL_VERSION` is **never** used (in mcp 1.30 it is `2025-11-25`).
  The `initialize` request is sent **manually** via `ClientSession.send_request`
  with `protocolVersion="2025-06-18"`; any other server-negotiated revision is
  rejected (`McpProtocolError`) — downgrade/upgrade both fail closed.
- SDK 1.30 API note: `streamablehttp_client(httpx_client_factory=...)` is
  deprecated (its `timeout`/`sse_read_timeout` params are ignored). M2 uses
  `streamable_http_client(url, http_client=<pre-built client>,
  terminate_on_close=True)`. A provided `http_client` is **not** closed by the
  SDK — `McpSession.close()` owns `aclose()`.
- Direct + transitive deps of mcp 1.30.0 (all already present via litellm or
  added by the extra; licenses MIT/BSD/Apache-2.0/PSF-compatible):
  anyio 4.16.0, httpx 0.28.1, httpx-sse 0.4.3, jsonschema 4.26.0 (+attrs,
  jsonschema-specifications, referencing, rpds-py), pydantic 2.14.3,
  pydantic-settings 2.16.3, pyjwt, python-multipart 0.0.33, sse-starlette 3.7.0,
  starlette 1.10.0, typing-extensions, typing-inspection, uvicorn 0.54.0,
  certifi/httpcore/h11/idna/sniffio. No new paid dependency or service.
  `jsonschema` (Draft 2020-12) is used lazily by `mcp_args` for argument
  validation — never at core import.
- CI installs `.[dev,docs,mcp]` on all four jobs (Ubuntu 3.11/3.12/3.13 +
  Windows 3.12); the suite `importorskip`s `mcp`, so a broken extra fails
  loudly, not silently.

## 3. Secure Streamable-HTTP transport (amendment E)

Single HTTP stack — no parallel engine, no SDK default transport:

```
SDK streamable_http_client
  -> httpx.AsyncClient(transport=HiveMcpHttpTransport, follow_redirects=False,
                        timeout=httpx.Timeout(op_timeout))
  -> HiveMcpHttpTransport.handle_async_request            (mcp_transport.py)
       cancel check, request-body cap, duplicate-query rejection,
       header allowlist (only mcp-protocol-version / mcp-session-id pass),
       cumulative per-session caps BEFORE any reservation,
       sync engine in a worker thread:
  -> ApiSession.request(headers=...)                      (outbound.py, M2: +headers)
       outbound authorization -> durable v9 reservation -> pre-dispatch
       revalidation -> single-use approval consume -> lease-fenced dispatch
       mark -> bounded audit (every physical request: initialize, discovery,
       notifications, tool calls, GET SSE, DELETE, retries)
  -> SecureHttpClient.request
       endpoint rule, pinned numeric destination, connected-peer verification,
       TLS hostname/SNI, SSRF classes, redirect+proxy rejection (3xx+Location
       raises), wire/decompression caps, total deadline, cancellation
```

- Redirects: the client sets `follow_redirects=False` **and** SecureHttpClient
  raises `RedirectRejectedError` on any 3xx+Location, so the SDK's own
  within-origin redirect following can never trigger. No proxies (no env
  proxies, no pooling).
- Headers: only `MCP-Protocol-Version` / `mcp-session-id` are forwarded per
  request; `SecureHttpClient._build_headers` validates every extra key against
  the endpoint rule's `extra_header_keys` allowlist and rejects control
  characters. Rule-fixed headers (Accept/Content-Type/User-Agent) are set by
  the transport layer. No credential forwarding (no OAuth in v0.9.0; `auth`
  is never attached).
- Session identifiers: the server-issued `mcp-session-id` is treated as
  sensitive — only its **sha256** is stored (`McpSessionState.session_id_hash`),
  bound to (server identity digest, endpoint, run); any substitution on a later
  response is rejected (`McpSessionError`). Raw ids are never logged/stored.
- Cumulative per-session caps (`McpServerConfig`): `max_session_requests` (64),
  `max_session_wire_bytes` (16 MiB), `session_deadline_seconds` (300), enforced
  in the transport on top of the durable per-request budgets.
- Responses are buffered by SecureHttpClient under its wire/decompression caps
  and served to the SDK as a bounded byte stream (`_BodyStream`), so SSE
  responses parse through the same secure path (demonstrated in tests). A
  long-lived GET stream is cut at the transport total deadline — **no unbounded
  SSE, no unbounded reconnects** (the SDK's GET reconnect loop is bounded by
  `MAX_RECONNECTION_ATTEMPTS=2` and each attempt is a reserved, audited
  request).
- Cancellation: `McpSession.close()` sets a shared cancel event; the transport
  refuses further requests, the SDK task group is cancelled, the session DELETE
  is attempted (best effort), streams and the httpx client are closed — all
  inside a **shielded** cancel scope so cleanup cannot be interrupted by the
  very cancellation it handles.
- Error translation: the SDK runs each POST in a task-group child, so a
  transport failure surfaces to the awaiter as a cancellation; the transport
  records the first HIVE error (`McpSessionState.last_error`) and
  `McpSessionManager._sdk_call` re-raises it (fail closed, type-preserving:
  e.g. `QuotaExhaustedError`, `OutboundNotAuthorizedError`,
  `ResponseLimitExceededError`, substitution `McpSessionError`).

## 4. Capability integrity and dispatch (amendments A/B/E)

- After manual initialize, `tools/list` is bound through
  `McpRegistry.bind_capabilities`: unknown/duplicate tools, out-of-subset or
  drifted `inputSchema` (vs the operator-pinned `input_schema_sha256`), and
  out-of-subset `outputSchema` all fail closed (`McpRegistryError`). Server
  `annotations` (readOnlyHint etc.) are never authorization evidence.
- `tools/list_changed` (or any stale flag) forces a re-bind before the next
  call; a vanished tool fails closed.
- Before every dispatch: lease revalidation (`lease_lookup(run_id)` owner +
  generation must match — stale workers rejected), `dispatch_decision` (DENIED
  raises; `ALLOWED_WITH_APPROVAL` is **blocked in M2** — human approval is not
  auto-granted), then actual arguments are validated by `validate_tool_args`
  (amendment G + F5 runtime): schema subset → 64 KiB byte cap → finite/bounded
  numeric walk (rejects inf/nan/>1e15, depth>16, >512 keys) →
  `jsonschema.Draft202012Validator`. No network, no codegen.
- Tool results are UNTRUSTED: bounded by `max_output_bytes`, secret-redacted
  (`redact_secrets`), never interpreted; the returned payload carries the
  `binding_digest` and an explicit "provenance only" note.
- Server-initiated sampling/elicitation/roots keep the SDK default "not
  supported" answers; `ClientCapabilities()` advertises none of them.

## 5. stdio transport (experimental; amendment C/D)

`mcp_stdio.py` spawns `python -I -S -c <bootstrap>` (no `.pth` execution, like
the W1 worker) which applies rlimits FIRST (fail-closed exit 3 if any fails),
sets `PR_SET_PDEATHSIG(SIGKILL)` (survives `execv`; kernel kills the child if
the HIVE parent dies), then `execv`s the operator-approved executable.
`start_new_session=True` + `killpg` on deadline/cancel; a parent-side watchdog
enforces the wall-clock deadline.

- Executable identity (`check_executable`): absolute path only, launcher/
  interpreter denylist by basename FIRST (sh/bash/npx/uvx/python/...), lstat
  rejects symlinks, regular file only, not group/world-writable.
- Env: minimal sanitized dict; only allowlisted `env_refs` NAMES are resolved
  from the parent environment (values never stored in config; tested:
  `SECRET_TOKEN` never reaches the child, `FIXTURE_ALLOWED` does).
- stderr → bounded file (1 MiB), never parsed. stdio framing: newline-delimited
  JSON-RPC, 4 MiB per-line cap, ≤64 pending messages, malformed frames fail
  closed (stream error, never executed).
- Non-POSIX: **fail closed** (`McpStdioError`); Windows Job Objects are a
  documented future path, no insecure fallback.

**Honest trust boundary (amendment C):** rlimits, the sanitized environment and
process groups are NOT filesystem or network isolation. The in-process network
guard in the bootstrap covers only the pre-`execv` window — it does NOT survive
exec (a fresh interpreter image is loaded). After exec the child is resource-
limited and killable as a group, but a same-user process can still reach host
files, credentials, local services or paid APIs. Production stdio activation
therefore remains BLOCKED.

## 6. Unsupported / not implemented in v0.9.0

- OAuth/authorization flows with MCP servers (no `auth` ever attached).
- Protocol revisions other than 2025-06-18 (incl. 2025-11-25, 2026-07-28).
- Server-initiated sampling, elicitation, roots, tasks (2026 features).
- Legacy HTTP+SSE transport, JSON-RPC batching, resumability (Last-Event-ID).
- True long-lived server→client SSE streams (GET stream cut at deadline).
- MCP tools in the production agent catalog; MCP-driven human approval flow.
- Automatic replay of uncertain external effects (dispatch marks are
  lease-fenced and single-use; reservations that cannot dispatch are
  reconciliation-only).

## 7. Warnings report (16, unchanged, NOT suppressed)

- 15 × `UserWarning` from pydantic's TypedDict `ReadOnly` qualifier handling
  inside **litellm's** `_OllamaGenerateResponse` / `_DeviceCode` /
  `_AccessTokenPoll` TypedDicts, surfaced while
  `test_a0_transport.py::test_dns_rebinding_at_connect_time_is_defeated`
  builds its models. Severity: informational (upstream library noise; no
  mutation-protection guarantee is relied upon). Remediation: none required
  for v0.9.0; revisit if litellm/pydantic change semantics.
- 1 × `UserWarning: Duplicate name: 'a.xml'` from `zipfile` in
  `test_documents.py::test_zip_guard_rejects_bomb_traversal_dupes_encrypted_bzip2`
  — the intentionally malicious duplicate-entry zip fixture.
  Severity: informational (the duplicate is what the guard test feeds on).

## 8. Verification

- `tests/test_mcp_m2.py`: 19 tests (SDK pin + missing-extra fail-closed in a
  subprocess, remote dispatch kill switch + zero-cost gate, no `mcp_` in the
  agent catalog, loopback end-to-end with protocol pin recorded server-side,
  downgrade rejection, discovery fail-closed variants, annotations never
  reclassify, session-id binding + mid-session substitution, malformed JSON /
  SSE / oversize, durable per-run budget exhaustion, lease loss before
  dispatch, cancellation, global disable, runtime arg bounds (F5),
  stdio end-to-end + env sanitization, symlink/relative/launcher rejection,
  stdio timeout group-kill, stdio malformed frame).
- Full suite: **1068 passed, 1 skipped, 16 warnings** locally (baseline after
  M1 fixes was 1049 passed + 1 skipped; +19 = M2). The 1 skip is the pre-existing
  IPv6-loopback-unavailable skip in `test_a0_transport.py`.
- `ruff check .` clean; `python -m compileall -q src tests examples` clean; all
  15 offline demos pass; fresh-process import with sockets denied OK; core
  import + `check_sdk()` fail-closed behavior with `mcp` blocked OK; secrets
  scan over changed files clean (only intentional F1 fake-secret test vectors).
- CI (`.github/workflows/ci.yml`) installs `.[dev,docs,mcp]` on all four jobs.

## 9. Zero-cost compliance

`ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)` is untouched and still the
authoritative gate (regression-tested). M2 adds no live external path: remote
dispatch is blocked by the kill switch; the only exercised transport in tests
is loopback with `allow_loopback`. A free SDK / HTTP 200 / attestation never
establishes verified zero-cost eligibility — `McpCostClass` attestations remain
operator-owned and unknown/paid classes stay denied.
