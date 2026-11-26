# Web & Document Engine v0.8.0 — W1: Architecture, Security Foundations, Threat Model

Status: W1 design + parser-isolation prototype (`parser_isolation.py`,
`_parser_worker.py`). W2 adds the source registry. W3–W8 are future,
separately authorized stages. **ZERO PAID API CONSUMPTION remains mandatory;
no live external request is made or authorized by this document or code.**

## 1. Reused infrastructure (no parallel systems)

Everything below reuses existing, audited components — no second permission,
networking, accounting, or execution system is introduced:

- `ToolGateway` / `ToolCatalog` / `ToolPolicy` / `ToolSpec` — tool registration,
  deny-by-default permissions, `SideEffect` classification (drives approval),
  per-run/agent/tool budgets, concurrency cap, execution timeouts, durable
  `tool_executions` ledger with lease fencing, `redact_secrets`.
- `ApprovalManager` — bound, single-use, expiring approvals (run/task/agent/
  tool version/canonical args/destination/policy/expiry).
- `WorkspaceSandbox` — realpath containment, deny-patterns, `O_NOFOLLOW`,
  hard-link refusal, post-open fd re-verification (`/proc/self/fd`, Linux);
  **all fs tools fail closed on platforms without fd re-verification** unless
  the operator explicitly accepts the residual risk (`allow_insecure_fs` is
  never silently enabled).
- `SecureHttpClient` — pinned numeric destinations + connected-peer check, TLS
  hostname/SNI, SSRF destination classes, redirect/proxy rejection, canonical
  path-encoding validation, size/decompression/total-deadline limits, bounded
  resolver pool, cancellation-safe semaphore.
- `OutboundAuthorizer` / `ApiSession` — DISABLED / APPROVAL_REQUIRED /
  AUTHORIZED policies, endpoint↔client binding verified at construction, durable
  v9 reservations (per-run/provider/endpoint budgets), revalidation before
  dispatch, metadata-only audit.
- `CheckpointStore` (schema v13 after W2) / `RunContext` — atomic transactions,
  lease fencing (owner + generation), replay protection, metadata-only privacy.
- `ZeroCostGate` / `live_dispatch_eligible` — v0.7.0 release gate: in zero-cost
  mode only local (`ollama`) inference may be dispatched.
- `ModelRouter` — recommends/selects only, never authorizes.

## 2. Threat model

Assets: workspace files, credentials/env, checkpoint DB, durable ledgers,
operator approvals, outbound network, agent process memory, parser workers.

Adversaries / inputs:
1. **Model output** (LLM-controlled): may propose tool names, arguments, URLs,
   file paths, citation ids. Must never reach execution/network/registration
   except through typed, validated, approval-bound tools.
2. **External content** (web pages, API responses, documents): untrusted DATA.
   Prompt injection must not grant permissions, register sources, or alter
   policy. Retrieved content is provenance, not factual verification.
3. **Malicious documents**: zip bombs, XXE payloads, malformed PDFs,
   formula-injection cells, symlink/hardlink plants, oversized inputs.
4. **Concurrent/stale workers**: replayed reservations, forged leases,
   cross-run access, destination substitution.
5. **Local privileged attacker**: out of scope (trusted-process boundary) —
   protection is against the supported public APIs/config/CLI, NOT against
   arbitrary privileged Python or direct DB writes. Documented honestly.

## 3. Trust boundaries

- **Agent/model boundary**: models call only registered tools; tool args are
  pydantic-validated (`extra='forbid'`); mutating tools require approval;
  outbound tools require outbound authorization + durable reservation.
- **Content boundary**: external/document content is data only; it can never
  register sources (only trusted retrieval/extraction runtime paths with a live
  lease can), never authorizes tools, never executes.
- **Process boundary (W1)**: untrusted documents are parsed only in isolated
  worker subprocesses, never in the main agent process.
- **Trusted-process boundary**: code with direct Python/DB access is trusted;
  private names/flags are NOT authorization mechanisms (lesson from the A2
  `_runtime_recorder` defect — fencing is lease-based, not name-based).

## 4. Parser-worker isolation — enforceable protections vs best-effort

`ParserWorkerPool` (`parser_isolation.py`) + trusted worker (`_parser_worker.py`):

**Enforceable on POSIX (tested):**
- fixed executable `sys.executable -I -c <hardcoded bootstrap>`; `shell=False`;
  no user/model-selected code or executable;
- rlimits applied inside the worker before parsing (RLIMIT_AS/CPU/FSIZE/CORE=0)
  — NOT via `preexec_fn` (unsafe in multithreaded parents); worker is
  single-threaded at that point;
- `start_new_session=True` (setsid in C) → whole process GROUP SIGKILLed on
  wall-clock deadline; per-job temp dir (0700) always removed;
- bounded IPC: length-prefixed JSON request/response caps + input/output file
  caps; bounded concurrency (worker semaphore) AND bounded pending queue
  (full = fail closed); deadline-aware slot acquisition;
- sanitized minimal env (no inherited credentials/secrets; HOME/TMPDIR inside
  the job dir);
- worker opens exactly the two per-job paths handed to it (code-level; the
  worker program is trusted).

**Best-effort (defense in depth, documented):**
- in-worker network guard (socket connect/getaddrinfo raise). A document cannot
  execute code, so the primary boundary is "untrusted content never runs
  in-process + no credentials in env"; the guard stops accidental network use
  by parser libraries. POSIX namespaces/seccomp are NOT used (privilege and
  portability) — a compromised *parser library* (not document) could in theory
  bypass the in-process guard; hence no credentials in the worker env and
  bounded rlimits are the load-bearing controls.
- OS-level orphan handling (W1 residual gate, tested): every worker runs a
  daemon **watchdog thread** armed before op execution that self-terminates
  the process via `os._exit(4)` on a wall-clock deadline (parent deadline +
  grace, capped at 1 h) and via `os._exit(5)` when `os.getppid()` no longer
  matches the parent (parent-death detection). This covers workers blocked on
  I/O or sleeping, which CPU rlimits alone do NOT terminate. The parent
  additionally kills the whole process group on its own deadline. Residual
  risk (documented, not claimed away): a process stuck in uninterruptible
  kernel sleep (D state) cannot be terminated by any userspace mechanism, and
  a watchdog thread could in theory be starved by a hostile parser library —
  which is why no credentials exist in the worker env and rlimits remain the
  load-bearing controls. We do NOT claim orphaned workers are guaranteed to
  terminate in every conceivable kernel state — only the tested watchdog +
  killpg mechanisms above.

**Fail closed:**
- non-POSIX platforms (Windows): `ParserWorkerUnavailable` on every `run()`.
  No timeout-only insecure mode, no silent `allow_insecure_fs`.
- queue full / unknown op / oversized payloads / rlimit failure (worker exit 3)
  all fail closed without spawning or with immediate kill.

### Windows compatibility assessment (future path, documented not implemented)
Windows lacks `resource.setrlimit` and POSIX process groups. The equivalent
mechanism is **Job Objects** (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
`JOB_OBJECT_LIMIT_PROCESS_MEMORY`, `JOB_OBJECT_LIMIT_ACTIVE_PROCESS`, CPU rate
control) assigned at creation via `STARTUPINFOEX`/`PROC_THREAD_ATTRIBUTE_JOB_LIST`,
plus handle-based filesystem security (no `O_NOFOLLOW`; use `FILE_FLAG_OPEN_REPARSE_POINT`
+ `GetFinalPathNameByHandle` verification, already fail-closed in the sandbox).
Until that is implemented AND tested on the Windows CI runner, parser
operations stay disabled on Windows. No insecure fallback is introduced.

## 5. Future contract — W3 secure web retrieval (design only, not implemented)

- No arbitrary model-supplied URL merely because its host is allowlisted.
  Operators define **endpoint policies**: scheme+host+port+path-prefixes+methods
  (segment-aware prefix matching, already in `http_client`), registered as
  endpoint ids bound 1:1 to authorizer config (construction-time binding check,
  as in `ApiSession._verify_endpoint_binding`).
- Each fetch is a fixed operation id with canonical, validated arguments;
  canonical request digest bound to the approval; outbound policy
  (DISABLED/APPROVAL_REQUIRED/AUTHORIZED) revalidated immediately before dispatch;
  every physical request consumes a durable v9 reservation (per-run/provider/
  endpoint budgets) and is audited metadata-only.
- No credential forwarding to arbitrary pages (credentials only to their own
  endpoint's `auth_env`, never cross-destination).
- Successful retrievals register a source in the W2 registry (trusted runtime
  path, lease-fenced) with URL, retrieval timestamp, content sha256, license/
  attribution, bounded excerpt policies.
- SearXNG stays disabled until a separately scoped private-instance transport
  policy exists (public transport rejects private/loopback destinations).
- The absence of an authorizable general-purpose free web search API is a
  CURRENT LIMITATION of this release, not a proven universal absence of free
  search services.

## 6. Future contract — W5 secure binary writes (design only, not implemented)

All document generation output passes through `WorkspaceSandbox` + ToolGateway
approval + durable ledger + exact-content binding:

1. **Generate once**: the generator produces the full binary in a secure staging
   location inside the workspace (per-run staging dir, 0700).
2. **Stage securely**: write to a unique temp name in the SAME directory as the
   destination (same filesystem → `os.replace` is atomic), fsync file AND
   directory; open the destination's parent directory by handle (POSIX
   `os.open(dir, O_RDONLY|O_DIRECTORY)`) and verify the staged file's fd path
   via `/proc/self/fd` to defeat parent-directory replacement; re-verify the
   destination is still a regular file inside the root (no symlink/hardlink:
   `O_NOFOLLOW`, `st_nlink == 1` checks before and after).
3. **Hash**: compute sha256 of the exact staged bytes.
4. **Approve**: approval binds destination relpath + existing destination
   content hash (if any) + canonical generation arguments + the exact generated
   content hash. Preview shows meaningful content, not just a hash.
5. **Publish the exact approved bytes**: single atomic `os.replace(staged,
   dest)`; after replace, re-open the destination fd and verify its sha256
   equals the approved hash (post-publish verification against overwrite races);
   stale workers (lease generation mismatch) cannot publish — the ledger
   records publication fenced by owner+generation.
6. **Cleanup**: staging file removed on any failure; orphaned staging files
   swept on recovery.

Atomic `os.replace` alone is NOT sufficient (TOCTOU on the parent directory,
symlink/hardlink plants, overwrite races) — hence directory handles, fd
re-verification, pre-publish hash binding, and post-publish verification.
Formula injection (cells starting `= + - @ \t \r`) and unsafe external links
are sanitized/omitted in writers.

## 7. Out of scope (separately authorized future stages)
OCR, arbitrary code execution, MCP, persistent semantic memory, dashboard/GUI,
deployment, live cloud inference.
