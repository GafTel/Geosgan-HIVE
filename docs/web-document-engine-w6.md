# Web & Document Engine v0.8.0 — W5 residual gate + W6 agent integration

Branch: `feat/web-document-engine`. Base: v0.7.0 (`641bd90`, schema v12 → v13 in W2).

## 1. W5 residual acceptance verdict: PASS

Audited the actual implementation and all reachable paths; defects found were fixed
failing-first (see `tests/test_docgen.py`, 40 tests).

### 1.1 Atomic approval + execution ledger
- `DocumentWriter.publish()` now performs the approval consumption AND the durable
  execution-ledger start in **one transaction** via `store.start_execution(...)`
  (never a standalone `consume_approval`). The consume is bound to approval id +
  APPROVED state + exact args hash + tool version + run + agent + tool + task +
  not-expired; the ledger insert shares the same transaction and is fenced by lease
  owner+generation. A duplicate `execution_id` aborts before any write.
- Start-before-effect: the ledger row is persisted before any filesystem effect.
- After publication + post-publish hash verification the row is marked `completed`
  (fenced by the same lease). A post-publication hash mismatch is **never** reported
  as success.
- Publication requires a run-bound lease (`run_id` + owner + generation); the
  durable bindings use the approval-bound agent, while ToolPolicy decides who may
  invoke `doc_publish` at all.
- Crash windows (documented, tested): before `start_execution` → nothing happened;
  after it → ledger row stays `started` (uncertain), approval consumed, so
  **auto-replay is impossible by construction**; a retried publish is rejected as
  `consumed` and never duplicates the mutation.

### 1.2 Filesystem publication / TOCTOU
- **Overwrite is DISABLED.** Rationale: a rename cannot atomically bind the
  pre-existing content hash, so a hash-bound overwrite cannot be made race-free
  under the documented threat model. Publication is **creation-only**.
- `WorkspaceSandbox.write_bytes_atomic` now performs an **atomic no-clobber**
  creation: parent directory opened by handle (`O_DIRECTORY` fd), tmp file created
  `O_CREAT|O_EXCL|O_NOFOLLOW` with its fd re-verified via `/proc/self/fd`, then
  **`os.link(tmp, dst, src_dir_fd, dst_dir_fd)`** — which fails with `EEXIST` if
  the destination appeared concurrently (symlink swap, racing creation) — followed
  by descriptor-relative `unlink` of the tmp name. `os.rename`/`os.replace` is never
  used for publication. Post-publish, the destination is re-opened `O_NOFOLLOW`,
  checked regular + single-link, and re-hashed.
- Remaining limitation (honest): this protects against destination substitution and
  parent-dir replacement by an attacker with workspace write access; it does NOT
  claim safety against arbitrary concurrent writers running as the same OS user
  outside the sandbox (documented threat model: enforcement is at the sandbox/
  gateway boundary, not against arbitrary privileged local processes).
- Staged-file identity: the published bytes are exactly the staged bytes re-hashed
  against the approved `output_sha256` immediately before consumption.

### 1.3 Staging lifecycle and privacy
- `.hive-staging/` (0700 dir, 0600 files) is in `DEFAULT_DENY_PATTERNS`: **no
  agent-facing file tool** (`fs_read`/`fs_list`/`fs_metadata`/`resolve`/`open_read`/
  `sha256_file`/...) can touch it; only the trusted staging methods can, through a
  containment-checked internal resolver.
- Staging contains SENSITIVE approved artifact bytes. It is bounded by operator
  config (`documents.staging_max_files`, `staging_max_bytes`,
  `staging_max_age_seconds`; `DocumentsConfig`); `stage_bytes` sweeps expired
  orphans first and then fails closed when the count/byte caps are full.
- `sweep_staging(max_age_seconds, protect=...)` removes only aged-out artifacts NOT
  referenced by a live pending/approved approval — no cleanup of another valid
  worker's active artifact. `DocumentWriter._sweep_staging()` expires stale
  approvals first and protects live ones; it runs on every `prepare`.
- Rejected/cancelled/expired approvals leave orphans that the age sweep recovers;
  failed publications after a staged-hash mismatch discard the corrupt staged file.
- Staged bytes are NEVER persisted to SQLite (metadata-only privacy verified by
  scanning the DB file for artifact markers), and no automatic persistent copies
  are created. chmod alone is not claimed to isolate against same-OS-identity
  processes — the isolation boundary is the sandbox deny + trusted-process boundary.
- If secure cross-process recovery of pending staging is not available, the pending
  approval simply expires and is swept; fresh generation + approval is required.
  Bytes are NEVER regenerated under an old approval.

### 1.4 Worker dependency trust (boundary statement)
- The worker bootstrap runs `sys.executable -I -c <hardcoded bootstrap>` with a
  sanitized env (no inherited credentials; HOME/TMPDIR inside the per-job dir) and
  `extra_paths` = [system site-packages, user-site] in that fixed order, so a
  user-site package cannot shadow a system one; `-I` skips `.pth` execution.
- Workspace paths, document contents, model arguments and environment variables
  cannot influence module resolution (tested: `syspath`/`probe` ops; job-dir/cwd/
  work-root shadow modules are not importable; `worker_blocked_modules` meta_path
  blocker proves fail-closed behavior when deps are missing).
- Trusted installation boundary: only packages installed by the operator into the
  interpreter's site dirs are trusted. We do NOT claim protection against arbitrary
  privileged Python execution or a compromised installed dependency (documented
  trusted-process boundary).

### 1.5 Generated-artifact provenance
- Artifacts register with `source_type='other'`, `provider='hive-docgen'`,
  `locator='generated:<dest>'`, `provenance.generated=true`, bound to artifact
  sha256 + spec sha256 + generator + approval/execution ids + destination.
- The `generated` marker is enforced on all read paths (get/list/cite/excerpt —
  tested); fixture relabeling remains impossible both directions; citations state
  "provenance only — not factual verification". Generated text is never presented
  as retrieved evidence and never invents citations.

## 2. W6 agent integration

### 2.1 What was added
- `src/geosgan_hive/source_tools.py`: `source_list`, `source_get`, `source_excerpt`
  — typed inputs (`extra=forbid`), lease-fenced `ReadBinding`, scope/expiry/integrity
  checks, bounded metadata-only output. Registered by `build_tool_gateway` whenever a
  durable store is wired; still **deny-by-default** through the same ToolPolicy
  (legacy mode without a `tools:` block never globally enables them — tested).
- `config/agents.webdocs.example.yaml`: opt-in example encoding the capability matrix
  (below), `tools.enabled` global allowlist, per-agent `allowed_tools`,
  `documents:` section, `apis:` web provider + `web_endpoints` policy, checkpoint
  block. All models `ollama/*` (zero-cost). Validated by tests.
- No second gateway/approval/supervisor/collaboration engine: everything reuses
  ToolGateway, ToolRegistry, ToolPolicy, ApprovalManager, RunContext,
  CollaborationEngine, SourceRegistry, WorkspaceSandbox, ModelGateway,
  CheckpointStore and the execution ledger.

### 2.2 Capability matrix (example; operators choose their own)
| agent      | tools |
|------------|-------|
| supervisor | doc_read, source_list/get/excerpt, doc_generate, doc_publish, delegation |
| researcher | web_get_page, doc_read, source_list/get/excerpt |
| developer  | calculator, doc_read, source_list/get/excerpt |
| analyst    | calculator, doc_read, doc_generate, source_list/get/excerpt |
| reviewer   | doc_read, source_list/get/excerpt (read-only; no mutations) |

Rules: an agent may use a tool only if it is in `tools.enabled` AND in its own
`allowed_tools`; explicit empty allowlists deny all; delegation never transfers
permissions (the gateway resolves the policy for the executing agent identity);
no agent can approve its own mutation (approvals are CLI-only operator decisions);
doc_publish additionally requires the exact-byte approved artifact + valid lease.

### 2.3 Approval lifecycle
- `doc_generate` = prepare: validate → generate once → stage → hash → create a
  PENDING approval bound to the exact artifact (structured runtime state in the
  approvals table, not model text). No destination write happens.
- Operator decides via CLI/store (`decide_approval`); `doc_publish` revalidates
  everything and publishes only the approved bytes (see §1.1/§1.2). Rejection,
  expiry, cancellation → no publication (tested). Gateway tool budgets/ledger
  wrap both calls; model-call semaphores are not held across human approval
  (approval is a store transition, not a suspended tool execution).

### 2.4 Untrusted content
- All web/document content is untrusted data (labeled in tool output). Tested:
  malicious document text requesting permission expansion / self-approval /
  secret exfiltration / fabricated citations has no effect (policy unchanged,
  invented `src_` ids rejected, no approvals created). Prompt labeling is
  defense-in-depth; enforcement lives in the gateway/policy/approval layers
  outside model control.

### 2.5 Privacy and routing
- Document contents are not sent anywhere automatically; model routing uses no
  document content. Zero-cost mode is unchanged: the v0.7.0 release gate
  (`ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)`) keeps ALL non-local cloud
  inference blocked regardless of agent routing or fallback (tested in W6 suite).
- Source excerpts are shared only as registry references within run/team
  visibility with current read authorization; metadata-only by default (no
  persisted content); nothing is stored as semantic memory.

### 2.6 Platform compatibility
- Parser-worker formats (PDF/DOCX/XLSX read+generate) and staging/publication are
  POSIX-only and fail closed elsewhere (Windows asserts fail-closed in CI);
  CSV/JSON/TXT/MD are bounded in-process but staging still requires fd
  re-verification. Windows Job Objects remain the documented future path; no
  `allow_insecure_fs`, no timeout-only mode. Missing optional deps → typed
  `DocumentUnavailableError`/`DocumentGenUnavailableError`, tools stay registered
  but are not advertised as usable unless the operator enables them.

## 3. Verification
- `tests/test_w6_integration.py` (13 tests, scenarios A–L) + `tests/test_docgen.py`
  (40) exercise the real gateway/approval/registry/ledger/sandbox paths on a
  deterministic loopback server (test-only transport; production config path is
  covered by `tests/test_web_retrieval.py`).
- Full suite, ruff, compileall, all offline demos (14 incl. `demo_agent_integration.py`),
  fresh-process network-denied import, secrets scan — see the delivery report for
  exact counts and the CI run URL.

## 4. Known limitations
- No general-purpose web search API is authorized (documented limitation, not a
  proven absence of free search); SearXNG stays disabled until a separately
  authorized private-instance transport policy exists.
- No OCR; no macro-enabled formats; no executable formulas in generated
  spreadsheets (documented CSV `'`-prefix policy; XLSX trigger cells stay literal).
- Overwrite publication disabled (§1.2); staging recovery is age-sweep based —
  pending approvals that cannot complete expire and are swept, then regenerated
  and re-approved.
- Worker-side watchdog is defense-in-depth; the parent-enforced deadline is
  mandatory; a worker stuck in uninterruptible kernel sleep (D state) cannot be
  killed by anyone (documented).
- Trusted-process boundary: no protection against arbitrary privileged Python
  execution, direct DB modification, or compromised installed dependencies.
