# CHANGELOG

## 0.8.0 - Web & Document Engine (branch feat/web-document-engine)

Phase 6 (W1-W8), built on v0.7.0 (main @ 641bd90). Schema v12 -> v13
(additive, transactional, rollback-tested; all historical data preserved).

- W1: trusted parser-worker isolation (POSIX): fixed `-I -S -c` bootstrap (no
  .pth execution), rlimits applied inside the worker before parsing, sanitized
  env (no inherited credentials), no shell, fixed op allowlist, bounded IPC
  both ways, bounded workers/pending queue, parent-enforced wall-clock deadline
  + process-group kill + worker-side watchdog (wall-clock + parent-death),
  per-job 0700 dirs cleaned up. Non-POSIX fails closed (Windows Job Objects =
  documented future path; no insecure/timeout-only mode).
- W2: SourceRegistry + citation integrity (schema v13): lease-fenced trusted
  origins (re-verified atomically inside the registration transaction), hashes
  computed from actual bytes, fixture/production anti-relabeling, scope/expiry/
  integrity checks on reads, metadata-only privacy (sensitive URL params,
  userinfo, fragments redacted; workspace-relative paths only), retention sweep.
- W3: secure web retrieval (`web_get_page`): disabled by default; operator
  endpoint policies (exact host/port, segment-aware path prefixes, query-param
  name allowlist, content types, size caps); canonical URL binding (traversal /
  double-encoding / userinfo / duplicate-query rejected); full A5 transport
  guarantees via SecureHttpClient + ApiSession durable budgets; stdlib
  html.parser bounded extraction (scripts/resources never fetched);
  sources registered only from actual successful retrievals. SearXNG stays
  disabled; no paid search.
- W4: local document reading (`doc_read`): PDF (pypdf), DOCX (python-docx),
  XLSX (openpyxl, read-only, formulas never evaluated), CSV/JSON/TXT/MD/HTML
  bounded; ZIP-bomb guards with actual decompression bounds + keyword prescan +
  defusedxml re-parse of every XML entry; worker-only complex formats;
  typed errors; optional deps lazy (core runs without them).
- W5: secure document generation (`doc_generate`/`doc_publish`): strict
  extra=forbid specs, worker-isolated complex formats, XML-escaped PDF text,
  literal formula-injection prevention (XLSX data_type='s', documented CSV
  `'`-prefix policy), exact-byte human approval (artifact sha256 + byte count
  bound), atomic approval-consume + ledger-start, creation-only no-clobber
  publication (dir-fd + /proc/self/fd verification + atomic link + post-publish
  re-hash), staged artifacts in .hive-staging (0700/0600, flock-serialized
  caps, terminal-state sweep, never in SQLite), generated artifacts marked
  generated=true (never retrieved evidence).
- W6: agent integration: deny-by-default per-agent tool permissions through the
  existing ToolGateway/ToolPolicy; source tools read-only + lease-fenced;
  real supervisor-driven offline orchestration; cross-agent publication only
  with exact artifact-bound human approval (identities kept distinct in audit).
- W7: operator CLI: --start-run/--run-id, --doc-read, --doc-prepare,
  --doc-status, --doc-publish (renew/takeover lease, never displaces a live
  foreign worker, no auto-approval), --capabilities (no secrets),
  --json, --sources/--source-id/--source-excerpt; exit codes 0/1/2;
  cross-process prepare->approve->publish tested via real subprocesses.
- W8: final adversarial audit + release prep. Confirmed defects fixed
  failing-first: model-supplied web extraction chars could raise the operator
  policy max (now clamped); missing destination parent now fails clean before
  the approval is consumed (no traceback leak, approval never burned); Windows
  CI skip marker on a POSIX-only worker test.

Optional deps (all permissive licenses, $0, local execution, Python 3.11-3.13):
pypdf BSD-3-Clause, python-docx MIT, openpyxl MIT, reportlab BSD-style,
defusedxml PSFL. Installed via the lazy `docs` extra; core works without them.

Zero authorized API expenditure; no live requests, model downloads or
generated-code execution in dev/tests/CI. Windows: document parse/generate/
publish fail closed by design. Known limitations: no OCR, no macros, no
executable spreadsheet formulas, no document overwrite (creation-only), no
authorized general-purpose search engine, no live non-local inference in
zero-cost mode, no GUI/MCP/persistent semantic memory yet; provenance is not
factual verification; trusted-process boundary unchanged (no defense against
arbitrary privileged local code or compromised installed dependencies).

## 0.7.0 - Free API & Model Intelligence Engine (branch feat/free-api-model-intelligence)

Phase 5 (A0-A7), built on v0.6.0 (main @ 121899b):

- A0: local-inference transport hardening (sync/async parity, env restoration,
  concurrent-gateway isolation, IPv6 loopback).
- A1: informational provider/capability catalog (never grants permissions).
- A5: SecureHttpClient (stdlib): pinned numeric destinations + connected-peer
  verification, TLS hostname/SNI, SSRF destination classes, redirect/proxy
  rejection, canonical percent-encoding validation, size/decompression/total
  deadlines, bounded resolver pool, cancellation-safe semaphore; durable API
  budgets (schema v9); outbound authorization via ToolGateway/ApprovalManager.
- A6: read-only adapters: Wikimedia, Crossref, GitHub REST, World Bank,
  Open-Meteo (non-commercial attestation default false); SearXNG disabled.
  Unicode titles / DOI special chars via canonical encoding; Crossref
  references from actual reference records.
- A2: deterministic benchmark engine; suite v2.0.0 (60 cases, en+es, 5 roles);
  evidence trust tiers; trusted_runtime only via the lease-fenced recorder.
- A3: policy-first routing engine (opt-in), mandatory eligibility filters,
  deterministic ranking, durable provenance (schema v11).
- A4: cloud evidence vault (schema v12), typed evidence classes, fail-closed
  eligibility oracle; fixtures can never mint provider_enforced records.
- A7: v0.7.0 production release gate - in zero-cost mode only local (ollama)
  inference may be dispatched; non-local cloud inference denied independently
  of evidence state. Full audit + release packaging.

Zero authorized API expenditure; no live requests in dev/tests; 811-test Linux
baseline preserved and extended (see CI).

## 0.6.0 - Zero-Cost Cloud Engine (merged to main @ 121899b)

See README "Phase 4" section.
