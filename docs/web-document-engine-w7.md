# Web & Document Engine — W7: CLI and Operator Workflows (v0.8.0)

Branch: `feat/web-document-engine` (target v0.8.0; version bump belongs to W8).
This stage extends the **existing** `geosgan-hive` CLI (`src/geosgan_hive/cli.py`).
There is no second CLI application and no parallel runtime: every W7 command
reuses the same configuration loader, `CheckpointStore`, `ApprovalManager`,
`SourceRegistry`, `WorkspaceSandbox`, `ToolGateway`/`ToolRegistry` and durable
execution ledger as the rest of the system.

All workflows are **offline-capable**: no live external requests, no paid API
consumption, no generated-code execution. In zero-cost mode the release gate
`ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)` keeps all non-local cloud
inference blocked.

---

## 1. Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Runtime error — a `HiveError` (unknown run/approval, denied, expired, stale lease, hash mismatch, live lease held by another worker, …) |
| `2`  | Usage / configuration error (missing required flags, invalid `--spec-file`, documents disabled in config, argparse errors) |

Errors are printed as one-line `Error: ...` messages on stderr. Raw exception
payloads, private document contents and credentials are never printed.
`--json` (global flag) switches supported inspection commands to structured
JSON on stdout for automation.

## 2. New W7 commands

All commands require a `checkpoint:` block in the agents config (they need the
durable store). Example config: `config/agents.webdocs.example.yaml`.

### 2.1 Run lifecycle

```
geosgan-hive -c CONFIG --start-run "quarterly report" [--run-id demo] [--operator alice] [--json]
```

Creates a run in state `RUNNING` owned by `cli:<operator>` with a live lease
(`create_run`). The run id is generated (UUID hex) unless `--run-id` is given.

### 2.2 Document reading (operator scope)

```
geosgan-hive -c CONFIG --doc-read PATH [--doc-max-chars N] [--json]
```

Reads an authorized workspace-relative document through `DocumentReader`
(sandbox-resolved, deny-patterns enforced, bounded extraction). Operator reads
do **not** register a source (`register_source=False`). Output includes format,
parser name, raw/extract SHA-256, byte count, truncation flag and bounded text.

### 2.3 Document preparation (creates a PENDING exact-byte approval)

```
geosgan-hive -c CONFIG --doc-prepare --doc-format csv \
    --doc-destination out/reporte.csv --spec-file spec.json \
    --run RUN_ID [--operator alice] [--json]
```

- `--spec-file` must be a JSON document validated strictly with
  `DocumentSpec.model_validate` (extra fields forbidden; invalid file → exit 2).
- Requires an existing run (`--run`); unknown run → exit 1.
- Generates the document **exactly once** in the isolated parser worker,
  stages the bytes in `<workspace>/.hive-staging/`, hashes them, and creates a
  **pending** approval bound to: run id, task id, agent, tool/version, canonical
  spec digest, destination, `expected_absent: true`, exact artifact SHA-256,
  byte count, generator and expiry (the configured `tools.approval.expiry_seconds`).
- Output shows `status=pending`, the full artifact SHA-256, byte count,
  spec digest and expiry, plus the explicit note that the artifact is
  **NOT published**.

### 2.4 Approval inspection and decision (existing commands, W7-enhanced)

```
geosgan-hive -c CONFIG --pending-approvals [--json]
geosgan-hive -c CONFIG --show-approval APPROVAL_ID [--json]
geosgan-hive -c CONFIG --approve APPROVAL_ID --operator alice
geosgan-hive -c CONFIG --deny APPROVAL_ID --operator alice
geosgan-hive -c CONFIG --cancel-approval APPROVAL_ID --operator alice
```

`--show-approval` for a `publish_artifact` approval now prints the **full
exact-byte binding** before any decision: format, generator, destination,
`artifact_sha256` (full 64-hex), `byte_count`, `spec_sha256`,
`expected_absent`, staging relpath and lifecycle state. There is **no
automatic approval** and no way for a model to impersonate the human operator.

### 2.5 Publication (exact approved bytes only)

```
geosgan-hive -c CONFIG --doc-publish APPROVAL_ID --run RUN_ID --operator alice [--json]
```

Sequence enforced by the CLI + `DocumentWriter.publish`:

1. The approval must exist, belong to `--run`, and be in state `approved`
   (pending/denied/expired/consumed → exit 1, nothing is published; **no
   auto-approval**).
2. Lease continuation: `renew_lease_gen(run, cli:<operator>)` if this operator
   already holds a live lease; otherwise `takeover_lease_gen(run,
   cli:<operator>)`, which is allowed only for a free/expired lease on a
   RUNNING or COMPLETED run and **bumps the generation**. A live lease held by
   another worker is never displaced → exit 1 (stale workers fail closed).
3. `publish` revalidates status/expiry/run/task/lease, re-hashes the staged
   bytes, atomically consumes the approval **and** starts the execution-ledger
   row in one transaction (`start_execution`), then publishes with a
   creation-only atomic no-clobber write (dir-fd + `/proc/self/fd` verify +
   `os.link` EEXIST + post-publish re-open/re-hash). A post-publication hash
   mismatch is never reported as success.
4. On success the ledger row is completed and provenance is registered
   (`generated: true`, `prepared_by`, `published_by`, `run_id`, `task_id`,
   approval + execution ids). Output prints destination, exact sha256, byte
   count, approval id, execution id, source id and the prepared/published
   identities.

Replaying `--doc-publish` after success fails: the approval is `consumed`.

### 2.6 Document approval lifecycle status

```
geosgan-hive -c CONFIG --doc-status APPROVAL_ID [--json]
```

Shows the full lifecycle — `pending`, `approved` (awaiting publication),
`published`, `denied`, `expired`, `cancelled`, `uncertain` (consumed but the
execution never completed; **never auto-replayed**) — plus effective expiry
(stored state plus a live `expires_at` check after `sweep_expired_approvals()`),
the artifact binding block, whether the staged file is still present, all
execution-ledger rows for the approval and any registered source ids.
A prepared or approved artifact is never displayed as already published.

### 2.7 Source and citation inspection (existing commands, W7-enhanced)

```
geosgan-hive -c CONFIG --sources [--source-run RUN_ID] [--json]
geosgan-hive -c CONFIG --source-id SOURCE_ID [--json]
geosgan-hive -c CONFIG --source-excerpt SOURCE_ID [--source-excerpt-chars N]
```

`--source-id` reports source type, provider, retrieval/generation time, raw and
extracted content hashes, extraction method and version, byte counts, and the
`generated` / `truncated` / `offline_fixture` flags plus excerpt availability.
Raw content is not printed by default. A citation establishes **provenance,
not factual verification**.

### 2.8 Capability and configuration inspection

```
geosgan-hive -c CONFIG --capabilities [--json]
```

Reports effective, never aspirational, capabilities — no secrets (no API keys,
no credential hashes, no private env values):

- `platform`: `os.name`, whether the POSIX parser worker and fd-verified
  staging/publication are available; on other platforms documents **fail
  closed** and the report says so (no insecure override).
- `agents` / `supervisor`: model and **effective** tool list (intersection of
  `tools.enabled` and the agent's `permissions.allowed_tools`; an empty
  allowlist denies everything), delegation flag.
- `model_providers` and `zero_cost_mode`: release gate local providers
  (`("ollama",)`), whether `cloud.zero_cost` is configured, and that non-local
  cloud inference is blocked in zero-cost mode.
- `outbound`: enabled providers, per-provider outbound policies, configured
  web endpoint policy ids (default: none → disabled).
- `documents`: enabled flag, optional dependency presence (pypdf, python-docx,
  openpyxl, reportlab, defusedxml), allowed formats, POSIX-worker-only formats,
  operator limits, and `publication: creation-only, atomic no-clobber;
  overwrite disabled`.
- `workspace`: root, allow_write, write extensions, deny name patterns.
- `approvals`: expiry seconds, `human_operator_only: true`.
- `budgets`: durable budget configuration.

## 3. Cross-process operator workflow

The intended lifecycle, exercised by `tests/test_w7_cli.py` (real
`python -m geosgan_hive` subprocesses) and `examples/demo_cli_workflow.py`:

1. **Process A**: `--start-run "task" --run-id demo --operator alice`
   → run RUNNING with a CLI-owned lease.
2. **Process B**: `--doc-prepare ... --run demo` → exact-byte pending approval
   + staged artifact.
3. **Process C**: `--pending-approvals`, `--show-approval <id>` → human review
   of the full sha256/bytes/format/destination/expiry.
4. **Process D**: `--approve <id> --operator alice` (or `--deny`).
5. **Process E**: `--doc-publish <id> --run demo --operator alice` → renews
   the same-operator lease (or takes over a free/expired one with a generation
   bump) and publishes only the exact previously prepared bytes.
6. **Any process**: `--doc-status <id>` → `published` with execution and
   provenance ids; `--doc-read` / `--source-id` verify the result.

### 3.1 Lease TTL and crash recovery

- Lease TTL is the store's configured lease expiry. Within the TTL the pending
  artifact and approval survive across CLI processes.
- When a store is opened, existing crash recovery marks RUNNING runs whose
  lease has expired as `FAILED` ("process crash detected…"). Existing semantics
  are never weakened: after that, `takeover_lease_gen` cannot resurrect the
  run, so a **fresh preparation + fresh approval is required**.
- **Bytes are never regenerated under an old approval.** If the staged artifact
  cannot be verified (missing, hash mismatch, staging swept), publication
  fails closed and a new prepare/approve cycle is required.

### 3.2 Staging lifecycle and privacy

- Staged artifacts live in `<workspace>/.hive-staging/` (0700 dir, 0600 files).
- `.hive-staging` is in the default deny patterns: agent file tools, source
  tools and other agents cannot read staging contents.
- Bounds: max files, max total bytes, max age; sweeps are serialized across
  processes with a flock and fire immediately on terminal approval state
  (denied/expired/consumed/cancelled) while protecting live approvals.
- Staged bytes are never persisted to SQLite; the database holds only hashes
  and metadata (metadata-only privacy).

## 4. Error taxonomy (user-facing)

Missing dependency (typed `DocumentGenUnavailableError` → exit 1 with a clear
message), unsupported platform (fail closed, reported by `--capabilities`),
disabled tool (`documents.enabled` false → exit 2), unauthorized workspace path
(sandbox deny → exit 1), unsupported format, invalid spec (exit 2),
resource-limit exhaustion, approval required / rejected / expired, destination
already exists (creation-only → exit 1), destination **parent directory missing**
(checked at prepare time and re-checked at publish time *before* the approval is
consumed → clean exit-1 error, the approval stays `approved` and the same bytes can
be published after the operator fixes the workspace; never a raw traceback),
artifact hash mismatch, stale lease /
live lease held by another worker, cancelled operation, uncertain execution
(never auto-replayed), unavailable source excerpt, blocked outbound request.

## 5. Zero-cost and outbound guarantees (unchanged, re-verified)

- ZERO paid API consumption: `ZERO_COST_RELEASE_LOCAL_PROVIDERS = ("ollama",)`;
  every non-local cloud provider stays blocked in zero-cost mode regardless of
  routing, fallback, delegation or recovery paths.
- Outbound web access stays default-disabled and requires operator-configured
  endpoint policies + durable outbound authorization; the CLI workflows make
  no network calls.
- `--capabilities` never reports an unavailable capability as operational and
  never prints secrets.

## 6. Tests and demo

- `tests/test_w7_cli.py` — 7 tests driving the real CLI in subprocesses:
  full cross-process lifecycle (start-run → prepare → pending → show → approve
  → publish → status published → replay refused), rejection and expiry,
  stale-lease/operator mismatch, usage errors and exit codes,
  documents-disabled config, capabilities report + secret-leak scan, and
  destination no-clobber.
- `examples/demo_cli_workflow.py` — offline demo of the same workflow through
  subprocesses, including a rejection path and a replay refusal, in a
  temporary workspace/database; asserts the spec title never lands in the
  SQLite file (staging privacy). Prints `DEMO_W7_OK`.

## 7. Known limitations (W7 scope)

- Windows / non-POSIX: document parsing, generation and publication fail
  closed (no fd re-verification available); `--capabilities` reports this
  explicitly. No `allow_insecure_fs`, no timeout-only mode.
- No document overwrite: publication is creation-only with atomic no-clobber.
- Pending artifacts do not survive owner-process death: crash recovery marks
  the run FAILED and a fresh prepare + approval is required (documented, fail
  closed).
- No general-purpose authorized search engine; no GUI; no persistent semantic
  memory; no live non-local inference in zero-cost mode.
- Trusted-process boundary unchanged: no defense against arbitrary privileged
  local code, direct DB edits, or compromised installed dependencies.
