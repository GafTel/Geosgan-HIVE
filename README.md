# GEOSGAN HIVE

Hybrid multi-agent AI system in Python: a central **Supervisor** routes each task
between **autonomous specialist agents**, and every role runs on its **own AI
model** through a single **Model Gateway** (LiteLLM).

```
user ──▶ Supervisor ──┬──▶ Researcher ──┐
                      ├──▶ Developer   ──┤
                      ├──▶ Analyst     ──┼──▶ final answer
                      └──▶ Reviewer ────┘
```

Each agent executes its own bounded tool-calling loop: if the model replies with
a tool call (JSON), it is executed and retried, up to a per-agent maximum.

## Architecture (Phase 1: MULTI-MODEL CORE)

```
                ┌─────────────────────────────┐
 user/CLI ──▶ Supervisor (own model) ────────┤
                └──────────────┬──────────────┘
                               │ routes
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
   Researcher            Developer                 Analyst/Reviewer
   (own model,           (own model,                (own model,
    permissions,          permissions,               permissions,
    limits)               limits)                    limits)
        └──────────────────────┬───────────────────────┘
                               │ complete() / acomplete()
                               ▼
                  ┌────────────────────────────┐
                  │  ModelGateway (LiteLLM)    │  provider + credentials,
                  │  sync + async              │  bounded retries (backoff
                  └──────────────┬─────────────┘  + jitter), timeouts,
                                 │                usage/cost capture
                                 ▼
                  ┌────────────────────────────┐
                  │  UsageTracker (JSONL opt.) │  tokens always, cost when known
                  └────────────────────────────┘
```

## Structure

```
src/geosgan_hive/
├── messages.py       # Message: unit of the conversation/transcript
├── tools.py          # Tools (safe AST calculator, UTC datetime, word count) + registry
├── errors.py         # HIVE error hierarchy (config, gateway, delegation, ...)
├── providers.py      # Provider registry: OpenAI, Mistral, Anthropic, Gemini, Ollama
├── retry.py          # Bounded RetryPolicy: exponential backoff + jitter, Retry-After
├── usage.py          # UsageRecord / UsageTracker (thread-safe, optional JSONL sink)
├── gateway.py        # ModelGateway: the single LiteLLM choke point (sync + async)
├── agent_config.py   # Pydantic specs + YAML loader + env overrides (5 core roles)
├── run_context.py    # Phase 2: CollaborationPolicy, RunContext, TaskNode, events
├── collaboration.py  # Phase 2: CollaborationEngine, HandoffRequest/Result, proposals
├── llm.py            # LLMClient + AsyncLLMClient protocols + FakeLLM (offline)
├── agents.py         # Agent: bounded tool/delegation loop, least privilege, arun
├── orchestrator.py   # Supervisor: sequential routing, run history, arun
├── factory.py        # build_team(): wires team + collaboration engine from HiveConfig
├── config.py         # Global Settings (pydantic-settings, prefix HIVE_)
└── cli.py            # Entry point `geosgan-hive` / `python -m geosgan_hive`
config/
├── agents.yaml       # Default 5-role team, 5 independent models + collaboration policy
└── agents.local.yaml # All-Ollama team: runs with zero API keys
tests/                # pytest suite, fully offline (FakeLLM / monkeypatched litellm)
examples/demo_offline.py        # Offline multi-model demo (scripted fake gateway)
examples/demo_collaboration.py  # Offline Phase 2 demo (parallel delegation, run history)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add only the provider keys you actually use
```

## Usage

```bash
# Offline demo (no API key needed, scripted responses):
python examples/demo_offline.py

# Real run with the default multi-model team:
geosgan-hive "Research the advantages of multi-model agents and write a summary"
python -m geosgan_hive "Explain what an agent supervisor is" --agents-config config/agents.local.yaml

# Optional: persist per-call usage/cost as JSONL
geosgan-hive "task" --usage-log logs/usage.jsonl
```

The output shows the **route** of agents that handled the task, the supervisor
steps, the final answer, and a **per-agent usage summary** (calls, tokens, cost).

## Configuration

The team is defined in a YAML file (`config/agents.yaml` by default, override with
`--agents-config` or `HIVE_CONFIG`) and validated with **pydantic**:

```yaml
supervisor:
  name: supervisor
  model: openai/gpt-4o-mini        # '<provider>/<model>' — required prefix
  max_steps: 8
  permissions: {allow_tools: false, allowed_tools: []}
agents:
  - name: researcher
    model: gemini/gemini-2.0-flash
    permissions: {allow_tools: true, allowed_tools: [calculator, word_count]}
    limits: {max_tool_steps: 5, max_calls_per_run: 20, timeout_seconds: 60,
             max_cost_usd_per_run: 1.0}
```

- The five core roles are exactly: **supervisor, researcher, developer, analyst,
  reviewer**.
- **No secrets in YAML.** Provider keys are read from environment variables at
  call time, and **only for providers actually used** (an all-Ollama team needs
  no key at all — see `config/agents.local.yaml`).
- Explicit env overrides (applied after YAML load, coerced by pydantic):
  `HIVE_SUPERVISOR_MODEL`, `HIVE_AGENT_DEVELOPER_MODEL`,
  `HIVE_AGENT_REVIEWER_TEMPERATURE`, `HIVE_SUPERVISOR_MAX_STEPS`, ...
  Pattern: `HIVE_AGENT_<NAME>_<FIELD>` (dashes in names become underscores).

### Providers

| Provider | Model prefix | Credential env var            | Notes                          |
| -------- | ------------ | ----------------------------- | ------------------------------ |
| OpenAI   | `openai/`    | `OPENAI_API_KEY`              |                                |
| Mistral  | `mistral/`   | `MISTRAL_API_KEY`             |                                |
| Anthropic| `anthropic/` | `ANTHROPIC_API_KEY`           |                                |
| Gemini   | `gemini/`    | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `google/` also accepted |
| Ollama   | `ollama/`    | none                          | `OLLAMA_API_BASE` (default `http://localhost:11434`) |

### Permissions (least privilege by default)

An agent gets **no tools unless explicitly allowed**: `permissions.allow_tools`
defaults to `false`, and `allowed_tools` lists the only tools visible to that
agent's model. Disallowed tool calls are rejected (`unknown tool`), never
executed. Unknown names in `allowed_tools` are a config error.

### Limits (per agent)

`max_tool_steps`, `max_calls_per_run`, `timeout_seconds`, `max_tokens`, and a
soft `max_cost_usd_per_run`. The cost budget is only enforced while costs are
reliably known — see below.

## Usage and cost tracking

Every model call produces a `UsageRecord` (agent, model, provider, prompt /
completion / total tokens, cost, latency, attempts, run_id) collected by the
thread-safe `UsageTracker`. Totals are available per agent, per run, and overall,
and can be persisted as JSONL (`--usage-log` / `HIVE_USAGE_LOG`).

**Cost semantics:** tokens are always recorded; `cost_usd` is the *actual* cost
reported by LiteLLM when the model is known, and **`None` when unknown** (local
Ollama models, unmapped models) — never `0`. Aggregated totals are `None` if any
contributing call has an unknown cost, and the cost budget is *not* enforced on
such unreliable data.

## Errors, retries and timeouts

- LiteLLM's own retry mechanism is disabled (`num_retries=0`): retries exist in
  exactly one place, the gateway.
- Bounded retries (`RetryPolicy`: `max_attempts`, exponential backoff with
  jitter, `Retry-After` respected, capped delays) apply **only to transient
  errors** (429, 5xx, timeouts, connection errors). Auth/400/context-window
  errors fail fast.
- Per-call timeout via `timeout_seconds` (global default `HIVE_DEFAULT_TIMEOUT_SECONDS`).
- Gateway failures surface as HIVE errors (`RateLimitError`, `GatewayTimeoutError`,
  `AuthenticationError`, ...); the supervisor degrades gracefully instead of
  crashing the run.

## Async-ready

`ModelGateway` exposes both `complete()` / `complete_response()` (sync) and
`acomplete()` / `acomplete_response()` (async, via `litellm.acompletion`) with the
same retry/timeout/usage semantics. Agents and the supervisor currently use the
sync path; the async path is tested and ready for Phase 2 (parallel inter-agent
collaboration).

## Tests and lint

```bash
pytest          # fully offline: FakeLLM + monkeypatched litellm
ruff check .
```

Real-provider smoke tests can be marked `@pytest.mark.integration`; they are
skipped by default (no credentials in CI).

## Global environment variables (prefix `HIVE_`)

| Variable                       | Default                | Description                                  |
| ------------------------------ | ---------------------- | -------------------------------------------- |
| `HIVE_CONFIG`                  | `config/agents.yaml`   | Team YAML config path                        |
| `HIVE_USAGE_LOG`               | —                      | JSONL file for per-call usage records        |
| `HIVE_DEFAULT_TIMEOUT_SECONDS` | `60`                   | Default per-call model timeout               |
| `HIVE_RETRY_MAX_ATTEMPTS`      | `3`                    | Total attempts per model call (bounded)      |
| `HIVE_RETRY_BASE_DELAY_SECONDS`| `1.0`                  | Backoff base delay                           |
| `HIVE_RETRY_MAX_DELAY_SECONDS` | `30.0`                 | Backoff delay cap                            |

Provider keys (`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`/`GOOGLE_API_KEY`) are read directly from the environment by the
gateway. Never print them or commit them (`.env` and `logs/` are in `.gitignore`).

## Adding an agent or a tool

Add an entry under `agents:` in the YAML config (name, model, permissions,
limits) — or override an existing role with an env var. To add a tool, create a
class with `name`, `description`, `parameters` (JSON Schema) and `run(args)`,
register it in the default registry (`tools.py`), then allow it explicitly in the
agents that may use it.


## Phase 2: HYBRID COLLABORATION ENGINE

Agents can now **delegate subtasks to each other** — but only through the central
`CollaborationEngine`; agents never call each other directly. The Supervisor keeps
sequential routing and ultimate authority; specialists may collaborate with bounded
parallel delegation.

```
Supervisor.run/arun (sequential, root authority)
  └─ root task in RunContext ──▶ Agent.arun
        └─ {"delegate": {...}} or {"delegations": [...]}
             └─ CollaborationEngine.adelegate / adelegate_many
                   ├─ build trusted HandoffRequest (runtime generates request_id,
                   │   run_id, task_id, parent_task_id, source_agent, depth —
                   │   NEVER taken from model JSON)
                   ├─ validate: known target, no self-delegation, permissions
                   │   (allow_delegation + allowed_agents + global allow_agent_to_agent),
                   │   no circular chain, max depth, max handoffs
                   ├─ events + task node in RunContext
                   └─ target Agent.arun (semaphore-bounded, timeout, cancellation-safe)
                        └─ HandoffResult fed back to the source agent
```

### Delegation protocol

The model may only choose `target`, `task`, `reason` and optional `context_hints`:

```json
{"delegate": {"target": "analyst", "task": "Analyze the results", "reason": "Quantitative analysis required"}}
{"delegations": [{"target": "researcher", "task": "Research topic A"}, {"target": "analyst", "task": "Analyze dataset B"}]}
```

Batch delegations are independent subtasks: executed with bounded concurrency
(`asyncio.Semaphore(max_parallel_agents)`), results returned in input order,
identical proposals executed only once. All metadata (`request_id`, `run_id`,
`task_id`, `parent_task_id`, `source_agent`, `delegation_depth`) is generated by
the runtime and validated with pydantic (`HandoffRequest` forbids extra keys);
malformed or unauthorized proposals are rejected with a structured
`HandoffResult(status=REJECTED)` and a `delegation_rejected` event.

### Permissions (least privilege)

`permissions.allow_delegation` (default `false`) and `permissions.allowed_agents`
(default `[]`) must both allow a handoff, plus the global
`collaboration.allow_agent_to_agent` switch (default `false`). The supervisor is
not restricted by `allowed_agents` (root authority) but obeys global limits.
Delegation instructions are added to the agent system prompt whenever
`allow_delegation` is set — even for agents with no tools.

### Run context, events and budgets

Each run gets an isolated `RunContext` (task tree, collaboration events, budgets),
closed and evicted by the engine after completion, error or cancellation
(bounded cache, no unbounded accumulation). Events: `delegation_requested`,
`delegation_approved`, `delegation_rejected`, `task_started`, `task_completed`,
`task_failed`, `result_returned` — identifiers + metadata only, never secrets or
full transcripts (subtask context is scoped to the ancestor task chain).

Every actual model request (supervisor, specialists, delegated, nested) atomically
reserves one unit of the run-wide `max_model_calls_per_run` budget **before** the
call — exact counts under concurrency, per task and per run. Per-agent limits are
enforced across the **whole run** (`RunContext.agent_calls`), so they cannot be
bypassed by repeatedly delegating to the same agent. The soft cost budget is
unchanged (unknown costs stay `None`).

**Logical vs physical requests** (two distinct budgets): `max_model_calls_per_run`
is the **hard logical budget** — one unit reserved atomically per `llm.complete`/
`acomplete` (before the call), covering supervisor + specialists + delegated + nested
calls, with a run-wide per-agent limit enforced in the same atomic step.
`RunContext.physical_requests` counts **physical provider requests** — the sum of
retry `attempts` in the usage records (failed/cancelled calls are recorded with
`error` set and `cost_usd=None`) — so a call retried 3 times is 1 logical call but
3 physical requests. Physical requests are tracked always, and can optionally be
hard-capped via `max_provider_requests_per_run` (default: off). Both are exact
under concurrency. Concurrency (`max_parallel_agents`) bounds **active model
requests**: the semaphore is taken per model call, never while an agent waits for
its delegated subtasks, so it cannot deadlock.

**Deadlock safety**: the `max_parallel_agents` semaphore is acquired per top-level
delegation; nested delegations inherit the ancestor's slot (`has_slot_ancestor`),
so a fully occupied semaphore can never deadlock on agents waiting for their own
subtasks. Batch cancellation uses `gather(return_exceptions=True)` + re-raise, so
no orphaned coroutines survive. Legacy sync-only LLM adapters run in a thread and
cannot be interrupted mid-call — the delegation wrapper is still cancelled on
timeout, and the eventual physical call remains counted.

### Async execution

Native async throughout: `AsyncLLMClient` protocol, `GatewayLLMClient.acomplete`,
`Agent.arun`, `Supervisor.arun`, `CollaborationEngine.adelegate/adelegate_many`.
A thread fallback exists only for legacy sync-only LLM adapters and never replaces
the native path. `geosgan-hive "task" --async` runs the whole team asynchronously.

### Collaboration policy (YAML / env)

| Setting | Default | Env override |
| --- | --- | --- |
| `allow_agent_to_agent` | `false` | `HIVE_COLLABORATION_ALLOW_AGENT_TO_AGENT` |
| `max_delegation_depth` | `2` | `HIVE_COLLABORATION_MAX_DELEGATION_DEPTH` |
| `max_handoffs_per_run` | `10` | `HIVE_COLLABORATION_MAX_HANDOFFS_PER_RUN` |
| `max_parallel_agents` | `3` | `HIVE_COLLABORATION_MAX_PARALLEL_AGENTS` |
| `default_delegation_timeout_seconds` | `120` | `HIVE_COLLABORATION_DEFAULT_DELEGATION_TIMEOUT_SECONDS` |
| `max_model_calls_per_run` | `200` | `HIVE_COLLABORATION_MAX_MODEL_CALLS_PER_RUN` |
| `max_provider_requests_per_run` | `null` (tracked, not capped) | `HIVE_COLLABORATION_MAX_PROVIDER_REQUESTS_PER_RUN` |

```bash
python examples/demo_collaboration.py   # offline: parallel delegation + run history
```

## Phase 2.5: INTELLIGENT MODEL FALLBACK ENGINE (v0.4.0)

Fallback lives inside the centralized `ModelGateway`, per logical call, so every
agent (including the supervisor) gets it through its bound client. It is **opt-in:
`mode=off` by default**, and the `off` path is behavior-identical to v0.3.0.

### Modes

| Mode | Behavior |
| --- | --- |
| `off` (default) | No fallback. Exactly the pre-2.5 behavior. |
| `continuity` | On a fallback-eligible error, serve the call from the next eligible model in the ordered chain; if the chain is exhausted, raise `FallbackExhaustedError`. |
| `wait` | Never switch models. On a fallback-eligible error, raise `ProviderUnavailableError` and suspend the run at a safe supervisor boundary (durable checkpoint). |
| `hybrid` | Try eligible alternatives first; if none is available, suspend like `wait`; after the cooldown a single-flight probe recovers the preferred primary model. |

### Error taxonomy and health

Every provider error is normalized (`ErrorClass`: `rate_limit`, `quota`, `outage`,
`network`, `timeout`, `auth`, `invalid_request`, `capability`, `context_window`,
`budget`, `provider_unavailable`, `fallback_exhausted`, `cancelled`, `unknown`) with
`retryable` / `fallback_eligible` / `terminal` flags. Retries (bounded, backoff +
jitter, `Retry-After` respected) happen only on the same model; fallback only on
fallback-eligible errors; cancellation, exhausted budgets and malformed requests
are terminal and never retried or fallen back. Unknown errors fail conservatively.

`HealthRegistry` keeps per-provider (`p:<provider>`) and per-model (`m:<model>`)
circuit breakers (CLOSED/OPEN/HALF_OPEN, monotonic clock, injectable for tests,
single-flight HALF_OPEN probes). Failure scope is correct by class: auth never
opens a provider-wide circuit (it is credential-specific); quota is account-wide;
outage/network are provider-wide; timeout is model-scoped. Quota reset timestamps
are never invented. On recovery the primary is preferred again on later calls.

### Fallback gates (restrictive by default)

- provider allowlist (`allowed_providers`); cross-provider transfer additionally
  requires the target provider in `allowed_fallback_providers` (context may not be
  sent to another provider without explicit eligibility);
- cost: `cost_tier()` (ollama=local/free, hosted=paid, else unknown). Unknown cost
  is **not zero**: unknown-cost candidates need `allow_unknown_cost_fallback`;
  switching to a paid model from a non-paid primary needs `allow_paid_fallback`;
- capability: tool support and a rough context-window token estimate are checked
  before dispatch;
- health: OPEN circuits are skipped;
- agent permissions, delegation rules, run/task ids are never altered by a fallback;
  no credentials or full prompts are logged.

### Budgets and concurrency (all Phase 2 guarantees preserved)

One logical model call = one `RunContext.reserve_model_call_for` reservation; every
physical dispatch (initial, retry, fallback, probe, failed, cancelled) reserves
atomically via `reserve_physical_request()` **before** the request, capped
optionally by `max_provider_requests_per_run`. Request-level concurrency is bounded
by the per-run semaphore acquired only around the in-flight request at the gateway
dispatch boundary — never during backoff, cooldowns, checkpoints, probe waits or
delegation waits — so nested collaboration cannot deadlock and slots are never
double-acquired. Reserved attempts are distinct from confirmed requests and from
provider billing. On resume, counters are restored from the checkpoint without
double counting; a crash after reservation conservatively retains it.

### Durable suspension and resumption (SQLite, stdlib only)

`CheckpointStore` (WAL, `schema_version` checked, one atomic transaction per
mutation) persists runs in states `running`, `waiting_for_provider`,
`ready_to_resume`, `completed`, `failed`, `cancelled`. Two privacy modes:

- `metadata_only`: state/counters/completed-step metadata only — **no**
  durable-resume claim; `resume()` rejects such checkpoints instead of silently
  reconstructing missing context;
- `resumable`: additionally stores the transcript (plaintext SQLite — **no
  encryption at rest is claimed**; protect the file with local permissions, delete
  when done; `*.db*` and `.geosgan_hive/` are gitignored). Credentials are never
  stored.

Suspension happens only at safe supervisor step boundaries — never while child
tasks are active; unsupported nested suspension fails safely with an explicit
diagnostic. Completed steps are persisted and never repeated on resume. Leases are
transactional (`UPDATE ... WHERE lease free/expired`), renewed during execution, and
all terminal commits are ownership-fenced, so duplicate/stale workers cannot commit.
Crashes during `RUNNING` are marked `FAILED` on next open; stale `WAITING` leases are
cleared and the run stays eligible. No exactly-once claim is made for side-effecting
tools (`Tool.readonly` distinguishes pure/read-only tools; built-ins are read-only).

```yaml
fallback:
  mode: hybrid                      # off | continuity | wait | hybrid
  fallback_models: [ollama/llama3.2, ollama/mistral]   # ordered chain
  max_fallback_attempts: 2
  max_wait_seconds: 300
  cooldown_seconds: 60
  failure_threshold: 3
  allowed_providers: [openai, ollama]
  allowed_fallback_providers: [ollama]
  allow_paid_fallback: false
  allow_unknown_cost_fallback: false
checkpoint:                         # required for wait/hybrid
  path: .geosgan_hive/checkpoints.db
  persist_mode: resumable           # metadata_only | resumable
  lease_ttl_seconds: 60
  worker_poll_seconds: 5
```

Env overrides: `HIVE_FALLBACK_*`, `HIVE_CHECKPOINT_*`, and per-agent
`HIVE_AGENT_<NAME>_FALLBACK_MODE` / `HIVE_AGENT_<NAME>_FALLBACK_MODELS`
(comma-separated; mode defaults to the global one). See `.env.example`.

### Recovery worker and CLI

Automatic recovery is opt-in and needs a running HIVE process (nothing is installed
as an OS service). The worker discovers eligible suspended runs (persisted
`next_attempt_at` and deadline honored, no busy polling, clean shutdown, same lease
mechanism as manual resume):

```bash
geosgan-hive --recovery-worker                 # run the worker until Ctrl+C
geosgan-hive --list-suspended                  # list suspended runs
geosgan-hive --show <run_id>                   # checkpoint details
geosgan-hive --resume <run_id>                 # manual resume (needs resumable mode)
geosgan-hive --cancel-run <run_id>             # cancel a run (terminal)
```

### Events

New structured run events (identifiers/metadata only, never secrets):
`fallback_selected`, `fallback_rejected`, `provider_failed`, `circuit_opened`,
`circuit_half_open`, `recovery_probe`, `budget_exhausted`, `task_suspended`,
`task_resumed`.

```bash
python examples/demo_fallback.py    # offline: fallback chain, circuit breaker, WAIT suspend + resume
```

### Known limitations

- The health registry is in-memory per process (by design in 2.5); restart-safe
  scheduling uses the persisted UTC timestamps in checkpoints.
- Durable resumption is supervisor-level only; nested agent suspension fails safely
  with a diagnostic instead of checkpointing mid-flight state.
- Legacy sync-only LLM adapters run in threads on the async path and cannot be
  interrupted mid-call (the wrapper still times out; the call stays counted).
- RESUMABLE checkpoints store the transcript in plaintext SQLite; protect the file
  locally. Reserved dispatch attempts are not provider billing.

### Approval-gate additions (C9)

- **Periodic lease renewal**: while a run is active, the supervisor runs an
  ownership-fenced lease keeper (thread on the sync path, asyncio task on the
  async path) that renews every `lease_ttl/3` (configurable via
  `lease_renew_interval_seconds`). Long model calls therefore never lose valid
  ownership; renewal failure marks the run's lease as lost, which blocks further
  dispatches (`reserve_physical_durable` only succeeds while RUNNING) and terminal
  commits (ownership-fenced `mark_*`). Keepers stop on completion, failure and
  cancellation with no orphaned tasks; crashed workers lose their persisted lease
  after `lease_ttl_seconds` and are recovered as FAILED on reopen.
- **Durable physical reservations**: schema v3 adds a monotonic
  `runs.physical_reserved` column. Every physical dispatch (initial, retry,
  fallback, probe, failed, cancelled) is committed to SQLite in ONE atomic
  transaction BEFORE the provider request is sent (wired via
  `ModelGateway.set_durable_store`, used by `build_team`/CLI). Counters never
  decrease; a crash after reservation conservatively retains it; resume reconciles
  `max(snapshot, column)` so resumed logical calls are never double counted.
  Reserved attempts remain distinct from confirmed requests and provider billing.
- **Credential scope of health keys**: exactly ONE credential per provider is
  resolved (first matching env var), so provider-scope circuits == that
  credential's scope. Auth failures never open provider-wide circuits; quota is
  account-wide and opens the provider scope; raw credentials never appear in keys,
  state or logs. Multi-credential-per-provider isolation is NOT supported and NOT
  claimed.
- **Resume authorization**: resumed runs always execute under the CURRENT config
  (permissions, fallback providers, policies); the checkpoint's `config_json` is
  only integrity-checked, never used to authorize anything. Incomplete
  checkpoints (resumable mode without transcript) and corrupt snapshots are
  rejected explicitly.
