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
├── errors.py         # HIVE error hierarchy (config, gateway transient/permanent, ...)
├── providers.py      # Provider registry: OpenAI, Mistral, Anthropic, Gemini, Ollama
├── retry.py          # Bounded RetryPolicy: exponential backoff + jitter, Retry-After
├── usage.py          # UsageRecord / UsageTracker (thread-safe, optional JSONL sink)
├── gateway.py        # ModelGateway: the single LiteLLM choke point (sync + async)
├── agent_config.py   # Pydantic specs + YAML loader + env overrides (5 core roles)
├── llm.py            # LLMClient protocol + FakeLLM (offline tests/demos)
├── agents.py         # Agent: bounded tool-calling loop, least-privilege permissions
├── orchestrator.py   # Supervisor: routes between agents until FINISH
├── factory.py        # build_team(): wires the whole team from a HiveConfig
├── config.py         # Global Settings (pydantic-settings, prefix HIVE_)
└── cli.py            # Entry point `geosgan-hive` / `python -m geosgan_hive`
config/
├── agents.yaml       # Default 5-role team, 5 independent models (needs provider keys)
└── agents.local.yaml # All-Ollama team: runs with zero API keys
tests/                # pytest suite, fully offline (FakeLLM / monkeypatched litellm)
examples/demo_offline.py  # Offline multi-model demo (scripted fake gateway)
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
