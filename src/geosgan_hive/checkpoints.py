from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import CheckpointError, QuotaExhaustedError

SCHEMA_VERSION = 11

#: Phase 3 (T4): durable tool-execution ledger + human approvals
_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_executions (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT,
    agent TEXT NOT NULL,
    tool TEXT NOT NULL,
    tool_version TEXT NOT NULL DEFAULT '1',
    args_sha256 TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    idempotency_key TEXT,
    approval_id TEXT,
    lease_owner TEXT,
    lease_generation INTEGER,
    requested_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    result_sha256 TEXT,
    error TEXT,
    output_bytes INTEGER
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT,
    agent TEXT NOT NULL,
    tool TEXT NOT NULL,
    tool_version TEXT NOT NULL DEFAULT '1',
    operation TEXT NOT NULL,
    target TEXT,
    args_json TEXT NOT NULL,
    args_sha256 TEXT NOT NULL,
    preconditions_json TEXT NOT NULL DEFAULT '{}',
    preview TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    consumed_at TEXT
);
"""

#: Phase 4 C3: durable, cross-process quota reservations + usage pools.
#: - quota_policies (v7): canonical quota policy per account pool — anchor,
#:   periods, COMPLETE canonical constraints (limit kinds + values) and reset
#:   semantics, all covered by policy_hash. Incompatible configurations are
#:   rejected across processes. Anchor/period/constraint changes are rejected at
#:   reservation time unless migrate_quota_policy is called explicitly; active
#:   usage is carried forward, never silently reset.
#: - quota_reservations: one row per unique dispatch_id, bound to its actual
#:   windows (items_json), limits (limits_json), policy (policy_hash) and lease
#:   binding. Idempotent reconciliation replays must match ALL of these.
#: - quota_usage: consumed usage per (pool, kind, period, window_start); historical
#:   windows are retained forever. Usage is only ever INCREMENTED at reservation
#:   time; there is deliberately no refund path (a crash after reservation
#:   conservatively retains the usage).
#: - quota_policy_audit: one row per explicit policy migration (operator, old/new
#:   hash, carried-usage summary), written in the SAME transaction as the change.
_QUOTA_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_policies (
    pool TEXT PRIMARY KEY,
    anchor_iso TEXT NOT NULL,
    periods_json TEXT NOT NULL,
    constraints_json TEXT NOT NULL DEFAULT '[]',
    reset_semantics TEXT NOT NULL DEFAULT 'unknown',
    policy_hash TEXT NOT NULL,
    operator TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_reservations (
    dispatch_id TEXT PRIMARY KEY,
    pool TEXT NOT NULL,
    anchor_iso TEXT NOT NULL,
    items_json TEXT NOT NULL,
    limits_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    run_id TEXT,
    lease_owner TEXT,
    lease_generation INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- v8: exact authorized-request binding (credentials are NEVER persisted,
    -- request_sha256 is a one-way digest of the SANITIZED payload)
    model TEXT,
    request_sha256 TEXT,
    prompt_tokens INTEGER,
    max_output_tokens INTEGER,
    cost_usd REAL,
    endpoint TEXT
);
CREATE TABLE IF NOT EXISTS quota_usage (
    pool TEXT NOT NULL,
    kind TEXT NOT NULL,
    period TEXT NOT NULL,
    window_start TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (pool, kind, period, window_start)
);
CREATE TABLE IF NOT EXISTS quota_policy_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool TEXT NOT NULL,
    old_policy_hash TEXT,
    new_policy_hash TEXT NOT NULL,
    operator TEXT NOT NULL,
    carried_json TEXT NOT NULL DEFAULT '[]',
    at TEXT NOT NULL
);
"""

#: Phase 5 (A5): durable public-API request reservations, usage and audit (v9).
#: Reservations bind the exact authorized operation (provider/endpoint/operation
#: + one-way sanitized request digest); usage is increment-only (never refunded);
#: audit rows carry BOUNDED metadata only (no credentials, auth headers or private
#: payloads). Local windows are HIVE-local accounting, never provider resets.
_API_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_reservations (
    reservation_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    max_wire_bytes INTEGER NOT NULL,
    run_id TEXT,
    lease_owner TEXT,
    lease_generation INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_usage (
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    window_start TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope, kind, window_start)
);
CREATE TABLE IF NOT EXISTS api_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    status_code INTEGER,
    bytes_in INTEGER NOT NULL,
    bytes_out INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    policy_decision TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL
);
"""

#: Phase 5 (A2): reproducible model-benchmark evidence (v10). Evidence records the
#: EXACT model identity (model_id with a pinned version) and execution_mode
#: (simulated vs measured). Simulated results are synthetic fixtures and must
#: NEVER be presented as real model measurements; measured records require an
#: execution transcript hash + operator attestation. Evidence expires (TTL) and
#: is swept; rows are bounded metadata + per-case scores (no raw prompts/outputs
#: beyond the bounded case results). v11 columns (trust_tier, evaluator_id) are
#: part of the base definition for fresh databases; existing v10 databases get
#: them via the additive, idempotent ALTER in _migrate_10_to_11.
_BENCHMARK_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_evidence (
    evidence_id TEXT PRIMARY KEY,
    suite_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    execution_mode TEXT NOT NULL CHECK (execution_mode IN ('simulated', 'measured')),
    aggregate_score REAL NOT NULL,
    cases_passed INTEGER NOT NULL,
    cases_total INTEGER NOT NULL,
    cases_json TEXT NOT NULL,
    hardware_profile TEXT NOT NULL DEFAULT '',
    run_config_json TEXT NOT NULL,
    provenance TEXT NOT NULL,
    transcript_sha256 TEXT,
    runner_attestation TEXT NOT NULL DEFAULT '',
    trust_tier TEXT NOT NULL DEFAULT 'synthetic_fixture',
    evaluator_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_evidence (model_id);
"""

#: Phase 5 (A2 gate + A3): routing decisions (v11) and benchmark trust tiers.
#: routing_decisions stores BOUNDED provenance only (candidate ids, eligibility
#: decisions, reason codes, selected model, mode, evidence refs, method/policy
#: versions) — never credentials, raw prompts or private response content.
#: benchmark_evidence gains trust_tier + evaluator_id via ALTER TABLE in the
#: migration (additive, idempotent). Trust tiers:
#:   infrastructure_test | synthetic_fixture  (simulated — never quality evidence)
#:   operator_reported                        (attested — NOT verified measurement)
#:   trusted_runtime                          (only via the lease-fenced runtime
#:                                             recorder; the sole tier eligible for
#:                                             automatic quality-based routing)
_ROUTING_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT,
    agent TEXT NOT NULL,
    required_capabilities TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    selected_model TEXT,
    selection_mode TEXT NOT NULL,
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    scoring_method TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_run ON routing_decisions (run_id);
"""

_QUOTA_KINDS = ("requests", "tokens")
_QUOTA_PERIODS = ("day", "week", "month")
#: bounded retries on SQLite write-lock contention (busy_timeout also applies)
_QUOTA_TX_ATTEMPTS = 3
#: SQLite INTEGER is a signed 64-bit value; reject anything outside its bounds
SQLITE_INT_MAX = 2**63 - 1


class ExecutionState(StrEnum):
    RUNNING = "running"
    WAITING_FOR_PROVIDER = "waiting_for_provider"
    READY_TO_RESUME = "ready_to_resume"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PersistMode(StrEnum):
    """Checkpoint privacy modes.

    - METADATA_ONLY: identifiers, counters, completed-step metadata and state are
      stored, but NOT the conversation transcript. No claim of complete durable
      resumption; resume() on such a checkpoint is rejected.
    - RESUMABLE: additionally stores the message transcript so a supervisor-level
      run can be resumed after a process restart. Stored in PLAINTEXT SQLite
      (no encryption at rest is claimed); protect the file with local filesystem
      permissions and delete it when no longer needed.
    """

    METADATA_ONLY = "metadata_only"
    RESUMABLE = "resumable"


class ReservationStatus(StrEnum):
    """Phase 4 C3 quota-reservation lifecycle. reserved -> dispatched is the only
    transition; there is no refund/cancel path that decrements quota_usage."""

    RESERVED = "reserved"
    DISPATCHED = "dispatched"


class QuotaReservationOutcome(StrEnum):
    """reserve_quota_batch result. CREATED: a NEW reservation was persisted (not yet
    dispatched). REPLAYED: an identical existing reservation was reconciled — this is
    idempotent bookkeeping, NOT dispatch authorization. Only mark_dispatched grants
    dispatch authorization, and exactly once."""

    CREATED = "created"
    REPLAYED = "replayed"


@dataclass
class RunCheckpoint:
    run_id: str
    state: ExecutionState
    persist_mode: PersistMode
    task: str
    agent: str
    model: str
    config_json: str
    counters_json: str
    steps_json: str
    route_json: str
    transcript_json: str | None
    step_count: int
    attempts: int
    next_attempt_at: str | None
    deadline_at: str | None
    error: str | None
    result: str | None
    lease_owner: str | None
    lease_expires_at: str | None
    created_at: str
    updated_at: str
    root_task_id: str | None = field(default=None)
    physical_reserved: int = field(default=0)  # durable, monotonic per-dispatch count
    lease_generation: int = field(default=0)  # bumped on every claim (fencing)
    tool_reservations: int = field(default=0)  # durable tool-execution reservations


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    persist_mode TEXT NOT NULL,
    task TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '',
    counters_json TEXT NOT NULL DEFAULT '{}',
    steps_json TEXT NOT NULL DEFAULT '[]',
    route_json TEXT NOT NULL DEFAULT '[]',
    transcript_json TEXT,
    step_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    deadline_at TEXT,
    error TEXT,
    result TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    root_task_id TEXT,
    physical_reserved INTEGER NOT NULL DEFAULT 0,
    lease_generation INTEGER NOT NULL DEFAULT 0,
    tool_reservations INTEGER NOT NULL DEFAULT 0
);
"""


def _add_months(dt: datetime, months: int) -> datetime:
    total = dt.month - 1 + months
    year = dt.year + total // 12
    month = total % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def quota_window_for(anchor_iso: str, period: str, at_iso: str) -> str:
    """Start of the quota window of `period` that contains `at`, anchored at the
    explicitly configured `anchor_iso`. Day = +1d steps, week = +7d steps,
    month = calendar-month arithmetic (day clamped). Unknown reset semantics fail
    closed: unparseable/naive timestamps, unknown periods and a FUTURE anchor
    (at < anchor) all raise CheckpointError — reset times are never invented."""
    try:
        anchor = datetime.fromisoformat(anchor_iso)
        at = datetime.fromisoformat(at_iso)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"unparseable ISO-8601 timestamp: {exc}") from exc
    if anchor.tzinfo is None or at.tzinfo is None:
        raise CheckpointError(
            "quota window timestamps must be timezone-aware (naive rejected)"
        )
    anchor = anchor.astimezone(UTC)
    at = at.astimezone(UTC)
    if at < anchor:
        raise CheckpointError(
            "quota window anchor is after 'at' — unknown reset semantics fail "
            "closed (reset times are never invented)"
        )
    if period == "day":
        k = int((at - anchor).total_seconds() // 86400)
        window = anchor + timedelta(days=k)
    elif period == "week":
        k = int((at - anchor).total_seconds() // (7 * 86400))
        window = anchor + timedelta(days=7 * k)
    elif period == "month":
        k = (at.year - anchor.year) * 12 + (at.month - anchor.month)
        window = _add_months(anchor, k)
        if window > at:
            window = _add_months(anchor, k - 1)
    else:
        raise CheckpointError(f"unknown quota period {period!r} — fail closed")
    return window.isoformat(timespec="milliseconds")


def _normalize_quota_items(items: Any) -> list[dict[str, Any]]:
    """Strict validation + dedupe of a quota batch by (kind, period). Integer
    accounting only: bools/floats/strings/None and negative amounts are rejected
    (nonfinite values are non-int and therefore rejected too)."""
    if not isinstance(items, (list, tuple)) or not items:
        raise CheckpointError("at least one quota item is required")
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}
    for it in items:
        if not isinstance(it, dict):
            raise CheckpointError("quota items must be dicts")
        kind, period, amount = it.get("kind"), it.get("period"), it.get("amount")
        if kind not in _QUOTA_KINDS:
            raise CheckpointError(f"unknown quota kind {kind!r} — fail closed")
        if period not in _QUOTA_PERIODS:
            raise CheckpointError(f"unknown quota period {period!r} — fail closed")
        if type(amount) is not int:  # bool/float/str/None rejected
            raise CheckpointError(
                f"quota amount must be a strict int, got {type(amount).__name__}"
            )
        if amount < 0:
            raise CheckpointError("negative quota amount rejected")
        if amount > SQLITE_INT_MAX:
            raise CheckpointError(
                f"quota amount exceeds SQLite int64 bounds ({SQLITE_INT_MAX})"
            )
        if amount == 0 and kind == "requests":
            raise CheckpointError("zero request reservations rejected")
        key = (kind, period)
        if key in seen:
            if seen[key] != amount:
                raise CheckpointError(f"conflicting duplicate quota items for {key}")
            continue
        seen[key] = amount
        out.append({"kind": kind, "period": period, "amount": amount})
    return out


def _normalize_quota_limits(limits: Any, items: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Every item must have an explicit strict-int limit (>= 1); missing limits
    fail closed."""
    if not isinstance(limits, dict):
        raise CheckpointError("limits must be a mapping of (kind, period) -> int")
    norm: dict[tuple[str, str], int] = {}
    for it in items:
        key = (it["kind"], it["period"])
        if key not in limits:
            raise CheckpointError(f"no limit provided for {key} — fail closed")
        v = limits[key]
        if type(v) is not int:
            raise CheckpointError(
                f"quota limit must be a strict int, got {type(v).__name__}"
            )
        if v < 1:
            raise CheckpointError("quota limit must be >= 1")
        if v > SQLITE_INT_MAX:
            raise CheckpointError(
                f"quota limit exceeds SQLite int64 bounds ({SQLITE_INT_MAX})"
            )
        norm[key] = v
    return norm


def _quota_items_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    def key(items: list[dict[str, Any]]) -> list:
        return sorted((i["kind"], i["period"], i["amount"], i.get("window_start"))
                      for i in items)
    return key(a) == key(b)


def _exec_schema(conn: sqlite3.Connection, script: str) -> None:
    """Execute a schema script statement-by-statement INSIDE the caller's already
    open transaction. Unlike Connection.executescript — which implicitly COMMITs
    any pending transaction first — this keeps schema creation and versioned
    migrations atomic: an interrupted migration rolls back completely and is
    retried (recoverable) on the next open."""
    for stmt in script.split(";"):
        # strip SQL comment lines first so a ';' inside a comment cannot
        # truncate a statement
        stmt = "\n".join(
            line for line in stmt.splitlines() if not line.strip().startswith("--")
        )
        if stmt.strip():
            conn.execute(stmt)


def _canonical_anchor(anchor_iso: str) -> str:
    """Normalize a quota window anchor to a canonical UTC ISO string (naive or
    unparseable anchors are rejected — fail closed)."""
    try:
        anchor = datetime.fromisoformat(anchor_iso)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"unparseable quota anchor: {exc}") from exc
    if anchor.tzinfo is None:
        raise CheckpointError("quota anchor must be timezone-aware (naive rejected)")
    return anchor.astimezone(UTC).isoformat(timespec="milliseconds")


def _quota_policy_hash(anchor_canon: str, periods: list[str],
                       constraints: list[dict[str, Any]], reset_semantics: str) -> str:
    """Canonical identity of a quota policy: window anchor + periods + COMPLETE
    constraints (limit kinds and values) + reset-semantics evidence level."""
    payload = json.dumps(
        {
            "anchor": anchor_canon,
            "periods": sorted(periods),
            "constraints": sorted(
                (c["kind"], c["period"], c["limit"]) for c in constraints
            ),
            "reset_semantics": reset_semantics,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_constraints(constraints: Any) -> list[dict[str, Any]]:
    if not isinstance(constraints, (list, tuple)):
        raise CheckpointError("constraints must be a list of {kind, period, limit}")
    out = []
    for c in constraints:
        if not isinstance(c, dict):
            raise CheckpointError("constraints must be dicts")
        kind, period, limit = c.get("kind"), c.get("period"), c.get("limit")
        if kind not in _QUOTA_KINDS or period not in _QUOTA_PERIODS:
            raise CheckpointError(f"invalid constraint kind/period in {c!r}")
        if type(limit) is not int or limit < 1 or limit > SQLITE_INT_MAX:
            raise CheckpointError("constraint limit must be a strict int in [1, 2^63-1]")
        out.append({"kind": kind, "period": period, "limit": limit})
    out.sort(key=lambda c: (c["kind"], c["period"]))
    return out


def _validate_binding(model: Any, request_sha256: Any, prompt_tokens: Any,
                      max_output_tokens: Any, cost_usd: Any, endpoint: Any) -> tuple:
    """Validate the optional exact-request binding of a reservation. Credentials
    are never persisted: request_sha256 must be a 64-hex digest of a SANITIZED
    payload (computed by the caller with secrets stripped)."""
    if model is not None and (not isinstance(model, str) or not model):
        raise CheckpointError("model binding must be a non-empty string when given")
    if request_sha256 is not None:
        if not isinstance(request_sha256, str) or len(request_sha256) != 64 \
                or any(c not in "0123456789abcdef" for c in request_sha256):
            raise CheckpointError("request_sha256 must be a 64-char hex digest")
    if prompt_tokens is not None and (type(prompt_tokens) is not int or prompt_tokens < 0
                                      or prompt_tokens > SQLITE_INT_MAX):
        raise CheckpointError("prompt_tokens binding must be a strict int >= 0")
    if max_output_tokens is not None and (type(max_output_tokens) is not int
                                          or max_output_tokens < 1
                                          or max_output_tokens > SQLITE_INT_MAX):
        raise CheckpointError("max_output_tokens binding must be a strict int >= 1")
    if cost_usd is not None:
        if isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float)) \
                or not math.isfinite(cost_usd) or cost_usd < 0:
            raise CheckpointError("cost_usd binding must be a finite number >= 0")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint
                                 or len(endpoint) > 512):
        raise CheckpointError("endpoint binding must be a non-empty string <= 512 chars")
    return model, request_sha256, prompt_tokens, max_output_tokens, cost_usd, endpoint


def _validate_reset_semantics(reset_semantics: Any) -> str:
    if not isinstance(reset_semantics, str) or not reset_semantics.replace("_", "").isalpha() \
            or not 1 <= len(reset_semantics) <= 32:
        raise CheckpointError("reset_semantics must be a short lowercase identifier")
    return reset_semantics


def _limits_json(limits: dict[tuple[str, str], int]) -> str:
    rows = [{"kind": k, "period": p, "limit": v}
            for (k, p), v in sorted(limits.items())]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _validate_quota_binding(run_id: Any, lease_owner: Any,
                            lease_generation: Any) -> None:
    """Audit rule: run-bound reservations require the COMPLETE lease binding.
    run_id, lease_owner and lease_generation must be provided together (all or
    none) — partial or malformed bindings are rejected."""
    given = [run_id is not None, lease_owner is not None, lease_generation is not None]
    if any(given) and not all(given):
        raise CheckpointError(
            "incomplete lease binding: run_id, lease_owner and lease_generation "
            "must be provided together (partial bindings rejected)"
        )
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id:
            raise CheckpointError("run_id must be a non-empty string")
        if not isinstance(lease_owner, str) or not lease_owner:
            raise CheckpointError("lease_owner must be a non-empty string")
        if type(lease_generation) is not int or lease_generation < 0:
            raise CheckpointError(
                f"lease_generation must be a strict non-negative int, got "
                f"{type(lease_generation).__name__}"
            )


class CheckpointStore:
    """Durable run checkpoints in SQLite (Python standard library only).

    - WAL mode; every public mutation is ONE atomic transaction (`with conn:`).
    - schema_version is stored and checked; incompatible databases raise
      CheckpointError instead of being silently migrated.
    - Leases: `claim_lease` is a single atomic UPDATE ... WHERE (free or expired),
      `renew_lease` and all terminal commits are fenced by `lease_owner`, so a stale
      worker cannot commit after losing ownership.
    - Crash recovery on open: RUNNING rows whose lease is absent/expired are marked
      FAILED (crash during RUNNING); stale leases on WAITING rows are cleared.
    - Timestamps are persisted as UTC ISO-8601 strings (restart-safe scheduling);
      in-process cooldowns elsewhere use a monotonic clock (health.py).
    - Credentials are NEVER stored: config_json is the YAML-derived config, which
      contains no secrets by construction.
    """

    def __init__(
        self,
        path: str,
        *,
        persist_mode: PersistMode = PersistMode.METADATA_ONLY,
        lease_ttl_seconds: float = 60.0,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be > 0")
        self.path = str(path)
        self.persist_mode = PersistMode(persist_mode)
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self._utcnow = utcnow
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            # cross-process write contention: wait instead of failing immediately
            self._conn.execute("PRAGMA busy_timeout=5000")
            # Schema creation + version check + migrations run in ONE atomic
            # transaction: BEGIN IMMEDIATE + statement-wise execution (NOT
            # executescript, which implicitly COMMITs first). An interrupted
            # migration rolls back fully and is retried on the next open.
            self._conn.execute("BEGIN IMMEDIATE")
            _exec_schema(self._conn, _SCHEMA)
            _exec_schema(self._conn, _LEDGER_SCHEMA)  # idempotent (IF NOT EXISTS)
            _exec_schema(self._conn, _QUOTA_SCHEMA)    # idempotent (IF NOT EXISTS)
            _exec_schema(self._conn, _API_SCHEMA)      # idempotent (IF NOT EXISTS)
            _exec_schema(self._conn, _BENCHMARK_SCHEMA)  # idempotent (IF NOT EXISTS)
            _exec_schema(self._conn, _ROUTING_SCHEMA)    # idempotent (IF NOT EXISTS)
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                v = int(row["value"])
                if v > SCHEMA_VERSION:
                    raise CheckpointError(
                        f"checkpoint database {self.path!r} has schema_version {v}, "
                        f"expected {SCHEMA_VERSION}; refusing to open an "
                        "incompatible store"
                    )
                while v < SCHEMA_VERSION:  # versioned forward migrations only
                    if v == 3:
                        self._migrate_3_to_4()
                    elif v == 4:
                        self._migrate_4_to_5()
                    elif v == 5:
                        self._migrate_5_to_6()
                    elif v == 6:
                        self._migrate_6_to_7()
                    elif v == 7:
                        self._migrate_7_to_8()
                    elif v == 8:
                        self._migrate_8_to_9()
                    elif v == 9:
                        self._migrate_9_to_10()
                    elif v == 10:
                        self._migrate_10_to_11()
                    else:
                        raise CheckpointError(
                            f"checkpoint database {self.path!r} has schema_version {v}, "
                            f"expected {SCHEMA_VERSION}; refusing to open an "
                            "incompatible store"
                        )
                    v += 1
        self._harden_permissions()
        self._recover_crashed()

    def _migrate_3_to_4(self) -> None:
        """Versioned migration v3 -> v4: adds lease_generation (ownership fencing by
        owner AND generation) and tool_reservations (durable per-dispatch tool budget
        reservations). Idempotent; runs inside the open transaction."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "lease_generation" not in cols:
            self._conn.execute(
                "ALTER TABLE runs ADD COLUMN lease_generation INTEGER NOT NULL DEFAULT 0"
            )
        if "tool_reservations" not in cols:
            self._conn.execute(
                "ALTER TABLE runs ADD COLUMN tool_reservations INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )

    def _migrate_4_to_5(self) -> None:
        """v4 -> v5: durable tool-execution ledger + approval records (Phase 3 T4)."""
        _exec_schema(self._conn, _LEDGER_SCHEMA)
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )

    def _migrate_5_to_6(self) -> None:
        """v5 -> v6 (Phase 4 C3): durable quota reservations + usage pools + policy
        registry. Additive only (new tables); all v5 data is preserved untouched,
        and historical quota windows are never reset by later config. Runs inside
        the caller's open transaction (statement-wise, no implicit commit)."""
        _exec_schema(self._conn, _QUOTA_SCHEMA)
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )

    def _migrate_6_to_7(self) -> None:
        """dev-v6 -> v7 (Phase 4 C3 hardening). Earlier DEVELOPMENT schema-v6
        databases (branch commits before the audit) are handled explicitly here;
        the official v5 -> v6 path above is preserved unchanged.

        Adds the complete canonical policy columns (constraints_json,
        reset_semantics) and the quota_policy_audit table, then recomputes
        policy hashes of existing rows with the v7 canonical formula so stored
        reservations stay consistent with their pool policy. All inside the
        caller's open transaction (statement-wise, no implicit commit)."""
        cols = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(quota_policies)").fetchall()}
        if "constraints_json" not in cols:
            self._conn.execute(
                "ALTER TABLE quota_policies ADD COLUMN constraints_json "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        if "reset_semantics" not in cols:
            self._conn.execute(
                "ALTER TABLE quota_policies ADD COLUMN reset_semantics "
                "TEXT NOT NULL DEFAULT 'unknown'"
            )
        _exec_schema(self._conn, _QUOTA_SCHEMA)  # creates audit table if missing
        for row in self._conn.execute(
            "SELECT pool, anchor_iso, periods_json, constraints_json, reset_semantics "
            "FROM quota_policies"
        ).fetchall():
            constraints = _normalize_constraints(json.loads(row["constraints_json"]))
            new_hash = _quota_policy_hash(
                row["anchor_iso"], json.loads(row["periods_json"]), constraints,
                row["reset_semantics"],
            )
            self._conn.execute(
                "UPDATE quota_policies SET policy_hash=? WHERE pool=?",
                (new_hash, row["pool"]),
            )
            self._conn.execute(
                "UPDATE quota_reservations SET policy_hash=? WHERE pool=? AND anchor_iso=?",
                (new_hash, row["pool"], row["anchor_iso"]),
            )
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )

    def _migrate_7_to_8(self) -> None:
        """v7 -> v8 (Phase 4 C4 hardening): quota_reservations gain the exact
        authorized-request binding columns (model, request_sha256, prompt_tokens,
        max_output_tokens, cost_usd, endpoint). Additive ALTERs; existing rows are
        preserved (NULL bindings). Credentials are never stored — only the
        one-way sanitized digest. Runs inside the caller's open transaction."""
        cols = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(quota_reservations)").fetchall()}
        for col, decl in (
            ("model", "TEXT"), ("request_sha256", "TEXT"),
            ("prompt_tokens", "INTEGER"), ("max_output_tokens", "INTEGER"),
            ("cost_usd", "REAL"), ("endpoint", "TEXT"),
        ):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE quota_reservations ADD COLUMN {col} {decl}"
                )
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )

    def _migrate_8_to_9(self) -> None:
        """v8 -> v9 (Phase 5 A5): adds the durable public-API tables (api_reservations,
        api_usage, api_audit). Purely additive (IF NOT EXISTS); all v8 data preserved.
        Runs inside the caller's open transaction; an interruption rolls back fully."""
        _exec_schema(self._conn, _API_SCHEMA)
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(9),)
        )

    def _migrate_9_to_10(self) -> None:
        """v9 -> v10 (Phase 5 A2): adds benchmark_evidence (model benchmark runs with
        exact model identity, simulated/measured separation, expiry). Purely additive
        (IF NOT EXISTS); all v9 data preserved. Runs inside the caller's open
        transaction; an interruption rolls back fully."""
        _exec_schema(self._conn, _BENCHMARK_SCHEMA)
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(10),)
        )

    def _migrate_10_to_11(self) -> None:
        """v10 -> v11 (A2 gate + A3): adds routing_decisions and the benchmark
        trust-tier/evaluator columns (ALTER TABLE, additive + idempotent); all
        v10 data preserved. Runs inside the caller's open transaction; an
        interruption rolls back fully."""
        _exec_schema(self._conn, _ROUTING_SCHEMA)
        cols = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(benchmark_evidence)").fetchall()}
        if "trust_tier" not in cols:
            self._conn.execute(
                "ALTER TABLE benchmark_evidence ADD COLUMN trust_tier TEXT NOT NULL "
                "DEFAULT 'synthetic_fixture'")
        if "evaluator_id" not in cols:
            self._conn.execute(
                "ALTER TABLE benchmark_evidence ADD COLUMN evaluator_id TEXT NOT NULL "
                "DEFAULT ''")
        self._conn.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(11),)
        )

    def _harden_permissions(self) -> None:
        """Best-effort: owner-only permissions on the database files (POSIX).
        Plaintext storage is documented; this limits local read access."""
        if self.path == ":memory:" or os.name != "posix":
            return
        for suffix in ("", "-wal", "-shm"):
            try:
                os.chmod(self.path + suffix, 0o600)
            except OSError:
                pass  # file may not exist yet (WAL/SHM) or FS may not support it

    # ---- time helpers (persisted timestamps are UTC ISO strings) ----
    def utcnow(self) -> datetime:
        return self._utcnow()

    def now_iso(self) -> str:
        return self._utcnow().isoformat(timespec="milliseconds")

    def _lease_expiry(self) -> str:
        from datetime import timedelta

        return (self._utcnow() + timedelta(seconds=self.lease_ttl_seconds)).isoformat(
            timespec="milliseconds"
        )

    def _recover_crashed(self) -> None:
        """Crash recovery at open: RUNNING with a stale/absent lease means the owning
        process died -> FAILED with a diagnostic. WAITING rows only get their stale
        lease cleared (they stay eligible for the recovery worker)."""
        now = self.now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET state=?, error=?, lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE state=? AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                (
                    ExecutionState.FAILED.value,
                    "process crash detected while RUNNING (stale lease); run marked FAILED",
                    now,
                    ExecutionState.RUNNING.value,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE state IN (?, ?) AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at <= ?",
                (
                    now,
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    now,
                ),
            )

    # ---- row mapping ----
    @staticmethod
    def _to_checkpoint(row: sqlite3.Row) -> RunCheckpoint:
        return RunCheckpoint(
            run_id=row["run_id"],
            state=ExecutionState(row["state"]),
            persist_mode=PersistMode(row["persist_mode"]),
            task=row["task"],
            agent=row["agent"],
            model=row["model"],
            config_json=row["config_json"],
            counters_json=row["counters_json"],
            steps_json=row["steps_json"],
            route_json=row["route_json"],
            transcript_json=row["transcript_json"],
            step_count=int(row["step_count"]),
            attempts=int(row["attempts"]),
            next_attempt_at=row["next_attempt_at"],
            deadline_at=row["deadline_at"],
            error=row["error"],
            result=row["result"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            root_task_id=row["root_task_id"],
            physical_reserved=int(row["physical_reserved"]),
            lease_generation=int(row["lease_generation"]),
            tool_reservations=int(row["tool_reservations"]),
        )

    def get(self, run_id: str) -> RunCheckpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return self._to_checkpoint(row) if row is not None else None

    def list_runs(self, *, state: ExecutionState | None = None) -> list[RunCheckpoint]:
        with self._lock:
            if state is None:
                rows = self._conn.execute(
                    "SELECT * FROM runs ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE state=? ORDER BY created_at", (state.value,)
                ).fetchall()
        return [self._to_checkpoint(r) for r in rows]

    def list_suspended(self) -> list[RunCheckpoint]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE state IN (?, ?) ORDER BY created_at",
                (ExecutionState.WAITING_FOR_PROVIDER.value, ExecutionState.READY_TO_RESUME.value),
            ).fetchall()
        return [self._to_checkpoint(r) for r in rows]

    def list_eligible(self, *, now_iso: str | None = None) -> list[RunCheckpoint]:
        """Suspended runs whose next attempt is due and whose deadline has not passed,
        and whose lease is free (no live worker). Used by the recovery worker (C5)."""
        now = now_iso or self.now_iso()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE state IN (?, ?) AND next_attempt_at IS NOT NULL "
                "AND next_attempt_at <= ? AND (deadline_at IS NULL OR deadline_at > ?) "
                "AND (lease_owner IS NULL OR lease_expires_at <= ?) ORDER BY next_attempt_at",
                (
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    now,
                    now,
                    now,
                ),
            ).fetchall()
        return [self._to_checkpoint(r) for r in rows]

    # ---- writes (each ONE atomic transaction) ----
    def create_run(
        self,
        *,
        run_id: str,
        owner: str,
        task: str,
        agent: str,
        model: str,
        config_json: str = "",
        root_task_id: str | None = None,
    ) -> RunCheckpoint:
        now = self.now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, state, persist_mode, task, agent, model, "
                "config_json, counters_json, steps_json, route_json, transcript_json, "
                "step_count, attempts, next_attempt_at, deadline_at, error, result, "
                "lease_owner, lease_expires_at, created_at, updated_at, root_task_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '[]', '[]', NULL, 0, 0, NULL, NULL, "
                "NULL, NULL, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    ExecutionState.RUNNING.value,
                    self.persist_mode.value,
                    task,
                    agent,
                    model,
                    config_json,
                    owner,
                    self._lease_expiry(),
                    now,
                    now,
                    root_task_id,
                ),
            )
        cp = self.get(run_id)
        assert cp is not None
        return cp

    def reserve_tool_durable(self, run_id: str, *, lease_owner: str | None = None,
                             lease_generation: int | None = None) -> bool:
        """Durably reserve ONE tool execution BEFORE the side effect is dispatched
        (Phase 3). Single atomic transaction, monotonic SQL increment (never
        decreases). Fenced by expected execution state (RUNNING), lease validity
        (unexpired) AND — when provided — the ACTUAL lease owner and generation, so
        a stale worker cannot dispatch after another worker acquired a fresh lease.
        A crash after this commit conservatively retains the reservation."""
        now = self.now_iso()
        cond = ["run_id=?", "state=?", "lease_expires_at > ?"]
        params: list[Any] = [run_id, ExecutionState.RUNNING.value, now]
        if lease_owner is not None:
            cond.append("lease_owner=?")
            params.append(lease_owner)
        if lease_generation is not None:
            cond.append("lease_generation=?")
            params.append(lease_generation)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET tool_reservations = tool_reservations + 1, updated_at=? "
                "WHERE " + " AND ".join(cond),
                [now, *params],
            )
            return cur.rowcount == 1

    def reserve_physical_durable(self, run_id: str, *, lease_owner: str | None = None,
                                 lease_generation: int | None = None) -> bool:
        """Durably commit ONE physical dispatch reservation to SQLite BEFORE the
        provider request is sent. Single atomic transaction; the counter is
        monotonic (SQL increment, never a read-modify-write), so concurrent
        connections/threads/workers cannot lose increments and persisted counters
        never decrease. Only allowed while the run is RUNNING: once the run is
        suspended/terminal (lease lost or recovered), dispatch is blocked.

        A crash after this commit but before dispatch conservatively retains the
        reservation. Returns False when the run row is gone, not RUNNING, or the
        lease has expired (ownership fencing beyond mere RUNNING state)."""
        now = self.now_iso()
        cond = ["run_id=?", "state=?", "lease_expires_at > ?"]
        params: list[Any] = [run_id, ExecutionState.RUNNING.value, now]
        if lease_owner is not None:
            cond.append("lease_owner=?")
            params.append(lease_owner)
        if lease_generation is not None:
            cond.append("lease_generation=?")
            params.append(lease_generation)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET physical_reserved = physical_reserved + 1, updated_at=? "
                "WHERE " + " AND ".join(cond),
                [now, *params],
            )
            return cur.rowcount == 1

    def update_progress(
        self, run_id: str, *, owner: str, counters_json: str, step_count: int
    ) -> None:
        """Persist budget counters + step progress at a step boundary (called before
        each supervisor model call for durable runs), fenced by lease ownership so a
        stale worker cannot overwrite newer state. One atomic transaction."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET counters_json=?, step_count=?, updated_at=? "
                "WHERE run_id=? AND lease_owner=?",
                (counters_json, step_count, now, run_id, owner),
            )
            if cur.rowcount != 1:
                raise CheckpointError(
                    f"cannot update progress of run {run_id!r}: lease lost (stale worker)"
                )

    def _transcript_or_none(self, transcript_json: str | None) -> str | None:
        # RESUMABLE stores the transcript; METADATA_ONLY never does (no silent
        # reconstruction of missing conversation content later)
        if self.persist_mode != PersistMode.RESUMABLE:
            return None
        return transcript_json

    def mark_waiting(
        self,
        run_id: str,
        *,
        owner: str,
        error: str,
        counters_json: str,
        steps_json: str,
        route_json: str,
        transcript_json: str | None,
        step_count: int,
        next_attempt_at: str,
        deadline_at: str,
    ) -> RunCheckpoint:
        now = self.now_iso()
        with self._lock, self._conn:
            # the transition is fenced by the current owner, but the lease is RELEASED
            # on suspension: a waiting run is not owned by a live process, so the
            # recovery worker / manual resume can claim it (single atomic UPDATE)
            cur = self._conn.execute(
                "UPDATE runs SET state=?, error=?, counters_json=?, steps_json=?, "
                "route_json=?, transcript_json=?, step_count=?, attempts=attempts+1, "
                "next_attempt_at=?, deadline_at=?, lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE run_id=? AND state=? AND lease_owner=?",
                (
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    error,
                    counters_json,
                    steps_json,
                    route_json,
                    self._transcript_or_none(transcript_json),
                    step_count,
                    next_attempt_at,
                    deadline_at,
                    now,
                    run_id,
                    ExecutionState.RUNNING.value,
                    owner,
                ),
            )
            if cur.rowcount != 1:
                raise CheckpointError(
                    f"cannot suspend run {run_id!r}: lease lost or run not RUNNING"
                )
        cp = self.get(run_id)
        assert cp is not None
        return cp

    def mark_running_gen(self, run_id: str, *, owner: str) -> int | None:
        """Like mark_running, but RETURNS the effective lease generation to the
        resuming worker (worker-owned identity; never re-read from SQLite later)."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET state=?, updated_at=?, lease_owner=?, lease_expires_at=?, "
                "lease_generation=lease_generation+1 "
                "WHERE run_id=? AND state IN (?, ?) AND lease_owner=?",
                (
                    ExecutionState.RUNNING.value, now, owner, self._lease_expiry(),
                    run_id,
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    owner,
                ),
            )
            if cur.rowcount != 1:
                return None
            row = self._conn.execute(
                "SELECT lease_generation FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return int(row["lease_generation"])

    def mark_running(self, run_id: str, *, owner: str) -> None:
        """Flip a claimed suspended run back to RUNNING (resume in progress)."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET state=?, updated_at=?, lease_owner=?, lease_expires_at=?, "
                "lease_generation=lease_generation+1 "
                "WHERE run_id=? AND state IN (?, ?) AND lease_owner=?",
                (
                    ExecutionState.RUNNING.value,
                    now,
                    owner,
                    self._lease_expiry(),
                    run_id,
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    owner,
                ),
            )
            if cur.rowcount != 1:
                raise CheckpointError(f"cannot resume run {run_id!r}: lease lost")

    def mark_completed(
        self,
        run_id: str,
        *,
        owner: str,
        lease_generation: int | None = None,
        result: str,
        counters_json: str,
        steps_json: str,
        route_json: str,
        transcript_json: str | None,
        step_count: int,
    ) -> None:
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET state=?, result=?, error=NULL, counters_json=?, "
                "steps_json=?, route_json=?, transcript_json=?, step_count=?, "
                "lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE run_id=? AND lease_owner=?"
                + (" AND lease_generation=?" if lease_generation is not None else ""),
                (
                    ExecutionState.COMPLETED.value,
                    result,
                    counters_json,
                    steps_json,
                    route_json,
                    self._transcript_or_none(transcript_json),
                    step_count,
                    now,
                    run_id,
                    owner,
                    *([lease_generation] if lease_generation is not None else []),
                ),
            )
            if cur.rowcount != 1:
                raise CheckpointError(
                    f"cannot complete run {run_id!r}: lease lost (stale worker fencing)"
                )

    def mark_failed(
        self,
        run_id: str,
        *,
        owner: str | None,
        error: str,
        lease_generation: int | None = None,
        counters_json: str = "{}",
        steps_json: str = "[]",
        route_json: str = "[]",
        transcript_json: str | None = None,
        step_count: int = 0,
    ) -> None:
        now = self.now_iso()
        with self._lock, self._conn:
            if owner is None:
                cur = self._conn.execute(
                    "UPDATE runs SET state=?, error=?, counters_json=?, steps_json=?, "
                    "route_json=?, transcript_json=?, step_count=?, lease_owner=NULL, "
                    "lease_expires_at=NULL, updated_at=? WHERE run_id=?",
                    (
                        ExecutionState.FAILED.value, error, counters_json, steps_json,
                        route_json, self._transcript_or_none(transcript_json),
                        step_count, now, run_id,
                    ),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE runs SET state=?, error=?, counters_json=?, steps_json=?, "
                    "route_json=?, transcript_json=?, step_count=?, lease_owner=NULL, "
                    "lease_expires_at=NULL, updated_at=? WHERE run_id=? AND lease_owner=?"
                    + (" AND lease_generation=?" if lease_generation is not None else ""),
                    (
                        ExecutionState.FAILED.value, error, counters_json, steps_json,
                        route_json, self._transcript_or_none(transcript_json),
                        step_count, now, run_id, owner,
                        *([lease_generation] if lease_generation is not None else []),
                    ),
                )
            if cur.rowcount != 1:
                raise CheckpointError(f"cannot fail run {run_id!r}: lease lost")

    def cancel(self, run_id: str, *, reason: str = "cancelled") -> None:
        """Cancellation is allowed at any time (no lease required) and is terminal."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET state=?, error=?, lease_owner=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE run_id=? AND state NOT IN (?, ?, ?)",
                (
                    ExecutionState.CANCELLED.value,
                    reason,
                    now,
                    run_id,
                    ExecutionState.COMPLETED.value,
                    ExecutionState.FAILED.value,
                    ExecutionState.CANCELLED.value,
                ),
            )
            if cur.rowcount != 1:
                raise CheckpointError(f"run {run_id!r} is already terminal")

    # ---- leases (transactional claim + renewal + ownership fencing) ----
    def claim_lease(self, run_id: str, owner: str) -> bool:
        """Atomically claim a suspended run for `owner` if its lease is free/expired.
        Single UPDATE ... WHERE: concurrent workers can never both win."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET lease_owner=?, lease_expires_at=?, "
                "lease_generation=lease_generation+1, updated_at=? "
                "WHERE run_id=? AND state IN (?, ?) "
                "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                (
                    owner,
                    self._lease_expiry(),
                    now,
                    run_id,
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    now,
                ),
            )
            return cur.rowcount == 1

    def claim_lease_gen(self, run_id: str, owner: str) -> int | None:
        """Atomically claim a suspended run for `owner` and RETURN the new lease
        generation to the claiming worker (worker-owned identity; it is never
        re-read from SQLite by other parties). Returns None when the lease is
        held by a live worker."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET lease_owner=?, lease_expires_at=?, "
                "lease_generation=lease_generation+1, updated_at=? "
                "WHERE run_id=? AND state IN (?, ?) "
                "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                (
                    owner,
                    self._lease_expiry(),
                    now,
                    run_id,
                    ExecutionState.WAITING_FOR_PROVIDER.value,
                    ExecutionState.READY_TO_RESUME.value,
                    now,
                ),
            )
            if cur.rowcount != 1:
                return None
            row = self._conn.execute(
                "SELECT lease_generation FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return int(row["lease_generation"])

    def renew_lease(self, run_id: str, owner: str) -> bool:
        """Extend the lease; fenced by ownership (a stale worker cannot renew)."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET lease_expires_at=?, updated_at=? "
                "WHERE run_id=? AND lease_owner=?",
                (self._lease_expiry(), now, run_id, owner),
            )
            return cur.rowcount == 1

    def release_lease(self, run_id: str, owner: str) -> None:
        now = self.now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE run_id=? AND lease_owner=?",
                (now, run_id, owner),
            )

    # ---- Phase 3: tool execution ledger ----

    def _verify_lease_locked(self, run_id: str, lease_owner: str | None,
                             lease_generation: int | None) -> bool:
        """Inside an open transaction: the run must be RUNNING with an unexpired
        lease AND (when given) match the expected owner AND generation. This is the
        atomic ownership-fencing check used by every durable tool dispatch and
        execution-start transition."""
        row = self._conn.execute(
            "SELECT state, lease_owner, lease_generation, lease_expires_at FROM runs "
            "WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return False
        if row["state"] != ExecutionState.RUNNING.value:
            return False
        # a run-bound dispatch requires a PRESENT, unexpired lease (missing expiry
        # is rejected — fail closed)
        if row["lease_expires_at"] is None or row["lease_expires_at"] <= self.now_iso():
            return False
        if lease_owner is not None and row["lease_owner"] != lease_owner:
            return False
        if lease_generation is not None and int(row["lease_generation"]) != int(lease_generation):
            return False
        return True

    def ledger_start(
        self,
        *,
        execution_id: str,
        run_id: str,
        agent: str,
        tool: str,
        args_sha256: str,
        status: str,
        task_id: str | None = None,
        tool_version: str = "1",
        args_json: str = "{}",
        idempotency_key: str | None = None,
        approval_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
    ) -> bool:
        """Insert a ledger row (execution-start is persisted BEFORE the side effect).
        Single atomic transaction; duplicate execution_id is rejected (no double
        dispatch). For mutating ops the row is fenced by lease owner+generation."""
        now = self.now_iso()
        with self._lock, self._conn:
            if lease_owner is not None and not self._verify_lease_locked(
                run_id, lease_owner, lease_generation
            ):
                return False  # stale worker / lost lease: no dispatch, no ledger row
            try:
                self._conn.execute(
                    "INSERT INTO tool_executions (execution_id, run_id, task_id, agent, "
                    "tool, tool_version, args_sha256, args_json, status, idempotency_key, "
                    "approval_id, lease_owner, lease_generation, requested_at, started_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        execution_id, run_id, task_id, agent, tool, tool_version,
                        args_sha256, args_json, status, idempotency_key, approval_id,
                        lease_owner, lease_generation, now,
                        now if status == ToolExecutionStatus.STARTED.value else None,
                    ),
                )
            except sqlite3.IntegrityError:
                return False  # duplicate execution id -> never double-dispatch
            return True

    def ledger_update(
        self,
        execution_id: str,
        *,
        status: str,
        result_sha256: str | None = None,
        error: str | None = None,
        output_bytes: int | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
    ) -> bool:
        """Transition a ledger row; fenced by owner+generation when provided (stale
        workers cannot commit results after losing ownership)."""
        now = self.now_iso()
        with self._lock, self._conn:
            if lease_owner is not None:
                cur = self._conn.execute(
                    "UPDATE tool_executions SET status=?, finished_at=?, result_sha256=?, "
                    "error=?, output_bytes=? "
                    "WHERE execution_id=? AND lease_owner=? AND lease_generation=?",
                    (status, now, result_sha256, error, output_bytes,
                     execution_id, lease_owner, lease_generation),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE tool_executions SET status=?, finished_at=?, result_sha256=?, "
                    "error=?, output_bytes=? WHERE execution_id=?",
                    (status, now, result_sha256, error, output_bytes, execution_id),
                )
            return cur.rowcount == 1

    def ledger_get(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
        return _ledger_row_to_dict(row) if row else None

    def ledger_list(
        self, *, run_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            query = "SELECT * FROM tool_executions"
            cond, params = [], []
            if run_id is not None:
                cond.append("run_id=?")
                params.append(run_id)
            if status is not None:
                cond.append("status=?")
                params.append(status)
            if cond:
                query += " WHERE " + " AND ".join(cond)
            query += " ORDER BY requested_at"
            rows = self._conn.execute(query, params).fetchall()
        return [_ledger_row_to_dict(r) for r in rows]

    def ledger_resolve(
        self, execution_id: str, *, decision: str, operator: str, evidence: str
    ) -> bool:
        """Operator resolution of an UNKNOWN outcome: records the decision + evidence
        without claiming the side effect was verified."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tool_executions SET status=?, error=?, finished_at=? "
                "WHERE execution_id=? AND status=?",
                (
                    ToolExecutionStatus.COMPLETED.value if decision == "completed"
                    else ToolExecutionStatus.FAILED.value,
                    f"operator-resolved by {operator}: {decision}; evidence: {evidence}",
                    now,
                    execution_id,
                    ToolExecutionStatus.UNKNOWN.value,
                ),
            )
            return cur.rowcount == 1

    # ---- Phase 3: approvals ----
    def create_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        agent: str,
        tool: str,
        operation: str,
        args_json: str,
        args_sha256: str,
        preview: str,
        expires_at: str,
        task_id: str | None = None,
        tool_version: str = "1",
        target: str | None = None,
        preconditions_json: str = "{}",
    ) -> None:
        now = self.now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO approvals (approval_id, run_id, task_id, agent, tool, "
                "tool_version, operation, target, args_json, args_sha256, "
                "preconditions_json, preview, status, requested_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id, run_id, task_id, agent, tool, tool_version, operation,
                    target, args_json, args_sha256, preconditions_json, preview,
                    ApprovalState.PENDING.value, now, expires_at,
                ),
            )

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
        return {k: row[k] for k in row.keys()} if row else None

    def list_approvals(
        self, *, run_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            query = "SELECT * FROM approvals"
            cond, params = [], []
            if run_id is not None:
                cond.append("run_id=?")
                params.append(run_id)
            if status is not None:
                cond.append("status=?")
                params.append(status)
            if cond:
                query += " WHERE " + " AND ".join(cond)
            query += " ORDER BY requested_at"
            rows = self._conn.execute(query, params).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def decide_approval(
        self, approval_id: str, *, decision: str, operator: str
    ) -> bool:
        """Operator decision (APPROVED/DENIED/CANCELLED). Only PENDING rows can be
        decided; the decider must be a trusted operator id (never an agent/model)."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE approvals SET status=?, decided_by=?, decided_at=? "
                "WHERE approval_id=? AND status=? AND expires_at > ?",
                (decision, operator, now, approval_id, ApprovalState.PENDING.value, now),
            )
            return cur.rowcount == 1

    def consume_approval(
        self, approval_id: str, *, expected_args_sha256: str, now_iso: str | None = None,
        expected_tool_version: str | None = None, expected_run_id: str | None = None,
        expected_agent: str | None = None, expected_tool: str | None = None,
        expected_task_id: str | None = None,
    ) -> bool:
        """Atomically consume an APPROVED approval: single transaction, single-use,
        bound to the exact argument hash (substitution -> rejected), not expired.
        When the expected_* bindings are given, they are checked in the SAME
        transaction, so an approval can never be consumed for a different
        tool version / run / agent / tool. This is the durable execution-start
        transition companion (see start_execution for the fully atomic variant)."""
        now = now_iso or self.now_iso()
        cond = ["approval_id=?", "status=?", "args_sha256=?", "expires_at > ?"]
        params: list[Any] = [
            approval_id, ApprovalState.APPROVED.value, expected_args_sha256, now,
        ]
        if expected_tool_version is not None:
            cond.append("tool_version=?")
            params.append(expected_tool_version)
        if expected_run_id is not None:
            cond.append("run_id=?")
            params.append(expected_run_id)
        if expected_agent is not None:
            cond.append("agent=?")
            params.append(expected_agent)
        if expected_tool is not None:
            cond.append("tool=?")
            params.append(expected_tool)
        if expected_task_id is not None:
            cond.append("task_id=?")
            params.append(expected_task_id)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE approvals SET status=?, consumed_at=? WHERE " + " AND ".join(cond),
                [ApprovalState.CONSUMED.value, now, *params],
            )
            return cur.rowcount == 1

    def start_execution(
        self, *, execution_id: str, run_id: str, agent: str, tool: str,
        args_sha256: str, status: str, approval_id: str, expected_args_sha256: str,
        task_id: str | None = None, tool_version: str = "1", args_json: str = "{}",
        lease_owner: str | None = None, lease_generation: int | None = None,
        expected_task_id: str | None = None,
    ) -> bool:
        """Atomic durable execution-start: consume the approval AND insert the ledger
        row in ONE transaction (both succeed or neither). The approval consume is
        bound to approval id + APPROVED state + exact args hash + tool version + run
        + agent + tool + not expired. A duplicate execution_id aborts before any
        write (no double dispatch, approval NOT consumed)."""
        now = self.now_iso()
        with self._lock, self._conn:
            if self._conn.execute(
                "SELECT 1 FROM tool_executions WHERE execution_id=?", (execution_id,)
            ).fetchone():
                return False
            # atomic ownership fencing: state + lease expiry + owner + generation
            if not self._verify_lease_locked(run_id, lease_owner, lease_generation):
                return False
            cond = [
                "approval_id=?", "status=?", "args_sha256=?", "tool_version=?",
                "run_id=?", "agent=?", "tool=?", "expires_at > ?",
            ]
            params: list[Any] = [
                approval_id, ApprovalState.APPROVED.value, expected_args_sha256,
                tool_version, run_id, agent, tool, now,
            ]
            if expected_task_id is not None:
                cond.append("task_id=?")
                params.append(expected_task_id)
            cur = self._conn.execute(
                "UPDATE approvals SET status=?, consumed_at=? WHERE " + " AND ".join(cond),
                [ApprovalState.CONSUMED.value, now, *params],
            )
            if cur.rowcount != 1:
                return False
            self._conn.execute(
                "INSERT INTO tool_executions (execution_id, run_id, task_id, agent, "
                "tool, tool_version, args_sha256, args_json, status, approval_id, "
                "lease_owner, lease_generation, requested_at, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    execution_id, run_id, task_id, agent, tool, tool_version,
                    args_sha256, args_json, status, approval_id, lease_owner,
                    lease_generation, now,
                    now if status == ToolExecutionStatus.STARTED.value else None,
                ),
            )
            return True

    def sweep_expired_approvals(self) -> int:
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE approvals SET status=? WHERE status IN (?, ?) AND expires_at <= ?",
                (
                    ApprovalState.EXPIRED.value,
                    ApprovalState.PENDING.value,
                    ApprovalState.APPROVED.value,
                    now,
                ),
            )
            return cur.rowcount

    # ---- Phase 4 C3: durable quota reservations (shared across runs/processes) ----

    def reserve_quota_batch(
        self,
        dispatch_id: str,
        *,
        pool: str,
        items: list[dict[str, Any]],
        anchor_iso: str,
        limits: dict[tuple[str, str], int],
        now_iso: str | None = None,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        constraints: list[dict[str, Any]] | None = None,
        reset_semantics: str = "unknown",
        model: str | None = None,
        request_sha256: str | None = None,
        prompt_tokens: int | None = None,
        max_output_tokens: int | None = None,
        cost_usd: float | None = None,
        endpoint: str | None = None,
    ) -> QuotaReservationOutcome:
        """Atomically reserve a batch of quota items for ONE unique dispatch_id,
        BEFORE dispatch. Everything happens in ONE BEGIN IMMEDIATE transaction, so
        concurrent processes/connections serialize the check-then-write and can
        never over-reserve. Returns CREATED for a new reservation, REPLAYED for an
        identical existing one (reconciliation only — NEVER dispatch
        authorization; only mark_dispatched grants that, exactly once).

        - items/limits are strictly validated (strict ints within SQLite int64
          bounds, no negatives, no zero request reservations, deduped by
          (kind, period); conflicting duplicates rejected);
        - the lease binding is all-or-none: run_id + lease_owner +
          lease_generation together or not at all (partial bindings rejected). When
          bound to a run, the existing lease fencing (_verify_lease_locked: RUNNING
          + unexpired lease + owner + generation) must pass — validated on the
          duplicate/replay path too — else CheckpointError (stale workers rejected);
        - a duplicate dispatch_id replays (REPLAYED) only when pool, canonical
          anchor, policy hash, limits, window-bound items and lease binding are ALL
          identical — otherwise CheckpointError (conflicting replay across windows,
          limits, policies or bindings);
        - the canonical account policy (anchor + periods hash) is persisted per
          pool; a differing anchor/periods is rejected unless migrate_quota_policy
          was called explicitly — active usage is never silently reset;
        - EVERY item is checked (used + amount <= limit) BEFORE any write; any
          exhaustion raises QuotaExhaustedError and the whole transaction rolls
          back (all-or-nothing, no partial accounting);
        - on success the reservation row (status=reserved, bound to its actual
          windows + policy) and the usage increments commit together. Usage is
          never decremented: there is no refund path, so a crash after reservation
          conservatively retains the consumed quota.

        SQLite write-lock contention (OperationalError busy/locked) is retried a
        bounded number of times on top of busy_timeout; persistent failure raises
        CheckpointError."""
        if not isinstance(dispatch_id, str) or not dispatch_id:
            raise CheckpointError("dispatch_id must be a non-empty unique string")
        if not isinstance(pool, str) or not pool:
            raise CheckpointError("pool must be a non-empty string")
        norm_items = _normalize_quota_items(items)
        norm_limits = _normalize_quota_limits(limits, norm_items)
        _validate_quota_binding(run_id, lease_owner, lease_generation)
        anchor_canon = _canonical_anchor(anchor_iso)
        limits_str = _limits_json(norm_limits)
        # constraints are account-config, NOT derived from the per-call limits
        # (the engine passes the complete canonical set); store-level default []
        norm_constraints = _normalize_constraints(
            constraints if constraints is not None else []
        )
        semantics = _validate_reset_semantics(reset_semantics)
        binding = _validate_binding(model, request_sha256, prompt_tokens,
                                    max_output_tokens, cost_usd, endpoint)
        for attempt in range(_QUOTA_TX_ATTEMPTS):
            try:
                return self._reserve_quota_batch_tx(
                    dispatch_id, pool=pool, items=norm_items, anchor_canon=anchor_canon,
                    limits=norm_limits, limits_str=limits_str, now_iso=now_iso,
                    run_id=run_id, lease_owner=lease_owner,
                    lease_generation=lease_generation,
                    constraints=norm_constraints, reset_semantics=semantics,
                    binding=binding,
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < _QUOTA_TX_ATTEMPTS - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise CheckpointError(
                    f"quota reservation transaction failed (SQLite contention): {exc}"
                ) from exc
        raise CheckpointError("quota reservation failed after bounded retries")

    def _reserve_quota_batch_tx(
        self, dispatch_id: str, *, pool: str, items: list[dict[str, Any]],
        anchor_canon: str, limits: dict[tuple[str, str], int], limits_str: str,
        now_iso: str | None, run_id: str | None, lease_owner: str | None,
        lease_generation: int | None, constraints: list[dict[str, Any]],
        reset_semantics: str, binding: tuple,
    ) -> QuotaReservationOutcome:
        now = now_iso or self.now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")  # serialize check+write across processes
            try:
                # lease fencing is validated BEFORE the duplicate check, so stale
                # workers are rejected even on identical replays
                if run_id is not None and not self._verify_lease_locked(
                        run_id, lease_owner, lease_generation):
                    raise CheckpointError(
                        f"lease fencing rejected quota reservation {dispatch_id!r} "
                        "(stale worker / lost lease)"
                    )
                periods = sorted({it["period"] for it in items})
                policy_hash = _quota_policy_hash(anchor_canon, periods, constraints,
                                                 reset_semantics)
                row = self._conn.execute(
                    "SELECT * FROM quota_reservations WHERE dispatch_id=?", (dispatch_id,)
                ).fetchone()
                if row is not None:
                    incoming = [
                        {**it, "window_start": quota_window_for(anchor_canon,
                                                                it["period"], now)}
                        for it in items
                    ]
                    same = (
                        row["pool"] == pool
                        and row["anchor_iso"] == anchor_canon
                        and row["policy_hash"] == policy_hash
                        and row["limits_json"] == limits_str
                        and _quota_items_equal(json.loads(row["items_json"]), incoming)
                        and row["run_id"] == run_id
                        and row["lease_owner"] == lease_owner
                        and (row["lease_generation"] is None) == (lease_generation is None)
                        and (lease_generation is None
                             or int(row["lease_generation"]) == int(lease_generation))
                        and row["model"] == binding[0]
                        and row["request_sha256"] == binding[1]
                        and row["prompt_tokens"] == binding[2]
                        and row["max_output_tokens"] == binding[3]
                        and row["cost_usd"] == binding[4]
                        and row["endpoint"] == binding[5]
                    )
                    if same:
                        self._conn.commit()
                        return QuotaReservationOutcome.REPLAYED
                    raise CheckpointError(
                        f"conflicting duplicate reservation for dispatch_id "
                        f"{dispatch_id!r} (window/limit/policy/binding mismatch) — "
                        "rejecting instead of double-counting or re-authorizing"
                    )
                # create path: persist/validate the canonical account policy first
                prow = self._conn.execute(
                    "SELECT policy_hash FROM quota_policies WHERE pool=?", (pool,)
                ).fetchone()
                if prow is None:
                    self._conn.execute(
                        "INSERT INTO quota_policies (pool, anchor_iso, periods_json, "
                        "constraints_json, reset_semantics, policy_hash, operator, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                        (pool, anchor_canon, json.dumps(periods),
                         json.dumps(constraints), reset_semantics, policy_hash, now, now),
                    )
                elif prow["policy_hash"] != policy_hash:
                    raise CheckpointError(
                        f"incompatible quota policy change for pool {pool!r}: "
                        "anchor/periods differ from the persisted policy — active "
                        "usage is never silently reset; call migrate_quota_policy "
                        "explicitly"
                    )
                checks: list[tuple[dict[str, Any], str]] = []
                for it in items:
                    window = quota_window_for(anchor_canon, it["period"], now)
                    limit = limits[(it["kind"], it["period"])]
                    urow = self._conn.execute(
                        "SELECT used FROM quota_usage WHERE pool=? AND kind=? "
                        "AND period=? AND window_start=?",
                        (pool, it["kind"], it["period"], window),
                    ).fetchone()
                    used = int(urow["used"]) if urow is not None else 0
                    if used + it["amount"] > limit:
                        raise QuotaExhaustedError(
                            f"quota exhausted for pool {pool!r} "
                            f"{it['kind']}/{it['period']} window {window}: "
                            f"used {used} + {it['amount']} > limit {limit}"
                        )
                    checks.append(({**it, "window_start": window}, window))
                self._conn.execute(
                    "INSERT INTO quota_reservations (dispatch_id, pool, anchor_iso, "
                    "items_json, limits_json, policy_hash, run_id, lease_owner, "
                    "lease_generation, status, created_at, model, request_sha256, "
                    "prompt_tokens, max_output_tokens, cost_usd, endpoint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dispatch_id, pool, anchor_canon,
                        json.dumps([c[0] for c in checks]), limits_str, policy_hash,
                        run_id, lease_owner, lease_generation,
                        ReservationStatus.RESERVED.value, now, *binding,
                    ),
                )
                for it, window in checks:
                    self._conn.execute(
                        "INSERT INTO quota_usage (pool, kind, period, window_start, used) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(pool, kind, period, "
                        "window_start) DO UPDATE SET used = used + excluded.used",
                        (pool, it["kind"], it["period"], window, it["amount"]),
                    )
                self._conn.commit()
                return QuotaReservationOutcome.CREATED
            except BaseException:
                self._conn.rollback()
                raise

    def get_reservation(self, dispatch_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quota_reservations WHERE dispatch_id=?", (dispatch_id,)
            ).fetchone()
        if row is None:
            return None
        d = {k: row[k] for k in row.keys()}
        d["items"] = json.loads(d.pop("items_json"))
        d["limits"] = json.loads(d.pop("limits_json"))
        return d

    def mark_dispatched(
        self,
        dispatch_id: str,
        *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now_iso: str | None = None,
        request_sha256: str | None = None,
    ) -> bool:
        """Atomically grant DISPATCH AUTHORIZATION for a reservation — exactly once.
        One BEGIN IMMEDIATE transaction covers: (a) lease-binding match — the
        supplied binding (all-or-none, INCLUDING all-None) must equal the
        reservation's stored binding, so a run-bound reservation can never be
        authorized by an unbound caller and vice versa; (b) the live run lease
        (owner + generation + present, unexpired expiry) when run-bound; (c) the
        reserved->dispatched transition; and (d) ATOMIC revalidation that the
        reservation still belongs to its original quota window and to the pool's
        CURRENT persisted policy — a reservation rolled into a new window, or one
        whose policy changed, must NOT authorize a request now. Only the single
        winner (UPDATE rowcount == 1) receives True. Never touches quota_usage
        (no refund, ever)."""
        _validate_quota_binding(run_id, lease_owner, lease_generation)
        for attempt in range(_QUOTA_TX_ATTEMPTS):
            try:
                return self._mark_dispatched_tx(
                    dispatch_id, run_id=run_id, lease_owner=lease_owner,
                    lease_generation=lease_generation, now_iso=now_iso,
                    request_sha256=request_sha256,
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < _QUOTA_TX_ATTEMPTS - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise CheckpointError(
                    f"mark_dispatched transaction failed (SQLite contention): {exc}"
                ) from exc
        raise CheckpointError("mark_dispatched failed after bounded retries")

    def _mark_dispatched_tx(
        self, dispatch_id: str, *, run_id: str | None, lease_owner: str | None,
        lease_generation: int | None, now_iso: str | None,
        request_sha256: str | None = None,
    ) -> bool:
        now = now_iso or self.now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM quota_reservations WHERE dispatch_id=?",
                    (dispatch_id,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return False
                # (a) supplied binding must EXACTLY match the stored binding,
                # including the all-None case (unbound caller vs run-bound row)
                if (row["run_id"] != run_id or row["lease_owner"] != lease_owner
                        or (row["lease_generation"] is None) != (lease_generation is None)
                        or (lease_generation is not None
                            and int(row["lease_generation"]) != int(lease_generation))):
                    self._conn.commit()
                    return False
                # (b) live run lease must be valid (present expiry + owner + gen)
                if run_id is not None and not self._verify_lease_locked(
                        run_id, lease_owner, lease_generation):
                    self._conn.commit()
                    return False
                # payload-change rejection: if the reservation is bound to a
                # sanitized request digest, the digest supplied at finalization
                # must match exactly — a changed payload is never authorized
                if request_sha256 is not None and row["request_sha256"] is not None \
                        and row["request_sha256"] != request_sha256:
                    self._conn.commit()
                    return False
                # (d) atomic window + current-policy revalidation at authorization
                prow = self._conn.execute(
                    "SELECT policy_hash FROM quota_policies WHERE pool=?",
                    (row["pool"],),
                ).fetchone()
                if prow is None or prow["policy_hash"] != row["policy_hash"]:
                    self._conn.commit()
                    return False  # policy changed since the reservation
                try:
                    for it in json.loads(row["items_json"]):
                        if quota_window_for(row["anchor_iso"], it["period"], now) \
                                != it["window_start"]:
                            self._conn.commit()
                            return False  # rolled into a new window: no authorization
                except CheckpointError:
                    self._conn.commit()
                    return False  # clock/anchor inconsistency: fail closed
                # (c) single-winner state transition
                cur = self._conn.execute(
                    "UPDATE quota_reservations SET status=? "
                    "WHERE dispatch_id=? AND status=?",
                    (ReservationStatus.DISPATCHED.value, dispatch_id,
                     ReservationStatus.RESERVED.value),
                )
                self._conn.commit()
                return cur.rowcount == 1  # True for exactly ONE caller ever
            except BaseException:
                self._conn.rollback()
                raise

    def quota_policy(self, pool: str) -> dict[str, Any] | None:
        """The persisted canonical quota policy of a pool: anchor, periods,
        complete constraints, reset semantics, policy hash, operator."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quota_policies WHERE pool=?", (pool,)
            ).fetchone()
        if row is None:
            return None
        d = {k: row[k] for k in row.keys()}
        d["periods"] = json.loads(d.pop("periods_json"))
        d["constraints"] = json.loads(d.pop("constraints_json"))
        return d

    def policy_audit(self, pool: str) -> list[dict[str, Any]]:
        """Audit trail of explicit policy migrations for a pool (transactional
        with the change itself)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, pool, old_policy_hash, new_policy_hash, operator, "
                "carried_json, at FROM quota_policy_audit WHERE pool=? ORDER BY id",
                (pool,),
            ).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            d["carried"] = json.loads(d.pop("carried_json"))
            out.append(d)
        return out

    def migrate_quota_policy(
        self, pool: str, *, anchor_iso: str, periods: list[str], operator: str,
        constraints: list[dict[str, Any]] | None = None,
        reset_semantics: str | None = None, now_iso: str | None = None,
    ) -> bool:
        """EXPLICIT, TRANSACTIONAL, AUDITED migration of a pool's canonical quota
        policy. One BEGIN IMMEDIATE transaction:
        - validates the new anchor/periods/constraints/reset semantics;
        - rejects UNSAFE migrations: a future anchor, or one that would move any
          active window BACKWARD (new window < old window), is refused;
        - PRESERVES EFFECTIVE ACTIVE USAGE: for every (kind, period), the usage of
          the currently active old window is carried forward (added) into the new
          active window. Usage is increment-only — historical rows are retained
          and never decreased, and no usage is ever silently dropped;
        - upserts the policy row (constraints + reset semantics + hash + operator)
          and appends a quota_policy_audit row with the carried-usage summary.
        Returns True when the policy changed, False for a no-op. Without this
        explicit call, incompatible configurations are rejected at reservation
        time across processes."""
        if not isinstance(pool, str) or not pool:
            raise CheckpointError("pool must be a non-empty string")
        if not isinstance(operator, str) or not operator:
            raise CheckpointError("migrate_quota_policy requires a non-empty operator id")
        anchor_canon = _canonical_anchor(anchor_iso)
        norm_periods = sorted(set(periods))
        for p in norm_periods:
            if p not in _QUOTA_PERIODS:
                raise CheckpointError(f"unknown quota period {p!r} — fail closed")
        # constraints/semantics not given -> PRESERVE the persisted ones (a window
        # migration must not silently change limit kinds/values or evidence level);
        # pass them explicitly to change them (which then requires this migration).
        now = now_iso or self.now_iso()
        for attempt in range(_QUOTA_TX_ATTEMPTS):
            try:
                return self._migrate_quota_policy_tx(
                    pool, anchor_canon=anchor_canon, norm_periods=norm_periods,
                    constraints=constraints, reset_semantics=reset_semantics,
                    operator=operator, now=now,
                )
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < _QUOTA_TX_ATTEMPTS - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise CheckpointError(
                    f"policy migration transaction failed (SQLite contention): {exc}"
                ) from exc
        raise CheckpointError("policy migration failed after bounded retries")

    def _migrate_quota_policy_tx(
        self, pool: str, *, anchor_canon: str, norm_periods: list[str],
        constraints: list[dict[str, Any]] | None, reset_semantics: str | None,
        operator: str, now: str,
    ) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM quota_policies WHERE pool=?", (pool,)
                ).fetchone()
                if row is not None:
                    old_hash = row["policy_hash"]
                    old_anchor = row["anchor_iso"]
                    if constraints is None:
                        norm_constraints = _normalize_constraints(
                            json.loads(row["constraints_json"]))
                    else:
                        norm_constraints = _normalize_constraints(constraints)
                    semantics = (_validate_reset_semantics(reset_semantics)
                                 if reset_semantics is not None
                                 else row["reset_semantics"])
                else:
                    old_hash = None
                    old_anchor = None
                    norm_constraints = _normalize_constraints(
                        constraints if constraints is not None else [])
                    semantics = _validate_reset_semantics(
                        reset_semantics if reset_semantics is not None else "unknown")
                new_hash = _quota_policy_hash(anchor_canon, norm_periods,
                                              norm_constraints, semantics)
                if row is not None:
                    if old_hash == new_hash:
                        self._conn.commit()
                        return False  # no-op
                    # carry the EFFECTIVE ACTIVE usage of old windows forward
                    carried: list[dict[str, Any]] = []
                    for period in norm_periods:
                        old_window = quota_window_for(old_anchor, period, now)
                        new_window = quota_window_for(anchor_canon, period, now)
                        if new_window < old_window:
                            raise CheckpointError(
                                f"unsafe policy migration for pool {pool!r}: period "
                                f"{period!r} would move the active window backward "
                                f"({old_window} -> {new_window}) — rejected"
                            )
                        if new_window == old_window:
                            continue  # same window: nothing to carry (no double count)
                        for kind in _QUOTA_KINDS:
                            urow = self._conn.execute(
                                "SELECT used FROM quota_usage WHERE pool=? AND kind=? "
                                "AND period=? AND window_start=?",
                                (pool, kind, period, old_window),
                            ).fetchone()
                            used = int(urow["used"]) if urow is not None else 0
                            if used > 0:
                                self._conn.execute(
                                    "INSERT INTO quota_usage (pool, kind, period, "
                                    "window_start, used) VALUES (?, ?, ?, ?, ?) "
                                    "ON CONFLICT(pool, kind, period, window_start) "
                                    "DO UPDATE SET used = used + excluded.used",
                                    (pool, kind, period, new_window, used),
                                )
                                carried.append({"kind": kind, "period": period,
                                                "from": old_window, "to": new_window,
                                                "used": used})
                else:
                    old_hash = None
                    carried = []
                self._conn.execute(
                    "INSERT INTO quota_policies (pool, anchor_iso, periods_json, "
                    "constraints_json, reset_semantics, policy_hash, operator, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(pool) DO UPDATE SET "
                    "anchor_iso=excluded.anchor_iso, periods_json=excluded.periods_json, "
                    "constraints_json=excluded.constraints_json, "
                    "reset_semantics=excluded.reset_semantics, "
                    "policy_hash=excluded.policy_hash, operator=excluded.operator, "
                    "updated_at=excluded.updated_at",
                    (pool, anchor_canon, json.dumps(norm_periods),
                     json.dumps(norm_constraints), semantics, new_hash, operator,
                     now, now),
                )
                self._conn.execute(
                    "INSERT INTO quota_policy_audit (pool, old_policy_hash, "
                    "new_policy_hash, operator, carried_json, at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (pool, old_hash, new_hash, operator, json.dumps(carried), now),
                )
                self._conn.commit()
                return True
            except BaseException:
                self._conn.rollback()
                raise

    def quota_status(self, pool: str) -> list[dict[str, Any]]:
        """All usage rows of a pool (every historical window retained, never reset)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT pool, kind, period, window_start, used FROM quota_usage "
                "WHERE pool=? ORDER BY window_start, kind, period",
                (pool,),
            ).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    # ------------------------------------------------------------------
    # Phase 5 (A5): durable public-API request reservations + audit (schema v9)
    # ------------------------------------------------------------------

    def reserve_api_request(
        self,
        reservation_id: str,
        *,
        provider: str,
        endpoint_id: str,
        operation_id: str,
        request_sha256: str,
        max_wire_bytes: int,
        budgets: Mapping[str, Mapping[str, int]],  # scope -> kind(requests|wire_bytes) -> limit
        window_start_iso: str,
        now_iso: str | None = None,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
    ) -> QuotaReservationOutcome:
        """Atomically reserve ONE durable API request budget unit BEFORE any physical
        HTTP request. One BEGIN IMMEDIATE transaction: lease fencing (when run-bound),
        replay check (identical -> REPLAYED reconciliation only; conflicting -> reject),
        budget check for EVERY scope (requests=1, wire_bytes=max_wire_bytes; conservative:
        the max is counted up front and NEVER refunded), insert reservation, increment
        usage. Returns CREATED/REPLAYED. Usage is increment-only; uncertain usage is
        never refunded."""
        _validate_quota_binding(run_id, lease_owner, lease_generation)
        if type(max_wire_bytes) is not int or max_wire_bytes < 1:
            raise CheckpointError("max_wire_bytes must be a strict positive int")
        if len(request_sha256) != 64 or any(c not in "0123456789abcdef" for c in request_sha256):
            raise CheckpointError("request_sha256 must be a sha256 hex digest")
        for attempt in range(_QUOTA_TX_ATTEMPTS):
            try:
                return self._reserve_api_tx(
                    reservation_id, provider=provider, endpoint_id=endpoint_id,
                    operation_id=operation_id, request_sha256=request_sha256,
                    max_wire_bytes=max_wire_bytes, budgets=dict(budgets),
                    window_start_iso=window_start_iso, now_iso=now_iso, run_id=run_id,
                    lease_owner=lease_owner, lease_generation=lease_generation)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < _QUOTA_TX_ATTEMPTS - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise CheckpointError(
                    f"reserve_api_request transaction failed (SQLite contention): {exc}"
                ) from exc
        raise CheckpointError("reserve_api_request failed after bounded retries")

    def _reserve_api_tx(self, reservation_id, *, provider, endpoint_id, operation_id,
                        request_sha256, max_wire_bytes, budgets, window_start_iso,
                        now_iso, run_id, lease_owner, lease_generation) -> QuotaReservationOutcome:
        now = now_iso or self.now_iso()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    "SELECT * FROM api_reservations WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()
                if existing is not None:
                    # replay: only identical bindings reconcile; lease fencing applies too
                    same = (
                        existing["provider"] == provider
                        and existing["endpoint_id"] == endpoint_id
                        and existing["operation_id"] == operation_id
                        and existing["request_sha256"] == request_sha256
                        and int(existing["max_wire_bytes"]) == max_wire_bytes
                        and existing["run_id"] == run_id
                        and existing["lease_owner"] == lease_owner
                        and existing["lease_generation"] == lease_generation
                    )
                    if not same:
                        raise CheckpointError(
                            f"conflicting replay of api reservation {reservation_id!r} "
                            "(substitution rejected)")
                    if run_id is not None and not self._verify_lease_locked(
                            run_id, lease_owner, lease_generation):
                        raise CheckpointError(
                            f"stale worker: lease fencing failed for api reservation "
                            f"{reservation_id!r} on run {run_id!r}")
                    self._conn.commit()
                    return QuotaReservationOutcome.REPLAYED
                if run_id is not None and not self._verify_lease_locked(
                        run_id, lease_owner, lease_generation):
                    raise CheckpointError(
                        f"stale worker: lease fencing failed for run {run_id!r}")
                scopes = {
                    f"run:{run_id}": budgets.get(f"run:{run_id}"),
                    f"provider:{provider}": budgets.get(f"provider:{provider}"),
                    f"endpoint:{endpoint_id}": budgets.get(f"endpoint:{endpoint_id}"),
                }
                for scope, kind_limits in scopes.items():
                    if kind_limits is None:
                        continue
                    for kind, amount in (("requests", 1), ("wire_bytes", max_wire_bytes)):
                        limit = kind_limits.get(kind)
                        if limit is None:
                            continue
                        row = self._conn.execute(
                            "SELECT used FROM api_usage WHERE scope=? AND kind=? "
                            "AND window_start=?", (scope, kind, window_start_iso),
                        ).fetchone()
                        used = int(row["used"]) if row else 0
                        if used + amount > int(limit):
                            raise QuotaExhaustedError(
                                f"API budget exhausted for {scope} ({kind}): "
                                f"used={used} +{amount} > limit={limit}")
                self._conn.execute(
                    "INSERT INTO api_reservations (reservation_id, provider, endpoint_id, "
                    "operation_id, request_sha256, max_wire_bytes, run_id, lease_owner, "
                    "lease_generation, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,'reserved',?)",
                    (reservation_id, provider, endpoint_id, operation_id, request_sha256,
                     max_wire_bytes, run_id, lease_owner, lease_generation, now))
                for scope, _kind_limits in scopes.items():
                    if scope == f"run:{run_id}" and run_id is None:
                        continue
                    for kind, amount in (("requests", 1), ("wire_bytes", max_wire_bytes)):
                        self._conn.execute(
                            "INSERT INTO api_usage (scope, kind, window_start, used) "
                            "VALUES (?,?,?,?) ON CONFLICT(scope, kind, window_start) "
                            "DO UPDATE SET used = used + excluded.used",
                            (scope, kind, window_start_iso, amount))
                self._conn.commit()
                return QuotaReservationOutcome.CREATED
            except Exception:
                self._conn.rollback()
                raise

    def mark_api_dispatched(
        self,
        reservation_id: str,
        *,
        run_id: str | None = None,
        lease_owner: str | None = None,
        lease_generation: int | None = None,
        now_iso: str | None = None,
        request_sha256: str | None = None,
    ) -> bool:
        """Single-winner dispatch authorization for an API reservation, immediately
        before the physical request. One BEGIN IMMEDIATE tx: exact binding match
        (incl. all-None), live lease fencing when run-bound, digest match, and the
        reserved->dispatched transition. Exactly one caller gets True. No refunds."""
        _validate_quota_binding(run_id, lease_owner, lease_generation)
        for attempt in range(_QUOTA_TX_ATTEMPTS):
            try:
                return self._mark_api_dispatched_tx(
                    reservation_id, run_id=run_id, lease_owner=lease_owner,
                    lease_generation=lease_generation, now_iso=now_iso,
                    request_sha256=request_sha256)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < _QUOTA_TX_ATTEMPTS - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise CheckpointError(
                    f"mark_api_dispatched transaction failed: {exc}") from exc
        raise CheckpointError("mark_api_dispatched failed after bounded retries")

    def _mark_api_dispatched_tx(self, reservation_id, *, run_id, lease_owner,
                                lease_generation, now_iso, request_sha256) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM api_reservations WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return False
                if (row["run_id"] != run_id or row["lease_owner"] != lease_owner
                        or row["lease_generation"] != lease_generation):
                    self._conn.commit()
                    return False
                if request_sha256 is not None and row["request_sha256"] != request_sha256:
                    self._conn.commit()
                    return False
                if run_id is not None and not self._verify_lease_locked(
                        run_id, lease_owner, lease_generation):
                    self._conn.commit()
                    return False
                cur = self._conn.execute(
                    "UPDATE api_reservations SET status='dispatched' "
                    "WHERE reservation_id=? AND status='reserved'",
                    (reservation_id,))
                self._conn.commit()
                return cur.rowcount == 1
            except Exception:
                self._conn.rollback()
                raise

    def record_api_audit(
        self,
        *,
        reservation_id: str,
        provider: str,
        endpoint_id: str,
        operation_id: str,
        outcome: str,
        status_code: int | None,
        bytes_in: int,
        bytes_out: int,
        duration_ms: int,
        policy_decision: str,
        provenance: str = "",
        now_iso: str | None = None,
    ) -> None:
        """Bounded audit metadata only: NEVER raw credentials, authorization headers
        or private query payloads."""
        now = now_iso or self.now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_audit (reservation_id, provider, endpoint_id, "
                "operation_id, outcome, status_code, bytes_in, bytes_out, duration_ms, "
                "policy_decision, provenance, at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (reservation_id, provider, endpoint_id, operation_id, outcome,
                 status_code, int(bytes_in), int(bytes_out), int(duration_ms),
                 policy_decision, provenance[:500], now))
            self._conn.commit()

    def api_usage_status(self, scope: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope, kind, window_start, used FROM api_usage WHERE scope=? "
                "ORDER BY window_start, kind", (scope,),
            ).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def api_audit_list(self, *, provider: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if provider is None:
                rows = self._conn.execute(
                    "SELECT * FROM api_audit ORDER BY id DESC LIMIT 500").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM api_audit WHERE provider=? ORDER BY id DESC LIMIT 500",
                    (provider,)).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    # ---- benchmark evidence (v10, Phase 5 A2) ----------------------------------

    #: maximum TTL for benchmark evidence (bounded retention)
    BENCHMARK_MAX_TTL_DAYS = 90
    #: maximum serialized size of the per-case results blob
    BENCHMARK_MAX_CASES_BYTES = 256 * 1024

    #: trust tiers for benchmark evidence (see _ROUTING_SCHEMA comment)
    TRUST_TIERS = ("infrastructure_test", "synthetic_fixture", "operator_reported",
                   "trusted_runtime")
    #: the only evaluator id allowed to produce trusted_runtime evidence
    RUNTIME_EVALUATOR_ID = "hive-runtime-evaluator"
    _EVALUATOR_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

    def record_benchmark_evidence(
        self,
        *,
        evidence_id: str,
        suite_version: str,
        model_id: str,
        model_version: str,
        execution_mode: str,
        aggregate_score: float,
        cases_passed: int,
        cases_total: int,
        cases_json: str,
        hardware_profile: str = "",
        run_config_json: str = "{}",
        provenance: str = "",
        transcript_sha256: str | None = None,
        runner_attestation: str = "",
        ttl_days: int = 30,
        trust_tier: str | None = None,
        evaluator_id: str = "",
        now_iso: str | None = None,
        _runtime_recorder: bool = False,
    ) -> None:
        """Atomically persist ONE benchmark evidence record. Validation is strict:
        - execution_mode is 'simulated' or 'measured';
        - TRUST TIERS: simulated -> infrastructure_test|synthetic_fixture;
          measured+attestation -> operator_reported ONLY (a caller can NEVER set
          trusted_runtime — that tier is reserved for the lease-fenced runtime
          recorder below). Operator-attested evidence is NOT verified
          measurement evidence and is ineligible for automatic quality routing;
        - a MEASURED record REQUIRES a transcript hash AND a non-empty operator
          attestation; arbitrary caller-provided hashes never elevate the tier;
        - evaluator_id is required; RUNTIME_EVALUATOR_ID is RESERVED and usable
          only through the private _runtime_recorder path (solely the
          lease-fenced recorder) — a caller naming the runtime evaluator
          never gains trusted_runtime status;
        - model_id must be an exact pinned identity (contains '@' version pin);
        - scores within [0,1], case counts consistent, cases_json bounded;
        - duplicate evidence_id is rejected (no replay/substitution);
        - expiry is in the future and within the max TTL.
        One transaction; fail closed on any violation."""
        now = now_iso or self.now_iso()
        if execution_mode not in ("simulated", "measured"):
            raise CheckpointError(f"invalid execution_mode {execution_mode!r}")
        # --- trust tier resolution (fail closed) ---
        if trust_tier is None:
            trust_tier = "synthetic_fixture" if execution_mode == "simulated" \
                else "operator_reported"
        if trust_tier not in self.TRUST_TIERS:
            raise CheckpointError(f"invalid trust_tier {trust_tier!r}")
        if trust_tier == "trusted_runtime" and not _runtime_recorder:
            raise CheckpointError(
                "trusted_runtime evidence can only be recorded through the "
                "lease-fenced runtime recorder (record_runtime_benchmark_evidence) "
                "— a caller-provided tier, hash or boolean never establishes it")
        if evaluator_id == self.RUNTIME_EVALUATOR_ID and not _runtime_recorder:
            raise CheckpointError(
                f"evaluator_id {self.RUNTIME_EVALUATOR_ID!r} is reserved for the "
                "lease-fenced runtime recorder — caller-supplied evidence never "
                "carries it")
        if execution_mode == "simulated" and trust_tier not in (
                "infrastructure_test", "synthetic_fixture"):
            raise CheckpointError(
                "simulated evidence can only have infrastructure_test/synthetic_fixture "
                "trust — relabeling simulated results is rejected")
        if execution_mode == "measured":
            if trust_tier != "operator_reported" and not _runtime_recorder:
                raise CheckpointError(
                    "caller-recorded measured evidence is operator_reported at most — "
                    "not verified measurement evidence")
            if not transcript_sha256 or not runner_attestation:
                raise CheckpointError(
                    "measured benchmark evidence requires transcript_sha256 AND "
                    "runner_attestation (operator); a hash alone never suffices")
        if not evaluator_id or not self._EVALUATOR_RE.match(evaluator_id):
            raise CheckpointError("evaluator_id is required (registered evaluator id)")
        if not re.fullmatch(r"[a-z0-9_./-]+@[A-Za-z0-9][A-Za-z0-9._:+-]{3,127}", model_id or ""):
            raise CheckpointError(
                "model_id must be an exact pinned identity like 'ollama/llama3.2@sha256:...'")
        if not model_version or len(model_version) > 128:
            raise CheckpointError("model_version must be a non-empty bounded string")
        if not 0.0 <= aggregate_score <= 1.0:
            raise CheckpointError("aggregate_score must be within [0, 1]")
        if not (0 <= cases_passed <= cases_total) or cases_total < 1 or cases_total > 10_000:
            raise CheckpointError("inconsistent case counts")
        if len(cases_json.encode("utf-8")) > self.BENCHMARK_MAX_CASES_BYTES:
            raise CheckpointError("cases_json too large")
        if not 1 <= ttl_days <= self.BENCHMARK_MAX_TTL_DAYS:
            raise CheckpointError(f"ttl_days must be within 1..{self.BENCHMARK_MAX_TTL_DAYS}")
        created = datetime.fromisoformat(now)
        expires = (created + timedelta(days=ttl_days)).isoformat(timespec="milliseconds")
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO benchmark_evidence (evidence_id, suite_version, "
                    "model_id, model_version, execution_mode, aggregate_score, "
                    "cases_passed, cases_total, cases_json, hardware_profile, "
                    "run_config_json, provenance, transcript_sha256, "
                    "runner_attestation, trust_tier, evaluator_id, created_at, "
                    "expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (evidence_id, suite_version, model_id, model_version, execution_mode,
                     float(aggregate_score), int(cases_passed), int(cases_total),
                     cases_json, hardware_profile[:500], run_config_json[:4096],
                     provenance[:500], transcript_sha256, runner_attestation[:200],
                     trust_tier, evaluator_id, now, expires),
                )
        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                f"benchmark evidence {evidence_id!r} already exists — replay/"
                "substitution rejected") from exc

    def record_runtime_benchmark_evidence(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_generation: int,
        evidence_id: str,
        suite_version: str,
        model_id: str,
        model_version: str,
        aggregate_score: float,
        cases_passed: int,
        cases_total: int,
        cases_json: str,
        transcript_sha256: str,
        hardware_profile: str = "",
        run_config_json: str = "{}",
        ttl_days: int = 30,
        now_iso: str | None = None,
    ) -> None:
        """The ONLY path to trusted_runtime evidence: lease-fenced (the run must
        be RUNNING with a live lease matching owner+generation — worker-owned
        identity, never caller-asserted), bound to the runtime evaluator, with a
        transcript hash. Model-generated content can never authorize or certify
        evidence: this method takes no model input. Atomic; duplicate ids and
        stale workers are rejected."""
        _validate_quota_binding(run_id, lease_owner, lease_generation)
        if not transcript_sha256 or len(transcript_sha256) != 64:
            raise CheckpointError("trusted runtime evidence requires a transcript sha256")
        with self._lock:
            if not self._verify_lease_locked(run_id, lease_owner, lease_generation):
                raise CheckpointError(
                    "stale worker: lease fencing failed for runtime benchmark evidence")
        self.record_benchmark_evidence(
            evidence_id=evidence_id, suite_version=suite_version, model_id=model_id,
            model_version=model_version, execution_mode="measured",
            aggregate_score=aggregate_score, cases_passed=cases_passed,
            cases_total=cases_total, cases_json=cases_json,
            hardware_profile=hardware_profile, run_config_json=run_config_json,
            provenance="trusted runtime measurement (lease-fenced recorder)",
            transcript_sha256=transcript_sha256,
            runner_attestation=self.RUNTIME_EVALUATOR_ID,
            ttl_days=ttl_days, trust_tier="trusted_runtime",
            evaluator_id=self.RUNTIME_EVALUATOR_ID, now_iso=now_iso,
            _runtime_recorder=True,
        )

    def benchmark_evidence_is_routing_eligible(self, row: dict[str, Any],
                                               suite_cases: int | None = None,
                                               now_iso: str | None = None) -> bool:
        """Only COMPLETE, UNEXPIRED, trusted_runtime evidence with the runtime
        evaluator may feed automatic quality-based routing. Simulated and
        operator-reported evidence NEVER qualify."""
        now = now_iso or self.now_iso()
        if row.get("trust_tier") != "trusted_runtime":
            return False
        if row.get("evaluator_id") != self.RUNTIME_EVALUATOR_ID:
            return False
        if row.get("execution_mode") != "measured" or not row.get("transcript_sha256"):
            return False
        if row.get("expires_at", "") <= now:
            return False
        try:
            cases = json.loads(row.get("cases_json") or "[]")
        except ValueError:
            return False
        if len(cases) != row.get("cases_total") or row.get("cases_total") < 1:
            return False  # incomplete evidence is not a verified measurement
        if suite_cases is not None and row.get("cases_total") != suite_cases:
            return False
        return True

    # ---- routing decisions (v11, Phase 5 A3) -------------------------------------

    ROUTING_DECISIONS_MAX_BYTES = 64 * 1024

    def record_routing_decision(
        self,
        *,
        decision_id: str,
        run_id: str | None,
        agent: str,
        required_capabilities: list[str],
        candidates_json: str,
        selected_model: str | None,
        selection_mode: str,
        evidence_refs: list[str] | None = None,
        scoring_method: str = "ranking-1.0",
        policy_version: str = "",
        now_iso: str | None = None,
    ) -> None:
        """Atomically persist one BOUNDED routing decision (provenance only: ids,
        reason codes, mode, evidence refs, method/policy versions — never
        credentials, raw prompts or private content). Duplicate ids rejected."""
        now = now_iso or self.now_iso()
        if len(candidates_json.encode("utf-8")) > self.ROUTING_DECISIONS_MAX_BYTES:
            raise CheckpointError("candidates_json too large")
        if selection_mode not in ("fixed", "eligible_default", "evidence_based", "none"):
            raise CheckpointError(f"invalid selection_mode {selection_mode!r}")
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO routing_decisions (decision_id, run_id, agent, "
                    "required_capabilities, candidates_json, selected_model, "
                    "selection_mode, evidence_refs, scoring_method, policy_version, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (decision_id, run_id, agent[:100],
                     json.dumps(list(required_capabilities))[:2048],
                     candidates_json, selected_model, selection_mode,
                     json.dumps(list(evidence_refs or []))[:2048],
                     scoring_method[:64], policy_version[:64], now),
                )
        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                f"routing decision {decision_id!r} already exists — replay rejected") from exc

    def list_routing_decisions(self, *, run_id: str | None = None,
                               limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if run_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM routing_decisions WHERE run_id=? "
                    "ORDER BY created_at DESC, decision_id DESC LIMIT ?",
                    (run_id, min(limit, 1000))).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM routing_decisions "
                    "ORDER BY created_at DESC, decision_id DESC LIMIT ?",
                    (min(limit, 1000),)).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def get_benchmark_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM benchmark_evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
        return {k: row[k] for k in row.keys()} if row else None

    def list_benchmark_evidence(
        self, *, model_id: str | None = None, include_expired: bool = False,
        now_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """List benchmark evidence (newest first). Expired records are hidden by
        default; pass include_expired=True to see them (marked expired)."""
        now = now_iso or self.now_iso()
        with self._lock:
            query = "SELECT * FROM benchmark_evidence"
            cond, params = [], []
            if model_id is not None:
                cond.append("model_id=?")
                params.append(model_id)
            if not include_expired:
                cond.append("expires_at > ?")
                params.append(now)
            if cond:
                query += " WHERE " + " AND ".join(cond)
            query += " ORDER BY created_at DESC, evidence_id DESC LIMIT 500"
            rows = self._conn.execute(query, params).fetchall()
        out = [{k: r[k] for k in r.keys()} for r in rows]
        for r in out:
            r["expired"] = r["expires_at"] <= now
        return out

    def sweep_expired_benchmark_evidence(self, now_iso: str | None = None) -> int:
        now = now_iso or self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM benchmark_evidence WHERE expires_at <= ?", (now,))
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def steps_to_json(steps: list[dict[str, Any]]) -> str:
    return json.dumps(steps, ensure_ascii=False)


def counters_to_json(counters: dict[str, Any]) -> str:
    return json.dumps(counters, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase 3 (T4): durable tool-execution ledger
# ---------------------------------------------------------------------------

class ApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


class ToolExecutionStatus(StrEnum):
    REJECTED = "rejected"
    WAITING_APPROVAL = "waiting_approval"
    RESERVED = "reserved"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"  # cancelled BEFORE execution
    UNKNOWN = "unknown"  # outcome uncertain (crash/timeout after side effect started)


def _ledger_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# methods are attached to CheckpointStore below (kept here for cohesion)
