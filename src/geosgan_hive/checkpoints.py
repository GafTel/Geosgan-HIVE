from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import CheckpointError, QuotaExhaustedError

SCHEMA_VERSION = 8

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
