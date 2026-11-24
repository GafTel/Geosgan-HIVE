from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import CheckpointError

SCHEMA_VERSION = 5

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
            self._conn.executescript(_SCHEMA)
            self._conn.executescript(_LEDGER_SCHEMA)  # idempotent (IF NOT EXISTS)
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
        self._conn.executescript(_LEDGER_SCHEMA)
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

    def reserve_tool_durable(self, run_id: str) -> bool:
        """Durably reserve ONE tool execution BEFORE the side effect is dispatched
        (Phase 3). Same guarantees as reserve_physical_durable: single atomic
        transaction, monotonic SQL increment (never decreases), only while RUNNING
        so stale/recovered workers cannot dispatch. A crash after this commit
        conservatively retains the reservation."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET tool_reservations = tool_reservations + 1, updated_at=? "
                "WHERE run_id=? AND state=?",
                (now, run_id, ExecutionState.RUNNING.value),
            )
            return cur.rowcount == 1

    def reserve_physical_durable(self, run_id: str) -> bool:
        """Durably commit ONE physical dispatch reservation to SQLite BEFORE the
        provider request is sent. Single atomic transaction; the counter is
        monotonic (SQL increment, never a read-modify-write), so concurrent
        connections/threads/workers cannot lose increments and persisted counters
        never decrease. Only allowed while the run is RUNNING: once the run is
        suspended/terminal (lease lost or recovered), dispatch is blocked.

        A crash after this commit but before dispatch conservatively retains the
        reservation. Returns False when the run row is gone or not RUNNING."""
        now = self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE runs SET physical_reserved = physical_reserved + 1, updated_at=? "
                "WHERE run_id=? AND state=?",
                (now, run_id, ExecutionState.RUNNING.value),
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
                "WHERE run_id=? AND lease_owner=?",
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
                    "lease_expires_at=NULL, updated_at=? WHERE run_id=? AND lease_owner=?",
                    (
                        ExecutionState.FAILED.value, error, counters_json, steps_json,
                        route_json, self._transcript_or_none(transcript_json),
                        step_count, now, run_id, owner,
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
        self, approval_id: str, *, expected_args_sha256: str, now_iso: str | None = None
    ) -> bool:
        """Atomically consume an APPROVED approval: single transaction, single-use,
        bound to the exact argument hash (substitution -> rejected), not expired.
        This is the durable execution-start transition companion."""
        now = now_iso or self.now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE approvals SET status=?, consumed_at=? "
                "WHERE approval_id=? AND status=? AND args_sha256=? AND expires_at > ?",
                (
                    ApprovalState.CONSUMED.value, now, approval_id,
                    ApprovalState.APPROVED.value, expected_args_sha256, now,
                ),
            )
            return cur.rowcount == 1

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
