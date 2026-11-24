from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from .checkpoints import CheckpointStore
from .errors import CheckpointError

#: resume_fn signature: (run_id, owner) -> awaitable result (e.g. Supervisor.aresume)
ResumeFn = Callable[[str, str], Awaitable[Any]]


class RecoveryWorker:
    """Opt-in automatic recovery worker for suspended (WAIT/HYBRID) runs.

    - Discovers eligible runs via `store.list_eligible()`: persisted
      `next_attempt_at` (conservative cooldown / Retry-After-derived) and
      `deadline_at` are honored; runs leased by a live worker are skipped.
    - No busy polling and no retry storms: the loop sleeps `poll_interval_seconds`
      (>= 1s) between scans and wakes early only when `wake()` is called.
    - Resumes through the injected callable, which uses the SAME lease mechanism
      as manual resume (transactional claim + renewal + ownership fencing inside
      the resume call), so automatic and manual recovery cannot duplicate work.
    - Clean shutdown: `stop()` or task cancellation ends the loop; a lease this
      worker holds on an interrupted run is released instead of lingering.
    - Automatic recovery requires a running HIVE process / worker: nothing is
      installed as a background OS service.
    """

    def __init__(
        self,
        store: CheckpointStore,
        resume_fn: ResumeFn,
        *,
        owner: str = "recovery-worker",
        poll_interval_seconds: float = 5.0,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if poll_interval_seconds < 1.0:
            raise ValueError("poll_interval_seconds must be >= 1.0 (no busy polling)")
        self.store = store
        self._resume_fn = resume_fn
        self.owner = owner
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._on_event = on_event
        self._wake: asyncio.Event | None = None  # created in the running loop
        self._stopping = False
        self.resumed = 0
        self.skipped = 0
        self.failed = 0

    def _emit(self, kind: str, **meta: Any) -> None:
        if self._on_event is not None:
            self._on_event(kind, meta)

    def wake(self) -> None:
        """Event-driven wake (e.g. after a new suspension): the next scan runs early."""
        if self._wake is not None:
            self._wake.set()

    def stop(self) -> None:
        """Request clean shutdown; the loop exits after the current scan."""
        self._stopping = True
        self.wake()

    async def run_once(self) -> int:
        """One scan: resume every eligible run. Returns the number resumed."""
        count = 0
        for cp in self.store.list_eligible(now_iso=self.store.now_iso()):
            if self._stopping:
                break
            owner = f"{self.owner}:{cp.run_id}"
            try:
                await self._resume_fn(cp.run_id, owner)
                self.resumed += 1
                count += 1
                self._emit("resumed", run_id=cp.run_id, owner=owner)
            except asyncio.CancelledError:
                # clean shutdown on cancellation: release our claim instead of
                # letting it linger until the lease TTL expires
                with contextlib.suppress(Exception):
                    self.store.release_lease(cp.run_id, owner)
                raise
            except CheckpointError as exc:
                # leased by another worker / not resumable / deadline passed: skip
                self.skipped += 1
                self._emit("skipped", run_id=cp.run_id, reason=str(exc))
            except Exception as exc:
                # one bad run must never crash the worker
                self.failed += 1
                self._emit("failed", run_id=cp.run_id,
                           reason=f"{type(exc).__name__}: {exc}")
        return count

    async def run_forever(self) -> None:
        """Scan/resume loop until `stop()` or cancellation. Sleeps between scans."""
        self._wake = asyncio.Event()
        self._stopping = False
        try:
            while not self._stopping:
                await self.run_once()
                if self._stopping:
                    break  # stop() during the scan: do not sleep a full interval
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self.poll_interval_seconds
                    )
                except TimeoutError:
                    pass  # scan cadence elapsed (not busy polling)
        except asyncio.CancelledError:
            self._stopping = True
            raise

    def run_blocking(self) -> None:
        """Blocking entry point (CLI): runs until Ctrl+C / cancellation, then stops."""
        try:
            asyncio.run(self.run_forever())
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.stop()
