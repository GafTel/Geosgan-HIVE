"""W1 (v0.8.0): trusted parser-worker isolation (prototype; real parsers land in W4).

Untrusted document bytes are NEVER parsed in the main agent process. A job is
executed by a dedicated short-lived subprocess:

- fixed executable: ``sys.executable`` with ``-I`` (isolated mode: no user
  site-packages, PYTHON* env vars ignored); the program is a HARDCODED bootstrap
  string in this module — never user/model-selected code, never a shell
  (``shell=False``, argv list only);
- resource limits are applied INSIDE the worker at startup (single-threaded,
  before any parsing) via ``resource.setrlimit`` — deliberately NOT via
  ``preexec_fn``, which is unsafe in a multithreaded parent;
- ``start_new_session=True`` (setsid, done in C in the child — safe in
  multithreaded parents) makes the worker a process-group leader so the whole
  group can be SIGKILLed on timeout/cancellation;
- environment is a minimal hardcoded dict (no inherited API credentials, no
  secrets; HOME/TMPDIR point inside the per-job directory);
- the worker-side bootstrap installs an in-process network guard (socket
  connect/getaddrinfo raise) — best-effort, because a document cannot execute
  code; the enforceable boundary is "untrusted content never runs in-process";
- filesystem: the worker opens exactly two paths handed to it (input, output)
  inside a per-job directory created with mode 0700; the job directory is
  removed after every job;
- IPC is bounded length-prefixed JSON frames (request/response caps) plus a
  bounded output file (content cap);
- concurrency is bounded by a fixed worker semaphore AND a bounded pending
  queue (full queue = fail closed, never unbounded);
- wall-clock deadline covers worker start -> response frame complete; on
  expiry the process GROUP is killed and the result is marked timed_out.

Enforceable vs best-effort (documented honestly):
- enforceable (POSIX): wall-clock deadline + process-group kill, rlimits
  (RLIMIT_AS/CPU/FSIZE/CORE), bounded IPC, bounded concurrency, sanitized env,
  no-shell, fixed argv[0], per-job fs paths only, cleanup.
- best-effort: the in-process network guard (a parser bug could in theory
  bypass it; combined with no credentials + no untrusted code execution this is
  defense in depth, not the primary boundary). POSIX namespaces/seccomp are NOT
  used (require privileges, not portable) — documented limitation.
- NOT available: pre-W1 platforms without ``resource``/process groups. On
  non-POSIX (Windows) the pool FAILS CLOSED (ParserWorkerUnavailable) — see
  docs/web-document-engine-w1.md for the Windows Job Objects future path. No
  timeout-only insecure mode exists.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _parser_worker
from .errors import HiveError


class ParserIsolationError(HiveError):
    """Base error for parser-worker isolation failures (all fail closed)."""


class ParserWorkerUnavailable(ParserIsolationError):
    """Isolation cannot be established on this platform — operation disabled."""


class ParserJobRejected(ParserIsolationError):
    """Job rejected before spawn: unknown op, oversized input/request, full queue."""


class ParserJobTimeout(ParserIsolationError):
    """No worker slot became available within the deadline."""


#: Hardcoded worker bootstrap. Runs with ``python -I -S -c``: -I removes the script
#: dir and user site and ignores PYTHON* env vars; -S additionally skips the `site`
#: module entirely, so NO .pth file executes anywhere (with -I alone, .pth files in
#: SYSTEM site-packages would still run — a real code-execution vector; -S closes
#: it because the bootstrap injects package dirs as PLAIN sys.path entries, which
#: never triggers .pth processing). The bootstrap applies rlimits FIRST (single-
#: threaded at that point — never preexec_fn in the multithreaded parent), installs
#: the network guard, then imports the trusted worker module and runs main().
_WORKER_BOOTSTRAP = r"""
import json as _json, os as _os, resource as _resource, socket as _socket, sys as _sys
_src = _sys.argv[1]
_cfg = _json.loads(_sys.argv[2])
_sys.path.insert(0, _src)
# interpreter-standard package locations only (site-packages/user-site), computed
# by the parent from ITS OWN interpreter — never from documents or env vars;
# needed because -I disables the user site where pip may install extras
for _p in _cfg.get("extra_paths", []):
    try:
        if _p and _os.path.isdir(_p):
            _sys.path.append(_p)
    except Exception:
        pass
# optional import blocklist (test/diagnostic use): modules that must never be
# importable inside the worker, regardless of where they are installed
_blocked = set(_cfg.get("blocked_modules", []))
if _blocked:
    import importlib.abc as _abc

    class _Blocker(_abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _blocked:
                raise ImportError(f"module blocked in parser worker: {name}")
            return None

    _sys.meta_path.insert(0, _Blocker())
for _name, _soft, _hard in _cfg["rlimits"]:
    try:
        _resource.setrlimit(getattr(_resource, _name), (_soft, _hard))
    except (ValueError, OSError, AttributeError, OverflowError) as _e:
        _sys.stderr.write("rlimit %s failed: %s\n" % (_name, _e))
        _sys.exit(3)  # fail closed: limits must apply before any parsing
class _NoNetSocket(_socket.socket):
    def connect(self, *a, **k):
        raise OSError("network disabled in parser worker")
    def connect_ex(self, *a, **k):
        return 111
def _no_net(*a, **k):
    raise OSError("network disabled in parser worker")
_socket.socket = _NoNetSocket
_socket.create_connection = _no_net
_socket.getaddrinfo = _no_net
_socket.gethostbyname = _no_net
from geosgan_hive import _parser_worker as _w
_sys.exit(_w.main(_sys.argv[3:]))
"""


@dataclass(frozen=True)
class ParserResourceLimits:
    """Limits applied inside the worker (POSIX rlimits) and enforced by the parent."""

    max_memory_bytes: int = 512 << 20      # RLIMIT_AS
    max_cpu_seconds: int = 10              # RLIMIT_CPU (SIGXCPU at soft, SIGKILL at hard)
    max_output_file_bytes: int = 32 << 20  # RLIMIT_FSIZE + parent-side content cap
    max_request_bytes: int = 256 << 10     # JSON request frame cap
    max_response_bytes: int = 4 << 20      # JSON response frame cap
    max_input_bytes: int = 32 << 20        # input staging file cap


@dataclass
class ParserJobResult:
    ok: bool
    op: str
    timed_out: bool = False
    error: str = ""
    returncode: int | None = None
    response: dict[str, Any] | None = None
    output_sha256: str | None = None
    output_bytes: int | None = None
    output: bytes | None = None  # bounded bulk payload (content cap), for parse results
    request_bytes: int = 0
    started: float = 0.0
    ended: float = 0.0

    @property
    def duration(self) -> float:
        return self.ended - self.started


@dataclass
class _PoolConfig:
    max_workers: int
    max_pending: int
    op_timeout_seconds: float
    limits: ParserResourceLimits
    work_root: str | None
    platform: str


class ParserWorkerPool:
    """Bounded pool of isolated parser-worker subprocesses.

    Fail-closed: on non-POSIX platforms every ``run`` raises
    ``ParserWorkerUnavailable`` (Windows Job Objects are a documented FUTURE
    path, not a timeout-only insecure mode). A full pending queue, an unknown
    op, or an oversized payload raises ``ParserJobRejected`` without spawning.
    """

    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_pending: int = 8,
        op_timeout_seconds: float = 20.0,
        limits: ParserResourceLimits | None = None,
        work_root: str | None = None,
        platform: str | None = None,
        worker_extra_paths: list[str] | None = None,
        worker_blocked_modules: list[str] | None = None,
    ) -> None:
        if max_workers < 1 or max_workers > 16:
            raise ValueError("max_workers must be in [1, 16]")
        if max_pending < 1 or max_pending > 64:
            raise ValueError("max_pending must be in [1, 64]")
        if op_timeout_seconds <= 0:
            raise ValueError("op_timeout_seconds must be > 0")
        self._cfg = _PoolConfig(
            max_workers=max_workers,
            max_pending=max_pending,
            op_timeout_seconds=float(op_timeout_seconds),
            limits=limits or ParserResourceLimits(),
            work_root=work_root,
            platform=platform or os.name,
        )
        self._sem = threading.Semaphore(max_workers)
        self._pending: queue.Queue[int] = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._active = 0
        self._closed = False
        self._src_dir = str(Path(__file__).resolve().parent.parent)
        #: None = interpreter-standard site dirs; [] = none (used by tests to
        #: prove fail-closed behavior when optional parser deps are missing)
        self._worker_extra_paths = worker_extra_paths
        self._worker_blocked = list(worker_blocked_modules or [])
        if work_root is not None:
            Path(work_root).mkdir(parents=True, exist_ok=True)

    # ---- introspection (tests/monitoring) ----
    @property
    def active_jobs(self) -> int:
        with self._lock:
            return self._active

    @property
    def pending_jobs(self) -> int:
        return self._pending.qsize()

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> ParserWorkerPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- internals ----
    def _minimal_env(self, job_dir: str) -> dict[str, str]:
        """Sanitized environment: no inherited credentials/secrets; HOME and
        TMPDIR point inside the per-job directory so nothing escapes it."""
        return {
            "PATH": os.defpath,
            "LC_ALL": "C.UTF-8",
            "HOME": job_dir,
            "TMPDIR": job_dir,
        }

    def _rlimits_cfg(self) -> list[list[Any]]:
        lim = self._cfg.limits
        cpu_hard = max(lim.max_cpu_seconds + 1, 2)
        return [
            ["RLIMIT_AS", lim.max_memory_bytes, lim.max_memory_bytes],
            ["RLIMIT_CPU", lim.max_cpu_seconds, cpu_hard],
            ["RLIMIT_FSIZE", lim.max_output_file_bytes, lim.max_output_file_bytes],
            ["RLIMIT_CORE", 0, 0],
        ]

    def _kill_group(self, proc: subprocess.Popen) -> None:
        """Kill the whole process group (worker is a group leader via
        start_new_session). Safe in a multithreaded parent (no preexec_fn)."""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    def _read_capped(self, path: str, cap: int) -> tuple[bytes, bool]:
        """Read a file up to cap+1 bytes; returns (data, truncated)."""
        try:
            with open(path, "rb") as fh:
                data = fh.read(cap + 1)
            return data, len(data) > cap
        except OSError:
            return b"", False

    def _parse_response_frame(self, raw: bytes, truncated: bool) -> dict[str, Any]:
        lim = self._cfg.limits
        if truncated or len(raw) < 4:
            raise ParserJobRejected(f"response frame exceeds cap {lim.max_response_bytes}")
        length = int.from_bytes(raw[:4], "big")
        if length > lim.max_response_bytes or length != len(raw) - 4:
            raise ParserJobRejected(f"response frame exceeds cap {lim.max_response_bytes}")
        resp = json.loads(raw[4:].decode("utf-8"))
        if not isinstance(resp, dict):
            raise ParserJobRejected("response frame is not a JSON object")
        return resp

    # ---- public API ----
    def run(self, op: str, input_bytes: bytes = b"", *,
            args: dict[str, Any] | None = None,
            timeout: float | None = None) -> ParserJobResult:
        """Run one trusted op in an isolated worker. ``args`` are bounded JSON
        op fields (each op re-validates them; the op table itself is fixed).
        Never raises for worker failures (they are reported in the result);
        raises only for pre-spawn rejections, queue exhaustion, slot timeout,
        or platform unavailability."""
        cfg = self._cfg
        if cfg.platform != "posix":
            raise ParserWorkerUnavailable(
                f"parser-worker isolation is disabled on platform {cfg.platform!r}: "
                "POSIX rlimits + process groups are required; the documented Windows "
                "Job Objects path is not yet implemented (fail closed)"
            )
        with self._lock:
            if self._closed:
                raise ParserIsolationError("pool is closed")
        if not isinstance(op, str) or op not in _parser_worker.OPS:
            raise ParserJobRejected(f"unknown parser op {op!r} (trusted ops only)")
        if not isinstance(input_bytes, (bytes, bytearray)):
            raise ParserJobRejected("input must be bytes")
        input_bytes = bytes(input_bytes)
        if len(input_bytes) > cfg.limits.max_input_bytes:
            raise ParserJobRejected(
                f"input {len(input_bytes)} exceeds cap {cfg.limits.max_input_bytes}")

        # bounded pending gate: full queue fails closed immediately
        try:
            self._pending.put_nowait(1)
        except queue.Full:
            raise ParserJobRejected(
                f"parser pending queue full ({cfg.max_pending}) — fail closed") from None

        deadline_s = cfg.op_timeout_seconds if timeout is None else float(timeout)
        if deadline_s <= 0:
            self._pending.get_nowait()
            raise ParserJobRejected("timeout must be > 0")
        started = time.monotonic()
        try:
            # deadline-aware slot acquisition: never wait past the deadline
            if not self._sem.acquire(timeout=deadline_s):
                raise ParserJobTimeout(
                    f"no parser worker slot within {deadline_s}s — fail closed")
            try:
                with self._lock:
                    self._active += 1
                return self._run_locked(op, input_bytes, args, started, deadline_s)
            finally:
                with self._lock:
                    self._active -= 1
                self._sem.release()
        finally:
            self._pending.get_nowait()

    def _run_locked(self, op: str, input_bytes: bytes, args: dict[str, Any] | None,
                    started: float, deadline_s: float) -> ParserJobResult:
        cfg = self._cfg
        lim = cfg.limits
        job_dir = tempfile.mkdtemp(prefix="job-", dir=cfg.work_root)
        try:
            os.chmod(job_dir, stat.S_IRWXU)  # 0700: owner-only per-job directory
        except OSError:
            pass
        input_path = os.path.join(job_dir, "input.bin")
        output_path = os.path.join(job_dir, "output.bin")
        stdout_path = os.path.join(job_dir, "stdout.bin")
        stderr_path = os.path.join(job_dir, "stderr.bin")
        with open(input_path, "wb") as fh:
            fh.write(input_bytes)

        if args:
            if not isinstance(args, dict) or not all(isinstance(k, str) for k in args):
                raise ParserJobRejected("op args must be a JSON object with string keys")
            request: dict[str, Any] = dict(args)  # fixed keys below always win
        else:
            request = {}
        # hard caps come from the POOL and can only be LOWERED by op args
        # (models/tools must never raise security limits); other op-specific
        # limits (e.g. max_extract_chars) pass through
        req_limits: dict[str, Any] = {
            "max_input_bytes": lim.max_input_bytes,
            "max_output_bytes": lim.max_output_file_bytes,
        }
        arg_limits = request.get("limits")
        if isinstance(arg_limits, dict):
            for k, v in arg_limits.items():
                if k in req_limits:
                    try:
                        req_limits[k] = min(req_limits[k], int(v))
                    except (TypeError, ValueError):
                        pass
                else:
                    req_limits[k] = v
        request.update({
            "op": op,
            "input_path": input_path,
            "output_path": output_path,
            "limits": req_limits,
            # worker-side watchdog: self-terminates on wall-clock deadline
            # (parent deadline + grace) or parent death, so orphaned/blocked
            # workers do not outlive their job even without a parent kill
            "watchdog": {
                "deadline_s": min(deadline_s + 5.0, 3600.0),
                "parent_pid": os.getpid(),
            },
        })
        req_payload = json.dumps(request).encode("utf-8")
        if len(req_payload) > lim.max_request_bytes:
            raise ParserJobRejected(
                f"request frame {len(req_payload)} exceeds cap {lim.max_request_bytes}")
        frame = len(req_payload).to_bytes(4, "big") + req_payload

        if self._worker_extra_paths is None:
            import site

            # W4 residual: SYSTEM site-packages come FIRST, user-site LAST, so a
            # (less trustworthy, user-writable) user-site package can never
            # shadow a system one. The worker runs with -I -S (see bootstrap), so
            # site never runs and NO .pth file executes; these dirs are injected
            # as plain module lookup paths only, in this fixed order.
            extra_paths = [*site.getsitepackages(), site.getusersitepackages()]
        else:
            extra_paths = list(self._worker_extra_paths)
        argv = [
            sys.executable, "-I", "-S", "-c", _WORKER_BOOTSTRAP,
            self._src_dir,
            json.dumps({"rlimits": self._rlimits_cfg(), "extra_paths": extra_paths,
                        "blocked_modules": self._worker_blocked}),
            input_path, output_path,
        ]
        deadline = started + deadline_s
        proc: subprocess.Popen | None = None
        timed_out = False
        try:
            with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=out_fh,
                    stderr=err_fh,
                    env=self._minimal_env(job_dir),
                    cwd=job_dir,
                    shell=False,               # never a shell
                    start_new_session=True,    # setsid in C: safe in threaded parent
                )
            try:
                assert proc.stdin is not None
                proc.stdin.write(frame)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # worker died early; wait() below reports the exit code

            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    self._kill_group(proc)
                    proc.wait(timeout=5)
                    break
                time.sleep(0.02)
            ended = time.monotonic()

            raw_out, out_trunc = self._read_capped(stdout_path, lim.max_response_bytes)
            err_data, _ = self._read_capped(stderr_path, 64 << 10)
            result = ParserJobResult(
                ok=False, op=op, timed_out=timed_out,
                returncode=proc.returncode, started=started, ended=ended,
                request_bytes=len(frame),
            )
            if timed_out:
                result.error = (f"parser worker exceeded deadline {deadline_s}s; "
                                "process group killed")
                return result
            if proc.returncode == 3:
                result.error = "worker failed closed: resource limits could not be applied"
                return result
            if proc.returncode == 4:
                result.error = "worker watchdog: wall-clock deadline self-termination"
                return result
            if proc.returncode == 5:
                result.error = "worker watchdog: parent death detected, self-terminated"
                return result
            try:
                resp = self._parse_response_frame(raw_out, out_trunc)
            except ParserJobRejected as exc:
                result.error = str(exc)
                self._kill_group(proc)
                return result
            if proc.returncode != 0 or not resp.get("ok"):
                result.error = str(resp.get("error") or err_data.decode("utf-8", "replace")[:500]
                                 or f"worker exit {proc.returncode}")
                return result
            result.ok = True
            result.response = resp
            out_data, truncated = self._read_capped(output_path, lim.max_output_file_bytes)
            if truncated:
                result.ok = False
                result.error = (f"output file exceeds cap {lim.max_output_file_bytes}")
                return result
            if out_data:
                import hashlib

                result.output_sha256 = hashlib.sha256(out_data).hexdigest()
                result.output_bytes = len(out_data)
                result.output = out_data
            return result
        finally:
            if proc is not None and proc.poll() is None:
                self._kill_group(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            shutil.rmtree(job_dir, ignore_errors=True)  # per-job cleanup, always
