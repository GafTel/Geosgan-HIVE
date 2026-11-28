"""M2 (v0.9.0, EXPERIMENTAL): local stdio MCP transport with isolated subprocess
spawning. Production stdio activation remains BLOCKED (amendment C/D) — this
module only provides testable infrastructure for trusted fixed fixtures.

Isolation pattern (mirrors the W1 parser worker, deliberately NOT preexec_fn,
which is unsafe in a multithreaded parent): the child is started as
``python -I -S -c <bootstrap>``; the bootstrap applies rlimits FIRST
(single-threaded at that point), installs an in-process network guard, then
``os.execv`` the operator-approved executable. HONEST BOUNDARY (amendment C):
rlimits and the process group survive execv; the in-process network guard does
NOT survive exec (a fresh interpreter image is loaded) — after exec the child
is NOT network-isolated by this mechanism. Stdio therefore provides resource
limits, sanitized env, and whole-group kill, but NOT filesystem or network
isolation; production activation stays BLOCKED on exactly that basis. The
process runs in its own session (start_new_session=True) so the whole process
GROUP can be SIGKILLed on timeout/cancellation; a watchdog thread also
self-terminates on wall-clock deadline or parent death.

Executable identity checks (amendment D review): absolute path only (no PATH
lookup), not a symlink, regular file, not group/world-writable, basename not a
shell/launcher/interpreter. Residual (documented, not hidden): a same-user
process can still replace the file between check and exec — the trusted-process
boundary applies; rlimits/env/process-groups are NOT a complete filesystem or
network sandbox.

Env: sanitized minimal dict + only the allowlisted env_ref NAMES resolved from
the parent environment (values never stored in config; nothing else inherited).
stderr is drained to a bounded file (never blocks, never parsed). stdio framing
is newline-delimited JSON-RPC with a hard per-line cap and bounded pending
messages.

Non-POSIX (Windows): FAIL CLOSED (no Job Objects yet — documented future path).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from .errors import HiveError
from .mcp_config import McpServerConfig

MAX_STDIO_LINE_BYTES = 4 << 20          # per JSON-RPC line cap
MAX_STDIO_PENDING = 64                  # bounded pending inbound messages
MAX_STDIO_STDERR_BYTES = 1 << 20        # stderr drain cap
STDIO_OP_TIMEOUT_DEFAULT = 60.0


class McpStdioError(HiveError):
    """stdio transport failure — fail closed."""


#: hardcoded bootstrap: rlimits -> PDEATHSIG -> network guard -> execv the
#: approved executable. PR_SET_PDEATHSIG is set BEFORE execv and survives it
#: (kernel attribute, cleared only by setuid exec): if the HIVE parent dies,
#: the kernel SIGKILLs the whole stdio child even though it exec'd a foreign
#: image (the parent-side watchdog thread below cannot see its own death).
_MCP_STDIO_BOOTSTRAP = r"""
import ctypes as _ctypes, json as _json, os as _os, resource as _resource
import signal as _signal, socket as _socket, sys as _sys
_cfg = _json.loads(_sys.argv[1])
for _name, _soft, _hard in _cfg["rlimits"]:
    try:
        _resource.setrlimit(getattr(_resource, _name), (_soft, _hard))
    except (ValueError, OSError, AttributeError, OverflowError) as _e:
        _sys.stderr.write("rlimit %s failed\n" % _name)
        _sys.exit(3)  # fail closed: limits must apply before exec
try:
    _ctypes.CDLL(None, use_errno=True).prctl(1, _signal.SIGKILL)  # PR_SET_PDEATHSIG
    if _os.getppid() == 1:
        _sys.exit(4)  # parent already gone before the prctl took effect
except Exception:
    _sys.exit(4)  # no prctl -> cannot guarantee parent-death cleanup -> fail closed
class _NoNet(_socket.socket):
    def connect(self, *a, **k):
        raise OSError("network disabled in MCP stdio bootstrap")
    def connect_ex(self, *a, **k):
        return 111
def _no_net(*a, **k):
    raise OSError("network disabled in MCP stdio bootstrap")
_socket.socket = _NoNet
_socket.create_connection = _no_net
_socket.getaddrinfo = _no_net
_socket.gethostbyname = _no_net
_exe = _sys.argv[2]
_os.execv(_exe, [_exe] + _sys.argv[3:])
"""


def check_executable(path: str) -> None:
    """Amendment-D executable identity checks (fail closed). The launcher
    denylist is checked FIRST (by basename, before any filesystem access) so an
    interpreter/launcher is rejected even when it is a symlink."""
    if not os.path.isabs(path):
        raise McpStdioError(f"executable must be an absolute path: {path!r}")
    base = os.path.basename(path).lower()
    if base in ("sh", "bash", "zsh", "cmd", "powershell", "pwsh", "npx", "uvx", "uv",
                "pip", "pip3", "python", "python3"):
        raise McpStdioError(f"executable {base!r} is a shell/launcher/interpreter — rejected")
    try:
        st = os.lstat(path)  # lstat: a symlink is rejected, not followed
    except OSError as exc:
        raise McpStdioError(f"executable not accessible: {path!r}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise McpStdioError(f"executable is a symlink — rejected: {path!r}")
    if not stat.S_ISREG(st.st_mode):
        raise McpStdioError(f"executable is not a regular file: {path!r}")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise McpStdioError(f"executable is group/world-writable — rejected: {path!r}")


def sanitized_env(job_dir: str, env_refs: list[str], environ: dict[str, str] | None) -> dict:
    """Minimal env: no inherited credentials; HOME/TMPDIR inside the per-job
    dir; only allowlisted env_ref NAMES are resolved from the parent env."""
    env = {"PATH": os.defpath, "LC_ALL": "C.UTF-8", "HOME": job_dir, "TMPDIR": job_dir}
    src = environ if environ is not None else os.environ
    for name in env_refs:
        if name in src:
            env[name] = src[name]
    return env


class StdioProcess:
    """One isolated MCP stdio subprocess (process-group leader)."""

    def __init__(self, server: McpServerConfig, *, work_root: str | None,
                 timeout_s: float, environ: dict[str, str] | None = None,
                 platform: str | None = None) -> None:
        if (platform or os.name) != "posix":
            raise McpStdioError(
                "MCP stdio isolation is POSIX-only (rlimits + process groups); "
                "fails closed on this platform — Windows Job Objects are a documented "
                "future path, no insecure fallback")
        t = server.transport
        assert t.kind == "stdio"
        check_executable(t.executable)
        self._server = server
        self.timeout_s = float(timeout_s)
        job_dir = tempfile.mkdtemp(prefix="mcp-job-", dir=work_root)
        try:
            os.chmod(job_dir, stat.S_IRWXU)
        except OSError:
            pass
        self.job_dir = job_dir
        err_path = os.path.join(job_dir, "stderr.log")
        self._err_fh = open(err_path, "wb")
        rlimits = [
            ["RLIMIT_AS", 512 << 20, 512 << 20],
            ["RLIMIT_CPU", max(int(timeout_s), 1), max(int(timeout_s) + 1, 2)],
            ["RLIMIT_FSIZE", 32 << 20, 32 << 20],
            ["RLIMIT_CORE", 0, 0],
            ["RLIMIT_NPROC", 64, 64],
        ]
        argv = [sys.executable, "-I", "-S", "-c", _MCP_STDIO_BOOTSTRAP,
                json.dumps({"rlimits": rlimits}), t.executable, *t.args]
        env = sanitized_env(job_dir, list(t.env_refs), environ)
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err_fh,
            env=env, cwd=job_dir, shell=False, start_new_session=True)
        self._deadline = time.monotonic() + self.timeout_s
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        # parent-side watchdog: wall-clock deadline + fast exit when the child
        # died. Parent-DEATH is handled kernel-side (PR_SET_PDEATHSIG in the
        # bootstrap, which survives execv) — a thread in the parent cannot
        # observe the parent's own death.
        while True:
            if time.monotonic() >= self._deadline:
                self.kill()
                return
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.kill()
            except OSError:
                pass

    def close(self) -> None:
        self.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        self._err_fh.close()
        shutil.rmtree(self.job_dir, ignore_errors=True)

    def stderr_bytes(self) -> bytes:
        """Bounded stderr snapshot (for diagnostics; never parsed)."""
        try:
            with open(os.path.join(self.job_dir, "stderr.log"), "rb") as fh:
                return fh.read(MAX_STDIO_STDERR_BYTES + 1)[:MAX_STDIO_STDERR_BYTES]
        except OSError:
            return b""


def read_bounded_line(read1) -> bytes | None:
    """Read one newline-terminated line with a hard cap (returns None on EOF).
    `read1(n)` is a callable reading up to n bytes."""
    buf = bytearray()
    while True:
        chunk = read1(65_536)
        if not chunk:
            return bytes(buf) if buf else None
        buf += chunk
        idx = buf.find(b"\n")
        if idx >= 0:
            if len(buf) > MAX_STDIO_LINE_BYTES:
                raise McpStdioError(f"stdio line exceeds {MAX_STDIO_LINE_BYTES}B cap")
            return bytes(buf[:idx])
        if len(buf) > MAX_STDIO_LINE_BYTES:
            raise McpStdioError(f"stdio line exceeds {MAX_STDIO_LINE_BYTES}B cap")


@contextlib.asynccontextmanager
async def stdio_transport(server: McpServerConfig, *, work_root: str | None = None,
                          timeout_s: float = STDIO_OP_TIMEOUT_DEFAULT,
                          environ: dict[str, str] | None = None,
                          platform: str | None = None):
    """Async context manager yielding (read_stream, write_stream, get_session_id)
    — the same shape the SDK's stdio_client/streamablehttp_client yield — fed
    from an isolated StdioProcess. Newline-delimited JSON-RPC framing with a
    hard per-line cap and bounded pending messages; malformed frames are
    reported as stream errors, never executed."""
    import anyio
    from mcp.shared.message import SessionMessage  # lazy: optional mcp extra

    proc = StdioProcess(server, work_root=work_root, timeout_s=timeout_s,
                        environ=environ, platform=platform)
    read_writer, read_stream = anyio.create_memory_object_stream(MAX_STDIO_PENDING)
    write_stream, write_reader = anyio.create_memory_object_stream(MAX_STDIO_PENDING)

    write_lock = threading.Lock()

    def _write_frame(payload: bytes) -> None:
        # in a worker thread: a pipe write to a stalled child must never block
        # the event loop; the lock preserves frame ordering
        with write_lock:
            assert proc.proc.stdin is not None
            proc.proc.stdin.write(payload + b"\n")
            proc.proc.stdin.flush()

    async def _pump_outbound():
        try:
            async with write_reader:
                async for msg in write_reader:
                    payload = json.dumps(
                        msg.message.model_dump(by_alias=True, mode="json",
                                                exclude_none=True)).encode("utf-8")
                    if len(payload) > MAX_STDIO_LINE_BYTES:
                        raise McpStdioError("outbound frame exceeds line cap")
                    await anyio.to_thread.run_sync(_write_frame, payload)
        except Exception as exc:
            await read_writer.send(McpStdioError(f"stdio write failed: {type(exc).__name__}"))
        finally:
            await read_writer.aclose()

    async def _pump_inbound():
        import mcp.types as types  # lazy: optional mcp extra

        try:
            while True:
                line = await anyio.to_thread.run_sync(
                    read_bounded_line, proc.proc.stdout.read1)
                if line is None:
                    break
                if not line.strip():
                    continue
                try:
                    msg = types.JSONRPCMessage.model_validate_json(line)
                except Exception as exc:
                    raise McpStdioError(f"malformed JSON-RPC frame: {exc}") from exc
                await read_writer.send(SessionMessage(msg))
        except Exception as exc:
            await read_writer.send(exc)
        finally:
            await read_writer.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_pump_inbound)
        tg.start_soon(_pump_outbound)
        try:
            yield read_stream, write_stream, lambda: None
        finally:
            tg.cancel_scope.cancel()
            proc.close()
