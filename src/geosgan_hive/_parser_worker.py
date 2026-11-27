"""W1 (v0.8.0 planning): TRUSTED parser-worker entrypoint.

This module runs ONLY inside an isolated parser-worker subprocess spawned by
``parser_isolation.ParserWorkerPool`` (see the hardcoded bootstrap there). It is
never imported for its side effects in the main agent process beyond reading the
``OPS`` registry for parent-side op validation; all isolation guards (rlimits,
network block) are installed by the BOOTSTRAP in the worker process before this
module is imported.

Trust boundary: the worker program is trusted HIVE code. Untrusted DOCUMENT
CONTENT is only ever passed to an op as data (file path / bytes). Documents
cannot select ops, code, executables, or arguments beyond the bounded JSON
fields each op validates. Real document parsers (W4) plug into ``OPS`` here.

IPC protocol (bounded, length-prefixed):
- request:  4-byte big-endian length + UTF-8 JSON  on stdin  (cap: request cap)
- response: 4-byte big-endian length + UTF-8 JSON  on stdout (cap: response cap)
- bulk content: written by the op to the output path given in the request;
  the parent reads it back with its own cap.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

#: hard caps enforced INSIDE the worker as well (defense in depth; the parent
#: enforces its own caps when writing/reading the frames)
MAX_REQUEST_BYTES = 1 << 20          # 1 MiB JSON request
MAX_RESPONSE_BYTES = 8 << 20         # 8 MiB JSON response
MAX_INPUT_BYTES = 64 << 20           # 64 MiB input file
MAX_OUTPUT_BYTES = 64 << 20          # 64 MiB output file


class WorkerProtocolError(Exception):
    """Malformed frame / op failure inside the worker."""


def _read_exact(stream: Any, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise WorkerProtocolError("eof in frame")
        buf += chunk
    return buf


def _read_frame(stream: Any, cap: int) -> bytes:
    length = int.from_bytes(_read_exact(stream, 4), "big")
    if length > cap:
        raise WorkerProtocolError(f"request frame {length} exceeds cap {cap}")
    return _read_exact(stream, length)


def _write_frame(stream: Any, payload: bytes, cap: int) -> None:
    if len(payload) > cap:
        raise WorkerProtocolError(f"response frame exceeds cap {cap}")
    stream.write(len(payload).to_bytes(4, "big") + payload)
    stream.flush()


def _read_input_capped(path: str, cap: int) -> bytes:
    with open(path, "rb") as fh:  # the ONLY file the worker opens for input
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise WorkerProtocolError(f"input file exceeds cap {cap}")
    return data


# ---- trusted ops ---------------------------------------------------------------
# Each op receives the validated request dict and returns a JSON-able dict.
# Ops may read the request's input_path (capped) and write output_path (capped).

def _op_ping(req: dict[str, Any]) -> dict[str, Any]:
    echo = str(req.get("echo", ""))[:256]
    return {"echo": echo, "argv0": sys.executable}


def _op_sha256(req: dict[str, Any]) -> dict[str, Any]:
    data = _read_input_capped(req["input_path"], MAX_INPUT_BYTES)
    return {"input_sha256": hashlib.sha256(data).hexdigest(), "input_bytes": len(data)}


def _op_echo_file(req: dict[str, Any]) -> dict[str, Any]:
    """Read input (capped) and write it to the output path (capped): exercises
    bounded bulk IPC in both directions."""
    data = _read_input_capped(req["input_path"], MAX_INPUT_BYTES)
    if len(data) > MAX_OUTPUT_BYTES:
        raise WorkerProtocolError("output exceeds cap")
    with open(req["output_path"], "wb") as fh:  # the ONLY file the worker writes
        fh.write(data)
    return {"output_sha256": hashlib.sha256(data).hexdigest(), "output_bytes": len(data)}


def _op_sleep(req: dict[str, Any]) -> dict[str, Any]:
    seconds = float(req.get("seconds", 0))
    if not (0 < seconds <= 600):
        raise WorkerProtocolError("invalid sleep")
    time.sleep(seconds)
    return {"slept": seconds}


def _op_env(req: dict[str, Any]) -> dict[str, Any]:
    """Report the (sanitized) environment the worker actually sees."""
    env = {k: v for k, v in sorted(os.environ.items())}
    return {"env": env, "cwd": os.getcwd()}


def _op_rlimits(req: dict[str, Any]) -> dict[str, Any]:
    import resource

    out = {}
    for name in req.get("names", ["RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE"]):
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
            out[name] = [soft, hard]
        except (AttributeError, ValueError, OSError) as exc:
            out[name] = f"error: {exc}"
    return {"rlimits": out}


def _op_net_probe(req: dict[str, Any]) -> dict[str, Any]:
    """Attempt a network connection; the worker-side guard must block it."""
    import socket

    blocked = False
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=2)
    except OSError:
        blocked = True
    return {"connect_blocked": blocked}


def _op_spam(req: dict[str, Any]) -> dict[str, Any]:
    """Produce a large JSON response (parent must reject over-cap responses)."""
    n = int(req.get("n", 0))
    if n < 0 or n > (1 << 30):
        raise WorkerProtocolError("invalid n")
    return {"blob": "x" * n}


def _op_spam_file(req: dict[str, Any]) -> dict[str, Any]:
    """Write n bytes to the output path (parent must reject over-cap content)."""
    n = int(req.get("n", 0))
    if n < 0 or n > (1 << 30):
        raise WorkerProtocolError("invalid n")
    with open(req["output_path"], "wb") as fh:
        fh.write(b"x" * n)
    return {"output_bytes": n}


def _op_syspath(req: dict[str, Any]) -> dict[str, Any]:
    """Report sys.path + cwd: proves the worker's module resolution cannot be
    influenced by the job dir / cwd / workspace / document contents."""
    return {"sys_path": list(sys.path), "cwd": os.getcwd(),
            "path0": sys.path[0] if sys.path else None}


def _op_probe(req: dict[str, Any]) -> dict[str, Any]:
    """Resolve well-known modules and report their __file__: any module
    resolving inside the job dir / work root / workspace is a shadowing attack."""
    import importlib

    out = {}
    for name in ("json", "hashlib", "pypdf", "docx", "openpyxl", "defusedxml"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__file__", None)
        except ImportError:
            out[name] = None
    return {"modules": out}


def _op_fds(req: dict[str, Any]) -> dict[str, Any]:
    """Report open fd targets (proves close_fds: no inherited descriptors)."""
    targets = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            targets.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            continue  # fd vanished (e.g. the dir-listing fd itself)
    return {"fds": sorted(targets)}


def _make_parse_op(fn_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """W4: document parse ops. The op reads the capped input, calls the trusted
    parser function (lazy optional-dependency import inside), and writes the
    bounded JSON result to the output path; the response carries only hashes."""

    def op(req: dict[str, Any]) -> dict[str, Any]:
        from geosgan_hive import documents as _docs

        fn = getattr(_docs, fn_name)
        lim = req["limits"]
        data = _read_input_capped(req["input_path"], int(lim["max_input_bytes"]))
        res = fn(data, req.get("scope") or {}, lim)
        payload = json.dumps(res, ensure_ascii=False, default=str).encode("utf-8")
        cap = int(lim["max_output_bytes"])
        if len(payload) > cap:
            raise WorkerProtocolError("parse result exceeds output cap")
        with open(req["output_path"], "wb") as fh:
            fh.write(payload)
        return {"output_bytes": len(payload),
                "output_sha256": hashlib.sha256(payload).hexdigest()}

    return op


#: the ONLY operations the worker will execute (trusted, hardcoded; W4 adds
#: real format parsers here). The parent validates op names against this table.
OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "ping": _op_ping,
    "sha256": _op_sha256,
    "echo_file": _op_echo_file,
    "sleep": _op_sleep,
    "env": _op_env,
    "rlimits": _op_rlimits,
    "net_probe": _op_net_probe,
    "spam": _op_spam,
    "spam_file": _op_spam_file,
    "fds": _op_fds,
    "syspath": _op_syspath,
    "probe": _op_probe,
    "parse_pdf": _make_parse_op("worker_parse_pdf"),
    "parse_docx": _make_parse_op("worker_parse_docx"),
    "parse_xlsx": _make_parse_op("worker_parse_xlsx"),
}


def _arm_watchdog(req: dict[str, Any]) -> None:
    """Worker-side wall-clock + parent-death watchdog (W1 residual gate).

    Runs as a daemon thread for the whole op: if the wall-clock deadline
    (duration from worker start, capped) expires, or the parent process is gone
    (os.getppid() != expected), the worker SELF-TERMINATES via os._exit — so an
    orphaned or I/O-blocked worker cannot run forever even when nobody kills
    it (rlimits alone do not cover sleeping/blocked workers). Residual risk:
    a process stuck in uninterruptible kernel sleep (D state) cannot be
    terminated by anyone — documented, not claimed away."""
    wd = req.get("watchdog") or {}
    try:
        deadline_s = float(wd.get("deadline_s", 0)) if wd else 0
    except (TypeError, ValueError):
        deadline_s = 0
    deadline_s = min(max(deadline_s, 0.0), 3600.0)  # hard cap
    parent_pid = wd.get("parent_pid") if wd else None
    try:
        parent_pid = int(parent_pid) if parent_pid is not None else None
    except (TypeError, ValueError):
        parent_pid = None
    deadline = time.monotonic() + deadline_s if deadline_s > 0 else None

    def _wd() -> None:
        while True:
            if parent_pid is not None and hasattr(os, "getppid") \
                    and os.getppid() != parent_pid:
                os._exit(5)  # parent death detected
            if deadline is not None and time.monotonic() >= deadline:
                os._exit(4)  # wall-clock deadline self-termination
            time.sleep(0.1)

    threading.Thread(target=_wd, daemon=True).start()


def main(argv: list[str]) -> int:
    """Worker entrypoint called by the bootstrap with [input_path, output_path]."""
    try:
        input_path, output_path = argv[0], argv[1]
        raw = _read_frame(sys.stdin.buffer, MAX_REQUEST_BYTES)
        req = json.loads(raw.decode("utf-8"))
        if not isinstance(req, dict):
            raise WorkerProtocolError("request must be a JSON object")
        op = req.get("op")
        if op not in OPS:
            raise WorkerProtocolError(f"unknown op {op!r}")
        req.setdefault("input_path", input_path)
        req.setdefault("output_path", output_path)
        _arm_watchdog(req)  # armed BEFORE any op execution
        result = OPS[op](req)
        resp = {"ok": True, "op": op, **result}
        payload = json.dumps(resp).encode("utf-8")
        _write_frame(sys.stdout.buffer, payload, MAX_RESPONSE_BYTES)
        return 0
    except WorkerProtocolError as exc:
        try:
            _write_frame(
                sys.stdout.buffer,
                json.dumps({"ok": False, "error": str(exc)[:500]}).encode("utf-8"),
                MAX_RESPONSE_BYTES,
            )
        except Exception:
            pass
        sys.stderr.write(f"worker error: {exc}\n")
        return 1
    except Exception as exc:  # never leak a traceback with sensitive data
        sys.stderr.write(f"worker failed: {type(exc).__name__}\n")
        return 2
