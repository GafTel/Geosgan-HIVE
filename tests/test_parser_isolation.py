"""W1: parser-worker isolation (v0.8.0 planning stage).

All tests use TRUSTED deterministic ops from _parser_worker.OPS only — no
malicious generated code is executed. POSIX-only behavior is asserted on
posix; on other platforms the pool must fail closed (see
test_non_posix_fails_closed, which runs everywhere).
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time

import pytest

from geosgan_hive import _parser_worker
from geosgan_hive.parser_isolation import (
    ParserIsolationError,
    ParserJobRejected,
    ParserJobTimeout,
    ParserResourceLimits,
    ParserWorkerPool,
    ParserWorkerUnavailable,
)

POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="parser isolation is POSIX-only")


def _pool(tmp_path, **kw) -> ParserWorkerPool:
    kw.setdefault("work_root", str(tmp_path))
    return ParserWorkerPool(**kw)


# ---- happy path ---------------------------------------------------------------

@posix_only
def test_ping_roundtrip_and_fixed_interpreter(tmp_path):
    with _pool(tmp_path) as pool:
        r = pool.run("ping")
    assert r.ok and r.timed_out is False
    assert r.response["op"] == "ping"
    assert r.response["argv0"] == sys.executable  # fixed executable, not a shell
    assert r.duration < 20


@posix_only
def test_sha256_op_hashes_input(tmp_path):
    payload = b"untrusted document bytes \x00\x01\xff" * 100
    with _pool(tmp_path) as pool:
        r = pool.run("sha256", payload)
    assert r.ok
    assert r.response["input_sha256"] == hashlib.sha256(payload).hexdigest()
    assert r.response["input_bytes"] == len(payload)


@posix_only
def test_echo_file_bounded_bulk_ipc(tmp_path):
    payload = os.urandom(100_000)
    with _pool(tmp_path) as pool:
        r = pool.run("echo_file", payload)
    assert r.ok
    assert r.output_sha256 == hashlib.sha256(payload).hexdigest()
    assert r.output_bytes == len(payload)


@posix_only
def test_sleep_op_within_deadline_succeeds(tmp_path):
    with _pool(tmp_path, op_timeout_seconds=10) as pool:
        r = pool.run("sleep", args={"seconds": 0.2})
    assert r.ok and r.response["slept"] == 0.2


# ---- isolation guarantees -------------------------------------------------------

@posix_only
def test_worker_env_is_sanitized_no_inherited_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_TEST_SECRET_ABC", "s3cr3t-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI")
    with _pool(tmp_path) as pool:
        r = pool.run("env")
    assert r.ok
    env = r.response["env"]
    keys = set(env)
    assert "HIVE_TEST_SECRET_ABC" not in keys
    assert "AWS_SECRET_ACCESS_KEY" not in keys
    for v in env.values():
        assert "s3cr3t-value" not in v and "wJalrXUtnFEMI" not in v
    assert "PATH" in keys  # minimal but functional
    # HOME/TMPDIR are contained in the per-job dir, not the real ones
    assert env["HOME"] != os.environ.get("HOME", "")


@posix_only
def test_rlimits_applied_inside_worker(tmp_path):
    lim = ParserResourceLimits(max_memory_bytes=256 << 20, max_cpu_seconds=5)
    with _pool(tmp_path, limits=lim) as pool:
        r = pool.run("rlimits")
    assert r.ok
    rl = r.response["rlimits"]
    assert rl["RLIMIT_AS"][0] == 256 << 20
    assert rl["RLIMIT_CPU"][0] == 5
    assert rl["RLIMIT_CORE"][0] == 0


@posix_only
def test_worker_network_is_blocked(tmp_path):
    with _pool(tmp_path) as pool:
        r = pool.run("net_probe")
    assert r.ok
    assert r.response["connect_blocked"] is True


@posix_only
def test_no_shell_interpretation_of_op_data(tmp_path):
    marker = tmp_path / "pwned-marker"
    with _pool(tmp_path) as pool:
        r = pool.run("ping", args={"echo": f"$(touch {marker}) `touch {marker}`"})
    assert r.ok
    assert "$(touch" in r.response["echo"]  # returned literally as data
    assert not marker.exists()  # never executed: shell=False, argv only


# ---- fail-closed rejections ------------------------------------------------------

@posix_only
def test_unknown_op_rejected_without_spawn(tmp_path):
    with _pool(tmp_path) as pool:
        with pytest.raises(ParserJobRejected, match="unknown parser op"):
            pool.run("os.system('touch /tmp/x')")
        with pytest.raises(ParserJobRejected, match="unknown parser op"):
            pool.run("../../etc/passwd")
    assert pool.active_jobs == 0 and pool.pending_jobs == 0


@posix_only
def test_oversized_input_rejected_before_spawn(tmp_path):
    lim = ParserResourceLimits(max_input_bytes=1024)
    with _pool(tmp_path, limits=lim) as pool:
        with pytest.raises(ParserJobRejected, match="exceeds cap"):
            pool.run("sha256", b"x" * 2048)
    assert pool.pending_jobs == 0


@posix_only
def test_bad_op_args_rejected(tmp_path):
    with _pool(tmp_path) as pool:
        with pytest.raises(ParserJobRejected, match="string keys"):
            pool.run("ping", args={1: "x"})  # type: ignore[dict-item]
        with pytest.raises(ParserJobRejected, match="string keys"):
            pool.run("ping", args=["not-a-dict"])  # type: ignore[arg-type]


@posix_only
def test_worker_timeout_kills_process_group_and_cleans_up(tmp_path):
    pool = _pool(tmp_path, op_timeout_seconds=30)
    r = pool.run("sleep", args={"seconds": 30}, timeout=0.6)
    pool.close()
    assert r.timed_out is True and r.ok is False
    assert "deadline" in r.error or "killed" in r.error
    # per-job directory removed
    assert not any(p.name.startswith("job-") for p in tmp_path.iterdir())
    # pool accounting restored: a fresh pool works (and this pool's slots freed)
    assert pool.active_jobs == 0 and pool.pending_jobs == 0
    with _pool(tmp_path) as pool2:
        assert pool2.run("ping").ok


@posix_only
def test_response_frame_cap_enforced(tmp_path):
    lim = ParserResourceLimits(max_response_bytes=1024)
    with _pool(tmp_path, limits=lim) as pool:
        r = pool.run("spam", args={"n": 5000})
    assert r.ok is False
    assert "cap" in r.error


@posix_only
def test_output_file_cap_enforced(tmp_path):
    lim = ParserResourceLimits(max_output_file_bytes=4096)
    with _pool(tmp_path, limits=lim) as pool:
        # RLIMIT_FSIZE=4096 in the worker kills writes beyond the cap
        r = pool.run("spam_file", args={"n": 10_000})
        r2 = pool.run("echo_file", b"x" * 10_000)
    assert r.ok is False and r2.ok is False


@posix_only
def test_pending_queue_full_fails_closed(tmp_path):
    # max_pending bounds TOTAL in-flight jobs (queued + running): while A is
    # in flight the queue is full and any further job fails closed immediately.
    pool = _pool(tmp_path, max_workers=1, max_pending=1, op_timeout_seconds=10)
    results: list = []

    def job_a():
        results.append(pool.run("sleep", args={"seconds": 1.0}, timeout=10))

    t = threading.Thread(target=job_a)
    t.start()
    deadline = time.monotonic() + 5
    while pool.active_jobs == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pool.active_jobs == 1 and pool.pending_jobs == 1
    with pytest.raises(ParserJobRejected, match="queue full"):
        pool.run("ping")
    t.join(timeout=10)
    pool.close()
    assert len(results) == 1 and results[0].ok
    # accounting restored after completion: a new job runs fine
    with _pool(tmp_path) as pool2:
        assert pool2.run("ping").ok
    assert pool.active_jobs == 0 and pool.pending_jobs == 0


@posix_only
def test_slot_timeout_raises_and_accounting_restored(tmp_path):
    pool = _pool(tmp_path, max_workers=1, max_pending=4, op_timeout_seconds=30)
    pool._sem.acquire()  # occupy the only worker slot (white-box test hook)
    try:
        with pytest.raises(ParserJobTimeout, match="no parser worker slot"):
            pool.run("ping", timeout=0.3)
        assert pool.pending_jobs == 0  # pending token returned on failure
    finally:
        pool._sem.release()
        pool.close()
    with _pool(tmp_path) as pool2:
        assert pool2.run("ping").ok


# ---- concurrency accounting ------------------------------------------------------

@posix_only
def test_worker_serialization_bounds_concurrency(tmp_path):
    pool = _pool(tmp_path, max_workers=1, max_pending=8, op_timeout_seconds=15)
    r1 = pool.run("ping")
    r2 = pool.run("ping")
    pool.close()
    assert r1.ok and r2.ok
    assert r2.started >= r1.ended - 0.001  # strictly serialized with 1 worker


@posix_only
def test_concurrent_jobs_all_succeed_and_counters_reset(tmp_path):
    pool = _pool(tmp_path, max_workers=2, max_pending=8, op_timeout_seconds=15)
    results = []
    lock = threading.Lock()

    def work():
        r = pool.run("ping")
        with lock:
            results.append(r)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    pool.close()
    assert len(results) == 6 and all(r.ok for r in results)
    assert pool.active_jobs == 0 and pool.pending_jobs == 0


# ---- platform behavior (runs everywhere) -----------------------------------------

def test_non_posix_fails_closed(tmp_path):
    for plat in ("nt", "java"):
        pool = ParserWorkerPool(platform=plat, work_root=str(tmp_path))
        with pytest.raises(ParserWorkerUnavailable, match="disabled on platform"):
            pool.run("ping")
        pool.close()


@posix_only
def test_rlimit_failure_fails_closed(tmp_path):
    # invalid limit value -> worker setrlimit raises -> exit 3 -> fail closed
    lim = ParserResourceLimits(max_memory_bytes=2**70)  # overflow -> worker exit 3
    with _pool(tmp_path, limits=lim) as pool:
        r = pool.run("ping")
    assert r.ok is False
    assert "limits" in r.error


def test_pool_validation():
    with pytest.raises(ValueError):
        ParserWorkerPool(max_workers=0)
    with pytest.raises(ValueError):
        ParserWorkerPool(max_pending=0)
    with pytest.raises(ValueError):
        ParserWorkerPool(op_timeout_seconds=0)


def test_closed_pool_rejects(tmp_path):
    pool = _pool(tmp_path)
    pool.close()
    with pytest.raises(ParserIsolationError, match="closed"):
        pool.run("ping")


# ---- W1 residual: worker-side watchdog + fd hygiene ------------------------------

def _spawn_worker_direct(tmp_path, req: dict):
    """Spawn the trusted worker directly (no pool) with a raw request frame."""
    import json as _json
    import subprocess as _sp

    from geosgan_hive.parser_isolation import _WORKER_BOOTSTRAP, ParserResourceLimits

    lim = ParserResourceLimits()
    cfg = _json.dumps({"rlimits": [
        ["RLIMIT_AS", lim.max_memory_bytes, lim.max_memory_bytes],
        ["RLIMIT_CPU", lim.max_cpu_seconds, lim.max_cpu_seconds + 1],
        ["RLIMIT_FSIZE", lim.max_output_file_bytes, lim.max_output_file_bytes],
        ["RLIMIT_CORE", 0, 0],
    ]})
    d = tmp_path / "direct"
    d.mkdir()
    ip, op = d / "in.bin", d / "out.bin"
    ip.write_bytes(b"x")
    payload = _json.dumps(req).encode()
    frame = len(payload).to_bytes(4, "big") + payload
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    proc = _sp.Popen(
        [sys.executable, "-I", "-c", _WORKER_BOOTSTRAP, src, cfg, str(ip), str(op)],
        stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
        env={"PATH": os.defpath, "LC_ALL": "C.UTF-8", "HOME": str(d), "TMPDIR": str(d)},
        cwd=str(d), shell=False, start_new_session=True)
    out, err = proc.communicate(frame, timeout=30)
    return proc.returncode, out, err


@posix_only
def test_worker_watchdog_self_terminates_on_deadline(tmp_path):
    # op sleeps 30s; worker watchdog deadline 0.5s -> worker exits 4 by itself,
    # with NO parent kill (parent just waits)
    rc, out, err = _spawn_worker_direct(tmp_path, {
        "op": "sleep", "seconds": 30,
        "watchdog": {"deadline_s": 0.5, "parent_pid": os.getpid()}})
    assert rc == 4


@posix_only
def test_worker_watchdog_detects_parent_death(tmp_path):
    # parent_pid points to a nonexistent process -> watchdog exits 5 quickly
    rc, out, err = _spawn_worker_direct(tmp_path, {
        "op": "sleep", "seconds": 30,
        "watchdog": {"deadline_s": 60, "parent_pid": 2**22 + 12345}})
    assert rc == 5


@posix_only
def test_worker_inherits_no_sensitive_file_descriptors(tmp_path):
    secret_fd_file = tmp_path / "parent-secret.db"
    secret_fd_file.write_bytes(b"parent-side secret handle")
    fh = open(secret_fd_file, "rb")  # stays open in the parent during the job
    try:
        with _pool(tmp_path) as pool:
            r = pool.run("fds")
    finally:
        fh.close()
    assert r.ok
    targets = r.response["fds"]
    assert all(not t.endswith("parent-secret.db") for t in targets)
    # only std streams + job-dir files + pipes/sockets/anon
    ok_prefixes = ("pipe:", "socket:", "anon_inode:", "/dev/")
    job_root = os.path.realpath(str(tmp_path))
    for t in targets:
        if t.startswith(ok_prefixes):
            continue
        assert os.path.realpath(t).startswith(job_root + os.sep) or t in ("/dev/null",), t


# ---- W4 residual: module resolution cannot be influenced by untrusted input ------

@posix_only
def test_worker_syspath_has_no_cwd_or_jobdir(tmp_path):
    with _pool(tmp_path) as pool:
        r = pool.run("syspath")
    assert r.ok
    sp = r.response["sys_path"]
    cwd = r.response["cwd"]
    assert "" not in sp and "." not in sp
    assert all(not p.startswith(cwd) for p in sp if p), sp
    # the package src dir is the only repo path; site dirs are appended after
    assert sp[0].endswith("src")


@posix_only
def test_worker_module_resolution_not_shadowable(tmp_path):
    # plant evil shadow modules in the work root (where job dirs live) and in a
    # fake cwd: the worker must resolve the REAL modules, never these
    (tmp_path / "pypdf.py").write_text("raise RuntimeError('shadowed pypdf')")
    (tmp_path / "json.py").write_text("raise RuntimeError('shadowed json')")
    (tmp_path / "hashlib.py").write_text("raise RuntimeError('shadowed hashlib')")
    with _pool(tmp_path) as pool:
        r = pool.run("probe")
        assert r.ok
        mods = r.response["modules"]
        for name in ("json", "hashlib"):
            assert mods[name] and "python3" in mods[name], (name, mods[name])  # stdlib
        for name in ("pypdf", "docx", "openpyxl", "defusedxml"):
            assert mods[name] is None or not mods[name].startswith(str(tmp_path)), \
                (name, mods[name])  # never from the work root
        # and the worker still works (stdlib not shadowed)
        assert pool.run("ping").ok


def test_ops_registry_is_fixed_and_trusted():
    expected = {"ping", "sha256", "echo_file", "sleep", "env", "rlimits",
                "net_probe", "spam", "spam_file", "fds", "syspath", "probe",
                "parse_pdf", "parse_docx", "parse_xlsx",
                "gen_pdf", "gen_docx", "gen_xlsx"}
    assert set(_parser_worker.OPS) == expected
    for fn in _parser_worker.OPS.values():
        assert callable(fn)
