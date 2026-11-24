import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.checkpoints import CheckpointStore, ExecutionState, PersistMode
from geosgan_hive.cli import main
from geosgan_hive.recovery import RecoveryWorker


class FakeClock:
    def __init__(self, start=1_700_000_000.0):
        self.t = start

    def __call__(self):
        return datetime.fromtimestamp(self.t, tz=UTC)

    def advance(self, seconds):
        self.t += seconds


def make_store(tmp_path, *, clock=None, persist_mode=PersistMode.RESUMABLE, name="cp.db"):
    clock = clock or FakeClock()
    store = CheckpointStore(
        str(tmp_path / name), persist_mode=persist_mode, lease_ttl_seconds=60.0,
        utcnow=clock,
    )
    return store, clock


def suspend_run(store, run_id, *, next_in=0.0, deadline_in=3600.0, owner="w",
                transcript_json=None):
    store.create_run(run_id=run_id, owner=owner, task="t", agent="supervisor", model="m")
    now = store.utcnow()
    store.mark_waiting(
        run_id, owner=owner, error="no model", counters_json="{}", steps_json="[]",
        route_json="[]", transcript_json=transcript_json, step_count=1,
        next_attempt_at=(now + timedelta(seconds=next_in)).isoformat(timespec="milliseconds"),
        deadline_at=(now + timedelta(seconds=deadline_in)).isoformat(timespec="milliseconds"),
    )


def test_worker_resumes_eligible_runs_via_shared_lease(tmp_path):
    store, clock = make_store(tmp_path)
    suspend_run(store, "due-1", next_in=0.0)
    suspend_run(store, "not-yet", next_in=600.0)
    resumed = []

    async def resume_fn(run_id, owner):
        # same lease mechanism as manual resume: claim, then fenced terminal commit
        assert store.claim_lease(run_id, owner)
        store.mark_completed(run_id, owner=owner, result="ok", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=2)
        resumed.append(run_id)

    worker = RecoveryWorker(store, resume_fn, poll_interval_seconds=5.0)
    count = asyncio.run(worker.run_once())
    assert count == 1 and resumed == ["due-1"]
    assert store.get("due-1").state == ExecutionState.COMPLETED
    assert store.get("not-yet").state == ExecutionState.WAITING_FOR_PROVIDER
    assert worker.resumed == 1 and worker.skipped == 0
    store.close()


def test_worker_skips_leased_and_deadline_expired_runs(tmp_path):
    store, clock = make_store(tmp_path)
    suspend_run(store, "leased", next_in=0.0)
    suspend_run(store, "expired", next_in=0.0, deadline_in=-1.0)  # deadline in the past
    assert store.claim_lease("leased", "other-worker")

    async def resume_fn(run_id, owner):
        raise AssertionError("must not be called")

    worker = RecoveryWorker(store, resume_fn)
    assert asyncio.run(worker.run_once()) == 0
    assert worker.resumed == 0
    store.close()


def test_worker_rejects_busy_polling_interval(tmp_path):
    store, _ = make_store(tmp_path)
    with pytest.raises(ValueError, match="busy polling"):
        RecoveryWorker(store, lambda r, o: asyncio.sleep(0), poll_interval_seconds=0.5)
    store.close()


def test_worker_run_forever_stops_cleanly_and_wakes_early(tmp_path):
    store, _ = make_store(tmp_path)
    suspend_run(store, "r1", next_in=0.0)
    events = []

    async def resume_fn(run_id, owner):
        assert store.claim_lease(run_id, owner)  # same lease mechanism as manual resume
        events.append(run_id)
        store.mark_completed(run_id, owner=owner, result="ok", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=2)

    worker = RecoveryWorker(store, resume_fn, poll_interval_seconds=60.0,
                            on_event=lambda kind, meta: (events.append(kind), worker.stop())
                            if kind == "resumed" else events.append(kind))
    # poll interval is 60s, but stop() wakes the loop -> returns immediately
    asyncio.run(worker.run_forever())
    assert worker.resumed == 1
    assert "resumed" in events
    store.close()


def test_worker_cancellation_releases_lease_and_stops(tmp_path):
    store, _ = make_store(tmp_path)
    suspend_run(store, "r-cancel", next_in=0.0)
    claimed = asyncio.Event()

    async def resume_fn(run_id, owner):
        assert store.claim_lease(run_id, owner)
        claimed.set()
        await asyncio.sleep(30)  # interrupted mid-resume

    worker = RecoveryWorker(store, resume_fn, poll_interval_seconds=60.0)

    async def go():
        task = asyncio.create_task(worker.run_forever())
        await claimed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    cp = store.get("r-cancel")
    assert cp.lease_owner is None  # lease released on cancellation, not left to TTL
    assert cp.state == ExecutionState.WAITING_FOR_PROVIDER
    assert store.list_eligible()  # immediately eligible again
    store.close()


def test_worker_survives_failing_resume(tmp_path):
    store, _ = make_store(tmp_path)
    suspend_run(store, "bad", next_in=0.0)

    async def resume_fn(run_id, owner):
        raise RuntimeError("boom")

    worker = RecoveryWorker(store, resume_fn)
    assert asyncio.run(worker.run_once()) == 0
    assert worker.failed == 1 and worker.resumed == 0
    store.close()


# ---- CLI management commands ----

CONFIG_YAML = """\
supervisor:
  name: supervisor
  description: d
  system_prompt: s
  model: ollama/sup
agents:
  - name: analyst
    description: d
    system_prompt: s
    model: ollama/analyst
collaboration:
  allow_agent_to_agent: true
fallback:
  mode: {mode}
checkpoint:
  path: {db}
  persist_mode: {persist}
  lease_ttl_seconds: 60
  worker_poll_seconds: 5
"""


def write_config(tmp_path, *, mode="wait", persist="resumable"):
    db = tmp_path / "cp.db"
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(CONFIG_YAML.format(mode=mode, db=db, persist=persist), encoding="utf-8")
    return cfg, db


def _quiet_env(monkeypatch):
    monkeypatch.setenv("HIVE_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("HIVE_RETRY_BASE_DELAY_SECONDS", "0")
    monkeypatch.setenv("HIVE_RETRY_MAX_DELAY_SECONDS", "0")


def test_cli_requires_task_unless_management(monkeypatch, tmp_path):
    cfg, _ = write_config(tmp_path)
    with pytest.raises(SystemExit):
        main(["--agents-config", str(cfg)])


def test_cli_management_without_checkpoint_config_errors(tmp_path, capsys):
    cfg = tmp_path / "plain.yaml"
    cfg.write_text(
        "supervisor: {name: supervisor, description: d, system_prompt: s, model: ollama/sup}\n"
        "agents:\n  - {name: analyst, description: d, system_prompt: s, model: ollama/analyst}\n",
        encoding="utf-8",
    )
    rc = main(["--list-suspended", "--agents-config", str(cfg)])
    assert rc == 2
    assert "checkpoint" in capsys.readouterr().err


def test_cli_list_show_cancel(tmp_path, capsys):
    cfg, db = write_config(tmp_path)
    store, _ = make_store(tmp_path, name="cp.db")
    suspend_run(store, "run-1", next_in=0.0,
                transcript_json='[{"role": "user", "content": "hi"}]')
    store.close()

    rc = main(["--list-suspended", "--agents-config", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run-1" in out and "waiting_for_provider" in out

    rc = main(["--show", "run-1", "--agents-config", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "counters:" in out and "transcript stored: True" in out

    rc = main(["--cancel-run", "run-1", "--agents-config", str(cfg)])
    assert rc == 0
    store2, _ = make_store(tmp_path, name="cp.db")
    assert store2.get("run-1").state == ExecutionState.CANCELLED
    store2.close()


def test_cli_suspend_then_resume_end_to_end(tmp_path, monkeypatch, capsys):
    _quiet_env(monkeypatch)
    cfg, db = write_config(tmp_path, mode="wait", persist="resumable")

    def failing(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    monkeypatch.setattr(litellm, "completion", failing)
    rc = main(["do the task", "--agents-config", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUSPENDED" in out

    store, _ = make_store(tmp_path, name="cp.db")
    suspended = store.list_suspended()
    assert len(suspended) == 1
    rid = suspended[0].run_id
    store.close()

    def finish(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"next": "FINISH", "answer": "resumed final"})))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs.get("model", "ollama/sup"),
        )

    monkeypatch.setattr(litellm, "completion", finish)
    rc = main(["--resume", rid, "--agents-config", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resumed final" in out and "SUSPENDED" not in out
    store2, _ = make_store(tmp_path, name="cp.db")
    assert store2.get(rid).state == ExecutionState.COMPLETED
    store2.close()


def test_cli_recovery_worker_flag_runs_worker(tmp_path, monkeypatch, capsys):
    cfg, db = write_config(tmp_path)
    ran = {}

    class FakeWorker:
        def __init__(self, store, resume_fn, **kwargs):
            ran["init"] = True
            self.resumed = 0
            self.skipped = 0
            self.failed = 0

        def run_blocking(self):
            ran["blocking"] = True

    monkeypatch.setattr("geosgan_hive.cli.RecoveryWorker", FakeWorker)
    rc = main(["--recovery-worker", "--agents-config", str(cfg)])
    assert rc == 0 and ran.get("blocking") is True
    assert "recovery-worker" in capsys.readouterr().out
