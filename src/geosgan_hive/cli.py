from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent_config import CheckpointConfig, load_hive_config
from .checkpoints import CheckpointStore
from .config import Settings
from .errors import ConfigError, HiveError
from .factory import build_team
from .gateway import ModelGateway
from .recovery import RecoveryWorker
from .retry import RetryPolicy
from .usage import UsageTracker


def _open_store(config, override_path: str | None) -> CheckpointStore:
    """Open the checkpoint store from config (path overridable via --checkpoint)."""
    cc = config.checkpoint or CheckpointConfig()
    if override_path:
        cc = cc.model_copy(update={"path": override_path})
    return CheckpointStore(
        cc.path,
        persist_mode=cc.persist_mode,
        lease_ttl_seconds=cc.lease_ttl_seconds,
    )


def _print_run(cp, *, detail: bool) -> None:
    print(
        f"  {cp.run_id}: state={cp.state.value} mode={cp.persist_mode.value} "
        f"agent={cp.agent} model={cp.model} attempts={cp.attempts} "
        f"next_attempt_at={cp.next_attempt_at} deadline_at={cp.deadline_at}"
    )
    if detail:
        print(f"    task: {cp.task}")
        print(f"    error: {cp.error}")
        print(f"    result: {cp.result}")
        print(f"    lease: owner={cp.lease_owner} expires_at={cp.lease_expires_at}")
        print(f"    counters: {cp.counters_json}")
        print(f"    steps: {cp.steps_json}")
        print(f"    route: {cp.route_json}")
        print(f"    transcript stored: {cp.transcript_json is not None}")


def _print_result(result) -> None:
    route = " -> ".join(result.route) if result.route else "(direct supervisor answer)"
    print(f"Route: {route}")
    print(f"Supervisor steps: {result.steps}")
    if result.suspended:
        print("Status: SUSPENDED (no eligible model; checkpointed, resumable later)")
    print("\nFinal answer:\n")
    print(result.final_answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geosgan-hive",
        description=(
            "Run the GEOSGAN HIVE multi-agent team (central supervisor + specialist "
            "agents, each on its own model) on a task. Management commands operate on "
            "suspended runs (Phase 2.5 WAIT/HYBRID fallback)."
        ),
    )
    parser.add_argument("task", nargs="?", default=None,
                        help="Task or question for the agent team")
    parser.add_argument(
        "--agents-config",
        default=None,
        help="Team YAML config (default: HIVE_CONFIG env var or config/agents.yaml)",
    )
    parser.add_argument(
        "--usage-log",
        default=None,
        help="Append per-call usage/cost records as JSONL to this file (or HIVE_USAGE_LOG)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override the supervisor max steps"
    )
    parser.add_argument(
        "--async",
        dest="async_",
        action="store_true",
        help="Run asynchronously via Supervisor.arun (native async execution)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the checkpoint database path from the config",
    )
    parser.add_argument(
        "--list-suspended", action="store_true",
        help="List suspended runs (waiting_for_provider / ready_to_resume) and exit",
    )
    parser.add_argument("--show", metavar="RUN_ID", default=None,
                        help="Show full checkpoint details for RUN_ID and exit")
    parser.add_argument("--resume", metavar="RUN_ID", default=None,
                        help="Resume a suspended run (requires persist_mode=resumable)")
    parser.add_argument("--cancel-run", metavar="RUN_ID", default=None,
                        help="Cancel a suspended/active run (terminal) and exit")
    parser.add_argument(
        "--recovery-worker", action="store_true",
        help="Run the opt-in automatic recovery worker until Ctrl+C",
    )
    args = parser.parse_args(argv)

    management = bool(
        args.list_suspended or args.show or args.resume or args.cancel_run
        or args.recovery_worker
    )
    if args.task is None and not management:
        parser.error("task is required unless a management command is used "
                     "(--list-suspended/--show/--resume/--cancel-run/--recovery-worker)")

    settings = Settings()
    config_path = Path(args.agents_config) if args.agents_config else settings.config_path
    usage_log = Path(args.usage_log) if args.usage_log else settings.usage_log
    try:
        config = load_hive_config(config_path)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.checkpoint:
        cc = config.checkpoint or CheckpointConfig()
        config = config.model_copy(
            update={"checkpoint": cc.model_copy(update={"path": args.checkpoint})}
        )
    if management and config.checkpoint is None:
        print(
            "Configuration error: no checkpoint store configured "
            "(set a 'checkpoint:' block in the config or pass --checkpoint PATH)",
            file=sys.stderr,
        )
        return 2

    store = _open_store(config, args.checkpoint) if config.checkpoint is not None else None
    try:
        if args.list_suspended and store is not None:
            suspended = store.list_suspended()
            print(f"Suspended runs: {len(suspended)}")
            for cp in suspended:
                _print_run(cp, detail=False)
            return 0
        if args.show and store is not None:
            cp = store.get(args.show)
            if cp is None:
                print(f"No checkpoint for run {args.show!r}", file=sys.stderr)
                return 1
            _print_run(cp, detail=True)
            return 0
        if args.cancel_run and store is not None:
            store.cancel(args.cancel_run, reason="cancelled via CLI")
            print(f"Run {args.cancel_run} cancelled.")
            return 0

        tracker = UsageTracker(sink=usage_log)
        gateway = ModelGateway(
            tracker=tracker,
            retry_policy=RetryPolicy(
                max_attempts=settings.retry_max_attempts,
                base_delay_seconds=settings.retry_base_delay_seconds,
                max_delay_seconds=settings.retry_max_delay_seconds,
            ),
            default_timeout_seconds=settings.default_timeout_seconds,
        )
        supervisor, _agents = build_team(config, gateway, tracker, checkpoint_store=store)
        if args.max_steps is not None:
            supervisor.max_steps = args.max_steps

        if args.recovery_worker:
            assert store is not None
            poll = (
                config.checkpoint.worker_poll_seconds if config.checkpoint else 5.0
            )

            def on_event(kind: str, meta: dict) -> None:
                print(f"[recovery-worker] {kind}: run={meta.get('run_id')} "
                      f"{meta.get('reason') or meta.get('owner') or ''}".rstrip())

            worker = RecoveryWorker(
                store,
                lambda run_id, owner: supervisor.aresume(run_id, owner=owner),
                poll_interval_seconds=poll,
                on_event=on_event,
            )
            worker.run_blocking()
            print(
                f"[recovery-worker] stopped: resumed={worker.resumed} "
                f"skipped={worker.skipped} failed={worker.failed}"
            )
            return 0

        if args.resume:
            if args.async_:
                result = asyncio.run(supervisor.aresume(args.resume))
            else:
                result = supervisor.resume(args.resume)
        elif args.async_:
            result = asyncio.run(supervisor.arun(args.task))
        else:
            result = supervisor.run(args.task)
    except HiveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()

    _print_result(result)
    print("\nUsage summary:")
    for name, summary in tracker.summary_by_agent().items():
        cost = f"${summary.cost_usd:.6f}" if summary.cost_usd is not None else "unknown"
        print(f"  {name}: calls={summary.calls} tokens={summary.total_tokens} cost_usd={cost}")
    total = tracker.total_cost()
    total_str = f"${total:.6f}" if total is not None else "unknown (some calls have no known cost)"
    print(f"  total: calls={tracker.summary().calls} cost_usd={total_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
