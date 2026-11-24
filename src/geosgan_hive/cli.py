from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent_config import CheckpointConfig, load_hive_config
from .approvals import ApprovalManager
from .checkpoints import CheckpointStore, ToolExecutionStatus
from .config import Settings
from .errors import ApprovalError, ConfigError, HiveError
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
    # Phase 3 (v0.5.0): tool approval + execution-ledger management (CLI-only
    # trusted operator interface; a model can never approve its own action)
    parser.add_argument(
        "--pending-approvals", nargs="?", const="", default=None, metavar="RUN_ID",
        help="List PENDING tool approvals (optionally filtered by RUN_ID) with a "
             "bounded preview, and exit",
    )
    parser.add_argument("--show-approval", metavar="APPROVAL_ID", default=None,
                        help="Show full details + preview of one approval and exit")
    parser.add_argument("--approve", metavar="APPROVAL_ID", default=None,
                        help="Approve an action (requires --operator)")
    parser.add_argument("--deny", metavar="APPROVAL_ID", default=None,
                        help="Deny an action (requires --operator)")
    parser.add_argument("--cancel-approval", metavar="APPROVAL_ID", default=None,
                        help="Cancel a pending approval (requires --operator)")
    parser.add_argument("--operator", default=None,
                        help="Trusted operator id for approve/deny/cancel/resolve")
    parser.add_argument(
        "--tool-executions", action="store_true",
        help="List durable tool executions (ledger); filter with --run/--status",
    )
    parser.add_argument("--run", default=None, help="Filter tool executions by RUN_ID")
    parser.add_argument("--status", default=None,
                        help="Filter tool executions by status (started/completed/...)")
    parser.add_argument("--resolve-execution", metavar="EXECUTION_ID", default=None,
                        help="Operator-resolve an UNKNOWN tool execution "
                             "(--decision completed|failed, --operator, --evidence)")
    parser.add_argument("--decision", default=None, choices=["completed", "failed"],
                        help="Resolution decision for --resolve-execution")
    parser.add_argument("--evidence", default="", help="Evidence note for --resolve-execution")
    args = parser.parse_args(argv)

    management = bool(
        args.list_suspended or args.show or args.resume or args.cancel_run
        or args.recovery_worker or args.pending_approvals is not None
        or args.show_approval or args.approve or args.deny or args.cancel_approval
        or args.tool_executions or args.resolve_execution
    )
    if args.task is None and not management:
        parser.error("task is required unless a management command is used "
                     "(--list-suspended/--show/--resume/--cancel-run/--recovery-worker/"
                     "--pending-approvals/--tool-executions/...)")

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
        if store is not None and (
            args.pending_approvals is not None or args.show_approval or args.approve
            or args.deny or args.cancel_approval or args.tool_executions
            or args.resolve_execution
        ):
            from .agent_config import ApprovalConfig

            ac = (
                config.tools.approval
                if config.tools is not None
                else ApprovalConfig()
            )
            mgr = ApprovalManager(
                store,
                expiry_seconds=ac.expiry_seconds,
                preview_bytes=ac.preview_bytes,
            )
            if args.pending_approvals is not None:
                run_id = args.pending_approvals or None
                pending = mgr.list_pending(run_id=run_id)
                print(f"Pending approvals: {len(pending)}")
                for row in pending:
                    print(f"  {row['approval_id']}  run={row['run_id']} "
                          f"agent={row['agent']} tool={row['tool']} "
                          f"op={row['operation']} target={row['target']} "
                          f"expires_at={row['expires_at']}")
                    print("    preview:")
                    for line in str(row["preview"]).splitlines():
                        print(f"      {line}")
                return 0
            if args.show_approval:
                row = mgr.get(args.show_approval)
                if row is None:
                    print(f"No approval {args.show_approval!r}", file=sys.stderr)
                    return 1
                for key in ("approval_id", "run_id", "task_id", "agent", "tool",
                            "tool_version", "operation", "target", "status",
                            "requested_at", "expires_at", "decided_by", "decided_at",
                            "consumed_at", "args_sha256", "preconditions_json"):
                    print(f"  {key}: {row[key]}")
                print("  preview:")
                for line in str(row["preview"]).splitlines():
                    print(f"    {line}")
                return 0
            decision_cmd = (
                ("approve", args.approve) if args.approve else
                ("deny", args.deny) if args.deny else
                ("cancel", args.cancel_approval) if args.cancel_approval else None
            )
            if decision_cmd is not None:
                action, approval_id = decision_cmd
                if not args.operator:
                    print("--operator is required for approval decisions",
                          file=sys.stderr)
                    return 2
                try:
                    mgr.decide(approval_id, approve=(action == "approve"),
                               cancel=(action == "cancel"), operator=args.operator)
                except ApprovalError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                print(f"Approval {approval_id} -> {action.upper()}D by {args.operator}")
                return 0
            if args.tool_executions:
                status = args.status
                if status is not None and status not in {s.value for s in ToolExecutionStatus}:
                    print(f"Invalid status {status!r}; valid: "
                          f"{[s.value for s in ToolExecutionStatus]}", file=sys.stderr)
                    return 2
                rows = store.ledger_list(run_id=args.run, status=status)
                print(f"Tool executions: {len(rows)}")
                for row in rows:
                    print(f"  {row['execution_id']}  run={row['run_id']} "
                          f"agent={row['agent']} tool={row['tool']} "
                          f"status={row['status']} approval={row['approval_id']} "
                          f"started_at={row['started_at']} finished_at={row['finished_at']}")
                    if row["error"]:
                        print(f"    error: {row['error']}")
                return 0
            if args.resolve_execution:
                if not args.operator or not args.decision:
                    print("--resolve-execution requires --decision and --operator",
                          file=sys.stderr)
                    return 2
                ok = store.ledger_resolve(args.resolve_execution,
                                          decision=args.decision, operator=args.operator,
                                          evidence=args.evidence or "(no evidence given)")
                if not ok:
                    print(f"Execution {args.resolve_execution!r} is not in UNKNOWN state",
                          file=sys.stderr)
                    return 1
                print(f"Execution {args.resolve_execution} resolved as "
                      f"{args.decision} by {args.operator}")
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
