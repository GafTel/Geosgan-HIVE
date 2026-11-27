from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from .agent_config import CheckpointConfig, load_hive_config
from .approvals import ApprovalManager
from .checkpoints import CheckpointStore, ToolExecutionStatus
from .cloud_profiles import LocalOllamaAdapter
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
        "--catalog-path",
        default=None,
        help="Override the informational provider catalog path (config "
             "'catalog_path:') for routing commands",
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
    # Phase 5 (v0.7.0, A2): model benchmark engine management (offline/simulated
    # by default; measured runs need separate authorization — never executed here)
    parser.add_argument(
        "--benchmark-simulate", action="store_true",
        help="Run the benchmark suite with the deterministic SIMULATED caller "
             "(synthetic fixtures, no model/network) and store the evidence; exit",
    )
    parser.add_argument("--benchmark-model", default="simulated/fixture@sha256:0000",
                        help="Exact pinned model identity recorded for --benchmark-simulate")
    parser.add_argument("--benchmark-suite", default=None,
                        help="Benchmark suite version (default: latest)")
    parser.add_argument("--benchmark-report", action="store_true",
                        help="Print the stored benchmark evidence report and exit")
    parser.add_argument("--benchmark-list", action="store_true",
                        help="Like --benchmark-report but includes expired records")
    parser.add_argument("--benchmark-model-filter", default=None,
                        help="Filter benchmark report by exact model id")
    parser.add_argument("--route", action="store_true",
                        help="Show the explainable routing decision for one agent "
                             "(offline; no dispatch)")
    parser.add_argument("--route-agent", default="researcher",
                        help="Agent name for --route (default: researcher)")
    parser.add_argument("--route-capabilities", default="",
                        help="Comma-separated required capabilities for --route")
    parser.add_argument("--route-needs-tools", action="store_true",
                        help="Require tool support for --route")
    parser.add_argument("--route-context-tokens", type=int, default=0,
                        help="Required context tokens for --route")
    parser.add_argument("--routing-decisions", action="store_true",
                        help="List durable routing decisions (bounded provenance)")
    parser.add_argument("--cloud-evidence", action="store_true",
                        help="List durable cloud evidence records (metadata only; "
                             "read-only inspection, never authorizes anything)")
    parser.add_argument("--cloud-evidence-all", action="store_true",
                        help="Include revoked/expired cloud evidence records")
    parser.add_argument("--routing-run", default=None,
                        help="Filter --routing-decisions by run id")
    parser.add_argument("--sources", action="store_true",
                        help="List registered sources (metadata only; read-only, "
                             "W2 source registry inspection)")
    parser.add_argument("--source-run", default=None,
                        help="Filter --sources by run id")
    parser.add_argument("--source-id", default=None, metavar="SOURCE_ID",
                        help="Show one source record (metadata only)")
    parser.add_argument("--source-excerpt", default=None, metavar="SOURCE_ID",
                        help="Show an integrity-verified bounded excerpt of a source "
                             "(operator scope; unavailable if content is not stored)")
    parser.add_argument("--source-excerpt-chars", type=int, default=500,
                        help="Max excerpt characters for --source-excerpt")
    # W7 (v0.8.0): document + capability operator workflows on top of the SAME CLI,
    # gateway, approval manager, registry, sandbox and ledger. Exit codes:
    # 0 ok, 1 runtime error (HiveError), 2 usage/configuration error.
    parser.add_argument("--json", dest="json_out", action="store_true",
                        help="Structured JSON output for inspection/automation commands")
    parser.add_argument("--start-run", metavar="TASK", default=None,
                        help="Start an authorized run (RUNNING, CLI-owned lease) and exit")
    parser.add_argument("--run-id", default=None,
                        help="Explicit run id for --start-run (default: generated)")
    parser.add_argument("--doc-read", metavar="PATH", default=None,
                        help="Read an authorized workspace document (operator scope; "
                             "bounded text + provenance) and exit")
    parser.add_argument("--doc-max-chars", type=int, default=2_000,
                        help="Max extracted characters for --doc-read")
    parser.add_argument("--doc-prepare", action="store_true",
                        help="Prepare a document from --spec-file (strictly validated "
                             "DocumentSpec JSON); creates a PENDING exact-byte approval; exit")
    parser.add_argument("--doc-format", default=None,
                        help="Generation format for --doc-prepare (pdf/docx/xlsx/csv/json/txt/md)")
    parser.add_argument("--doc-destination", default=None,
                        help="Workspace-relative destination for --doc-prepare (creation-only)")
    parser.add_argument("--spec-file", default=None,
                        help="JSON file with the DocumentSpec for --doc-prepare")
    parser.add_argument("--doc-status", metavar="APPROVAL_ID", default=None,
                        help="Show the full lifecycle of a document approval/artifact "
                             "(prepared/pending/approved/rejected/expired/cancelled/"
                             "published/failed/uncertain) and exit")
    parser.add_argument("--doc-publish", metavar="APPROVAL_ID", default=None,
                        help="Publish the exact APPROVED artifact of --run (requires "
                             "--operator; takes over a free/expired lease; never "
                             "auto-approves) and exit")
    parser.add_argument("--capabilities", action="store_true",
                        help="Report effective capabilities (agents, tool permissions, "
                             "providers, zero-cost mode, outbound mode, document deps/"
                             "formats, platform support, workspace, limits; no secrets) "
                             "and exit")
    args = parser.parse_args(argv)

    management = bool(
        args.list_suspended or args.show or args.resume or args.cancel_run
        or args.recovery_worker or args.pending_approvals is not None
        or args.show_approval or args.approve or args.deny or args.cancel_approval
        or args.tool_executions or args.resolve_execution
        or args.benchmark_simulate or args.benchmark_report or args.benchmark_list
        or args.route or args.routing_decisions or args.cloud_evidence
        or args.sources or args.source_id or args.source_excerpt
        or args.start_run or args.doc_read or args.doc_prepare or args.doc_status
        or args.doc_publish or args.capabilities
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
    if args.catalog_path:
        config = config.model_copy(update={"catalog_path": args.catalog_path})
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
                if args.json_out:
                    print(json.dumps(pending, default=str))
                    return 0
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
                if args.json_out:
                    print(json.dumps(row, default=str))
                    return 0
                for key in ("approval_id", "run_id", "task_id", "agent", "tool",
                            "tool_version", "operation", "target", "status",
                            "requested_at", "expires_at", "decided_by", "decided_at",
                            "consumed_at", "args_sha256", "preconditions_json"):
                    print(f"  {key}: {row[key]}")
                # W7: for document artifacts, show the FULL exact-byte binding before
                # the operator decides (never display a prepared/approved artifact as
                # already published — the lifecycle status is shown separately)
                if row["operation"] == "publish_artifact":
                    try:
                        pre = json.loads(row["preconditions_json"] or "{}")
                    except ValueError:
                        pre = {}
                    print("  artifact (exact-byte binding):")
                    print(f"    format: {pre.get('format')}")
                    print(f"    generator: {pre.get('generator')}")
                    print(f"    destination: {row['target']}")
                    print(f"    artifact_sha256 (full): {pre.get('output_sha256')}")
                    print(f"    byte_count: {pre.get('byte_count')}")
                    print(f"    spec_sha256: {pre.get('spec_sha256')}")
                    print(f"    expected_absent (creation-only): {pre.get('expected_absent')}")
                    print(f"    staging_relpath: {pre.get('staging_relpath')}")
                    execs = [e for e in store.ledger_list(run_id=row["run_id"] or None)
                             if e["approval_id"] == row["approval_id"]]
                    print(f"    lifecycle: approval={row['status']}"
                          + (f" execution={execs[0]['status']}" if execs else
                             " execution=not-started (not published)"))
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
                if args.json_out:
                    print(json.dumps(rows, default=str))
                    return 0
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
        # Phase 4: a cloud block must always run with the zero-cost gate wired.
        # The only production adapter today is local Ollama inference (cost 0.0,
        # loopback-only); live cloud providers stay fail-closed blocked because
        # no live-capable adapter is registered for them.
        cloud_adapter = LocalOllamaAdapter() if config.cloud is not None else None
        supervisor, _agents = build_team(config, gateway, tracker, checkpoint_store=store,
                                         cloud_adapter=cloud_adapter)
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

        if args.benchmark_simulate and store is not None:
            from .benchmarks import SUITE_VERSION, SimulatedCaller, format_report, run_benchmark

            suite = args.benchmark_suite or SUITE_VERSION
            result = run_benchmark(
                store, caller=SimulatedCaller(), model_id=args.benchmark_model,
                model_version=args.benchmark_model.split("@", 1)[1], suite_version=suite,
                execution_mode="simulated",
                hardware_profile="offline-simulated",
                run_config={"caller": "SimulatedCaller", "suite": suite},
                provenance="CLI --benchmark-simulate (synthetic fixtures; NOT a real "
                           "model measurement)",
            )
            print(f"Simulated benchmark recorded: {result.evidence_id} "
                  f"score={result.aggregate_score:.3f} "
                  f"cases={result.cases_passed}/{result.cases_total} "
                  f"mode={result.execution_mode.upper()}")
            print(format_report(store.list_benchmark_evidence()))
            return 0
        if (args.benchmark_report or args.benchmark_list) and store is not None:
            from .benchmarks import format_report

            rows = store.list_benchmark_evidence(
                model_id=args.benchmark_model_filter,
                include_expired=args.benchmark_list)
            print(format_report(rows))
            return 0
        if args.route or args.routing_decisions:
            from .catalog import load_catalog
            from .routing import ModelRouter, RoutingPolicy

            catalog = load_catalog(config.catalog_path) if config.catalog_path else None
            policy = config.routing or RoutingPolicy()  # fixed mode by default
            router = ModelRouter(
                policy, catalog=catalog,
                cloud_registry=getattr(gateway, "_cloud_registry", None),
                health=getattr(gateway, "health", None),
                store=store)
            if args.route:
                agent = config.supervisor if args.route_agent == config.supervisor.name \
                    else config.agent(args.route_agent)
                caps = tuple(c.strip() for c in args.route_capabilities.split(",")
                             if c.strip())
                decision = router.select(
                    agent=agent.name, configured_model=agent.model,
                    required_capabilities=caps,
                    needs_tools=args.route_needs_tools,
                    context_tokens=args.route_context_tokens)
                print(f"Routing decision {decision.decision_id}")
                print(f"  agent={decision.agent} mode={decision.selection_mode} "
                      f"selected={decision.selected_model} reason={decision.reason}")
                print(f"  policy={decision.policy_version} method={decision.scoring_method} "
                      f"evidence={list(decision.evidence_refs) or '[]'}")
                for c in decision.candidates:
                    score = f"{c.score:.4f}" if c.score is not None else "-"
                    print(f"    {'ELIGIBLE' if c.eligible else 'rejected'} "
                          f"{c.model} reason={c.reason} score={score}")
                return 0
            rows = store.list_routing_decisions(run_id=args.routing_run) \
                if store is not None else []
            print(f"Routing decisions: {len(rows)} record(s)")
            for r in rows:
                print(f"  {r['decision_id'][:12]} run={r['run_id']} agent={r['agent']} "
                      f"mode={r['selection_mode']} selected={r['selected_model']} "
                      f"method={r['scoring_method']} at={r['created_at']}")
            return 0
        if args.cloud_evidence:
            reg = getattr(gateway, "_cloud_registry", None)
            vault = getattr(reg, "evidence_vault", None) if reg is not None else None
            if vault is None:
                print("Cloud evidence vault: not wired "
                      "(set cloud.evidence_vault: true in the config)")
                return 0
            rows = vault.list_evidence(include_revoked=args.cloud_evidence_all,
                                       include_expired=args.cloud_evidence_all)
            print(f"Cloud evidence: {len(rows)} record(s) (metadata only; "
                  "listing never authorizes dispatch)")
            for r in rows:
                print(f"  {r['evidence_id'][:12]} provider={r['provider_id']} "
                      f"class={r['evidence_class']} model={r['model_id']} "
                      f"restriction={r['spending_restriction']} "
                      f"overage={r['overage_behavior']} revoked={bool(r['revoked'])} "
                      f"expires={r['expires_at']}")
            return 0
        if (args.sources or args.source_id or args.source_excerpt) and store is not None:
            # W2: minimal read-only source inspection (operator scope; metadata
            # only — the CLI has no sandbox, so workspace excerpts report
            # 'unavailable' rather than reading files; integrity is still enforced)
            from .sources import SourceRegistry

            reg = SourceRegistry(store)
            if args.source_excerpt:
                try:
                    ex = reg.excerpt(args.source_excerpt,
                                     max_chars=max(0, args.source_excerpt_chars))
                except HiveError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                print(f"source={ex['source_id']} available={ex['available']} "
                      f"integrity={ex['integrity']}")
                print(ex["excerpt"] if ex["excerpt"] is not None else "(no stored content)")
                return 0
            if args.source_id:
                d = reg.get(args.source_id)
                if d is None:
                    print(f"No source {args.source_id!r}", file=sys.stderr)
                    return 1
                if args.json_out:
                    print(json.dumps(d, default=str))
                    return 0
                try:
                    prov = json.loads(d.get("provenance_json") or "{}")
                except ValueError:
                    prov = {}
                for key in ("source_id", "run_id", "task_id", "agent_id", "origin_kind",
                            "source_type", "provider", "locator", "retrieved_at",
                            "content_sha256", "content_bytes", "license", "attribution",
                            "visibility", "storage", "workspace_relpath", "expires_at",
                            "expired", "policy_version", "created_at"):
                    print(f"  {key}: {d[key]}")
                # W7: provenance detail without printing content by default
                print(f"  extract_method: {d.get('extract_method')}")
                print(f"  extract_version: {d.get('extract_version')}")
                print(f"  raw_sha256: {d.get('raw_sha256')}")
                print(f"  generated_artifact: {bool(prov.get('generated', False))}")
                print(f"  truncated: {bool(prov.get('truncated', False))}")
                print(f"  offline_fixture: {d.get('origin_kind') == 'offline_fixture'}")
                ex = reg.excerpt(args.source_id, max_chars=0)
                print(f"  excerpt: available={ex['available']} integrity={ex['integrity']} "
                      f"(content is never printed by default; use --source-excerpt)")
                print("  note: provenance only — not factual verification")
                return 0
            rows = reg.list(run_id=args.source_run)
            if args.json_out:
                print(json.dumps(rows, default=str))
                return 0
            print(f"Sources: {len(rows)} record(s) (metadata only; provenance, "
                  "not factual verification)")
            for d in rows:
                print(f"  {d['source_id']} type={d['source_type']} "
                      f"provider={d['provider']} origin={d['origin_kind']} "
                      f"run={d['run_id']} visibility={d['visibility']} "
                      f"storage={d['storage']} expired={d['expired']} "
                      f"locator={d['locator'][:80]}")
            return 0
        # ---- W7: run start / document workflows / capability report (same CLI) ----
        if args.start_run and store is not None:
            run_id = args.run_id or uuid4().hex
            owner = f"cli:{args.operator or 'operator'}"
            try:
                store.create_run(run_id=run_id, owner=owner, task=args.start_run,
                                 agent="operator", model="", root_task_id=run_id)
            except HiveError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            payload = {"run_id": run_id, "owner": owner, "state": "running",
                       "task": args.start_run}
            if args.json_out:
                print(json.dumps(payload))
            else:
                print(f"Run started: {run_id} (owner={owner}, state=running)")
            return 0

        if args.capabilities:
            import importlib.util

            from .cloud_profiles import ZERO_COST_RELEASE_LOCAL_PROVIDERS

            docs = config.documents
            deps = {m: importlib.util.find_spec(m) is not None
                    for m in ("pypdf", "docx", "openpyxl", "reportlab", "defusedxml")}
            worker_ok = os.name == "posix" and os.path.isdir("/proc/self/fd")
            tools_cfg = config.tools
            enabled = (set(tools_cfg.enabled) if tools_cfg is not None
                       else {"calculator", "current_datetime", "word_count"})

            def eff_tools(spec):
                if not spec.permissions.allow_tools:
                    return []
                return [t for t in spec.permissions.allowed_tools if t in enabled]

            apis = config.apis
            report = {
                "platform": {
                    "os": os.name,
                    "parser_worker_available": worker_ok,
                    "staging_publication_available": worker_ok,
                    "note": ("POSIX parser worker + fd-verified staging/publication"
                             if worker_ok else
                             "document parsing/generation/publication FAIL CLOSED on "
                             "this platform (no fd re-verification); no insecure override"),
                },
                "agents": [{"name": s.name, "model": s.model,
                            "effective_tools": eff_tools(s),
                            "delegation": s.permissions.allow_delegation}
                           for s in config.agents],
                "supervisor": {"name": config.supervisor.name,
                               "effective_tools": eff_tools(config.supervisor)},
                "model_providers": sorted(
                    {config.supervisor.model.split("/", 1)[0]}
                    | {s.model.split("/", 1)[0] for s in config.agents}),
                "zero_cost_mode": {
                    "release_gate_local_providers": list(ZERO_COST_RELEASE_LOCAL_PROVIDERS),
                    "cloud_zero_cost_configured": bool(config.cloud is not None
                                                       and config.cloud.zero_cost),
                    "non_local_cloud_inference": "blocked in zero-cost mode",
                },
                "outbound": {
                    "enabled_providers": list(apis.enabled) if apis is not None else [],
                    "provider_policies": ({p.provider: p.outbound_policy
                                           for p in apis.providers}
                                          if apis is not None else {}),
                    "web_endpoint_policies": ([p.endpoint_id for p in apis.web_endpoints]
                                              if apis is not None else []),
                },
                "documents": ({
                    "enabled": docs.enabled,
                    "optional_dependencies": deps,
                    "formats": list(docs.allowed_formats),
                    "worker_formats_posix_only": ["pdf", "docx", "xlsx"],
                    "limits": {k: getattr(docs, k) for k in (
                        "max_input_bytes", "max_extract_chars", "max_pdf_pages",
                        "max_xlsx_cells", "staging_max_files", "staging_max_bytes",
                        "staging_max_age_seconds")},
                    "publication": "creation-only, atomic no-clobber; overwrite disabled",
                } if docs is not None else {"enabled": False}),
                "workspace": {
                    "root": tools_cfg.workspace_root if tools_cfg is not None else "workspace",
                    "allow_write": bool(tools_cfg and tools_cfg.allow_write),
                    "write_extensions": (list(tools_cfg.write_extensions)
                                         if tools_cfg is not None else [".txt", ".md"]),
                    "deny_name_patterns": (list(tools_cfg.deny_name_patterns)
                                           if tools_cfg is not None else []),
                },
                "approvals": {
                    "expiry_seconds": (tools_cfg.approval.expiry_seconds
                                       if tools_cfg is not None else 600.0),
                    "human_operator_only": True,
                },
                "budgets": (tools_cfg.budgets.model_dump() if tools_cfg is not None
                            else None),
            }
            if args.json_out:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(f"Platform: os={os.name} parser_worker={worker_ok} "
                      f"staging/publication={worker_ok}")
                print(f"Zero-cost: local providers only "
                      f"{list(ZERO_COST_RELEASE_LOCAL_PROVIDERS)}; non-local cloud "
                      f"inference blocked")
                print(f"Model providers configured: {report['model_providers']}")
                print(f"Outbound: enabled={report['outbound']['enabled_providers']} "
                      f"policies={report['outbound']['provider_policies']}")
                docs_rep = report["documents"]
                if docs_rep.get("enabled"):
                    print(f"Documents: enabled; deps={docs_rep['optional_dependencies']} "
                          f"formats={docs_rep['formats']}")
                    print(f"  publication: {docs_rep['publication']}")
                else:
                    print("Documents: DISABLED (documents.enabled not set)")
                print(f"Workspace: root={report['workspace']['root']} "
                      f"allow_write={report['workspace']['allow_write']}")
                for a in report["agents"]:
                    print(f"  agent {a['name']}: model={a['model']} "
                          f"tools={a['effective_tools']} delegation={a['delegation']}")
                print(f"  supervisor {report['supervisor']['name']}: "
                      f"tools={report['supervisor']['effective_tools']}")
            return 0

        if args.doc_read or args.doc_prepare or args.doc_status or args.doc_publish:
            from .docgen import DocumentSpec, DocumentWriter
            from .documents import DocumentReader
            from .sources import SourceRegistry
            from .tool_factory import build_tool_catalog

            docs_cfg = config.documents
            if docs_cfg is None or not docs_cfg.enabled:
                print("Error: document support is disabled in this configuration "
                      "(set documents.enabled: true)", file=sys.stderr)
                return 2
            _catalog, sb = build_tool_catalog(config)
            from .agent_config import ApprovalConfig

            ac = (config.tools.approval if config.tools is not None
                  else ApprovalConfig())
            mgr = ApprovalManager(store, expiry_seconds=ac.expiry_seconds,
                                  preview_bytes=ac.preview_bytes)
            reader = DocumentReader(store=store, sandbox=sb, config=docs_cfg)
            writer = DocumentWriter(store=store, sandbox=sb, config=docs_cfg,
                                    approval_manager=mgr)  # configured expiry applies

            if args.doc_read:
                try:
                    r = reader.read(args.doc_read, max_chars=args.doc_max_chars,
                                    register_source=False)  # operator scope: no registration
                except HiveError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                payload = {"path": r.path, "format": r.format, "parser": r.parser,
                           "raw_sha256": r.raw_sha256, "extract_sha256": r.extract_sha256,
                           "bytes_in": r.bytes_in, "truncated": r.truncated,
                           "unsupported": r.unsupported, "metadata": r.metadata,
                           "text": r.text, "note": r.note}
                if args.json_out:
                    print(json.dumps(payload, ensure_ascii=False, default=str))
                else:
                    print(f"{r.path}: format={r.format} parser={r.parser} "
                          f"bytes_in={r.bytes_in} truncated={r.truncated} "
                          f"raw_sha256={r.raw_sha256}")
                    print(f"note: {r.note}")
                    print(r.text[: args.doc_max_chars])
                return 0

            if args.doc_prepare:
                if not (args.doc_format and args.doc_destination and args.spec_file
                        and args.run):
                    print("--doc-prepare requires --doc-format, --doc-destination, "
                          "--spec-file and --run RUN_ID", file=sys.stderr)
                    return 2
                if store.get(args.run) is None:
                    print(f"Error: no run {args.run!r} (start one with --start-run)",
                          file=sys.stderr)
                    return 1
                try:
                    with open(args.spec_file, encoding="utf-8") as fh:
                        spec = DocumentSpec.model_validate(json.load(fh))
                    art = writer.prepare(format=args.doc_format,
                                         destination=args.doc_destination, spec=spec,
                                         agent=args.operator or "cli-operator",
                                         run_id=args.run, task_id=args.run)
                except HiveError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                except (ValueError, OSError) as exc:
                    print(f"Error: invalid specification file: {exc}", file=sys.stderr)
                    return 2
                row = store.get_approval(art.approval_id)
                payload = {"status": art.status, "approval_id": art.approval_id,
                           "artifact_sha256": art.artifact_sha256,
                           "byte_count": art.byte_count, "spec_sha256": art.spec_sha256,
                           "destination": art.destination, "format": art.format,
                           "generator": art.generator, "run_id": args.run,
                           "expires_at": row["expires_at"] if row else None,
                           "note": "prepared + staged; NOT published — review with "
                                   "--show-approval, decide with --approve/--deny, "
                                   "publish with --doc-publish"}
                if args.json_out:
                    print(json.dumps(payload, default=str))
                else:
                    for k, v in payload.items():
                        print(f"  {k}: {v}")
                return 0

            if args.doc_status:
                store.sweep_expired_approvals()  # make terminal expiry authoritative
                row = store.get_approval(args.doc_status)
                if row is None:
                    print(f"No approval {args.doc_status!r}", file=sys.stderr)
                    return 1
                pre = json.loads(row["preconditions_json"] or "{}")
                execs = [e for e in store.ledger_list(run_id=row["run_id"] or None)
                         if e["approval_id"] == row["approval_id"]]
                staged = bool(pre.get("staging_relpath")) and os.path.exists(
                    os.path.join(sb.root, pre["staging_relpath"]))
                status = row["status"]
                if status in ("pending", "approved") and row["expires_at"] \
                        and row["expires_at"] <= store.now_iso():
                    status = "expired"  # effective terminal state, not just stored
                if status == "consumed":
                    lifecycle = ("published" if any(e["status"] == "completed"
                                                    for e in execs)
                                 else "uncertain (consumed but execution not completed — "
                                      "never auto-replayed)")
                elif status == "approved":
                    lifecycle = "approved — awaiting publication (--doc-publish)"
                else:
                    lifecycle = status  # pending / denied / expired / cancelled
                srcs = [s for s in SourceRegistry(store).list(run_id=row["run_id"] or None)
                        if json.loads(s.get("provenance_json") or "{}").get(
                            "approval_id") == row["approval_id"]]
                payload = {"approval_id": row["approval_id"], "lifecycle": lifecycle,
                           "status": status, "run_id": row["run_id"],
                           "task_id": row["task_id"], "agent": row["agent"],
                           "tool": row["tool"], "operation": row["operation"],
                           "target": row["target"], "expires_at": row["expires_at"],
                           "decided_by": row["decided_by"], "consumed_at": row["consumed_at"],
                           "artifact": {"format": pre.get("format"),
                                        "output_sha256": pre.get("output_sha256"),
                                        "byte_count": pre.get("byte_count"),
                                        "spec_sha256": pre.get("spec_sha256"),
                                        "generator": pre.get("generator"),
                                        "expected_absent": pre.get("expected_absent"),
                                        "staging_relpath": pre.get("staging_relpath"),
                                        "staged_file_present": staged},
                           "executions": execs,
                           "source_ids": [s["source_id"] for s in srcs]}
                if args.json_out:
                    print(json.dumps(payload, default=str))
                else:
                    print(f"approval {row['approval_id']}: lifecycle={lifecycle}")
                    print(f"  run={row['run_id']} task={row['task_id']} "
                          f"agent={row['agent']} tool={row['tool']} op={row['operation']}")
                    print(f"  target={row['target']} expires_at={row['expires_at']} "
                          f"decided_by={row['decided_by']} consumed_at={row['consumed_at']}")
                    print("  artifact:")
                    for k, v in payload["artifact"].items():
                        print(f"    {k}: {v}")
                    for e in execs:
                        print(f"  execution {e['execution_id']}: status={e['status']} "
                              f"result_sha256={e['result_sha256']}")
                    for sid in payload["source_ids"]:
                        print(f"  source: {sid}")
                return 0

            if args.doc_publish:
                if not (args.run and args.operator):
                    print("--doc-publish requires --run RUN_ID and --operator OP",
                          file=sys.stderr)
                    return 2
                row = store.get_approval(args.doc_publish)
                if row is None:
                    print(f"No approval {args.doc_publish!r}", file=sys.stderr)
                    return 1
                if row["run_id"] != args.run:
                    print(f"Error: approval belongs to run {row['run_id']!r}, not "
                          f"{args.run!r} — rejected", file=sys.stderr)
                    return 1
                if row["status"] != "approved":
                    print(f"Error: approval is {row['status']}, not approved — nothing "
                          f"published (no auto-approval)", file=sys.stderr)
                    return 1
                owner = f"cli:{args.operator}"
                # continue through an authorized runtime path: renew our own live
                # lease (same operator, separate CLI process), else take over a
                # free/expired lease on a COMPLETED agent run; a live lease held by
                # ANOTHER worker is never displaced — fail closed
                gen = store.renew_lease_gen(args.run, owner)  # generation 0 is falsy!
                if gen is None:
                    gen = store.takeover_lease_gen(args.run, owner)
                if gen is None:
                    print("Error: run lease is held by a live worker or the run is in a "
                          "non-continuable state — fail closed (stale worker rejected)",
                          file=sys.stderr)
                    return 1
                try:
                    res = writer.publish(approval_id=args.doc_publish,
                                         agent=f"operator:{args.operator}",
                                         run_id=args.run, task_id=row["task_id"],
                                         lease_owner=owner, lease_generation=gen)
                except HiveError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                if args.json_out:
                    print(json.dumps(res, default=str))
                else:
                    print(f"Published: {res['destination']} "
                          f"sha256={res['artifact_sha256']} bytes={res['byte_count']}")
                    print(f"  approval={res['approval_id']} execution={res['execution_id']} "
                          f"source={res['source_id']} prepared_by={res['prepared_by']} "
                          f"published_by={res['published_by']}")
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
