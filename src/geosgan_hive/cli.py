from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent_config import load_hive_config
from .config import Settings
from .errors import ConfigError
from .factory import build_team
from .gateway import ModelGateway
from .retry import RetryPolicy
from .usage import UsageTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geosgan-hive",
        description=(
            "Run the GEOSGAN HIVE multi-agent team (central supervisor + specialist "
            "agents, each on its own model) on a task."
        ),
    )
    parser.add_argument("task", help="Task or question for the agent team")
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
    args = parser.parse_args(argv)

    settings = Settings()
    config_path = Path(args.agents_config) if args.agents_config else settings.config_path
    usage_log = Path(args.usage_log) if args.usage_log else settings.usage_log
    try:
        config = load_hive_config(config_path)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

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
    supervisor, _agents = build_team(config, gateway, tracker)
    if args.max_steps is not None:
        supervisor.max_steps = args.max_steps

    result = supervisor.run(args.task)

    route = " -> ".join(result.route) if result.route else "(direct supervisor answer)"
    print(f"Route: {route}")
    print(f"Supervisor steps: {result.steps}")
    print("\nFinal answer:\n")
    print(result.final_answer)
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
