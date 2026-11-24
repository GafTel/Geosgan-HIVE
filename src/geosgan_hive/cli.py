from __future__ import annotations

import argparse

from .config import Settings
from .factory import build_supervisor
from .llm import LiteLLMClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geosgan-hive", description="Run the multi-agent system on a task."
    )
    parser.add_argument("task", help="Task or question for the agent team")
    parser.add_argument("--model", default=None, help="Model to use (overrides HIVE_MODEL)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum supervisor steps (HIVE_MAX_STEPS)",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    updates: dict[str, object] = {}
    if args.model:
        updates["model"] = args.model
    if args.max_steps is not None:
        updates["max_steps"] = args.max_steps
    if updates:
        settings = settings.model_copy(update=updates)

    llm = LiteLLMClient(model=settings.model, api_key=settings.api_key)
    supervisor = build_supervisor(llm, settings)
    result = supervisor.run(args.task)

    route = " -> ".join(result.route) if result.route else "(direct supervisor answer)"
    print(f"Route: {route}")
    print(f"Supervisor steps: {result.steps}")
    print("\nFinal answer:\n")
    print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
