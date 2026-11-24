from __future__ import annotations

import argparse

from .config import Settings
from .factory import build_supervisor
from .llm import LiteLLMClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-setup", description="Ejecuta el sistema multi-agente sobre una tarea."
    )
    parser.add_argument("tarea", help="Tarea o pregunta para el equipo de agentes")
    parser.add_argument("--model", default=None, help="Modelo a usar (sobrescribe AGENT_MODEL)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Máximo de pasos del supervisor (AGENT_MAX_STEPS)",
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
    result = supervisor.run(args.tarea)

    route = " -> ".join(result.route) if result.route else "(respuesta directa del supervisor)"
    print(f"Ruta: {route}")
    print(f"Pasos del supervisor: {result.steps}")
    print("\nRespuesta final:\n")
    print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
