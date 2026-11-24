from __future__ import annotations

from .agents import Agent
from .config import Settings, load_settings
from .llm import LLMClient
from .orchestrator import Supervisor
from .tools import default_registry


def build_default_agents(llm: LLMClient, settings: Settings) -> dict[str, Agent]:
    """Equipo de agentes especialistas por defecto (se puede ampliar o sustituir)."""
    common = {
        "llm": llm,
        "tools": default_registry(),
        "max_tool_steps": settings.max_tool_steps,
        "temperature": settings.temperature,
    }
    return {
        "investigador": Agent(
            name="investigador",
            description="Investiga y organiza información, usando herramientas si necesita datos.",
            system_prompt=(
                "Eres un investigador meticuloso. Analiza la tarea, usa las herramientas "
                "disponibles cuando aporten datos concretos y entrega un resumen estructurado."
            ),
            **common,
        ),
        "redactor": Agent(
            name="redactor",
            description="Redacta textos claros a partir de la información disponible.",
            system_prompt=(
                "Eres un redactor profesional. Escribe en español, de forma clara y bien "
                "estructurada, usando la información del contexto cuando exista."
            ),
            **common,
        ),
        "revisor": Agent(
            name="revisor",
            description="Revisa y mejora respuestas: corrección, claridad y cobertura de la tarea.",
            system_prompt=(
                "Eres un revisor exigente. Evalúa la respuesta del contexto, detecta problemas "
                "de claridad, corrección o cobertura y entrega la versión final mejorada."
            ),
            **common,
        ),
    }


def build_supervisor(llm: LLMClient, settings: Settings | None = None) -> Supervisor:
    settings = settings or load_settings()
    return Supervisor(
        build_default_agents(llm, settings),
        llm,
        max_steps=settings.max_steps,
        temperature=settings.temperature,
    )
