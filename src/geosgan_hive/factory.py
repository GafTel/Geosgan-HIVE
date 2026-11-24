from __future__ import annotations

from .agents import Agent
from .config import Settings, load_settings
from .llm import LLMClient
from .orchestrator import Supervisor
from .tools import default_registry


def build_default_agents(llm: LLMClient, settings: Settings) -> dict[str, Agent]:
    """Default team of specialist agents (can be extended or replaced)."""
    common = {
        "llm": llm,
        "tools": default_registry(),
        "max_tool_steps": settings.max_tool_steps,
        "temperature": settings.temperature,
    }
    return {
        "researcher": Agent(
            name="researcher",
            description="Investigates and organizes information, using tools if it needs data.",
            system_prompt=(
                "You are a meticulous researcher. Analyze the task, use the available tools "
                "when they provide concrete data, and deliver a structured summary."
            ),
            **common,
        ),
        "writer": Agent(
            name="writer",
            description="Writes clear texts based on the available information.",
            system_prompt=(
                "You are a professional writer. Write in clear, well-structured English, "
                "using the context information when it exists."
            ),
            **common,
        ),
        "reviewer": Agent(
            name="reviewer",
            description="Reviews and improves answers: correctness, clarity and task coverage.",
            system_prompt=(
                "You are a demanding reviewer. Evaluate the answer in the context, detect clarity, "
                "correctness or coverage issues, and deliver the improved final version."
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
