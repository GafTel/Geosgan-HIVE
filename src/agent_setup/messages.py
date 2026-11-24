from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    """Mensaje de la conversación. `agent` indica quién lo produjo (traza)."""

    role: str  # "system" | "user" | "assistant"
    content: str
    agent: str | None = None

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
