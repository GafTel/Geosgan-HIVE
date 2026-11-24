from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    """Conversation message. `agent` records who produced it (trace)."""

    role: str  # "system" | "user" | "assistant"
    content: str
    agent: str | None = None

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
