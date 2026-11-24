from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import ToolValidationError


class SideEffect(StrEnum):
    """Behavioral classification of a tool (drives approval + recovery policy)."""

    PURE = "pure"  # no side effects at all
    READ_ONLY = "read_only"  # reads, no mutation
    LOCAL_MUTATION = "local_mutation"  # mutates local state (files)
    EXTERNAL_MUTATION = "external_mutation"  # mutates external systems


class ToolBudgets(BaseModel):
    """Independent tool budgets (separate from model-call / physical-request budgets).
    All hard reservations are atomic (RunContext lock / SQLite transaction)."""

    max_tool_calls_per_run: int = Field(default=50, ge=0)
    max_tool_calls_per_agent: int = Field(default=20, ge=0)
    max_tool_calls_per_tool: int = Field(default=10, ge=0)
    max_concurrent_tool_executions: int = Field(default=4, ge=1, le=32)
    max_tool_execution_seconds: float = Field(default=30.0, gt=0)
    max_tool_input_bytes: int = Field(default=65_536, ge=1)
    max_tool_output_bytes: int = Field(default=100_000, ge=1)


class ToolSpec(BaseModel):
    """Typed metadata for one tool. `input_model` (a pydantic model with
    extra='forbid') gives strict validation and rejects unexpected arguments."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    version: str = "1"
    category: str = "data"
    side_effect: SideEffect = SideEffect.READ_ONLY
    idempotent: bool = True
    requires_approval: bool | None = None  # None -> derived from side_effect
    input_model: Any = None  # type[BaseModel] | None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_input_bytes: int = Field(default=65_536, ge=1)
    max_output_bytes: int = Field(default=100_000, ge=1)

    @property
    def readonly(self) -> bool:
        return self.side_effect in (SideEffect.PURE, SideEffect.READ_ONLY)

    @property
    def approval_required(self) -> bool:
        if self.requires_approval is not None:
            return self.requires_approval
        return self.side_effect in (SideEffect.LOCAL_MUTATION, SideEffect.EXTERNAL_MUTATION)


def canonical_args(args: Any) -> str:
    """Canonical JSON (sorted keys) used for hashing/binding approvals to exact args."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def args_sha256(args: Any) -> str:
    return hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()


def validate_args(spec: ToolSpec, args: Any) -> dict[str, Any]:
    """Strict validation: must be an object, within the input size cap, and match the
    input model (extra='forbid' rejects unexpected arguments)."""
    if not isinstance(args, dict):
        raise ToolValidationError(
            f"tool '{spec.name}' expects an object of arguments, got {type(args).__name__}"
        )
    raw = canonical_args(args)
    if len(raw.encode("utf-8")) > spec.max_input_bytes:
        raise ToolValidationError(
            f"tool '{spec.name}' input exceeds max_input_bytes={spec.max_input_bytes}"
        )
    if spec.input_model is None:
        return dict(args)
    try:
        model = spec.input_model.model_validate(args)
    except ValidationError as exc:
        raise ToolValidationError(
            f"tool '{spec.name}' rejected invalid arguments: {exc}"
        ) from exc
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return dict(args)
