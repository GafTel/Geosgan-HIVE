"""M2 (v0.9.0): strict validation of ACTUAL tool arguments against the
operator-authorized, subset-validated JSON Schema (amendment G + F5 runtime).

Pipeline: (1) the schema must be inside the supported subset
(`validate_input_schema` — fail closed otherwise); (2) the args payload is
size-capped and walked for non-finite numbers (inf/nan are not JSON, but
Python-side callers could smuggle them), excessive depth/keys; (3) the args are
validated with a standards-compliant JSON Schema Draft 2020-12 validator
(`jsonschema`, pulled in by the optional `mcp` extra — imported lazily, core
never requires it). Validation never performs network requests and never
executes or interprets returned/generated content.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import HiveError
from .mcp_contracts import (
    MAX_NUMERIC_MAGNITUDE,
    McpSchemaError,
    McpUnavailableError,
    validate_input_schema,
)
from .tool_spec import canonical_args

MAX_ARGS_BYTES = 64 * 1024
MAX_ARGS_DEPTH = 16
MAX_ARGS_KEYS = 512


class McpArgsError(HiveError):
    """Tool arguments rejected (schema subset / bounds / type validation)."""


def _walk_finite(value: Any, *, depth: int, keys: list[int]) -> None:
    if depth > MAX_ARGS_DEPTH:
        raise McpArgsError(f"arguments nest deeper than {MAX_ARGS_DEPTH}")
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > MAX_NUMERIC_MAGNITUDE:
            raise McpArgsError("non-finite or out-of-bounds numeric argument (F5)")
    elif isinstance(value, dict):
        keys[0] += len(value)
        if keys[0] > MAX_ARGS_KEYS:
            raise McpArgsError(f"arguments exceed {MAX_ARGS_KEYS} total keys")
        for k, v in value.items():
            if not isinstance(k, str):
                raise McpArgsError("argument keys must be strings")
            _walk_finite(v, depth=depth + 1, keys=keys)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_finite(v, depth=depth + 1, keys=keys)


def validate_tool_args(schema: Any, args: Any, *,
                       max_args_bytes: int = MAX_ARGS_BYTES) -> dict[str, Any]:
    """Validate actual arguments against the operator-authorized schema.
    Returns the args (as a dict) on success; raises McpArgsError/McpSchemaError
    fail-closed otherwise. Pure local validation — no network, no codegen."""
    validate_input_schema(schema)  # schema must be inside the supported subset
    if not isinstance(args, dict):
        raise McpArgsError("tool arguments must be a JSON object")
    raw = canonical_args(args)
    if len(raw.encode("utf-8")) > max_args_bytes:
        raise McpArgsError(f"arguments exceed {max_args_bytes} bytes")
    _walk_finite(args, depth=0, keys=[0])
    try:
        import jsonschema
    except ImportError as exc:
        raise McpUnavailableError(
            "jsonschema is not installed (install the optional 'mcp' extra)") from exc
    try:
        validator = jsonschema.Draft202012Validator(schema)  # no format checkers, no refs
        validator.validate(args)
    except jsonschema.ValidationError as exc:
        raise McpArgsError(f"arguments do not match the authorized schema: "
                           f"{exc.message[:200]} (at {list(exc.absolute_path)})") from exc
    except jsonschema.SchemaError as exc:
        raise McpSchemaError(f"invalid authorized schema: {exc}") from exc
    return args
