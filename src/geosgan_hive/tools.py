from __future__ import annotations

import ast
import operator
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments

    def run(self, args: dict[str, Any]) -> str: ...


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

_MAX_MAGNITUDE = 10**100
_MAX_EXPONENT = 1000


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("expression not allowed")
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        value = _BIN_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    else:
        raise ValueError("expression not allowed")
    if abs(value) > _MAX_MAGNITUDE:
        raise ValueError("result too large")
    return value


class CalculatorTool:
    name = "calculator"
    description = "Evaluates a math expression (e.g. '2 + 3 * 4'). Basic arithmetic only."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Math expression"}},
        "required": ["expression"],
    }

    def run(self, args: dict[str, Any]) -> str:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return "Error: missing 'expression' parameter"
        try:
            result = _eval_node(ast.parse(expression, mode="eval"))
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero"
        except (SyntaxError, ValueError) as exc:
            return f"Error: {exc}"


class CurrentDateTimeTool:
    name = "current_datetime"
    description = "Returns the current date and time in UTC (ISO 8601 format)."
    parameters = {"type": "object", "properties": {}}

    def run(self, args: dict[str, Any]) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")


class WordCountTool:
    name = "word_count"
    description = "Counts words and characters of a text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to measure"}},
        "required": ["text"],
    }

    def run(self, args: dict[str, Any]) -> str:
        text = str(args.get("text", ""))
        return f"words={len(text.split())} characters={len(text)}"


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {t.name: t for t in tools}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def run(self, name: str, args: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.run(args)
        except Exception as exc:
            return f"Error running '{name}': {exc}"


def default_registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool(), CurrentDateTimeTool(), WordCountTool()])
