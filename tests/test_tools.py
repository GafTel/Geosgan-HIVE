from datetime import datetime

from agent_setup.tools import (
    CalculatorTool,
    CurrentDateTimeTool,
    WordCountTool,
    default_registry,
)


def test_calculator_basic():
    tool = CalculatorTool()
    assert tool.run({"expression": "2 + 3 * 4"}) == "14"
    assert tool.run({"expression": "(2 + 3) * 4"}) == "20"
    assert tool.run({"expression": "10 / 4"}) == "2.5"
    assert tool.run({"expression": "2 ** 10"}) == "1024"


def test_calculator_rejects_unsafe_expressions():
    tool = CalculatorTool()
    assert tool.run({"expression": "__import__('os').system('ls')"}).startswith("Error")
    assert tool.run({"expression": "open('x.txt')"}).startswith("Error")
    assert tool.run({"expression": "nombre_variable"}).startswith("Error")
    assert tool.run({"expression": "True + 1"}).startswith("Error")
    # protección contra DoS: exponentes o resultados enormes deben fallar rápido
    assert tool.run({"expression": "9**9**9"}).startswith("Error")
    assert tool.run({"expression": "10**1000000"}).startswith("Error")


def test_calculator_division_by_zero():
    assert CalculatorTool().run({"expression": "1 / 0"}).startswith("Error")


def test_calculator_missing_param():
    assert CalculatorTool().run({}).startswith("Error")


def test_word_count():
    assert WordCountTool().run({"text": "hola mundo"}) == "palabras=2 caracteres=10"


def test_current_datetime_is_iso():
    out = CurrentDateTimeTool().run({})
    datetime.fromisoformat(out)


def test_registry_unknown_tool_and_schemas():
    registry = default_registry()
    assert registry.run("no_existe", {}).startswith("Error")
    assert {t["name"] for t in registry.schemas()} == {
        "calculator",
        "current_datetime",
        "word_count",
    }
