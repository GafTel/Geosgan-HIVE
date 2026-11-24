"""Phase 3 T2: bounded data tools (json/csv) — strict schemas, resource bounds."""


from geosgan_hive.tool_factory import build_tool_catalog
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tools_data import (
    MAX_CSV_ROWS,
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    register_data_tools,
)


def make_gw() -> ToolGateway:
    catalog = ToolCatalog()
    register_data_tools(catalog)
    policy = build_policy(
        global_enabled=["json_parse", "json_validate", "csv_parse", "csv_validate"],
        agent_permissions={"a": (True, ["json_parse", "json_validate", "csv_parse",
                                       "csv_validate"])},
    )
    return ToolGateway(catalog, policy)


def ctx():
    return ToolExecutionContext(agent="a", run_id="r")


def test_json_parse_summary():
    gw = make_gw()
    r = gw.execute("json_parse", {"text": '{"a": [1,2,3], "b": "x"}'}, ctx=ctx())
    assert r.ok and "object(2 keys)" in r.output and "keys=" in r.output


def test_json_validate_valid_and_invalid():
    gw = make_gw()
    r = gw.execute("json_validate", {"text": "[1, 2]"}, ctx=ctx())
    assert r.ok and "VALID JSON" in r.output and "array(2 items)" in r.output
    r = gw.execute("json_validate", {"text": "{bad"}, ctx=ctx())
    assert r.ok and r.output.startswith("INVALID JSON")
    r = gw.execute("json_parse", {"text": "{bad"}, ctx=ctx())
    assert r.ok is False and r.status == "rejected"


def test_json_depth_and_node_bounds():
    gw = make_gw()
    deep = "[" * (MAX_JSON_DEPTH + 2) + "]" * (MAX_JSON_DEPTH + 2)
    r = gw.execute("json_validate", {"text": deep}, ctx=ctx())
    assert r.ok and "INVALID" in r.output or (not r.ok and "nesting" in r.error)
    big = "[" + ",".join(["1"] * (MAX_JSON_NODES + 5)) + "]"
    r = gw.execute("json_parse", {"text": big}, ctx=ctx())
    assert (not r.ok and "nodes" in r.error) or (r.ok and "INVALID" in r.output)


def test_csv_parse_and_validate():
    gw = make_gw()
    r = gw.execute("csv_parse", {"text": "a,b\n1,2\n3,4"}, ctx=ctx())
    assert r.ok and "2 data rows x 2 columns" in r.output and '["a", "b"]' in r.output
    r = gw.execute("csv_validate", {"text": "a,b\n1,2\n3"}, ctx=ctx())
    assert r.ok and "INVALID CSV: ragged" in r.output
    r = gw.execute("csv_validate", {"text": "a,b\n1,2"}, ctx=ctx())
    assert r.ok and r.output.startswith("VALID CSV")


def test_csv_row_bound():
    gw = make_gw()
    big = "a\n" + "\n".join("x" for _ in range(MAX_CSV_ROWS + 5))
    r = gw.execute("csv_parse", {"text": big}, ctx=ctx())
    assert not r.ok and "rows" in r.error


def test_unexpected_argument_rejected():
    gw = make_gw()
    r = gw.execute("json_parse", {"text": "{}", "extra": 1}, ctx=ctx())
    assert r.ok is False and "invalid arguments" in r.error


def test_catalog_registers_data_tools_but_policy_denies_by_default():
    catalog, _sb = build_tool_catalog()
    assert "json_parse" in catalog.names() and "csv_validate" in catalog.names()
    from geosgan_hive.tool_policy import build_policy
    policy = build_policy(global_enabled=[], agent_permissions={"a": (True, ["json_parse"])})
    gw = ToolGateway(catalog, policy)
    r = gw.execute("json_parse", {"text": "{}"}, ctx=ctx())
    assert r.ok is False and "not allowed" in r.error  # global enable required
