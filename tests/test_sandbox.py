"""Phase 3 T3: workspace sandbox + read-only file tools — traversal, symlink
escapes, denied names, special files, size caps, listing caps."""

import os

import pytest

from geosgan_hive.errors import SandboxViolationError
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tools_files import register_file_tools


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "hello.txt").write_text("hello world", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "inner.md").write_text("# Title\n\ntext", encoding="utf-8")
    (root / ".env").write_text("SECRET=abc", encoding="utf-8")
    (root / "checkpoints.db").write_bytes(b"\x00" * 16)
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP SECRET OUTSIDE", encoding="utf-8")
    os.symlink(outside, root / "escape_link.txt")  # symlink escaping the workspace
    os.symlink(root / "sub", root / "inside_link")  # symlink staying inside
    return WorkspaceSandbox(str(root))


def make_gw(sandbox):
    catalog = ToolCatalog()
    register_file_tools(catalog, sandbox)
    policy = build_policy(
        global_enabled=["fs_list", "fs_read", "fs_metadata"],
        agent_permissions={"a": (True, ["fs_list", "fs_read", "fs_metadata"])},
    )
    return ToolGateway(catalog, policy)


def ctx():
    return ToolExecutionContext(agent="a", run_id="r")


def test_fs_read_and_metadata_and_list(ws):
    gw = make_gw(ws)
    r = gw.execute("fs_read", {"path": "hello.txt"}, ctx=ctx())
    assert r.ok and r.output == "hello world"
    r = gw.execute("fs_metadata", {"path": "sub/inner.md"}, ctx=ctx())
    assert r.ok and '"is_dir": false' in r.output and "sha256" in r.output
    r = gw.execute("fs_list", {"path": "."}, ctx=ctx())
    assert r.ok and "hello.txt" in r.output and "sub" in r.output


def test_path_traversal_rejected(ws):
    gw = make_gw(ws)
    for bad in ("../outside.txt", "sub/../../outside.txt", "..\\..\\outside.txt"):
        r = gw.execute("fs_read", {"path": bad}, ctx=ctx())
        assert r.ok is False and r.status == "rejected", bad


def test_absolute_path_outside_root_rejected(ws, tmp_path):
    gw = make_gw(ws)
    r = gw.execute("fs_read", {"path": str(tmp_path / "outside.txt")}, ctx=ctx())
    assert r.ok is False and "outside the workspace" in r.error


def test_symlink_escape_rejected_but_inside_symlink_ok(ws):
    gw = make_gw(ws)
    r = gw.execute("fs_read", {"path": "escape_link.txt"}, ctx=ctx())
    assert r.ok is False and "escape" in r.error.lower()
    r = gw.execute("fs_read", {"path": "inside_link/inner.md"}, ctx=ctx())
    assert r.ok and "# Title" in r.output


def test_denied_names_never_exposed(ws):
    gw = make_gw(ws)
    for name in (".env", "checkpoints.db"):
        r = gw.execute("fs_read", {"path": name}, ctx=ctx())
        assert r.ok is False and "denied" in r.error, name
        r = gw.execute("fs_metadata", {"path": name}, ctx=ctx())
        assert r.ok is False and "denied" in r.error, name
    r = gw.execute("fs_list", {"path": "."}, ctx=ctx())
    # denied entries are still listed by name (metadata only), but unreadable;
    # listing itself must not fail
    assert r.ok


def test_size_caps_enforced(ws, tmp_path):
    target = os.path.join(ws.root, "big.bin")
    with open(target, "wb") as f:
        f.write(b"x" * 4096)
    gw = make_gw(ws)
    r = gw.execute("fs_read", {"path": "big.bin", "max_bytes": 100}, ctx=ctx())
    assert r.ok is False and "bytes > limit" in r.error
    r = gw.execute("fs_metadata", {"path": "big.bin"}, ctx=ctx())
    assert r.ok  # metadata allowed, sha256 computed (small file)


def test_special_files_rejected(ws):
    if os.name != "posix":
        pytest.skip("device files are POSIX-specific")
    gw = make_gw(ws)
    # /dev/null is outside the workspace -> absolute-outside rejection covers it;
    # a fifo inside the workspace must be rejected as non-regular
    fifo = os.path.join(ws.root, "pipe")
    os.mkfifo(fifo)
    r = gw.execute("fs_read", {"path": "pipe"}, ctx=ctx())
    assert r.ok is False and "not a regular file" in r.error


def test_listing_cap(ws):
    for i in range(300):
        open(os.path.join(ws.root, f"f{i:03d}.txt"), "w").close()
    gw = make_gw(ws)
    r = gw.execute("fs_list", {"path": ".", "max_entries": 10}, ctx=ctx())
    assert r.ok and "truncated" in r.output
    r = gw.execute("fs_list", {"path": ".", "max_entries": 10_000}, ctx=ctx())
    assert r.ok is False  # beyond tool-level cap


def test_race_between_resolve_and_open_fails_closed(ws, monkeypatch, tmp_path):
    """TOCTOU: if the path is swapped for an escaping symlink after resolution, the
    open fails (O_NOFOLLOW -> ELOOP) or the post-open fd re-verification rejects it."""
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    target = os.path.join(ws.root, "race.txt")
    os.symlink(outside, target)  # the on-disk entry is an escaping symlink
    # simulate resolve() having seen the pre-swap regular file (TOCTOU window)
    import geosgan_hive.sandbox as sb

    real_realpath = os.path.realpath
    monkeypatch.setattr(
        sb.os.path, "realpath",
        lambda p: p if str(p).endswith("race.txt") else real_realpath(p),
    )
    r = make_gw(ws).execute("fs_read", {"path": "race.txt"}, ctx=ctx())
    assert r.ok is False and r.status == "rejected"
    assert "TOP SECRET" not in (r.output or "")


def test_sandbox_rejects_empty_and_nul_paths(ws):
    with pytest.raises(SandboxViolationError):
        ws.resolve("")
    with pytest.raises(SandboxViolationError):
        ws.resolve("a\x00b")
