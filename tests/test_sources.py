"""W2 (v0.8.0): source registry + citation integrity (schema v13).

Covers: v12->v13 migration + interrupted-migration rollback + historical data
preservation, trusted-origin/lease fencing, forged registration, stale workers,
replay, provenance-distinct dedup, cross-run access, expiry, integrity-verified
excerpts, sensitive-locator redaction, metadata-only privacy, concurrent
registration, CLI inspection. All offline; fixtures only.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from geosgan_hive.checkpoints import SCHEMA_VERSION, CheckpointStore
from geosgan_hive.errors import CheckpointError
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.sources import (
    SourceRegistry,
    SourceType,
    StoragePolicy,
    Visibility,
    sanitize_locator,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _store(tmp_path, clock=None):
    clock = clock if clock is not None else [NOW]
    return CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: clock[0]), clock


def _run(store, run_id="run-1", owner="w1"):
    store.create_run(run_id=run_id, owner=owner, task="t", agent="supervisor",
                     model="ollama/llama3.2")
    cp = store.get(run_id)
    assert cp is not None
    return cp.lease_generation


def _origin(reg, store, run_id="run-1", owner="w1", gen=None):
    if gen is None:
        gen = _run(store, run_id, owner)
    return reg.trusted_origin(run_id=run_id, lease_owner=owner,
                              lease_generation=gen, agent_id="researcher")


# ---- schema v13 migration ---------------------------------------------------------

def test_v12_to_v13_migration_additive_preserves_data_and_rolls_back(tmp_path):
    db = str(tmp_path / "v12.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '12')")
    conn.execute("CREATE TABLE legacy_marker (id TEXT PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO legacy_marker VALUES ('keep-me', 'historical v12 data')")
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        legacy = store._conn.execute(
            "SELECT note FROM legacy_marker WHERE id='keep-me'").fetchone()
    assert v == "13" and SCHEMA_VERSION == 13
    assert {"sources", "source_audit"} <= tables
    assert legacy is not None and legacy[0] == "historical v12 data"
    store.close()

    # interrupted migration rolls back fully and recovers on next open
    import geosgan_hive.checkpoints as ckpt

    db2 = str(tmp_path / "v12b.db")
    conn = sqlite3.connect(db2)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '12')")
    conn.commit()
    conn.close()
    real = ckpt.CheckpointStore._migrate_12_to_13

    def boom(self):
        self._conn.execute("CREATE TABLE IF NOT EXISTS sources (x TEXT)")  # partial
        raise RuntimeError("interrupted")

    ckpt.CheckpointStore._migrate_12_to_13 = boom
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            CheckpointStore(db2, utcnow=lambda: NOW)
    finally:
        ckpt.CheckpointStore._migrate_12_to_13 = real
    conn = sqlite3.connect(db2)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert v == "12" and "sources" not in tables  # rolled back fully
    store = CheckpointStore(db2, utcnow=lambda: NOW)  # recovers through the chain
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "13"
    store.close()


# ---- trust boundary: registration requires a lease-fenced trusted origin ------------

def test_register_without_trusted_origin_rejected(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    with pytest.raises(CheckpointError, match="trusted origin"):
        reg.register(origin=None, source_type=SourceType.WEB_PAGE, provider="wikimedia",
                     locator="https://example.org/x")  # type: ignore[arg-type]
    with pytest.raises(CheckpointError, match="trusted origin"):
        reg.register(origin={"kind": "trusted_runtime"}, source_type="web_page",  # type: ignore[arg-type]
                     provider="p", locator="l")
    store.close()


def test_fixtures_disabled_by_default(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store)  # allow_fixtures=False
    with pytest.raises(CheckpointError, match="fixtures are disabled"):
        reg.register_fixture(locator="https://example.org/f")
    store.close()


def test_stale_worker_registration_rejected(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    gen = _run(store)
    origin = reg.trusted_origin(run_id="run-1", lease_owner="w1",
                                lease_generation=gen, agent_id="researcher")
    # stale generation at minting time
    with pytest.raises(CheckpointError, match="stale worker"):
        reg.trusted_origin(run_id="run-1", lease_owner="w1",
                           lease_generation=gen + 5, agent_id="researcher")
    # run cancelled -> lease invalid -> atomic in-tx re-verification rejects
    store.cancel("run-1")
    with pytest.raises(CheckpointError, match="stale worker"):
        reg.register(origin=origin, source_type=SourceType.WEB_PAGE,
                     provider="wikimedia", locator="https://example.org/a")
    store.close()


def test_register_and_cite_roundtrip(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(
        locator="https://en.wikipedia.org/wiki/Test",
        content=b"fixture body", provider="wikimedia",
        license="CC BY-SA 4.0", attribution="Wikipedia contributors",
        provenance={"page": 1, "section": "Intro"})
    assert sid.startswith("src_")
    d = reg.get(sid)
    assert d is not None
    assert d["origin_kind"] == "offline_fixture"
    assert d["source_type"] == "offline_fixture"
    assert d["license"] == "CC BY-SA 4.0"
    assert d["content_sha256"] == hashlib.sha256(b"fixture body").hexdigest()
    assert d["content_bytes"] == 12
    assert d["provenance_json"] == '{"page": 1, "section": "Intro"}'
    cite = reg.cite(sid)
    assert sid in cite and "CC BY-SA 4.0" in cite
    assert "provenance only" in cite and "offline_fixture" in cite
    store.close()


def test_unknown_license_stays_explicitly_unknown(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(locator="https://example.org/u")
    d = reg.get(sid)
    assert d["license"] == "unknown"  # NULL in DB, surfaced as unknown — never fabricated
    with store._lock:
        raw = store._conn.execute(
            "SELECT license FROM sources WHERE source_id=?", (sid,)).fetchone()[0]
    assert raw is None
    store.close()


def test_duplicate_source_id_replay_rejected(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    reg.register_fixture(locator="https://example.org/1", source_id="src_replay1")
    with pytest.raises(CheckpointError, match="replay rejected"):
        reg.register_fixture(locator="https://example.org/2", source_id="src_replay1")
    store.close()


def test_same_content_different_provenance_stays_distinct(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    content = b"identical bytes"
    sha = hashlib.sha256(content).hexdigest()
    a = reg.register_fixture(locator="https://a.example.org/doc", content=content)
    b = reg.register_fixture(locator="https://b.example.org/doc", content=content)
    assert a != b
    found = reg.find_by_content_sha256(sha)
    assert {r["source_id"] for r in found} == {a, b}
    assert {r["locator"] for r in found} == {
        "https://a.example.org/doc", "https://b.example.org/doc"}  # not merged
    store.close()


# ---- authorization on reads ----------------------------------------------------------

def test_cross_run_access_denied_unless_team_visibility(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    gen1 = _run(store, "run-1", "w1")
    gen2 = _run(store, "run-2", "w2")
    o1 = reg.trusted_origin(run_id="run-1", lease_owner="w1",
                            lease_generation=gen1, agent_id="a1")
    sid_run = reg.register(origin=o1, source_type=SourceType.WEB_PAGE,
                           provider="wikimedia", locator="https://x.example.org/1",
                           visibility=Visibility.RUN)
    sid_team = reg.register(origin=o1, source_type=SourceType.WEB_PAGE,
                            provider="wikimedia", locator="https://x.example.org/2",
                            visibility=Visibility.TEAM)
    b2 = reg.read_binding(run_id="run-2", lease_owner="w2", lease_generation=gen2)
    with pytest.raises(CheckpointError, match="cross-run"):
        reg.get(sid_run, binding=b2)
    assert reg.get(sid_team, binding=b2) is not None  # team visibility
    assert reg.get(sid_run) is not None  # operator scope (no binding)
    # unbound fixture is not run-visible
    fix = reg.register_fixture(locator="https://x.example.org/fix")
    with pytest.raises(CheckpointError, match="denied"):
        reg.get(fix, binding=b2)
    # listing with binding only shows visible rows
    ids = {d["source_id"] for d in reg.list(binding=b2)}
    assert sid_team in ids and sid_run not in ids and fix not in ids
    store.close()


def test_read_binding_stale_worker_rejected(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store)
    gen = _run(store)
    with pytest.raises(CheckpointError, match="stale worker"):
        reg.read_binding(run_id="run-1", lease_owner="someone-else",
                         lease_generation=gen)
    with pytest.raises(CheckpointError, match="incomplete lease binding"):
        reg.read_binding(run_id="run-1", lease_owner="w1", lease_generation=None)  # type: ignore[arg-type]
    store.close()


# ---- expiry / retention ---------------------------------------------------------------

def test_expired_source_rejected_and_swept(tmp_path):
    store, clock = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(locator="https://x.example.org/e", ttl_seconds=60)
    assert reg.get(sid)["expired"] is False
    clock[0] = NOW + timedelta(seconds=120)
    d = reg.get(sid)
    assert d["expired"] is True
    with pytest.raises(CheckpointError, match="expired"):
        reg.cite(sid)
    with pytest.raises(CheckpointError, match="expired"):
        reg.excerpt(sid)
    n = reg.sweep_expired()
    assert n == 1 and reg.get(sid) is None
    with store._lock:
        audit = store._conn.execute(
            "SELECT event FROM source_audit WHERE source_id=?", (sid,)).fetchall()
    assert any(r[0] == "expired" for r in audit)
    store.close()


# ---- excerpts: integrity-verified, never invented --------------------------------------

def test_metadata_only_excerpt_unavailable_never_invented(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(locator="https://x.example.org/m",
                                content=b"secret-body-not-stored")
    ex = reg.excerpt(sid, max_chars=100)
    assert ex["available"] is False and ex["excerpt"] is None
    assert ex["integrity"] == "not_stored"
    assert "unavailable" in reg.cite(sid, max_chars=100)
    store.close()


def test_workspace_storage_excerpt_verified_and_substitution_detected(tmp_path):
    ws = tmp_path / "ws"
    sandbox = WorkspaceSandbox(str(ws))
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, sandbox=sandbox, allow_fixtures=True)
    (ws / "doc.txt").write_text("hello provenance", encoding="utf-8")
    gen = _run(store)
    origin = reg.trusted_origin(run_id="run-1", lease_owner="w1",
                                lease_generation=gen, agent_id="researcher")
    sid = reg.register(origin=origin, source_type=SourceType.DOCUMENT,
                       provider="local", locator="workspace:doc.txt",
                       storage=StoragePolicy.WORKSPACE, workspace_relpath="doc.txt")
    d = reg.get(sid)
    assert d["storage"] == "workspace" and d["workspace_relpath"] == "doc.txt"
    assert d["content_sha256"] == hashlib.sha256(b"hello provenance").hexdigest()
    ex = reg.excerpt(sid, max_chars=50)
    assert ex["available"] is True and ex["integrity"] == "verified"
    assert ex["excerpt"] == "hello provenance"
    # substitution: modify the file -> hash mismatch -> rejected
    (ws / "doc.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(CheckpointError, match="hash mismatch"):
        reg.excerpt(sid)
    # content removed -> explicit unavailable, never reconstructed
    (ws / "doc.txt").unlink()
    ex2 = reg.excerpt(sid)
    assert ex2["available"] is False and ex2["excerpt"] is None
    store.close()


def test_workspace_relpath_must_be_relative_and_inside(tmp_path):
    ws = tmp_path / "ws"
    sandbox = WorkspaceSandbox(str(ws))
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, sandbox=sandbox, allow_fixtures=True)
    gen = _run(store)
    origin = reg.trusted_origin(run_id="run-1", lease_owner="w1",
                                lease_generation=gen, agent_id="r")
    with pytest.raises(CheckpointError, match="RELATIVE"):
        reg.register(origin=origin, source_type=SourceType.DOCUMENT, provider="p",
                     locator="x", storage=StoragePolicy.WORKSPACE,
                     workspace_relpath=str(ws / "abs.txt"))
    with pytest.raises(CheckpointError):
        reg.register(origin=origin, source_type=SourceType.DOCUMENT, provider="p",
                     locator="x", storage=StoragePolicy.WORKSPACE,
                     workspace_relpath="../outside.txt")
    with pytest.raises(CheckpointError, match="storage='workspace'"):
        reg.register(origin=origin, source_type=SourceType.DOCUMENT, provider="p",
                     locator="x", workspace_relpath="doc.txt")  # metadata_only default
    store.close()


# ---- privacy ---------------------------------------------------------------------------

def test_sensitive_locator_params_redacted_in_storage(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(
        locator="https://api.example.org/works?query=cats&api_key=SUPERSECRET123&token=abc")
    with store._lock:
        row = store._conn.execute(
            "SELECT locator, locator_sha256 FROM sources WHERE source_id=?",
            (sid,)).fetchone()
    assert "SUPERSECRET123" not in row["locator"] and "abc" not in row["locator"]
    assert "api_key=%5BREDACTED%5D" in row["locator"] or "api_key=[REDACTED]" in row["locator"]
    assert "query=cats" in row["locator"]
    assert row["locator_sha256"] == hashlib.sha256(
        b"https://api.example.org/works?query=cats&api_key=SUPERSECRET123&token=abc"
    ).hexdigest()  # raw locator hashed, not stored
    # the raw secret is nowhere in the DB file
    db_bytes = open(str(tmp_path / "cp.db"), "rb").read()
    assert b"SUPERSECRET123" not in db_bytes
    store.close()


def test_metadata_only_never_persists_content_or_absolute_paths(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    marker = b"UNIQUE-CONTENT-MARKER-9f8e7d"
    sid = reg.register_fixture(locator="https://x.example.org/p", content=marker)
    d = reg.get(sid)
    assert d["storage"] == "metadata_only" and d["workspace_relpath"] is None
    assert d["content_sha256"] == hashlib.sha256(marker).hexdigest()  # hash only
    db_bytes = open(str(tmp_path / "cp.db"), "rb").read()
    assert marker not in db_bytes  # raw content never persisted
    assert str(tmp_path).encode() not in db_bytes  # no absolute paths
    store.close()


def test_sanitize_locator_unit():
    out = sanitize_locator("https://h.example.org/p?a=1&password=x&sig=y&b=2")
    assert "password=[REDACTED]" in out.replace("%5B", "[").replace("%5D", "]")
    assert "a=1" in out and "b=2" in out
    assert sanitize_locator("doi:10.1000/xyz\x00") == "doi:10.1000/xyz"
    assert len(sanitize_locator("https://h/" + "p" * 5000)) <= 2048


# ---- W2 residual gate ---------------------------------------------------------------

def test_fixture_origin_cannot_be_relabeled_as_production(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    # grab a fixture origin via register_fixture's internal path is not public;
    # emulate forgery: build the private origin object directly (trusted-process
    # boundary: Python-level access is trusted; the check is defense in depth)
    from geosgan_hive.sources import OriginKind, _TrustedOrigin

    forged = _TrustedOrigin(kind=OriginKind.OFFLINE_FIXTURE, run_id=None, task_id=None,
                            lease_owner=None, lease_generation=None, agent_id="x")
    with pytest.raises(CheckpointError, match="relabeled"):
        reg.register(origin=forged, source_type=SourceType.WEB_PAGE, provider="web",
                     locator="https://evil.example.org/")
    # and the reverse: trusted runtime origin cannot mint a fixture record
    gen = _run(store)
    origin = reg.trusted_origin(run_id="run-1", lease_owner="w1",
                                lease_generation=gen, agent_id="r")
    with pytest.raises(CheckpointError, match="offline_fixture"):
        reg.register(origin=origin, source_type=SourceType.OFFLINE_FIXTURE,
                     provider="p", locator="l")
    store.close()


def test_userinfo_and_sensitive_fragment_redacted(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(
        locator="https://alice:hunter2@proxy.example.org/p?x=1#access_token=TOPSECRET999")
    with store._lock:
        raw = store._conn.execute(
            "SELECT locator FROM sources WHERE source_id=?", (sid,)).fetchone()[0]
    assert "hunter2" not in raw and "alice" not in raw and "TOPSECRET999" not in raw
    assert "[REDACTED]@proxy.example.org" in raw
    assert "#[REDACTED]" in raw  # whole sensitive fragment redacted
    assert "x=1" in raw
    db_bytes = open(str(tmp_path / "cp.db"), "rb").read()
    assert b"hunter2" not in db_bytes and b"TOPSECRET999" not in db_bytes
    store.close()


def test_cite_display_locator_is_bounded(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    long_path = "p/" + "a" * 3000
    sid = reg.register_fixture(locator=f"https://h.example.org/{long_path}")
    cite = reg.cite(sid)
    assert len(cite) < 600  # display locator truncated, not the full 3KB path
    assert "a" * 3000 not in cite
    store.close()


# ---- caller hash cross-checks ------------------------------------------------------------

def test_caller_content_hash_mismatch_rejected(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    with pytest.raises(CheckpointError, match="does not match"):
        reg.register_fixture(locator="https://x/", content=b"real",
                             content_sha256="0" * 64)
    with pytest.raises(CheckpointError, match="sha256 hex"):
        reg.register_fixture(locator="https://x/", content_sha256="nothex")
    store.close()


# ---- concurrency --------------------------------------------------------------------------

def test_concurrent_registration_atomic(tmp_path):
    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    results: list = []
    errors: list = []

    def work(i):
        try:
            results.append(reg.register_fixture(
                locator=f"https://x.example.org/{i}", content=f"c{i}".encode()))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(set(results)) == 20
    assert len(reg.list()) == 20
    # duplicate id across threads: exactly one wins
    outcomes = []

    def dup():
        try:
            reg.register_fixture(locator="https://x/dup", source_id="src_dup00000000001")
            outcomes.append("ok")
        except CheckpointError:
            outcomes.append("rejected")

    ts = [threading.Thread(target=dup) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert outcomes.count("ok") == 1 and outcomes.count("rejected") == 3
    store.close()


# ---- CLI (read-only operator inspection) ----------------------------------------------------

def test_cli_sources_inspection(tmp_path, capsys, no_network):
    from geosgan_hive.cli import main

    store, _ = _store(tmp_path)
    reg = SourceRegistry(store, allow_fixtures=True)
    sid = reg.register_fixture(locator="https://en.wikipedia.org/wiki/Cat",
                                content=b"cats", license=None)
    store.close()
    db = str(tmp_path / "cp.db")
    rc = main(["--sources", "--agents-config", "config/agents.local.yaml",
               "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out and "offline_fixture" in out
    rc = main(["--source-id", sid, "--agents-config", "config/agents.local.yaml",
               "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "license: unknown" in out and "storage: metadata_only" in out
    rc = main(["--source-excerpt", sid, "--agents-config", "config/agents.local.yaml",
               "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "available=False" in out and "no stored content" in out
    rc = main(["--source-id", "src_missing", "--agents-config",
               "config/agents.local.yaml", "--checkpoint", db])
    assert rc == 1
