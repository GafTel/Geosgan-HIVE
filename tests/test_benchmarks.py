"""A2 tests: reproducible model benchmark engine (offline; no model calls, no
network, no downloads, no execution of generated code).

Covers: suite integrity (roles x en/es), deterministic scoring of every kind,
exact model identity + version tracking, simulated/measured separation,
evidence persistence + expiry + sweep, schema v9->v10 migration (+ rollback),
CLI report/simulate commands, and the pre-A2 A6 encoding regression checks."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from geosgan_hive.benchmarks import (
    LANGUAGES,
    ROLES,
    SUITE_VERSION,
    BenchmarkCase,
    ScoringSpec,
    SimulatedCaller,
    format_report,
    get_suite,
    run_benchmark,
    score_case,
)
from geosgan_hive.checkpoints import SCHEMA_VERSION, CheckpointStore
from geosgan_hive.errors import CheckpointError

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
MODEL = "ollama/llama3.2@sha256:deadbeef"


def _store(tmp_path, **kw):
    return CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: NOW, **kw)


# ---- suite integrity --------------------------------------------------------------

def test_suite_covers_all_roles_and_both_languages():
    cases = get_suite()
    assert {c.role for c in cases} == set(ROLES)
    for role in ROLES:
        langs = {c.language for c in cases if c.role == role}
        assert set(LANGUAGES) <= langs, role
    assert len({c.case_id for c in cases}) == len(cases)  # unique ids
    assert SUITE_VERSION in ("1.0.0",)


def test_case_validation_rejects_unknown_role_or_language():
    with pytest.raises(ValueError):
        BenchmarkCase("x", "hacker", "en", "p", ScoringSpec("exact", "y"))
    with pytest.raises(ValueError):
        BenchmarkCase("x", "analyst", "fr", "p", ScoringSpec("exact", "y"))
    with pytest.raises(ValueError):
        ScoringSpec("llm_judge", "x")


# ---- deterministic scoring ----------------------------------------------------------

def _case(spec):
    return BenchmarkCase("c1", "analyst", "en", "prompt", spec)


def test_scoring_exact_contains_json_regex():
    c = _case(ScoringSpec("exact", "396"))
    assert score_case(c, "396").passed
    assert score_case(c, " 396\n").passed
    assert not score_case(c, "397").passed
    ci = _case(ScoringSpec("exact", "ABC", case_sensitive=False))
    assert score_case(ci, "abc").passed  # case-insensitive exact matches
    assert not score_case(_case(ScoringSpec("exact", "ABC")), "abc").passed

    c = _case(ScoringSpec("contains", ["foo", "bar"]))
    assert score_case(c, "xx bar yy").passed
    assert not score_case(c, "xx baz yy").passed

    c = _case(ScoringSpec("json_fields", {"a.b": 1, "c": None}))
    assert score_case(c, json.dumps({"a": {"b": 1}, "c": "anything"})).passed
    assert not score_case(c, json.dumps({"a": {"b": 2}, "c": 1})).passed
    assert not score_case(c, "not json").passed
    assert not score_case(c, json.dumps([1, 2])).passed

    c = _case(ScoringSpec("regex", r"\b\d{3}\b"))
    assert score_case(c, "result is 396 ok").passed
    assert not score_case(c, "result is 39 ok").passed
    with pytest.raises(ValueError):
        score_case(_case(ScoringSpec("regex", "([bad")), "x")


def test_scoring_is_deterministic():
    cases = get_suite()
    caller = SimulatedCaller()
    for case in cases:
        out = caller.complete(case, MODEL)
        r1, r2 = score_case(case, out), score_case(case, out)
        assert r1.passed == r2.passed and r1.detail == r2.detail


# ---- runner + evidence --------------------------------------------------------------

def test_simulated_run_persists_labeled_evidence(tmp_path):
    store = _store(tmp_path)
    r = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                      model_version="sha256:deadbeef", hardware_profile="test-box",
                      run_config={"temperature": 0.0})
    assert r.execution_mode == "simulated"
    assert r.aggregate_score == 1.0 and r.cases_total == len(get_suite())
    row = store.get_benchmark_evidence(r.evidence_id)
    assert row["execution_mode"] == "simulated"
    assert row["model_id"] == MODEL and row["model_version"] == "sha256:deadbeef"
    assert row["transcript_sha256"] is None  # simulated: no measured transcript
    assert "simulated" in row["provenance"].lower()
    assert row["expires_at"] > row["created_at"]
    cases = json.loads(row["cases_json"])
    assert len(cases) == r.cases_total and all(c["passed"] for c in cases)
    store.close()


def test_run_is_deterministic_across_runs(tmp_path):
    store = _store(tmp_path)
    a = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL, model_version="v1")
    b = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL, model_version="v1")
    assert a.aggregate_score == b.aggregate_score
    assert [c.passed for c in a.cases] == [c.passed for c in b.cases]
    store.close()


def test_unpinned_model_identity_rejected(tmp_path):
    store = _store(tmp_path)
    for bad in ("llama3", "ollama/llama3.2", "", "ollama/llama3.2@"):
        with pytest.raises(CheckpointError, match="pinned"):
            run_benchmark(store, caller=SimulatedCaller(), model_id=bad, model_version="v")
    store.close()


def test_measured_requires_attestation_and_transcript(tmp_path):
    store = _store(tmp_path)
    # measured without operator attestation -> refused
    with pytest.raises(CheckpointError, match="attestation"):
        run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                      model_version="v", execution_mode="measured")
    # measured with attestation -> stored with transcript hash
    r = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                      model_version="v", execution_mode="measured",
                      runner_attestation="operator-1")
    row = store.get_benchmark_evidence(r.evidence_id)
    assert row["execution_mode"] == "measured"
    assert row["transcript_sha256"] and len(row["transcript_sha256"]) == 64
    assert row["runner_attestation"] == "operator-1"
    # store-level: measured insert without proof is rejected outright
    with pytest.raises(CheckpointError, match="transcript_sha256"):
        store.record_benchmark_evidence(
            evidence_id="bad1", suite_version=SUITE_VERSION, model_id=MODEL,
            model_version="v", execution_mode="measured", aggregate_score=1.0,
            cases_passed=1, cases_total=1, cases_json="[]")
    # simulated record can never be relabeled measured
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(
            evidence_id="bad2", suite_version=SUITE_VERSION, model_id=MODEL,
            model_version="v", execution_mode="measured", aggregate_score=1.0,
            cases_passed=1, cases_total=1, cases_json="[]", transcript_sha256="ab" * 32,
            runner_attestation="")  # no operator attestation
    store.close()


def test_evidence_expiry_and_sweep(tmp_path):
    store = _store(tmp_path)
    run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                  model_version="v", ttl_days=30)
    assert store.list_benchmark_evidence() != []  # visible while fresh
    # advance the clock past expiry
    later = NOW + timedelta(days=31)
    store2 = CheckpointStore(store.path, utcnow=lambda: later)
    assert store2.list_benchmark_evidence() == []  # expired hidden by default
    rows = store2.list_benchmark_evidence(include_expired=True)
    assert len(rows) == 1 and rows[0]["expired"] is True
    report = format_report(rows)
    assert "EXPIRED" in report
    assert "SIMULATED" in report and "NOT a real model measurement" in report
    assert store2.sweep_expired_benchmark_evidence() == 1
    assert store2.list_benchmark_evidence(include_expired=True) == []
    store2.close()
    store.close()


def test_evidence_validation_bounds(tmp_path):
    store = _store(tmp_path)
    base = dict(evidence_id="x", suite_version=SUITE_VERSION, model_id=MODEL,
                model_version="v", execution_mode="simulated", aggregate_score=0.5,
                cases_passed=1, cases_total=2, cases_json="[]")
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**{**base, "aggregate_score": 1.5})
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**{**base, "cases_passed": 5})
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**{**base, "ttl_days": 9999})
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**{**base, "execution_mode": "guessed"})
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**{**base, "cases_json": "x" * (300 * 1024)})
    store.close()


# ---- schema v9 -> v10 migration ------------------------------------------------------

def test_v9_to_v10_migration_preserves_data(tmp_path):
    db = str(tmp_path / "v9.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '9')")
    conn.execute(
        "CREATE TABLE quota_reservations (dispatch_id TEXT PRIMARY KEY, pool TEXT NOT NULL, "
        "anchor_iso TEXT NOT NULL, items_json TEXT NOT NULL, limits_json TEXT NOT NULL, "
        "policy_hash TEXT NOT NULL, run_id TEXT, lease_owner TEXT, lease_generation INTEGER, "
        "status TEXT NOT NULL, created_at TEXT NOT NULL, model TEXT, request_sha256 TEXT, "
        "prompt_tokens INTEGER, max_output_tokens INTEGER, cost_usd REAL, endpoint TEXT)")
    conn.execute(
        "INSERT INTO quota_reservations VALUES ('d-old', 'p', 'a', '[]', '[]', 'h', NULL,"
        " NULL, NULL, 'reserved', 't', NULL, NULL, NULL, NULL, NULL, NULL)")
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "10" and SCHEMA_VERSION == 10
    assert "benchmark_evidence" in tables
    assert store.get_reservation("d-old")["dispatch_id"] == "d-old"  # v9 data preserved
    store.close()


def test_interrupted_v9_v10_migration_rolls_back(tmp_path):
    db = str(tmp_path / "v9b.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '9')")
    conn.commit()
    conn.close()
    import geosgan_hive.checkpoints as ckpt

    real = ckpt.CheckpointStore._migrate_9_to_10

    def boom(self):
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS benchmark_evidence (x TEXT)")  # partial write
        raise RuntimeError("interrupted")

    ckpt.CheckpointStore._migrate_9_to_10 = boom
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            CheckpointStore(db, utcnow=lambda: NOW)
    finally:
        ckpt.CheckpointStore._migrate_9_to_10 = real
    conn = sqlite3.connect(db)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert v == "9" and "benchmark_evidence" not in tables  # rolled back fully
    store = CheckpointStore(db, utcnow=lambda: NOW)  # recovers on next open
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "10"
    store.close()


# ---- CLI -----------------------------------------------------------------------------

def _cli(tmp_path, *argv):
    from geosgan_hive.cli import main

    cfg = REPO_ROOT / "config" / "agents.local.yaml"
    return main([*argv, "--agents-config", str(cfg),
                 "--checkpoint", str(tmp_path / "cli.db")])


def test_cli_benchmark_simulate_and_report(tmp_path, capsys, no_network):
    rc = _cli(tmp_path, "--benchmark-simulate")
    assert rc == 0
    out = capsys.readouterr().out
    assert "SIMULATED" in out and "score=1.000" in out
    rc = _cli(tmp_path, "--benchmark-report")
    assert rc == 0
    out = capsys.readouterr().out
    assert "Benchmark evidence: 1 record(s)" in out
    assert "NOT a real model measurement" in out
    assert "simulated/fixture@sha256:0000" in out
    # report filtered by model
    rc = _cli(tmp_path, "--benchmark-report", "--benchmark-model-filter", "other@x:1")
    assert rc == 0
    assert "0 record(s)" in capsys.readouterr().out


# ---- pre-A2 A6 encoding regression checks ----------------------------------------------

def test_wikimedia_accented_title_fails_closed_without_request():
    """Non-ASCII titles cannot be put on the transport request line (http.client
    is ASCII; the path charset is deliberately narrow) — they are rejected
    fail-closed before any request. Documented limitation, not a bypass."""
    from pydantic import ValidationError

    from geosgan_hive.api_adapters import WikimediaGetPageInput

    for t in ("Híve", "Açaí", "abeja🐝"):
        with pytest.raises(ValidationError):
            WikimediaGetPageInput(title=t)
    assert WikimediaGetPageInput(title="abeja reina").title == "abeja_reina"


def test_crossref_reference_preserves_required_fields():
    from geosgan_hive.api_adapters import _crossref_item

    item = _crossref_item({
        "DOI": "10.1234/abc", "title": ["T"], "author": [{"family": "Doe", "given": "J"}],
        "published": {"date-parts": [[2024, 3, 1]]}, "container-title": ["C"],
        "URL": "https://doi.org/10.1234/abc",
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        "type": "journal-article", "publisher": "P"})
    for key in ("doi", "title", "authors", "published", "container", "url", "license",
                "type", "publisher"):
        assert key in item and item[key] not in ("", None, [])
