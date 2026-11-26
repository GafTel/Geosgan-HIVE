"""A2 tests: reproducible model benchmark engine (offline; no model calls, no
network, no downloads, no execution of generated code).

Covers: suite integrity (roles x en/es), deterministic scoring of every kind,
exact model identity + version tracking, simulated/measured separation,
evidence persistence + expiry + sweep, schema v9->v11 chain + v10->v11 trust
columns (+ rollback), the A2-gate evidence trust boundary (trust tiers, lease-fenced
runtime recorder, replay/substitution/expiry), CLI report/simulate commands, and
the pre-A2 A6 encoding regression checks."""

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
    trusted_quality_evidence,
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
    assert SUITE_VERSION == "2.0.0"


def test_suite_v2_expansion_coverage():
    """v2.0.0: >=50 meaningful cases, every role >=10, en+es per role, and all
    required categories (planning/dependency, citation, prompt injection,
    programming/debugging, math, missing values, security/error detection)."""
    cases = get_suite()
    assert len(cases) >= 50
    for role in ROLES:
        role_cases = [c for c in cases if c.role == role]
        assert len(role_cases) >= 10, role
        assert {c.language for c in role_cases} == set(LANGUAGES), role
    tags = {tag for c in cases for tag in c.tags}
    assert {"planning", "dependency_order", "citation", "prompt_injection",
            "programming", "debugging", "math", "missing_value", "security",
            "error_detection", "tool_selection"} <= tags
    # every case is deterministically scorable by the simulated caller
    caller = SimulatedCaller()
    for c in cases:
        assert score_case(c, caller.complete(c, MODEL)).passed, c.case_id


def test_suite_v1_preserved_for_evidence_provenance():
    v1 = get_suite("1.0.0")
    assert len(v1) == 10
    assert {c.case_id for c in v1}.isdisjoint({c.case_id for c in get_suite()})
    with pytest.raises(CheckpointError):
        get_suite("9.9.9")


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
            cases_passed=1, cases_total=1, cases_json="[]", evaluator_id="test-evaluator")
    # simulated record can never be relabeled measured
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(
            evidence_id="bad2", suite_version=SUITE_VERSION, model_id=MODEL,
            model_version="v", execution_mode="measured", aggregate_score=1.0,
            cases_passed=1, cases_total=1, cases_json="[]", transcript_sha256="ab" * 32,
            runner_attestation="", evaluator_id="test-evaluator")  # no operator attestation
    # evaluator_id is required: caller cannot record without a registered evaluator
    with pytest.raises(CheckpointError, match="evaluator_id"):
        store.record_benchmark_evidence(
            evidence_id="bad3", suite_version=SUITE_VERSION, model_id=MODEL,
            model_version="v", execution_mode="simulated", aggregate_score=1.0,
            cases_passed=1, cases_total=1, cases_json="[]")
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
                cases_passed=1, cases_total=2, cases_json="[]",
                evaluator_id="test-evaluator")
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


# ---- schema migrations (v9 -> ... -> v11 chain, v10 -> v11 trust columns) ---------------

def test_v9_to_v11_migration_chain_preserves_data(tmp_path):
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
    assert v == "12" and SCHEMA_VERSION == 12  # v9 migrates through the full chain
    assert {"benchmark_evidence", "routing_decisions"} <= tables
    with store._lock:
        bcols = {r["name"] for r in store._conn.execute(
            "PRAGMA table_info(benchmark_evidence)").fetchall()}
    assert {"trust_tier", "evaluator_id"} <= bcols
    assert store.get_reservation("d-old")["dispatch_id"] == "d-old"  # v9 data preserved
    store.close()


def test_interrupted_v9_v10_migration_rolls_back(tmp_path):  # chain recovers to v11
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
    store = CheckpointStore(db, utcnow=lambda: NOW)  # recovers on next open (to v11)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "12"
    store.close()


def test_v10_to_v11_migration_adds_trust_columns_and_routing_table(tmp_path):
    """A pre-v11 (v10) database has benchmark_evidence WITHOUT trust_tier/
    evaluator_id; opening must add them additively (ALTER), create
    routing_decisions, preserve all v10 rows, and land on schema_version 11."""
    db = str(tmp_path / "v10.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '10')")
    conn.execute(
        "CREATE TABLE benchmark_evidence (evidence_id TEXT PRIMARY KEY, "
        "suite_version TEXT NOT NULL, model_id TEXT NOT NULL, model_version TEXT NOT NULL, "
        "execution_mode TEXT NOT NULL, aggregate_score REAL NOT NULL, "
        "cases_passed INTEGER NOT NULL, cases_total INTEGER NOT NULL, "
        "cases_json TEXT NOT NULL, hardware_profile TEXT NOT NULL DEFAULT '', "
        "run_config_json TEXT NOT NULL, provenance TEXT NOT NULL, "
        "transcript_sha256 TEXT, runner_attestation TEXT NOT NULL DEFAULT '', "
        "created_at TEXT NOT NULL, expires_at TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO benchmark_evidence VALUES ('ev-old', '1.0.0', ?, 'v', 'simulated', "
        "1.0, 10, 10, '[]', '', '{}', 'old run', NULL, '', "
        "'2026-11-01T00:00:00.000+00:00', '2026-12-01T00:00:00.000+00:00')", (MODEL,))
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        cols = {r["name"] for r in store._conn.execute(
            "PRAGMA table_info(benchmark_evidence)").fetchall()}
    assert v == "12" and "routing_decisions" in tables
    assert {"trust_tier", "evaluator_id"} <= cols  # added by v10->v11 ALTER
    row = store.get_benchmark_evidence("ev-old")
    assert row is not None and row["evidence_id"] == "ev-old"  # v10 row preserved
    assert row["trust_tier"] == "synthetic_fixture"  # migration default (simulated)
    # the migrated row is simulated -> NOT routing eligible
    assert store.benchmark_evidence_is_routing_eligible(row) is False
    # new records work on the migrated schema
    store.record_benchmark_evidence(
        evidence_id="ev-new", suite_version=SUITE_VERSION, model_id=MODEL,
        model_version="v", execution_mode="simulated", aggregate_score=0.5,
        cases_passed=1, cases_total=2, cases_json="[]", evaluator_id="test-evaluator")
    assert store.get_benchmark_evidence("ev-new")["trust_tier"] == "synthetic_fixture"
    store.close()


def test_interrupted_v10_v11_migration_rolls_back(tmp_path):
    """An interruption inside v10->v11 rolls back fully: no routing_decisions,
    no trust columns, meta stays '10'; the next open retries and lands on 11."""
    db = str(tmp_path / "v10b.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '10')")
    conn.execute(
        "CREATE TABLE benchmark_evidence (evidence_id TEXT PRIMARY KEY, "
        "suite_version TEXT NOT NULL, model_id TEXT NOT NULL, model_version TEXT NOT NULL, "
        "execution_mode TEXT NOT NULL, aggregate_score REAL NOT NULL, "
        "cases_passed INTEGER NOT NULL, cases_total INTEGER NOT NULL, "
        "cases_json TEXT NOT NULL, hardware_profile TEXT NOT NULL DEFAULT '', "
        "run_config_json TEXT NOT NULL, provenance TEXT NOT NULL, "
        "transcript_sha256 TEXT, runner_attestation TEXT NOT NULL DEFAULT '', "
        "created_at TEXT NOT NULL, expires_at TEXT NOT NULL)")
    conn.commit()
    conn.close()
    import geosgan_hive.checkpoints as ckpt

    real = ckpt.CheckpointStore._migrate_10_to_11

    def boom(self):
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS routing_decisions (x TEXT)")  # partial write
        raise RuntimeError("interrupted")

    ckpt.CheckpointStore._migrate_10_to_11 = boom
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            CheckpointStore(db, utcnow=lambda: NOW)
    finally:
        ckpt.CheckpointStore._migrate_10_to_11 = real
    conn = sqlite3.connect(db)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(benchmark_evidence)").fetchall()}
    conn.close()
    assert v == "10" and "routing_decisions" not in tables  # rolled back fully
    assert "trust_tier" not in cols
    store = CheckpointStore(db, utcnow=lambda: NOW)  # recovers to v11 on next open
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        cols = {r["name"] for r in store._conn.execute(
            "PRAGMA table_info(benchmark_evidence)").fetchall()}
    assert v == "12" and {"trust_tier", "evaluator_id"} <= cols
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

def test_wikimedia_unicode_titles_canonically_encoded_and_safe():
    """Unicode titles are supported via CANONICAL UTF-8 percent-encoding: the
    request line stays pure ASCII and the transport re-validates every escape
    (traversal, encoded separators, double encoding, control chars all fail
    closed). Dangerous raw input is rejected at the input validator."""
    from pydantic import ValidationError

    from geosgan_hive.api_adapters import WikimediaGetPageAdapter, WikimediaGetPageInput
    from geosgan_hive.http_client import EndpointNotAllowedError, _validate_path_encoding

    # accepted and canonically encoded end-to-end
    for title in ("H\u00edve", "A\u00e7a\u00ed", "abeja\U0001f41d", "C# rocks!",
                  "abeja reina"):
        inp = WikimediaGetPageInput(title=title)
        adapter = WikimediaGetPageAdapter.__new__(WikimediaGetPageAdapter)
        path, _ = adapter.build_request({"title": inp.title, "language": "es"})
        assert path.isascii(), path  # request line is pure ASCII
        assert "%" not in inp.title  # raw % never accepted
        _validate_path_encoding(path)  # transport accepts the canonical form
    assert WikimediaGetPageInput(title="abeja reina").title == "abeja_reina"
    assert (WikimediaGetPageAdapter.__new__(WikimediaGetPageAdapter)
            .build_request({"title": "H\u00edve", "language": "es"})[0]
            .endswith("/page/H%C3%ADve/bare"))
    # rejected fail-closed at the input validator (no request can be built)
    for bad in ("../secret", "a/b", "a%2fb", "..", ".", "a\\b", "x\x00y",
                "a\x7fb", "%41"):
        with pytest.raises(ValidationError):
            WikimediaGetPageInput(title=bad)
    # and even if a forged encoded path reached the transport, it fails closed
    for bad_path in ("/page/%2e%2e/x", "/page/a%2Fb", "/page/%252e", "/page/%c3%28",
                     "/page/%00", "/page/%2f"):
        with pytest.raises(EndpointNotAllowedError):
            _validate_path_encoding(bad_path)


def test_crossref_references_come_from_reference_records():
    """Bibliographic references are extracted from the publication's actual
    `reference` records (never confused with the publication's own metadata),
    with bounded count and bounded fields."""
    from geosgan_hive.api_adapters import _crossref_item

    msg = {
        "DOI": "10.1234/parent", "title": ["Parent paper"],
        "reference-count": 2,
        "reference": [
            {"DOI": "10.5555/ref1", "article-title": "Ref one", "author": "Smith, A",
             "year": "2001", "journal-title": "J1"},
            {"unstructured": "Doe 2020, unstructured ref"},
            {"DOI": "10.5555/" + "x" * 400},  # oversized field must be bounded
            "not-a-dict",
        ] + [{"DOI": f"10.5555/filler{i}"} for i in range(40)],  # > 25 -> bounded
    }
    item = _crossref_item(msg)
    assert item["reference_count"] == 2
    assert len(item["references"]) == 24  # first 25 entries, one non-dict skipped
    r0 = item["references"][0]
    assert r0["doi"] == "10.5555/ref1" and r0["article_title"] == "Ref one"
    assert r0["author"] == "Smith, A" and r0["year"] == "2001"
    assert r0["journal_title"] == "J1"
    assert item["references"][1]["unstructured"].startswith("Doe 2020")
    assert len(item["references"][2]["doi"]) <= 300  # bounded
    # no references present -> empty list, never fabricated
    assert _crossref_item({"DOI": "10.1/x"})["references"] == []
    assert _crossref_item({"DOI": "10.1/x"})["reference_count"] is None


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


# ---- A2 gate: benchmark evidence trust boundary -------------------------------------
# Simulated / operator-reported evidence must NEVER influence real quality rankings;
# only lease-fenced trusted_runtime evidence from the runtime evaluator qualifies.

def _cases_json(n=60):
    return json.dumps([{"case_id": f"c{i}", "passed": True, "detail": ""} for i in range(n)])


def _live_lease(store, run_id="r-bench", owner="worker-1"):
    store.create_run(run_id=run_id, owner=owner, task="bench", agent="supervisor",
                     model=MODEL)
    cp = store.get(run_id)
    assert cp is not None and cp.state == "running"
    return cp.lease_owner, int(cp.lease_generation)


def _record_kwargs(**over):
    kw = dict(evidence_id="ev-x", suite_version=SUITE_VERSION, model_id=MODEL,
              model_version="v", execution_mode="simulated", aggregate_score=0.9,
              cases_passed=60, cases_total=60, cases_json=_cases_json(60),
              evaluator_id="test-evaluator")
    kw.update(over)
    return kw


def test_caller_cannot_create_trusted_runtime_evidence(tmp_path):
    store = _store(tmp_path)
    # a caller-provided trusted_runtime tier is rejected outright
    with pytest.raises(CheckpointError, match="trusted_runtime"):
        store.record_benchmark_evidence(**_record_kwargs(trust_tier="trusted_runtime"))
    # the reserved runtime evaluator id cannot be claimed by a caller, even with
    # a forged transcript hash + attestation
    with pytest.raises(CheckpointError, match="reserved"):
        store.record_benchmark_evidence(**_record_kwargs(
            execution_mode="measured", transcript_sha256="ab" * 32,
            runner_attestation="op",
            evaluator_id=CheckpointStore.RUNTIME_EVALUATOR_ID))
    # a caller-provided hash alone never elevates a simulated record
    with pytest.raises(CheckpointError):
        store.record_benchmark_evidence(**_record_kwargs(
            execution_mode="measured", transcript_sha256="ab" * 32))
    store.close()


def test_simulated_and_operator_reported_evidence_never_routing_eligible(tmp_path):
    store = _store(tmp_path)
    r = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL, model_version="v")
    row = store.get_benchmark_evidence(r.evidence_id)
    assert row["trust_tier"] == "synthetic_fixture"
    assert store.benchmark_evidence_is_routing_eligible(row, suite_cases=60) is False
    assert trusted_quality_evidence(store, MODEL) == []
    # operator-attested measured evidence: transcript hash + attestation, still
    # operator_reported (NOT verified measurement) and never routing-eligible
    r2 = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL, model_version="v",
                      execution_mode="measured", runner_attestation="operator-1")
    row2 = store.get_benchmark_evidence(r2.evidence_id)
    assert row2["trust_tier"] == "operator_reported"
    assert row2["evaluator_id"] == "operator-cli"
    assert row2["transcript_sha256"] and len(row2["transcript_sha256"]) == 64
    assert store.benchmark_evidence_is_routing_eligible(row2, suite_cases=60) is False
    assert trusted_quality_evidence(store, MODEL) == []  # nothing ranks on fixtures
    store.close()


def test_runtime_recorder_flag_is_not_a_caller_parameter(tmp_path):
    """Failing-first regression for a confirmed defect: the public recorder used
    to accept _runtime_recorder=True, letting a caller mint trusted_runtime
    evidence with NO lease fencing. Now the public method has NO such parameter
    (TypeError) and the tier is unconditionally rejected (CheckpointError). The
    trusted-process boundary is documented: this defends the API boundary, not
    against arbitrary privileged local code or direct DB writes."""
    store = _store(tmp_path)
    kw = _record_kwargs(execution_mode="measured", transcript_sha256="ab" * 32,
                        runner_attestation="op", trust_tier="trusted_runtime",
                        evaluator_id=CheckpointStore.RUNTIME_EVALUATOR_ID)
    with pytest.raises(TypeError):
        store.record_benchmark_evidence(**{**kw, "_runtime_recorder": True})
    with pytest.raises(CheckpointError, match="trusted_runtime"):
        store.record_benchmark_evidence(**kw)
    # reserved evaluator id alone (no tier) is also rejected on the caller path
    with pytest.raises(CheckpointError, match="reserved"):
        store.record_benchmark_evidence(**_record_kwargs(
            execution_mode="measured", transcript_sha256="ab" * 32,
            runner_attestation="op",
            evaluator_id=CheckpointStore.RUNTIME_EVALUATOR_ID))
    assert store.list_benchmark_evidence(include_expired=True) == []
    store.close()


def test_runtime_recorder_requires_live_lease(tmp_path):
    store = _store(tmp_path)
    kw = dict(run_id="r1", lease_owner="w1", lease_generation=0, evidence_id="ev-rt",
              suite_version=SUITE_VERSION, model_id=MODEL, model_version="v",
              aggregate_score=0.95, cases_passed=60, cases_total=60,
              cases_json=_cases_json(60), transcript_sha256="cd" * 32)
    with pytest.raises(CheckpointError, match="stale worker"):
        store.record_runtime_benchmark_evidence(**kw)  # unknown run
    owner, gen = _live_lease(store, "r1", "w1")
    with pytest.raises(CheckpointError, match="stale worker"):
        store.record_runtime_benchmark_evidence(**{**kw, "lease_generation": gen + 5})
    with pytest.raises(CheckpointError, match="stale worker"):
        store.record_runtime_benchmark_evidence(**{**kw, "lease_owner": "w2"})
    # correct worker-owned lease -> the ONLY trusted_runtime path
    store.record_runtime_benchmark_evidence(
        **{**kw, "lease_owner": owner, "lease_generation": gen})
    row = store.get_benchmark_evidence("ev-rt")
    assert row["trust_tier"] == "trusted_runtime"
    assert row["evaluator_id"] == CheckpointStore.RUNTIME_EVALUATOR_ID
    assert store.benchmark_evidence_is_routing_eligible(row, suite_cases=60) is True
    assert [r["evidence_id"] for r in trusted_quality_evidence(store, MODEL)] == ["ev-rt"]
    store.close()


def test_trusted_evidence_replay_completeness_expiry(tmp_path):
    store = _store(tmp_path)
    owner, gen = _live_lease(store)
    base = dict(run_id="r-bench", lease_owner=owner, lease_generation=gen,
                suite_version=SUITE_VERSION, model_id=MODEL, model_version="v",
                aggregate_score=1.0, cases_passed=60, cases_total=60,
                cases_json=_cases_json(60), transcript_sha256="ef" * 32, ttl_days=1)
    store.record_runtime_benchmark_evidence(evidence_id="ev-a", **base)
    # replay / substitution: duplicate evidence id rejected
    with pytest.raises(CheckpointError, match="already exists"):
        store.record_runtime_benchmark_evidence(evidence_id="ev-a", **base)
    # incomplete evidence (cases_json shorter than cases_total) is not eligible
    store.record_runtime_benchmark_evidence(
        evidence_id="ev-b", **{**base, "cases_json": _cases_json(10)})
    row_b = store.get_benchmark_evidence("ev-b")
    assert row_b["trust_tier"] == "trusted_runtime"
    assert store.benchmark_evidence_is_routing_eligible(row_b, suite_cases=60) is False
    # expiry: advance the clock beyond the TTL
    later = NOW + timedelta(days=2)
    store2 = CheckpointStore(store.path, utcnow=lambda: later)
    row_a = store2.get_benchmark_evidence("ev-a")
    assert store2.benchmark_evidence_is_routing_eligible(row_a, suite_cases=60) is False
    assert trusted_quality_evidence(store2, MODEL) == []
    assert store2.sweep_expired_benchmark_evidence() >= 1
    store2.close()
    store.close()


def test_evidence_bound_to_exact_model_and_suite(tmp_path):
    store = _store(tmp_path)
    owner, gen = _live_lease(store)
    model_b = "ollama/qwen3@sha256:cafe"
    for mid, ev in ((MODEL, "ev-m1"), (model_b, "ev-m2")):
        store.record_runtime_benchmark_evidence(
            run_id="r-bench", lease_owner=owner, lease_generation=gen,
            evidence_id=ev, suite_version=SUITE_VERSION, model_id=mid, model_version="v",
            aggregate_score=0.9, cases_passed=60, cases_total=60,
            cases_json=_cases_json(60), transcript_sha256="ab" * 32)
    # v1-suite evidence for the same model must not leak into v2 routing queries
    store.record_runtime_benchmark_evidence(
        run_id="r-bench", lease_owner=owner, lease_generation=gen,
        evidence_id="ev-v1", suite_version="1.0.0", model_id=MODEL, model_version="v",
        aggregate_score=0.9, cases_passed=10, cases_total=10,
        cases_json=_cases_json(10), transcript_sha256="ab" * 32)
    assert {r["evidence_id"] for r in trusted_quality_evidence(store, MODEL)} == {"ev-m1"}
    assert {r["evidence_id"] for r in trusted_quality_evidence(store, model_b)} == {"ev-m2"}
    store.close()
