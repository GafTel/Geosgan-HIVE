"""A2: reproducible model benchmark engine (v0.7.0).

Design guarantees:

- DETERMINISTIC SCORING: every case has a machine-checkable scoring spec
  (exact / contains / json_fields / regex). The same model output always
  produces the same score. No LLM-as-judge, no subjective ranking.
- ROLES + LANGUAGES: suites cover supervisor / researcher / developer /
  analyst / reviewer, each with English and Spanish cases.
- EXACT MODEL IDENTITY: evidence records a pinned model id
  ('provider/model@version', e.g. an ollama sha256 pin) plus model_version,
  suite version, hardware profile, run config, provenance, timestamp, TTL.
- SIMULATED vs MEASURED separation: `execution_mode='simulated'` marks
  synthetic fixture results (they are NOT real model measurements and can
  never be relabeled). `measured` records are accepted only with a transcript
  hash AND an operator runner attestation (enforced by the store).
- OFFLINE: the built-in simulated caller never touches a model or network.
  Real runs require a separately authorized caller (gateway + zero-cost gate);
  this phase ships no live execution and downloads no weights.
- NO arbitrary generated-code execution: benchmark cases never execute
  model-generated code; the filesystem sandbox is not a code-execution sandbox.
- Persistence: schema v10 `benchmark_evidence` (atomic insert, expiry, sweep).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .checkpoints import CheckpointStore
from .errors import CheckpointError

ROLES = ("supervisor", "researcher", "developer", "analyst", "reviewer")
LANGUAGES = ("en", "es")
SUITE_VERSION = "1.0.0"

#: a model caller executes one case and returns the raw model output text.
#: In production this is wired to the ModelGateway (zero-cost gate enforced);
#: tests and the offline demo use scripted/simulated callers.
class ModelCaller(Protocol):
    def complete(self, case: BenchmarkCase, model_id: str) -> str: ...


@dataclass(frozen=True)
class ScoringSpec:
    """Machine-checkable expectation. kind:
    - exact:        output.strip() == expected (case_sensitive honored)
    - contains:     expected (or any of expected_list) appears in output
    - json_fields:  output parses as JSON object and expected {path: value}
                    matches (paths are dot-separated keys; values compared
                    exactly, with expected null meaning "key present")
    - regex:        re.search(expected_pattern, output) is not None
    """

    kind: str  # exact | contains | json_fields | regex
    expected: Any
    case_sensitive: bool = True

    def __post_init__(self):
        if self.kind not in ("exact", "contains", "json_fields", "regex"):
            raise ValueError(f"unknown scoring kind {self.kind!r}")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    role: str
    language: str
    prompt: str
    scoring: ScoringSpec
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        if self.language not in LANGUAGES:
            raise ValueError(f"unknown language {self.language!r}")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    detail: str  # bounded machine-checkable detail, never a raw payload dump


def score_case(case: BenchmarkCase, output: str) -> CaseResult:
    """Deterministic, pure scoring. Same (case, output) -> same result, always."""
    spec = case.scoring
    out = output if spec.case_sensitive else output.lower()
    if spec.kind == "exact":
        a, b = out.strip(), str(spec.expected).strip()
        if not spec.case_sensitive:
            a, b = a.lower(), b.lower()
        ok = a == b
        return CaseResult(case.case_id, ok,
                          "exact match" if ok else "output differs from expected")
    if spec.kind == "contains":
        needles = spec.expected if isinstance(spec.expected, list) else [spec.expected]
        if not spec.case_sensitive:
            needles = [str(n).lower() for n in needles]
        ok = any(str(n) in out for n in needles)
        return CaseResult(case.case_id, ok,
                          "required content present" if ok else "required content missing")
    if spec.kind == "json_fields":
        try:
            data = json.loads(output)
        except ValueError:
            return CaseResult(case.case_id, False, "output is not valid JSON")
        if not isinstance(data, dict):
            return CaseResult(case.case_id, False, "output JSON is not an object")
        for path, expected in dict(spec.expected).items():
            cur: Any = data
            for key in str(path).split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return CaseResult(case.case_id, False, f"missing field {path!r}")
                cur = cur[key]
            if expected is None:
                continue  # presence only
            if cur != expected:
                return CaseResult(case.case_id, False, f"field {path!r} mismatch")
        return CaseResult(case.case_id, True, "all expected JSON fields match")
    if spec.kind == "regex":
        try:
            flags = 0 if spec.case_sensitive else re.IGNORECASE
            ok = re.search(str(spec.expected), output, flags) is not None
        except re.error as exc:
            raise ValueError(f"invalid scoring regex {spec.expected!r}") from exc
        return CaseResult(case.case_id, ok, "pattern matched" if ok else "pattern not found")
    raise ValueError(f"unhandled scoring kind {spec.kind!r}")


# ---- the versioned suite ---------------------------------------------------------

def _suite_v1() -> list[BenchmarkCase]:
    cases = [
        # supervisor: planning / delegation / coordination (instruction following)
        BenchmarkCase("supervisor_plan_en", "supervisor", "en",
                      "List the three steps to research a topic, as a numbered JSON "
                      "object {\"steps\": [...]}.",
                      ScoringSpec("json_fields", {"steps": None}), ("planning",)),
        BenchmarkCase("supervisor_delegate_es", "supervisor", "es",
                      "Indica en JSON {\"delegado_a\": \"...\"} a qué agente delegar "
                      "una tarea de búsqueda bibliográfica.",
                      ScoringSpec("json_fields", {"delegado_a": None}), ("delegation",)),
        # researcher: source interpretation / citation accuracy / tool selection
        BenchmarkCase("researcher_cite_en", "researcher", "en",
                      "Given source [1] 'Doe 2024, Journal X', produce a citation "
                      "string starting with 'Doe, J. (2024)'.",
                      ScoringSpec("contains", "Doe, J. (2024)"), ("citation",)),
        BenchmarkCase("researcher_source_es", "researcher", "es",
                      "Clasifica la fuente 'blog anónimo sin fecha' como tipo de "
                      "fuente en JSON {\"tipo\": \"...\"}.",
                      ScoringSpec("json_fields", {"tipo": None}), ("source_eval",)),
        # developer: programming / debugging / code review / test generation
        # (cases check STRUCTURED answers, never execute generated code)
        BenchmarkCase("developer_debug_en", "developer", "en",
                      "A function returns None instead of 0. Answer in JSON "
                      "{\"bug\": \"...\", \"fix\": \"...\"}.",
                      ScoringSpec("json_fields", {"bug": None, "fix": None}), ("debugging",)),
        BenchmarkCase("developer_test_es", "developer", "es",
                      "Escribe en JSON {\"tests\": [...]} dos casos de prueba para "
                      "una función que suma dos números.",
                      ScoringSpec("json_fields", {"tests": None}), ("test_gen",)),
        # analyst: mathematical reasoning / structured analysis / numerical accuracy
        BenchmarkCase("analyst_math_en", "analyst", "en",
                      "Compute 17 * 23 + 5. Answer with just the number.",
                      ScoringSpec("exact", "396"), ("math",)),
        BenchmarkCase("analyst_table_es", "analyst", "es",
                      "Dado [2, 4, 6], devuelve la media en JSON {\"media\": ...}.",
                      ScoringSpec("json_fields", {"media": 4}), ("math", "es")),
        # reviewer: error detection / quality assessment (structured verdicts)
        BenchmarkCase("reviewer_error_en", "reviewer", "en",
                      "Review the claim '2+2=5'. Answer in JSON {\"verdict\": "
                      "\"correct\"|\"incorrect\", \"reason\": \"...\"}.",
                      ScoringSpec("json_fields", {"verdict": "incorrect"}), ("error_detection",)),
        BenchmarkCase("reviewer_quality_es", "reviewer", "es",
                      "Revisa el resumen 'La Tierra es plana'. Responde en JSON "
                      "{\"veredicto\": \"correcto\"|\"incorrecto\"}.",
                      ScoringSpec("json_fields", {"veredicto": "incorrecto"}),
                      ("error_detection", "es")),
    ]
    return cases


SUITES: dict[str, list[BenchmarkCase]] = {SUITE_VERSION: _suite_v1()}


def get_suite(version: str = SUITE_VERSION) -> list[BenchmarkCase]:
    if version not in SUITES:
        raise CheckpointError(f"unknown benchmark suite version {version!r}")
    return SUITES[version]


@dataclass
class BenchmarkRunResult:
    evidence_id: str
    suite_version: str
    model_id: str
    model_version: str
    execution_mode: str  # 'simulated' | 'measured'
    aggregate_score: float
    cases_passed: int
    cases_total: int
    cases: list[CaseResult]
    transcript_sha256: str | None
    created_at: str
    expires_at: str


class SimulatedCaller:
    """Deterministic synthetic caller for OFFLINE simulated runs. Produces
    canned outputs derived from each case's scoring spec — these results are
    SIMULATED and must never be presented as real model measurements."""

    def __init__(self, *, pass_all: bool = True):
        self._pass_all = pass_all
        self.calls: list[str] = []

    def complete(self, case: BenchmarkCase, model_id: str) -> str:
        self.calls.append(case.case_id)
        spec = case.scoring
        if not self._pass_all:
            return "simulated-wrong-output"
        if spec.kind == "exact":
            return str(spec.expected)
        if spec.kind == "contains":
            needle = spec.expected[0] if isinstance(spec.expected, list) else spec.expected
            return f"simulated answer containing {needle}"
        if spec.kind == "json_fields":
            obj: dict[str, Any] = {}
            for path, expected in dict(spec.expected).items():
                keys = str(path).split(".")
                cur = obj
                for k in keys[:-1]:
                    cur = cur.setdefault(k, {})
                cur[keys[-1]] = expected if expected is not None else "simulated"
            return json.dumps(obj)
        if spec.kind == "regex":
            return "simulated output matching the pattern"
        raise ValueError(spec.kind)


def run_benchmark(
    store: CheckpointStore,
    *,
    caller: ModelCaller,
    model_id: str,
    model_version: str,
    suite_version: str = SUITE_VERSION,
    execution_mode: str = "simulated",
    hardware_profile: str = "",
    run_config: dict[str, Any] | None = None,
    provenance: str = "",
    runner_attestation: str = "",
    ttl_days: int = 30,
    evidence_id: str | None = None,
) -> BenchmarkRunResult:
    """Run one suite against one exact model identity and persist the evidence.

    execution_mode='simulated' (default, offline) uses synthetic callers.
    execution_mode='measured' is accepted only with a transcript hash and an
    operator runner attestation (store-enforced) — it is intended for separately
    authorized real runs through the zero-cost-gated gateway, which this phase
    does not perform."""
    import uuid

    cases = get_suite(suite_version)
    if not re.fullmatch(r"[a-z0-9_./-]+@[A-Za-z0-9][A-Za-z0-9._:+-]{3,127}", model_id):
        raise CheckpointError(
            "model_id must be an exact pinned identity like 'ollama/llama3.2@sha256:...'")
    if execution_mode not in ("simulated", "measured"):
        raise CheckpointError(f"invalid execution_mode {execution_mode!r}")

    results: list[CaseResult] = []
    transcript = hashlib.sha256()
    for case in cases:
        output = caller.complete(case, model_id)
        transcript.update(case.case_id.encode())
        transcript.update(b"\0")
        transcript.update(output.encode("utf-8", "ignore")[:65_536])
        transcript.update(b"\0")
        results.append(score_case(case, output))
    passed = sum(1 for r in results if r.passed)
    aggregate = passed / len(results) if results else 0.0
    transcript_hex = transcript.hexdigest()
    if execution_mode == "measured" and not runner_attestation:
        raise CheckpointError(
            "measured runs require an operator runner_attestation (separate "
            "authorization); refusing to record a measurement without it")

    ev_id = evidence_id or uuid.uuid4().hex
    cases_json = json.dumps(
        [{"case_id": r.case_id, "passed": r.passed, "detail": r.detail[:200]} for r in results],
        sort_keys=True)
    store.record_benchmark_evidence(
        evidence_id=ev_id, suite_version=suite_version, model_id=model_id,
        model_version=model_version, execution_mode=execution_mode,
        aggregate_score=aggregate, cases_passed=passed, cases_total=len(results),
        cases_json=cases_json, hardware_profile=hardware_profile,
        run_config_json=json.dumps(run_config or {}, sort_keys=True, default=str)[:4000],
        provenance=provenance or ("simulated offline run (synthetic fixtures)"
                                  if execution_mode == "simulated" else "gateway run"),
        transcript_sha256=transcript_hex if execution_mode == "measured" else None,
        runner_attestation=runner_attestation, ttl_days=ttl_days,
    )
    row = store.get_benchmark_evidence(ev_id)
    assert row is not None
    return BenchmarkRunResult(
        evidence_id=ev_id, suite_version=suite_version, model_id=model_id,
        model_version=model_version, execution_mode=execution_mode,
        aggregate_score=aggregate, cases_passed=passed, cases_total=len(results),
        cases=results, transcript_sha256=transcript_hex if execution_mode == "measured" else None,
        created_at=row["created_at"], expires_at=row["expires_at"],
    )


def format_report(evidence: list[dict[str, Any]]) -> str:
    """Human-readable benchmark report (CLI). Simulated rows are always marked."""
    lines = [f"Benchmark evidence: {len(evidence)} record(s)"]
    for e in evidence:
        mode = e["execution_mode"].upper()
        if e["execution_mode"] == "simulated":
            mode += " (synthetic fixtures — NOT a real model measurement)"
        expired = " EXPIRED" if e.get("expired") else ""
        lines.append(
            f"  {e['evidence_id'][:12]}  model={e['model_id']} v={e['model_version']} "
            f"suite={e['suite_version']} mode={mode} score={e['aggregate_score']:.3f} "
            f"cases={e['cases_passed']}/{e['cases_total']} created={e['created_at']} "
            f"expires={e['expires_at']}{expired}")
    return "\n".join(lines)
