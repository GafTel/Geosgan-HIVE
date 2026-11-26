"""Offline demo of the Phase 5 (A2) MODEL BENCHMARK ENGINE (no model, no network).

Runs the versioned benchmark suite (supervisor/researcher/developer/analyst/
reviewer, English + Spanish cases) with the deterministic SIMULATED caller,
persists the evidence in a temporary SQLite store (schema v10), prints the CLI
report, and shows the simulated/measured separation guarantees:

1. Simulated evidence is stored labeled SIMULATED (synthetic fixtures).
2. A 'measured' record without an operator attestation + transcript hash is
   refused — simulated results can never be relabeled as real measurements.
3. Evidence expires (TTL) and is swept; expired rows are hidden by default.
4. Scoring is deterministic: identical runs produce identical scores.

Real benchmark execution requires separate authorization (zero-cost-gated
gateway); this demo performs none and downloads no weights.

Run: PYTHONPATH=src python examples/demo_benchmarks.py
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta

from geosgan_hive.benchmarks import (
    SUITE_VERSION,
    SimulatedCaller,
    format_report,
    get_suite,
    run_benchmark,
)
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.errors import CheckpointError

MODEL = "ollama/llama3.2@sha256:demo0000"


def main():
    cases = get_suite()
    print(f"1. suite {SUITE_VERSION}: {len(cases)} cases, "
          f"roles={sorted({c.role for c in cases})}, "
          f"languages={sorted({c.language for c in cases})}")

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(f"{tmp}/bench.db",
                                utcnow=lambda: datetime(2026, 11, 26, 12, tzinfo=UTC))
        r1 = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                           model_version="sha256:demo0000", hardware_profile="demo-box",
                           run_config={"temperature": 0.0, "max_tokens": 512},
                           provenance="demo simulated run")
        r2 = run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                           model_version="sha256:demo0000")
        print(f"2. simulated run: score={r1.aggregate_score:.3f} "
              f"cases={r1.cases_passed}/{r1.cases_total} mode={r1.execution_mode.upper()}")
        print(f"   deterministic: identical rerun score={r2.aggregate_score:.3f} "
              f"(equal={r1.aggregate_score == r2.aggregate_score})")

        print("3. report (fresh evidence):")
        print(format_report(store.list_benchmark_evidence()))

        # measured without attestation -> refused
        try:
            run_benchmark(store, caller=SimulatedCaller(), model_id=MODEL,
                          model_version="sha256:demo0000", execution_mode="measured")
            raise SystemExit("expected refusal")
        except CheckpointError as exc:
            print(f"4. measured without operator attestation refused: {str(exc)[:70]}...")

        # unpinned model identity -> refused
        try:
            run_benchmark(store, caller=SimulatedCaller(), model_id="llama3.2",
                          model_version="latest")
            raise SystemExit("expected refusal")
        except CheckpointError as exc:
            print(f"5. unpinned model identity refused: {str(exc)[:70]}...")

        # expiry: advance the clock past the TTL
        later = datetime(2026, 11, 26, 12, tzinfo=UTC) + timedelta(days=31)
        store2 = CheckpointStore(store.path, utcnow=lambda: later)
        hidden = store2.list_benchmark_evidence()
        shown = store2.list_benchmark_evidence(include_expired=True)
        print(f"6. expiry: fresh view shows {len(hidden)} record(s), "
              f"with-expired shows {len(shown)} (marked EXPIRED); "
              f"swept={store2.sweep_expired_benchmark_evidence()}")
        store2.close()
        store.close()

    print("\nDemo finished. Fully offline; simulated results are NOT real model measurements.")


if __name__ == "__main__":
    main()
