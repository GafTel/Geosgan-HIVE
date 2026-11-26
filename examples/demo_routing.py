"""Offline demo of the Phase 5 (A3) INTELLIGENT ROUTING ENGINE (no network, no
model calls, no dispatch).

Shows, fully offline against a temporary SQLite store and the informational
catalog:

1. Mandatory eligibility filters BEFORE any ranking: the blocked cloud
   candidates (groq / openrouter) are rejected with reason codes, and unknown
   billing exposure fails closed.
2. Routing modes: 'fixed' (default), 'eligible_default' (explicit configured
   default), and 'evidence_based' - which is UNAVAILABLE while no trusted
   runtime benchmark evidence exists (fail closed, never fabricated scores).
3. Deterministic, explainable decisions: every candidate carries an
   eligibility verdict + reason code; provenance is durably recorded
   (schema v11 routing_decisions) with bounded metadata only.
4. The router SELECTS but never AUTHORIZES: no dispatch happens here; the
   zero-cost gate still revalidates at dispatch time in production.

Run: PYTHONPATH=src python examples/demo_routing.py
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from geosgan_hive.catalog import load_catalog
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.routing import ModelRouter, RoutingMode, RoutingPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)


def show(decision):
    print(f"  decision {decision.decision_id[:12]} agent={decision.agent} "
          f"mode={decision.selection_mode} selected={decision.selected_model} "
          f"reason={decision.reason}")
    for c in decision.candidates:
        score = f"{c.score:.4f}" if c.score is not None else "-"
        state = "ELIGIBLE" if c.eligible else "rejected"
        print(f"    {state:8s} {c.model:28s} reason={c.reason:28s} score={score}")


def main():
    catalog = load_catalog(REPO_ROOT / "config" / "catalog.yaml")
    print(f"1. catalog {catalog.catalog_version}: "
          f"{[p.id for p in catalog.providers]}")

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(f"{tmp}/routing.db", utcnow=lambda: NOW)

        print("\n2. fixed mode (default; configured model must pass all filters):")
        fixed = ModelRouter(RoutingPolicy(), catalog=catalog, store=store,
                            utcnow=lambda: NOW)
        d = fixed.select(agent="researcher", configured_model="ollama/llama3.2")
        show(d)

        print("\n3. eligible_default mode (explicit operator-configured default):")
        policy = RoutingPolicy(mode=RoutingMode.ELIGIBLE_DEFAULT,
                               defaults={"*": "ollama/llama3.2"})
        router = ModelRouter(policy, catalog=catalog, store=store, utcnow=lambda: NOW)
        for agent in ("researcher", "developer", "analyst"):
            show(router.select(agent=agent, configured_model="ollama/llama3.2"))

        print("\n4. evidence_based mode with NO trusted evidence (fail closed):")
        eb = ModelRouter(RoutingPolicy(mode=RoutingMode.EVIDENCE_BASED),
                         catalog=catalog, store=store, utcnow=lambda: NOW)
        d = eb.select(agent="analyst", configured_model="ollama/llama3.2")
        show(d)
        assert d.selected_model is None and d.reason == "NO_TRUSTED_EVIDENCE"

        print("\n5. blocked cloud candidates are never eligible (fail closed):")
        for m in ("groq/llama-3.3-70b-versatile", "openrouter/auto"):
            ev = router.evaluate(m)
            print(f"    {m}: eligible={ev.eligible} reason={ev.reason}")
            assert not ev.eligible

        print("\n6. durable bounded provenance (schema v11 routing_decisions):")
        rows = store.list_routing_decisions()
        print(f"  {len(rows)} decision(s) recorded; sample fields:")
        r = rows[0]
        print(f"    agent={r['agent']} mode={r['selection_mode']} "
              f"selected={r['selected_model']} method={r['scoring_method']} "
              f"policy={r['policy_version']}")
        assert "candidates_json" in r and "prompt" not in str(r).lower()
        store.close()

    print("\nOK: routing is deterministic, explainable and fail-closed; "
          "no dispatch and no network access occurred.")


if __name__ == "__main__":
    main()
