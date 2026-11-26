"""Offline demo of the Phase 5 (A4) CLOUD EVIDENCE & BILLING VERIFICATION
infrastructure (no network, no live provider requests, nothing activated).

Shows, against a temporary SQLite store (schema v12):

1. Evidence classes and why only provider_enforced can ever authorize: public
   docs and operator declarations are recorded but never eligible.
2. Test fixtures can NEVER mint provider_enforced evidence: the default vault
   rejects that class (the accept flag is code-level only, never config/CLI).
3. Even with the code-level ingestion path, eligibility requires the EXACT
   binding (provider + model + endpoint + credential fingerprint), a
   provider-enforced spending restriction and non-chargeable overage; expiry,
   revocation and substitution all fail closed.
4. Groq/OpenRouter stay blocked for live dispatch in v0.7.0.
5. Metadata-only privacy: only sha256 fingerprints and labels are stored.

Run: PYTHONPATH=src python examples/demo_cloud_evidence.py
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta

from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.cloud_evidence import (
    AUTHENTICATED_SOURCES,
    CloudEvidenceRecord,
    CloudEvidenceVault,
    EvidenceClass,
    OverageBehavior,
    SpendingRestriction,
)
from geosgan_hive.errors import CheckpointError

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
FP = "a" * 64


def _rec(**over):
    base = dict(
        evidence_id="ev-demo", provider_id="groq",
        evidence_class=EvidenceClass.PROVIDER_ENFORCED,
        account_org_label="org-main", credential_fingerprint=FP,
        model_id="groq/llama-3.3-70b-versatile",
        endpoint="https://api.groq.com/openai/v1",
        billing_plan="free", quota_pool="groq:acct:" + FP,
        spending_restriction=SpendingRestriction.HARD_LIMIT_USD0,
        overage_behavior=OverageBehavior.BLOCK,
        source=AUTHENTICATED_SOURCES[0], collection_method="signed_export_v1",
        docs_url="https://console.groq.com/docs", policy_version="groq-tos-2026-11",
        collected_at=NOW - timedelta(days=1), ttl_days=30,
        limitations="demo fixture - never authorizes production dispatch",
    )
    base.update(over)
    return CloudEvidenceRecord.model_validate(base)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(f"{tmp}/evidence.db", utcnow=lambda: NOW)

        print("1. informational evidence is recordable but never eligible:")
        vault = CloudEvidenceVault(store)  # default: accept_provider_enforced=False
        vault.record(_rec(evidence_id="docs-1", evidence_class=EvidenceClass.PUBLIC_DOCS,
                          spending_restriction=SpendingRestriction.NONE,
                          overage_behavior=OverageBehavior.UNKNOWN,
                          credential_fingerprint=None))
        vault.record(_rec(evidence_id="op-1",
                          evidence_class=EvidenceClass.OPERATOR_DECLARATION,
                          spending_restriction=SpendingRestriction.NONE,
                          overage_behavior=OverageBehavior.UNKNOWN,
                          credential_fingerprint=None))
        ok, why = vault.eligible_for_live_dispatch(
            provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
            endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP)
        print(f"   docs+declaration recorded; eligible={ok} reason={why}")
        assert not ok

        print("2. a fixture cannot mint provider_enforced evidence:")
        try:
            vault.record(_rec())
            raise SystemExit("BUG: fixture minted provider_enforced")
        except CheckpointError as exc:
            print(f"   rejected: {str(exc)[:70]}...")

        print("3. code-level ingestion path + exact binding rules:")
        vault2 = CloudEvidenceVault(store, accept_provider_enforced=True)
        vault2.record(_rec(evidence_id="pe-1"))
        base = dict(provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
                    endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP)
        print(f"   exact binding -> {vault2.eligible_for_live_dispatch(**base)}")
        for label, over in (("credential", {"account_fingerprint": "b" * 64}),
                            ("model", {"model_id": "groq/other"}),
                            ("endpoint", {"endpoint": "https://evil.example.com"})):
            ok, why = vault2.eligible_for_live_dispatch(**{**base, **over})
            print(f"   {label} substitution -> eligible={ok} reason={why}")
            assert not ok
        vault2.revoke("pe-1", reason="demo rotation", operator="operator-1")
        ok, why = vault2.eligible_for_live_dispatch(**base)
        print(f"   after revocation -> eligible={ok} reason={why}")
        assert not ok

        print("4. metadata-only: only fingerprints/labels stored, audit trail kept:")
        row = vault2.get("pe-1")
        fp = row["credential_fingerprint"][:12]
        print(f"   class={row['evidence_class']} fingerprint={fp}... "
              f"revoked={bool(row['revoked'])}")
        print(f"   audit: {[t['event'] for t in vault2.audit_trail('pe-1')]}")
        assert "sk-" not in str(row)

        print("5. Groq/OpenRouter remain blocked for live dispatch in v0.7.0 "
              "(no provider_enforced evidence via the accepted path).")
        store.close()

    print("\nOK: evidence infrastructure is durable, bound, fail-closed and "
          "metadata-only; nothing was activated and no request was made.")


if __name__ == "__main__":
    main()
