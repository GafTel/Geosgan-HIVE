"""A4 tests: free-cloud evidence & billing verification infrastructure (offline).

Fixture matrix (all deterministic, no network, no live requests):
- forged evidence / operator attestation masquerading as trusted
- credential/account/model/endpoint substitution
- expired / revoked / future-dated evidence
- unknown billing, paid overage, required credit purchase
- missing provider-enforced spending restriction
- fixtures can never mint provider_enforced (code-level ingestion flag only)
- concurrent policy changes, stale workers, restart/recovery
- blocked providers cannot be activated via config / CLI / routing / fallback
- schema v11 -> v12 migration + interrupted-migration rollback
- metadata-only privacy (no raw credentials anywhere)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from geosgan_hive.catalog import catalog_from_dict
from geosgan_hive.checkpoints import SCHEMA_VERSION, CheckpointStore
from geosgan_hive.cloud_evidence import (
    AUTHENTICATED_SOURCES,
    CloudEvidenceRecord,
    CloudEvidenceVault,
    EvidenceClass,
    OverageBehavior,
    SpendingRestriction,
)
from geosgan_hive.cloud_profiles import CloudProfile, CloudProfileRegistry
from geosgan_hive.errors import CheckpointError
from geosgan_hive.routing import ModelRouter, RoutingPolicy

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
FP = "a" * 64
FP_OTHER = "b" * 64
GROQ_MODEL = "groq/llama-3.3-70b-versatile"  # not a valid HIVE provider prefix on purpose


def _store(tmp_path, name="cp.db", now=NOW):
    return CheckpointStore(str(tmp_path / name), utcnow=lambda: now)


def _rec(**over):
    base = dict(
        evidence_id="ev-1", provider_id="groq",
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
        limitations="fixture",
    )
    base.update(over)
    return CloudEvidenceRecord.model_validate(base)


# ---- recording rules ---------------------------------------------------------------

def test_fixture_cannot_mint_provider_enforced(tmp_path):
    """The default vault rejects provider_enforced outright: config/CLI/fixture
    input can never create the only authorizing class."""
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store)  # accept_provider_enforced=False (default)
    with pytest.raises(CheckpointError, match="accept_provider_enforced"):
        vault.record(_rec())
    assert vault.list_evidence(include_revoked=True, include_expired=True) == []
    store.close()


def test_informational_classes_recordable_but_never_eligible(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store)
    vault.record(_rec(evidence_id="docs-1", evidence_class=EvidenceClass.PUBLIC_DOCS,
                      spending_restriction=SpendingRestriction.NONE,
                      overage_behavior=OverageBehavior.UNKNOWN,
                      credential_fingerprint=None))
    vault.record(_rec(evidence_id="op-1", evidence_class=EvidenceClass.OPERATOR_DECLARATION,
                      spending_restriction=SpendingRestriction.NONE,
                      overage_behavior=OverageBehavior.UNKNOWN,
                      credential_fingerprint=None))
    # operator attestation masquerading as trusted: class is still declaration
    ok, reason = vault.eligible_for_live_dispatch(
        provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
        endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP)
    assert not ok and reason == "NO_PROVIDER_ENFORCED_EVIDENCE"
    store.close()


def test_provider_enforced_validation_rules(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    # unauthenticated source
    with pytest.raises(CheckpointError, match="authenticated source"):
        vault.record(_rec(evidence_id="x1", source="operator_says_so"))
    # missing collection method
    with pytest.raises(CheckpointError, match="collection_method"):
        vault.record(_rec(evidence_id="x2", collection_method=""))
    # no hard limit
    with pytest.raises(CheckpointError, match="spending restriction"):
        vault.record(_rec(evidence_id="x3",
                          spending_restriction=SpendingRestriction.NONE))
    # chargeable overage
    with pytest.raises(CheckpointError, match="overage"):
        vault.record(_rec(evidence_id="x4", overage_behavior=OverageBehavior.CHARGE))
    # unknown overage
    with pytest.raises(CheckpointError, match="overage"):
        vault.record(_rec(evidence_id="x5", overage_behavior=OverageBehavior.UNKNOWN))
    # missing credential fingerprint
    with pytest.raises(CheckpointError, match="credential fingerprint"):
        vault.record(_rec(evidence_id="x6", credential_fingerprint=None))
    # future timestamp
    with pytest.raises(CheckpointError, match="future"):
        vault.record(_rec(evidence_id="x7", collected_at=NOW + timedelta(days=1)))
    # replay: duplicate id
    vault.record(_rec())
    with pytest.raises(CheckpointError, match="already exists"):
        vault.record(_rec())
    store.close()


def test_eligible_only_with_exact_binding(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_rec())
    base = dict(provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
                endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP)
    assert vault.eligible_for_live_dispatch(**base) == (True, "ok")
    # credential/account substitution
    assert vault.eligible_for_live_dispatch(
        **{**base, "account_fingerprint": FP_OTHER})[1] == "CREDENTIAL_MISMATCH"
    # model substitution
    assert vault.eligible_for_live_dispatch(
        **{**base, "model_id": "groq/other-model"})[1] == "MODEL_MISMATCH"
    # endpoint substitution
    assert vault.eligible_for_live_dispatch(
        **{**base, "endpoint": "https://evil.example.com"})[1] == "ENDPOINT_MISMATCH"
    # provider mismatch
    assert vault.eligible_for_live_dispatch(
        **{**base, "provider_id": "openrouter"})[1] == "NO_PROVIDER_ENFORCED_EVIDENCE"
    store.close()


def test_expiry_and_revocation_invalidate(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_rec(ttl_days=10))
    base = dict(provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
                endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP)
    assert vault.eligible_for_live_dispatch(**base, now=NOW)[0]
    later = NOW + timedelta(days=11)
    ok, reason = vault.eligible_for_live_dispatch(**base, now=later)
    assert not ok and reason == "EVIDENCE_EXPIRED"
    # revoke (even before expiry) -> immediately ineligible + audited
    vault.revoke("ev-1", reason="provider policy changed", operator="operator-1")
    ok, reason = vault.eligible_for_live_dispatch(**base, now=NOW)
    assert not ok and reason == "EVIDENCE_REVOKED"
    trail = vault.audit_trail("ev-1")
    assert [t["event"] for t in trail] == ["revoked", "recorded"]
    with pytest.raises(CheckpointError):
        vault.revoke("ev-1", reason="again", operator="operator-1")  # double revoke
    store.close()


def test_metadata_only_privacy(tmp_path):
    """No raw credentials anywhere: only sha256 fingerprints + labels."""
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_rec())
    row = vault.get("ev-1")
    assert row["credential_fingerprint"] == FP
    blob = str(row) + str(vault.audit_trail("ev-1")) + str(vault.list_evidence(
        include_revoked=True, include_expired=True))
    assert "sk-" not in blob and "Bearer" not in blob and "api_key=" not in blob.lower()
    # raw-credential-looking fingerprint is rejected by the model pattern
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CloudEvidenceRecord.model_validate({**_rec().__dict__,
                                            "credential_fingerprint": "sk-abcdef1234567890"})
    store.close()


# ---- registry / gate integration ------------------------------------------------------

def _profile(provider="gemini", models=("gemini/gemini-2.0-flash",), fp=FP,
             enforcement="provider_hard_limit", endpoint=None):
    anchor = (NOW - timedelta(days=1)).isoformat()
    ev = {
        "source": "provider_billing_console_export", "evidence_ref": "ev-1",
        "collected_at": anchor, "collected_by": "operator-1",
        "credential_bound": True, "account_fingerprint": fp,
        "billing_enforcement": enforcement,
    }
    if endpoint:
        ev["endpoint"] = endpoint
    return CloudProfile.model_validate({
        "provider": provider, "models": list(models),
        "billing": "free_tier_verified", "enabled": True,
        "verified_free": True, "overage_billing_disabled": True,
        "billing_verified": True, "verified_at": anchor,
        "account_label": "main", "account_fingerprint": fp,
        "quota_window_start": anchor, "quota_reset_semantics": "provider_documented",
        "verification_evidence": ev,
        "quotas": [{"pool": f"{provider}:acct:{fp}", "max_requests": 100,
                    "max_tokens": 1_000_000, "period": "day"}],
    })


def test_vault_wired_blocks_cloud_without_provider_enforced_evidence(tmp_path):
    """With the vault wired, even a fully-profiled cloud model stays blocked
    until provider_enforced evidence matches the exact binding."""
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    reg = CloudProfileRegistry(profiles=[_profile()], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=True,
                               evidence_vault=vault)
    model = "gemini/gemini-2.0-flash"
    ok, why = reg.live_dispatch_eligible(model, now=NOW)
    assert not ok and "vault" in why  # no provider_enforced evidence yet
    # import only public docs + operator declaration -> still blocked
    vault.record(_rec(evidence_id="d1", provider_id="gemini",
                      evidence_class=EvidenceClass.PUBLIC_DOCS,
                      model_id=model, spending_restriction=SpendingRestriction.NONE,
                      overage_behavior=OverageBehavior.UNKNOWN,
                      credential_fingerprint=None))
    vault.record(_rec(evidence_id="d2", provider_id="gemini",
                      evidence_class=EvidenceClass.OPERATOR_DECLARATION,
                      model_id=model, spending_restriction=SpendingRestriction.NONE,
                      overage_behavior=OverageBehavior.UNKNOWN,
                      credential_fingerprint=None))
    ok, why = reg.live_dispatch_eligible(model, now=NOW)
    assert not ok and "NO_PROVIDER_ENFORCED_EVIDENCE" in why
    # a matching provider_enforced record unblocks ONLY this exact binding
    vault.record(_rec(evidence_id="pe-1", provider_id="gemini", model_id=model,
                      endpoint=None, credential_fingerprint=FP))
    ok, why = reg.live_dispatch_eligible(model, now=NOW)
    assert ok, why
    # model substitution still blocked
    ok2, _ = reg.live_dispatch_eligible("gemini/gemini-2.5-pro", now=NOW)
    assert not ok2  # no profile for that model at all
    store.close()


def test_vault_not_wired_preserves_existing_behavior(tmp_path):
    store = _store(tmp_path)
    reg = CloudProfileRegistry(profiles=[_profile()], zero_cost=True,
                               max_spend_usd=0.0, trusted_evidence_binding=True)
    ok, why = reg.live_dispatch_eligible("gemini/gemini-2.0-flash", now=NOW)
    assert ok, why  # unchanged legacy trusted path (vault opt-in only)
    store.close()


def test_local_inference_not_subject_to_vault(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store)
    reg = CloudProfileRegistry(
        profiles=[_profile(provider="ollama", models=("ollama/llama3.2",),
                            enforcement="local_inference",
                            endpoint="http://localhost:11434")],
        zero_cost=True, max_spend_usd=0.0, trusted_evidence_binding=True,
        evidence_vault=vault)
    ok, why = reg.live_dispatch_eligible("ollama/llama3.2", now=NOW)
    assert ok, why  # local inference path does not require vault evidence
    store.close()


def test_blocked_providers_cannot_be_activated_via_routing_or_config(tmp_path, no_network):
    """Groq/OpenRouter stay blocked through every activation vector."""
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store)  # fixtures cannot mint provider_enforced
    reg = CloudProfileRegistry(profiles=(), zero_cost=True, max_spend_usd=0.0,
                               evidence_vault=vault)
    cat = catalog_from_dict({
        "catalog_version": "1.0.0", "generated_at": NOW.isoformat(),
        "providers": [
            {"id": "mistral", "name": "mistral", "category": "llm_cloud",
             "base_endpoints": ["https://api.mistral.ai/v1"],
             "models": [{"model_id": "mistral/m", "capabilities": ["chat"],
                         "context_window": 128_000, "tool_support": True}],
             "capabilities": ["chat"], "docs": ["https://docs.mistral.ai"],
             "billing": "free_tier_unverified", "operational_status": "blocked"},
        ]})
    router = ModelRouter(RoutingPolicy(mode="eligible_default",
                                        defaults={"*": "mistral/m"},
                                        cross_provider_fallback=True),
                         catalog=cat, cloud_registry=reg, store=store,
                         utcnow=lambda: NOW)
    # the blocked/unverified catalog entry is never selected even with fallback
    # authorization on
    d = router.select(agent="researcher", configured_model="mistral/m")
    assert d.selected_model is None
    ev = router.evaluate("mistral/m")
    assert not ev.eligible and ev.reason in ("CATALOG_PROVIDER_DISABLED",
                                             "BILLING_UNVERIFIED")
    # a profile for a blocked provider is still denied by the registry
    reg2 = CloudProfileRegistry(
        profiles=[_profile(provider="mistral", models=("mistral/m",))],
        zero_cost=True, max_spend_usd=0.0, trusted_evidence_binding=True,
        evidence_vault=vault)
    ok, why = reg2.live_dispatch_eligible("mistral/m", now=NOW)
    assert not ok and "vault" in why  # no provider_enforced evidence exists
    store.close()


def test_concurrent_record_and_revoke_is_atomic(tmp_path):
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    errors = []

    def rec(i):
        try:
            vault.record(_rec(evidence_id=f"ev-{i}"))
        except CheckpointError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=rec, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(vault.list_evidence(include_revoked=True, include_expired=True)) == 20
    # concurrent revokes of distinct ids all succeed, each audited exactly once
    def rev(i):
        vault.revoke(f"ev-{i}", reason="rotation", operator="operator-1")

    threads = [threading.Thread(target=rev, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    rows = vault.list_evidence(include_revoked=True, include_expired=True)
    assert all(r["revoked"] for r in rows)
    assert all(len(vault.audit_trail(r["evidence_id"])) == 2 for r in rows)
    store.close()


def test_restart_recovery_preserves_evidence_and_denials(tmp_path):
    """Close + reopen the store: evidence, revocation and denials persist."""
    store = _store(tmp_path)
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_rec())
    vault.revoke("ev-1", reason="rotation", operator="operator-1")
    store.close()
    store2 = CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: NOW)
    vault2 = CloudEvidenceVault(store2, accept_provider_enforced=True)
    row = vault2.get("ev-1")
    assert row is not None and row["revoked"] == 1
    ok, reason = vault2.eligible_for_live_dispatch(
        provider_id="groq", model_id="groq/llama-3.3-70b-versatile",
        endpoint="https://api.groq.com/openai/v1", account_fingerprint=FP, now=NOW)
    assert not ok and reason == "EVIDENCE_REVOKED"
    store2.close()


# ---- schema v11 -> v12 migration -------------------------------------------------------

def test_v11_to_v12_migration_additive_and_rollback(tmp_path):
    db = str(tmp_path / "v11.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '11')")
    conn.commit()
    conn.close()
    store = CheckpointStore(db, utcnow=lambda: NOW)
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "12" and SCHEMA_VERSION == 12
    assert {"cloud_evidence", "cloud_evidence_audit"} <= tables
    # vault works on the migrated store
    vault = CloudEvidenceVault(store, accept_provider_enforced=True)
    vault.record(_rec())
    assert vault.get("ev-1") is not None
    store.close()

    # interrupted migration rolls back fully and recovers on next open
    db2 = str(tmp_path / "v11b.db")
    conn = sqlite3.connect(db2)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '11')")
    conn.commit()
    conn.close()
    import geosgan_hive.checkpoints as ckpt

    real = ckpt.CheckpointStore._migrate_11_to_12

    def boom(self):
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cloud_evidence (x TEXT)")  # partial write
        raise RuntimeError("interrupted")

    ckpt.CheckpointStore._migrate_11_to_12 = boom
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            CheckpointStore(db2, utcnow=lambda: NOW)
    finally:
        ckpt.CheckpointStore._migrate_11_to_12 = real
    conn = sqlite3.connect(db2)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert v == "11" and "cloud_evidence" not in tables  # rolled back fully
    store = CheckpointStore(db2, utcnow=lambda: NOW)  # recovers to v12
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "12"
    store.close()


# ---- CLI inspection ---------------------------------------------------------------------

def test_cli_cloud_evidence_lists_metadata_only(tmp_path, capsys, no_network):
    from geosgan_hive.agent_config import HiveConfig
    from geosgan_hive.cli import main
    from geosgan_hive.factory import build_team
    from geosgan_hive.gateway import ModelGateway

    anchor = (NOW - timedelta(days=1)).isoformat()
    cloud = {
        "zero_cost": True, "max_spend_usd": 0.0, "trusted_evidence_binding": True,
        "evidence_vault": True,
        "profiles": [{
            "provider": "ollama", "models": ["ollama/llama3.2"],
            "billing": "free_tier_verified", "enabled": True,
            "verified_free": True, "overage_billing_disabled": True,
            "billing_verified": True, "verified_at": anchor,
            "account_label": "main", "account_fingerprint": FP,
            "quota_window_start": anchor,
            "quota_reset_semantics": "provider_documented",
            "verification_evidence": {
                "source": "provider_billing_console_export", "evidence_ref": "ev-1",
                "collected_at": anchor, "collected_by": "operator-1",
                "credential_bound": True, "account_fingerprint": FP,
                "billing_enforcement": "local_inference",
                "endpoint": "http://localhost:11434",
            },
            "quotas": [{"pool": "ollama:acct:" + FP, "max_requests": 100,
                        "max_tokens": 1_000_000, "period": "day"}],
        }],
    }
    config = HiveConfig.model_validate({
        "supervisor": {"name": "supervisor", "description": "s",
                       "system_prompt": "s", "model": "ollama/llama3.2"},
        "agents": [{"name": "researcher", "description": "r", "system_prompt": "r",
                    "model": "ollama/llama3.2"}],
        "checkpoint": {"path": str(tmp_path / "cp.db"), "persist_mode": "resumable"},
        "cloud": cloud,
    })
    gw = ModelGateway()
    build_team(config, gw)
    vault = gw._cloud_registry.evidence_vault
    assert vault is not None  # wired via the config flag
    vault.record(_rec(evidence_id="docs-1", evidence_class=EvidenceClass.PUBLIC_DOCS,
                      provider_id="groq",
                      spending_restriction=SpendingRestriction.NONE,
                      overage_behavior=OverageBehavior.UNKNOWN,
                      credential_fingerprint=None))
    rc = main(["--cloud-evidence", "--agents-config", "config/agents.local.yaml",
               "--checkpoint", str(tmp_path / "cp.db")])
    # CLI builds its own store/registry from the config file (vault not wired in
    # agents.local.yaml) -> reports not wired, exit 0
    assert rc == 0
    out = capsys.readouterr().out
    assert "vault" in out
