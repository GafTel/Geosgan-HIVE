"""A4: provider-independent free-cloud EVIDENCE and billing verification (v0.7.0).

Infrastructure only. NO live provider request is made or authorized here, and
Groq / OpenRouter stay blocked for live dispatch throughout v0.7.0: no
provider currently supplies trustworthy provider-enforced zero-cost evidence
through the accepted ingestion path.

Evidence classes (strictly ordered by trust):
- public_docs:           informational only (official documentation snapshot)
- operator_declaration:  operator-entered claim; NEVER authorization
- account_authenticated: account-specific, authenticated collection; still not
                         by itself provider-enforced proof
- provider_enforced:     the ONLY class that can ever support live-dispatch
                         eligibility (provider-enforced hard spending limit,
                         no chargeable overage). Recording this class requires
                         CloudEvidenceVault(accept_provider_enforced=True),
                         which is a code-level constructor flag that is NEVER
                         set from YAML config, CLI, routing or fallback. Test
                         fixtures therefore cannot authorize production dispatch.

Bindings: provider identity, account/org label, credential fingerprint (sha256
of the credential - the raw credential is NEVER stored), exact model id+version,
exact endpoint, billing plan, quota pool, spending restriction, overage
behavior, source + collection method, collected_at/expiry, revocation, policy
version. A boolean flag, a "free" model name, a zero displayed price or a
manually entered record never authorizes live dispatch.

Durability: schema v12 (cloud_evidence + cloud_evidence_audit), atomic inserts,
duplicate-id replay rejected, revocation audited, expiry enforced at read time.
Metadata-only privacy: fingerprints/labels/refs only, never raw secrets or
payloads. Eligibility checks are fail-closed and re-evaluated at dispatch time;
evidence import NEVER auto-activates anything.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .checkpoints import CheckpointStore
from .errors import CheckpointError

#: sources whose collection method is considered authenticated for
#: account_authenticated / provider_enforced classes (allowlist, fail closed)
AUTHENTICATED_SOURCES = (
    "provider_billing_api_signed_export",
    "provider_billing_console_export",
    "provider_admin_api_audit_log",
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_HTTPS_URL = re.compile(
    r"^https://(\[[0-9a-fA-F:.]+\]|[a-zA-Z0-9.-]{1,253})(:[0-9]{1,5})?"
    r"(/[a-zA-Z0-9_./:?#&=%~+-]*)?$")


class EvidenceClass(StrEnum):
    PUBLIC_DOCS = "public_docs"
    OPERATOR_DECLARATION = "operator_declaration"
    ACCOUNT_AUTHENTICATED = "account_authenticated"
    PROVIDER_ENFORCED = "provider_enforced"


class SpendingRestriction(StrEnum):
    NONE = "none"
    HARD_LIMIT_USD0 = "hard_limit_usd0"            # provider-enforced $0 cap
    ALLOWANCE_NO_OVERAGE = "allowance_no_overage"  # free allowance, overage disabled


class OverageBehavior(StrEnum):
    UNKNOWN = "unknown"
    BLOCK = "block"        # provider blocks requests beyond the allowance
    THROTTLE = "throttle"  # provider throttles; never charges
    CHARGE = "charge"      # provider charges overage -> disqualifies


class CloudEvidenceRecord(BaseModel):
    """One typed evidence record. Frozen; extra fields forbidden; timestamps
    tz-aware. Never contains raw credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")
    evidence_class: EvidenceClass
    account_org_label: str = Field(default="", max_length=100)
    credential_fingerprint: str | None = Field(default=None, pattern=_SHA256_HEX.pattern)
    model_id: str | None = Field(default=None, max_length=160)
    # exact model binding; None = provider-wide
    endpoint: str | None = Field(default=None, pattern=r"^https?://[!-~]{1,2000}$")
    billing_plan: str = Field(default="", max_length=100)
    quota_pool: str = Field(default="", max_length=100)
    spending_restriction: SpendingRestriction = SpendingRestriction.NONE
    overage_behavior: OverageBehavior = OverageBehavior.UNKNOWN
    source: str = Field(pattern=r"^[a-z0-9][a-z0-9_ -]{0,63}$")
    collection_method: str = Field(default="", max_length=100)
    docs_url: str = Field(default="", pattern=_HTTPS_URL.pattern)
    policy_version: str = Field(default="", max_length=64)
    collected_at: datetime
    ttl_days: int = Field(default=30, ge=1, le=3650)
    limitations: str = Field(default="", max_length=500)

    @field_validator("collected_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware (naive rejected)")
        return v


#: eligibility reason codes
E_OK = "ok"
E_NO_PROVIDER_ENFORCED = "NO_PROVIDER_ENFORCED_EVIDENCE"
E_REVOKED = "EVIDENCE_REVOKED"
E_EXPIRED = "EVIDENCE_EXPIRED"
E_PROVIDER_MISMATCH = "PROVIDER_MISMATCH"
E_MODEL_MISMATCH = "MODEL_MISMATCH"
E_ENDPOINT_MISMATCH = "ENDPOINT_MISMATCH"
E_CREDENTIAL_MISMATCH = "CREDENTIAL_MISMATCH"
E_NO_HARD_LIMIT = "NO_PROVIDER_ENFORCED_SPENDING_LIMIT"
E_CHARGEABLE_OVERAGE = "CHARGEABLE_OVERAGE"
E_UNAUTHENTICATED_SOURCE = "UNAUTHENTICATED_SOURCE"
E_FUTURE_TIMESTAMP = "FUTURE_COLLECTION_TIMESTAMP"


class CloudEvidenceVault:
    """Durable store + eligibility oracle for cloud evidence (schema v12).

    `accept_provider_enforced` is a CODE-LEVEL flag (default False): without
    it, recording provider_enforced evidence is rejected outright, so test
    fixtures / imported files can never mint the only authorizing class.
    It is deliberately NOT exposed via YAML config, CLI, routing or fallback.
    """

    MAX_ROWS = 1000

    def __init__(self, store: CheckpointStore, *, accept_provider_enforced: bool = False,
                 utcnow: Any = None) -> None:
        self.store = store
        self.accept_provider_enforced = bool(accept_provider_enforced)
        self._utcnow = utcnow or store.utcnow

    # ---- recording ------------------------------------------------------------

    def record(self, rec: CloudEvidenceRecord) -> None:
        """Atomically persist one evidence record. Class rules:
        - provider_enforced requires accept_provider_enforced=True (code-level
          only) AND an authenticated source AND a hard spending restriction
          AND non-chargeable overage AND a credential fingerprint;
        - account_authenticated requires an authenticated source;
        - public_docs / operator_declaration are informational (recordable) but
          can never become eligible;
        - duplicate evidence_id rejected (replay); future collected_at
          rejected; expiry bounded by ttl_days.
        One transaction; fail closed on any violation."""
        now = self._utcnow()
        if rec.evidence_class == EvidenceClass.PROVIDER_ENFORCED:
            if not self.accept_provider_enforced:
                raise CheckpointError(
                    "recording provider_enforced evidence requires the code-level "
                    "CloudEvidenceVault(accept_provider_enforced=True) ingestion "
                    "path \u2014 config/CLI/fixture input can never mint this class; "
                    "live dispatch stays blocked")
            if rec.source not in AUTHENTICATED_SOURCES:
                raise CheckpointError(
                    f"provider_enforced evidence requires an authenticated source "
                    f"(one of {AUTHENTICATED_SOURCES}), got {rec.source!r}")
            if not rec.collection_method:
                raise CheckpointError(
                    "provider_enforced evidence requires a collection_method")
            if rec.spending_restriction not in (
                    SpendingRestriction.HARD_LIMIT_USD0,
                    SpendingRestriction.ALLOWANCE_NO_OVERAGE):
                raise CheckpointError(
                    "provider_enforced evidence requires a provider-enforced "
                    "spending restriction (hard_limit_usd0 or allowance_no_overage)")
            if rec.overage_behavior == OverageBehavior.CHARGE:
                raise CheckpointError(
                    "chargeable overage disqualifies provider_enforced evidence")
            if rec.overage_behavior == OverageBehavior.UNKNOWN:
                raise CheckpointError(
                    "provider_enforced evidence requires a KNOWN non-chargeable "
                    "overage behavior (block/throttle); unknown is rejected")
            if not rec.credential_fingerprint:
                raise CheckpointError(
                    "provider_enforced evidence requires a credential fingerprint "
                    "(sha256; the raw credential is never stored)")
        if rec.evidence_class == EvidenceClass.ACCOUNT_AUTHENTICATED:
            if rec.source not in AUTHENTICATED_SOURCES:
                raise CheckpointError(
                    "account_authenticated evidence requires an authenticated source")
        if rec.collected_at > now:
            raise CheckpointError("collected_at is in the future \u2014 rejected")
        expires = (rec.collected_at + timedelta(days=rec.ttl_days)).isoformat(
            timespec="milliseconds")
        try:
            with self.store._lock, self.store._conn:
                self.store._conn.execute(
                    "INSERT INTO cloud_evidence (evidence_id, provider_id, "
                    "evidence_class, account_org_label, credential_fingerprint, "
                    "model_id, endpoint, billing_plan, quota_pool, "
                    "spending_restriction, overage_behavior, source, "
                    "collection_method, docs_url, policy_version, collected_at, "
                    "expires_at, revoked, revocation_reason, limitations, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', ?, ?)",
                    (rec.evidence_id, rec.provider_id, rec.evidence_class.value,
                     rec.account_org_label, rec.credential_fingerprint, rec.model_id,
                     rec.endpoint, rec.billing_plan, rec.quota_pool,
                     rec.spending_restriction.value, rec.overage_behavior.value,
                     rec.source, rec.collection_method, rec.docs_url,
                     rec.policy_version, rec.collected_at.isoformat(timespec="milliseconds"),
                     expires, rec.limitations[:500], now.isoformat(timespec="milliseconds")),
                )
                self.store._conn.execute(
                    "INSERT INTO cloud_evidence_audit (evidence_id, event, detail, at) "
                    "VALUES (?, 'recorded', ?, ?)",
                    (rec.evidence_id, rec.evidence_class.value,
                     now.isoformat(timespec="milliseconds")))
        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                f"cloud evidence {rec.evidence_id!r} already exists \u2014 replay/"
                "substitution rejected") from exc

    def revoke(self, evidence_id: str, *, reason: str, operator: str = "operator",
               now_iso: str | None = None) -> None:
        """Atomically revoke one record (audited). Revocation immediately
        invalidates eligibility for everything bound to it."""
        now = now_iso or self._utcnow().isoformat(timespec="milliseconds")
        if not reason or len(reason) > 300:
            raise CheckpointError("revocation requires a bounded reason")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", operator or ""):
            raise CheckpointError("invalid operator id")
        with self.store._lock, self.store._conn:
            cur = self.store._conn.execute(
                "UPDATE cloud_evidence SET revoked=1, revocation_reason=? WHERE "
                "evidence_id=? AND revoked=0", (reason[:300], evidence_id))
            if cur.rowcount != 1:
                raise CheckpointError(
                    f"cloud evidence {evidence_id!r} not found or already revoked")
            self.store._conn.execute(
                "INSERT INTO cloud_evidence_audit (evidence_id, event, detail, at) "
                "VALUES (?, 'revoked', ?, ?)", (evidence_id, reason[:300], now))

    # ---- reading ----------------------------------------------------------------

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT * FROM cloud_evidence WHERE evidence_id=?",
                (evidence_id,)).fetchone()
        return {k: row[k] for k in row.keys()} if row else None

    def list_evidence(self, *, provider_id: str | None = None,
                      include_revoked: bool = False, include_expired: bool = False,
                      limit: int = 200) -> list[dict[str, Any]]:
        now = self._utcnow().isoformat(timespec="milliseconds")
        with self.store._lock:
            q = "SELECT * FROM cloud_evidence"
            cond, params = [], []
            if provider_id is not None:
                cond.append("provider_id=?")
                params.append(provider_id)
            if not include_revoked:
                cond.append("revoked=0")
            if not include_expired:
                cond.append("expires_at > ?")
                params.append(now)
            if cond:
                q += " WHERE " + " AND ".join(cond)
            q += " ORDER BY collected_at DESC, evidence_id DESC LIMIT ?"
            rows = self.store._conn.execute(q, (*params, min(limit, self.MAX_ROWS))).fetchall()
        out = [{k: r[k] for k in r.keys()} for r in rows]
        for r in out:
            r["expired"] = r["expires_at"] <= now
        return out

    def audit_trail(self, evidence_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT seq, evidence_id, event, detail, at FROM cloud_evidence_audit "
                "WHERE evidence_id=? ORDER BY seq DESC LIMIT ?",
                (evidence_id, min(limit, 200))).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    # ---- eligibility (fail closed) ------------------------------------------------

    def eligible_for_live_dispatch(
        self, *, provider_id: str, model_id: str, endpoint: str | None,
        account_fingerprint: str, now: datetime | None = None,
    ) -> tuple[bool, str]:
        """May live dispatch be authorized for this exact binding RIGHT NOW?

        Requires at least one provider_enforced, non-revoked, unexpired record
        whose provider matches, whose model binding is exact-or-provider-wide,
        whose endpoint binding is exact-or-provider-wide, whose credential
        fingerprint matches the account fingerprint binding, with a
        provider-enforced spending restriction and non-chargeable overage from
        an authenticated source. Everything else fails closed. Importing
        evidence never activates anything by itself: this is only an oracle
        consulted by the existing gate at dispatch time."""
        now = now or self._utcnow()
        rows = self.list_evidence(provider_id=provider_id, include_revoked=True,
                                  include_expired=True, limit=self.MAX_ROWS)
        candidates = [r for r in rows if r["evidence_class"] == "provider_enforced"]
        if not candidates:
            return False, E_NO_PROVIDER_ENFORCED
        now_iso = now.isoformat(timespec="milliseconds")
        for r in candidates:
            if r["revoked"]:
                return False, E_REVOKED
            if r["expires_at"] <= now_iso:
                return False, E_EXPIRED
            if r["provider_id"] != provider_id:
                return False, E_PROVIDER_MISMATCH
            if r["model_id"] is not None and r["model_id"] != model_id:
                return False, E_MODEL_MISMATCH
            if r["endpoint"] is not None and endpoint is not None and r["endpoint"] != endpoint:
                return False, E_ENDPOINT_MISMATCH
            if r["credential_fingerprint"] != account_fingerprint:
                return False, E_CREDENTIAL_MISMATCH
            if r["spending_restriction"] not in (
                    "hard_limit_usd0", "allowance_no_overage"):
                return False, E_NO_HARD_LIMIT
            if r["overage_behavior"] in ("charge", "unknown"):
                return False, E_CHARGEABLE_OVERAGE
            if r["source"] not in AUTHENTICATED_SOURCES or not r["collection_method"]:
                return False, E_UNAUTHENTICATED_SOURCE
            if r["collected_at"] > now_iso:
                return False, E_FUTURE_TIMESTAMP
        return True, E_OK


__all__ = [
    "AUTHENTICATED_SOURCES", "CloudEvidenceRecord", "CloudEvidenceVault",
    "EvidenceClass", "OverageBehavior", "SpendingRestriction",
]
