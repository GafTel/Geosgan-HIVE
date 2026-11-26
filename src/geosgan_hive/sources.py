"""W2 (v0.8.0): Source Registry and Citation Integrity (schema v13).

A source record binds: stable source id, run/task identity, creating agent or
trusted-runtime identity, source type + provider, canonical (sanitized) locator,
retrieval timestamp, raw/extracted content SHA-256, extraction method+version,
byte counts, page/section/sheet/cell provenance, license + attribution (NULL =
explicitly UNKNOWN, never fabricated), expiry, visibility, storage policy.

Trust boundary (mirrors the A2 benchmark-recorder lesson — a private name or
boolean is NOT proof of trusted execution):
- registration requires a `_TrustedOrigin`, which can only be minted by
  `SourceRegistry.trusted_origin(...)` AFTER the run lease (owner+generation)
  is verified against the store — and re-verified atomically inside the write
  transaction. Agents/models cannot register invented URLs or content: no tool
  registers sources in W2 (tools arrive in W3+), and the only trusted runtime
  paths are lease-fenced.
- test fixtures use `register_fixture` (requires allow_fixtures=True at
  construction) and are recorded with origin_kind='offline_fixture' so fixture
  provenance is always distinguishable from production retrieval.
- trusted-process boundary: this protects the supported public APIs/config/CLI,
  NOT arbitrary privileged Python or direct DB writes.

Citation integrity: `cite`/`excerpt` accept only registered, authorized,
unexpired sources whose stored content still verifies against its recorded
sha256. Unknown ids, cross-run access (unless visibility='team'), expired
sources, substitutions and hash mismatches are rejected. A valid citation
establishes PROVENANCE, not factual correctness.

Privacy/retention: metadata-only by default — raw content, excerpts,
credentials, sensitive URL query parameters (redacted in the stored locator)
and private payloads are NEVER persisted; workspace_relpath is workspace-
relative only. Persistent content requires an explicit storage='workspace'
policy and is integrity-verified on every read. sweep_expired implements
retention; delete_content additionally removes workspace files (sandbox-checked).
No hidden persistent semantic memory is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .checkpoints import CheckpointStore, _validate_quota_binding
from .errors import CheckpointError

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
#: query parameter names whose values are NEVER persisted
_SENSITIVE_PARAM = re.compile(
    r"(key|token|secret|password|passwd|sig|signature|auth|credential|session|cookie)",
    re.IGNORECASE,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

MAX_LOCATOR_CHARS = 2048
MAX_TEXT_META = 1024
MAX_PROVENANCE_BYTES = 4096
MAX_ROWS = 500
MAX_CONTENT_READ_BYTES = 8 << 20  # 8 MiB cap when verifying workspace content


class SourceType(StrEnum):
    WEB_PAGE = "web_page"
    API_RESPONSE = "api_response"
    DOCUMENT = "document"
    OFFLINE_FIXTURE = "offline_fixture"
    OTHER = "other"


class OriginKind(StrEnum):
    TRUSTED_RUNTIME = "trusted_runtime"
    OFFLINE_FIXTURE = "offline_fixture"


class StoragePolicy(StrEnum):
    METADATA_ONLY = "metadata_only"  # default: no persistent content at all
    WORKSPACE = "workspace"          # explicit policy: content lives in the sandbox


class Visibility(StrEnum):
    RUN = "run"      # only the owning run (default)
    TEAM = "team"    # also other runs of the same store (explicit opt-in)


def _bounded_text(value: Any, cap: int) -> str:
    """Bound + strip control characters; never fabricate."""
    s = _CONTROL_CHARS.sub("", str(value))
    return s[:cap]


def sanitize_locator(locator: str) -> str:
    """Canonical locator for storage (W2 residual gate): sensitive query
    parameter VALUES, URL userinfo (user:pass@host) and sensitive FRAGMENTS
    (OAuth-style #access_token=...) are redacted — never persisted; control
    chars stripped; length capped. Paths are bounded but may still contain
    operator-sensitive segments, so display surfaces truncate them."""
    raw = str(locator)
    if not raw:
        return ""
    cleaned = _CONTROL_CHARS.sub("", raw)[: MAX_LOCATOR_CHARS * 2]
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned[:MAX_LOCATOR_CHARS]
    if parts.scheme and parts.netloc:
        host = parts.hostname or ""
        try:
            port = f":{parts.port}" if parts.port else ""
        except ValueError:
            port = ""
        netloc = f"[REDACTED]@{host}{port}" if parts.username or parts.password \
            else f"{host}{port}"
        if parts.query:
            kept = [
                (k, "[REDACTED]" if _SENSITIVE_PARAM.search(k) else v)
                for k, v in parse_qsl(parts.query, keep_blank_values=True)
            ]
            query = urlencode(kept)
        else:
            query = ""
        fragment = "[REDACTED]" if _SENSITIVE_PARAM.search(parts.fragment or "") \
            else parts.fragment
        cleaned = urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
    return cleaned[:MAX_LOCATOR_CHARS]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _TrustedOrigin:
    """Private: proof of a trusted registration context. Only constructible via
    SourceRegistry.trusted_origin (lease-verified). Not a caller-assertable flag."""

    kind: OriginKind
    run_id: str | None
    task_id: str | None
    lease_owner: str | None
    lease_generation: int | None
    agent_id: str


@dataclass(frozen=True)
class ReadBinding:
    """A validated read/access context for source lookups (current lease)."""

    run_id: str
    lease_owner: str
    lease_generation: int


def _validate_binding(run_id: Any, lease_owner: Any, lease_generation: Any) -> None:
    _validate_quota_binding(run_id, lease_owner, lease_generation)


class SourceRegistry:
    """Durable source registry on top of CheckpointStore (schema v13 tables
    `sources` + `source_audit`). All mutations are single atomic transactions
    under the store lock; duplicate source ids (replay) are rejected."""

    def __init__(
        self,
        store: CheckpointStore,
        *,
        sandbox: Any = None,
        allow_fixtures: bool = False,
        policy_version: str = "sources-1.0.0",
    ) -> None:
        self.store = store
        self.sandbox = sandbox
        self.allow_fixtures = bool(allow_fixtures)
        self.policy_version = policy_version[:64]

    # ---- trusted origin minting (lease-fenced) ----
    def trusted_origin(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_generation: int,
        agent_id: str,
        task_id: str | None = None,
    ) -> _TrustedOrigin:
        """Mint a trusted registration origin. Fails closed unless the run is
        RUNNING with a live lease matching owner+generation (worker-owned
        identity, never caller-asserted)."""
        _validate_binding(run_id, lease_owner, lease_generation)
        with self.store._lock:
            if not self.store._verify_lease_locked(run_id, lease_owner, lease_generation):
                raise CheckpointError(
                    "stale worker: lease fencing failed for source registration")
        return _TrustedOrigin(
            kind=OriginKind.TRUSTED_RUNTIME,
            run_id=run_id,
            task_id=task_id,
            lease_owner=lease_owner,
            lease_generation=int(lease_generation),
            agent_id=_bounded_text(agent_id, 100),
        )

    def read_binding(self, *, run_id: str, lease_owner: str,
                     lease_generation: int) -> ReadBinding:
        """Validated read context for agent/run-scoped lookups (current lease)."""
        _validate_binding(run_id, lease_owner, lease_generation)
        with self.store._lock:
            if not self.store._verify_lease_locked(run_id, lease_owner, lease_generation):
                raise CheckpointError("stale worker: lease fencing failed for source read")
        return ReadBinding(run_id=run_id, lease_owner=lease_owner,
                           lease_generation=int(lease_generation))

    # ---- registration ----
    def register(
        self,
        *,
        origin: _TrustedOrigin,
        source_type: SourceType | str,
        provider: str,
        locator: str,
        raw_bytes: bytes | None = None,
        raw_sha256: str | None = None,
        content: bytes | None = None,
        content_sha256: str | None = None,
        extract_method: str = "",
        extract_version: str = "",
        provenance: dict[str, Any] | None = None,
        license: str | None = None,  # None = explicitly unknown
        attribution: str = "",
        visibility: Visibility | str = Visibility.RUN,
        storage: StoragePolicy | str = StoragePolicy.METADATA_ONLY,
        workspace_relpath: str | None = None,
        ttl_seconds: float | None = None,
        source_id: str | None = None,
        now_iso: str | None = None,
    ) -> str:
        """Atomically register one source. Requires a trusted origin (lease-
        fenced). Content hashes are computed HERE from the actual bytes/file —
        caller-supplied hashes are cross-checked, never trusted. Duplicate
        source ids are rejected (replay)."""
        if not isinstance(origin, _TrustedOrigin):
            raise CheckpointError(
                "source registration requires a trusted origin (lease-fenced "
                "trusted_origin()); agents cannot register invented sources")
        st = SourceType(str(source_type))
        vis = Visibility(str(visibility))
        sto = StoragePolicy(str(storage))
        # W2 residual gate: origin kind and source type cannot be relabeled —
        # a fixture origin can only register fixture sources, and a trusted
        # runtime origin can never mint a fixture source (or vice versa)
        if origin.kind is OriginKind.OFFLINE_FIXTURE and st is not SourceType.OFFLINE_FIXTURE:
            raise CheckpointError(
                "fixture origin cannot be relabeled as production source type "
                f"{st.value!r} — rejected")
        if origin.kind is OriginKind.TRUSTED_RUNTIME and st is SourceType.OFFLINE_FIXTURE:
            raise CheckpointError(
                "trusted-runtime origin cannot register an offline_fixture source "
                "— rejected")
        sid = source_id or ("src_" + secrets.token_hex(8))
        if not _SOURCE_ID.match(sid):
            raise CheckpointError(f"invalid source_id {sid!r}")
        clean_locator = sanitize_locator(locator)
        locator_hash = _sha256_hex(str(locator).encode("utf-8")) if locator else None
        if raw_bytes is not None:
            raw_sha256_calc = _sha256_hex(raw_bytes)
            if raw_sha256 is not None and raw_sha256 != raw_sha256_calc:
                raise CheckpointError("raw_sha256 does not match the provided raw_bytes")
            raw_sha256 = raw_sha256_calc
            raw_len: int | None = len(raw_bytes)
        else:
            raw_len = None
            if raw_sha256 is not None and not _SHA256_HEX.match(raw_sha256):
                raise CheckpointError("raw_sha256 must be sha256 hex")
        content_len: int | None = None
        if content is not None:
            content_sha256_calc = _sha256_hex(content)
            if content_sha256 is not None and content_sha256 != content_sha256_calc:
                raise CheckpointError(
                    "content_sha256 does not match the provided content")
            content_sha256 = content_sha256_calc
            content_len = len(content)
        if content_sha256 is not None and not _SHA256_HEX.match(content_sha256):
            raise CheckpointError("content_sha256 must be sha256 hex")

        prov_json = "{}"
        if provenance is not None:
            if not isinstance(provenance, dict) or not all(
                    isinstance(k, str) for k in provenance):
                raise CheckpointError("provenance must be a JSON object with string keys")
            prov_json = json.dumps(provenance, ensure_ascii=True)[:MAX_PROVENANCE_BYTES]

        relpath: str | None = None
        if sto is StoragePolicy.WORKSPACE:
            if self.sandbox is None:
                raise CheckpointError("workspace storage requires a WorkspaceSandbox")
            if not workspace_relpath:
                raise CheckpointError("workspace storage requires workspace_relpath")
            if os.path.isabs(workspace_relpath):
                raise CheckpointError("workspace_relpath must be workspace-RELATIVE")
            # verify the content actually exists in the sandbox and hash it
            try:
                resolved = self.sandbox.resolve(workspace_relpath)  # containment checks
            except Exception as exc:
                raise CheckpointError(
                    f"invalid workspace_relpath {workspace_relpath!r}: {exc}") from exc
            try:
                with open(resolved, "rb") as fh:
                    data = fh.read(MAX_CONTENT_READ_BYTES + 1)
            except OSError as exc:
                raise CheckpointError(
                    f"workspace content unreadable: {exc}") from exc
            if len(data) > MAX_CONTENT_READ_BYTES:
                raise CheckpointError("workspace content exceeds read cap")
            file_hash = _sha256_hex(data)
            if content_sha256 is not None and content_sha256 != file_hash:
                raise CheckpointError(
                    "content hash mismatch: registered file content does not match "
                    "content_sha256 — invented/substituted content rejected")
            content_sha256 = file_hash
            content_len = len(data)
            relpath = _bounded_text(workspace_relpath, MAX_TEXT_META)
        elif workspace_relpath is not None:
            raise CheckpointError("workspace_relpath requires storage='workspace'")

        now = now_iso or self.store.now_iso()
        expires_at = None
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise CheckpointError("ttl_seconds must be > 0")
            from datetime import timedelta

            expires_at = (
                self.store.utcnow() + timedelta(seconds=float(ttl_seconds))
            ).isoformat(timespec="milliseconds")

        row = (
            sid, origin.run_id, origin.task_id, origin.agent_id, origin.kind.value,
            st.value, _bounded_text(provider, MAX_TEXT_META), clean_locator, locator_hash,
            now, raw_sha256, content_sha256,
            _bounded_text(extract_method, 100), _bounded_text(extract_version, 64),
            raw_len, content_len, prov_json,
            _bounded_text(license, MAX_TEXT_META) if license is not None else None,
            _bounded_text(attribution, MAX_TEXT_META), vis.value, sto.value, relpath,
            expires_at, self.policy_version, now,
        )
        try:
            with self.store._lock, self.store._conn:
                # re-verify the lease ATOMICALLY with the insert (stale workers
                # cannot register even with a once-valid origin)
                if origin.run_id is not None:
                    if not self.store._verify_lease_locked(
                            origin.run_id, origin.lease_owner, origin.lease_generation):
                        raise CheckpointError(
                            "stale worker: lease fencing failed inside the "
                            "registration transaction")
                self.store._conn.execute(
                    "INSERT INTO sources (source_id, run_id, task_id, agent_id, "
                    "origin_kind, source_type, provider, locator, locator_sha256, "
                    "retrieved_at, raw_sha256, content_sha256, extract_method, "
                    "extract_version, raw_bytes, content_bytes, provenance_json, "
                    "license, attribution, visibility, storage, workspace_relpath, "
                    "expires_at, policy_version, created_at) VALUES ("
                    "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
                self.store._conn.execute(
                    "INSERT INTO source_audit (source_id, event, actor, detail, at) "
                    "VALUES (?, 'registered', ?, ?, ?)",
                    (sid, origin.agent_id,
                     json.dumps({"origin_kind": origin.kind.value,
                                 "source_type": st.value})[:512], now),
                )
        except sqlite3.IntegrityError as exc:
            raise CheckpointError(
                f"source {sid!r} already exists — replay rejected") from exc
        return sid

    def register_fixture(
        self,
        *,
        locator: str,
        content: bytes | None = None,
        provider: str = "offline-fixture",
        source_id: str | None = None,
        agent_id: str = "test-fixture",
        now_iso: str | None = None,
        **kw: Any,
    ) -> str:
        """Deterministic OFFLINE fixture registration (tests/demos only). Requires
        allow_fixtures=True at construction; recorded with
        origin_kind='offline_fixture' so it is always distinguishable from
        production retrieval provenance. Fixtures never authorize anything."""
        if not self.allow_fixtures:
            raise CheckpointError(
                "offline fixtures are disabled (allow_fixtures=False); fixtures "
                "can never represent production retrieval")
        origin = _TrustedOrigin(
            kind=OriginKind.OFFLINE_FIXTURE, run_id=None, task_id=None,
            lease_owner=None, lease_generation=None, agent_id=agent_id)
        return self.register(
            origin=origin, source_type=SourceType.OFFLINE_FIXTURE, provider=provider,
            locator=locator, content=content, source_id=source_id, now_iso=now_iso,
            **kw)

    # ---- authorized reads ----
    def _check_access(self, row: dict[str, Any], binding: ReadBinding | None) -> None:
        if binding is None:
            return  # operator/CLI scope (trusted local operator)
        if row["run_id"] is not None and row["run_id"] != binding.run_id \
                and row["visibility"] != Visibility.TEAM.value:
            raise CheckpointError(
                f"cross-run access to source {row['source_id']!r} denied "
                "(visibility='run')")
        if row["run_id"] is None and row["visibility"] != Visibility.TEAM.value:
            raise CheckpointError(
                f"run-scoped access to unbound source {row['source_id']!r} denied")

    def _row_to_dict(self, r: Any) -> dict[str, Any]:
        d = {k: r[k] for k in r.keys()}
        d["expired"] = bool(d["expires_at"] and d["expires_at"] <= self.store.now_iso())
        d["license"] = d["license"] if d["license"] else "unknown"
        return d

    def get(self, source_id: str, *, binding: ReadBinding | None = None) -> dict[str, Any] | None:
        with self.store._lock:
            r = self.store._conn.execute(
                "SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
        if r is None:
            return None
        d = self._row_to_dict(r)
        self._check_access(d, binding)
        return d

    def list(self, *, binding: ReadBinding | None = None, run_id: str | None = None,
             limit: int = 200) -> list[dict[str, Any]]:
        with self.store._lock:
            if binding is not None and run_id is None:
                rows = self.store._conn.execute(
                    "SELECT * FROM sources WHERE run_id=? OR visibility='team' "
                    "ORDER BY created_at DESC, source_id DESC LIMIT ?",
                    (binding.run_id, min(limit, MAX_ROWS))).fetchall()
            elif run_id is not None:
                rows = self.store._conn.execute(
                    "SELECT * FROM sources WHERE run_id=? "
                    "ORDER BY created_at DESC, source_id DESC LIMIT ?",
                    (run_id, min(limit, MAX_ROWS))).fetchall()
            else:
                rows = self.store._conn.execute(
                    "SELECT * FROM sources ORDER BY created_at DESC, source_id DESC "
                    "LIMIT ?", (min(limit, MAX_ROWS),)).fetchall()
        out = [self._row_to_dict(r) for r in rows]
        for d in out:
            self._check_access(d, binding)
        return out

    def find_by_content_sha256(self, content_sha256: str, *,
                               binding: ReadBinding | None = None) -> list[dict[str, Any]]:
        """Content dedup DETECTION: identical content keeps DISTINCT source ids
        and provenance — dedup never merges authorization boundaries."""
        if not _SHA256_HEX.match(content_sha256 or ""):
            raise CheckpointError("content_sha256 must be sha256 hex")
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT * FROM sources WHERE content_sha256=? "
                "ORDER BY created_at DESC, source_id DESC LIMIT ?",
                (content_sha256, MAX_ROWS)).fetchall()
        out = [self._row_to_dict(r) for r in rows]
        for d in out:
            self._check_access(d, binding)
        return out

    def excerpt(self, source_id: str, *, binding: ReadBinding | None = None,
                max_chars: int = 2000, max_bytes: int = MAX_CONTENT_READ_BYTES) -> dict[str, Any]:
        """Bounded, integrity-verified excerpt. Content that is not stored
        (metadata_only) or no longer available returns available=False with
        metadata — missing excerpts are NEVER reconstructed or invented."""
        if max_chars < 0 or max_chars > 100_000:
            raise CheckpointError("max_chars out of range")
        d = self.get(source_id, binding=binding)
        if d is None:
            raise CheckpointError(f"unknown source {source_id!r}")
        if d["expired"]:
            raise CheckpointError(f"source {source_id!r} expired")
        base = {"source_id": d["source_id"], "content_sha256": d["content_sha256"],
                "content_bytes": d["content_bytes"], "available": False,
                "integrity": "not_stored", "excerpt": None}
        if d["storage"] != StoragePolicy.WORKSPACE.value or not d["workspace_relpath"]:
            return base
        if self.sandbox is None:
            base["integrity"] = "no_sandbox"
            return base
        try:
            resolved = self.sandbox.resolve(d["workspace_relpath"])
        except Exception as exc:
            raise CheckpointError(f"workspace content unavailable: {exc}") from exc
        try:
            with open(resolved, "rb") as fh:
                data = fh.read(min(max_bytes, MAX_CONTENT_READ_BYTES) + 1)
        except OSError:
            base["integrity"] = "unavailable"
            return base
        if len(data) > min(max_bytes, MAX_CONTENT_READ_BYTES):
            raise CheckpointError("content exceeds read cap")
        actual = _sha256_hex(data)
        if d["content_sha256"] and actual != d["content_sha256"]:
            raise CheckpointError(
                f"content hash mismatch for source {source_id!r} — substitution detected")
        text = data.decode("utf-8", "replace")
        return {**base, "available": True, "integrity": "verified",
                "excerpt": text[:max_chars]}

    def cite(self, source_id: str, *, binding: ReadBinding | None = None,
             max_chars: int = 0) -> str:
        """Citation string for a registered, authorized, unexpired source.
        Establishes PROVENANCE only — not factual correctness."""
        d = self.get(source_id, binding=binding)
        if d is None:
            raise CheckpointError(f"unknown citation id {source_id!r}")
        if d["expired"]:
            raise CheckpointError(f"citation {source_id!r} expired")
        sha = (d["content_sha256"] or d["raw_sha256"] or "")[:16]
        display_locator = (d["locator"] or "unknown")[:256]  # bounded display metadata
        parts = [
            f"[{d['source_id']}]",
            f"type={d['source_type']} provider={d['provider'] or 'unknown'}",
            f"locator={display_locator}",
            f"retrieved_at={d['retrieved_at']}",
            f"sha256:{sha or 'unknown'}",
            f"license:{d['license']}",
        ]
        if d["attribution"]:
            parts.append(f"attribution={d['attribution']}")
        if d["origin_kind"] == OriginKind.OFFLINE_FIXTURE.value:
            parts.append("origin=offline_fixture(not production retrieval)")
        parts.append("(provenance only — not factual verification)")
        text = " ".join(parts)
        if max_chars > 0:
            ex = self.excerpt(source_id, binding=binding, max_chars=max_chars)
            if ex["available"]:
                text += f"\nexcerpt(integrity={ex['integrity']}): {ex['excerpt']}"
            else:
                text += f"\nexcerpt unavailable (integrity={ex['integrity']})"
        return text

    # ---- retention ----
    def sweep_expired(self, now_iso: str | None = None, *,
                      delete_content: bool = False) -> int:
        """Delete expired source records (audited). With delete_content=True,
        workspace-stored content files are removed too (sandbox-checked)."""
        now = now_iso or self.store.now_iso()
        with self.store._lock, self.store._conn:
            rows = self.store._conn.execute(
                "SELECT source_id, storage, workspace_relpath FROM sources "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,)).fetchall()
            n = 0
            for r in rows:
                if delete_content and r["storage"] == StoragePolicy.WORKSPACE.value \
                        and r["workspace_relpath"] and self.sandbox is not None:
                    try:
                        resolved = self.sandbox.resolve(r["workspace_relpath"])
                        if os.path.isfile(resolved):
                            os.remove(resolved)
                    except OSError:
                        pass  # content already gone; metadata still deleted
                self.store._conn.execute(
                    "INSERT INTO source_audit (source_id, event, actor, detail, at) "
                    "VALUES (?, 'expired', '', ?, ?)", (r["source_id"], now, now))
                self.store._conn.execute(
                    "DELETE FROM sources WHERE source_id=?", (r["source_id"],))
                n += 1
        return n
