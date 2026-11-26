"""A5 tests: outbound authorization, durable v9 API accounting, migrations.

Offline only (loopback servers, controlled fixtures, subprocess crash tests).
"""

import json
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import (
    ApiConfig,
    ApiEndpointConfig,
    ApiProviderConfig,
    load_hive_config,
)
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode, QuotaReservationOutcome
from geosgan_hive.errors import CheckpointError, QuotaExhaustedError
from geosgan_hive.http_client import EndpointRule, HttpLimits, SecureHttpClient, TransportConfig
from geosgan_hive.outbound import (
    OutboundAuthorizer,
    OutboundNotAuthorizedError,
    OutboundPolicy,
    build_api_session,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
DIGEST = "a" * 64


# ---- config schema ----

def _api_provider(**overrides):
    base = {
        "provider": "wikimedia",
        "billing": "public_api_no_charge",
        "outbound_policy": "authorized",
        "allowed_operations": ["search"],
        "endpoints": [{
            "endpoint_id": "wikimedia-core", "scheme": "https",
            "host": "api.wikimedia.org", "path_prefixes": ["/core/v1/wikipedia/en"],
            "methods": ["GET"],
        }],
    }
    base.update(overrides)
    return base


def _apis(**overrides):
    base = {"enabled": ["wikimedia"], "providers": [_api_provider()]}
    base.update(overrides)
    return ApiConfig.model_validate(base)


def test_default_all_disabled():
    assert build_api_session(load_hive_config(REPO_ROOT / "config" / "agents.local.yaml",
                                              environ={}), CheckpointStore(":memory:")) is None
    auth = OutboundAuthorizer(CheckpointStore(":memory:"), _apis())
    d = auth.check(provider="wikimedia", endpoint_id="wikimedia-core",
                   operation_id="search", canonical_args={}, agent="a",
                   run_id=None, task_id=None)
    assert d.ok  # authorized by scoped policy
    auth2 = OutboundAuthorizer(CheckpointStore(":memory:"),
                               _apis(enabled=[]))  # nothing enabled
    d2 = auth2.check(provider="wikimedia", endpoint_id="wikimedia-core",
                     operation_id="search", canonical_args={}, agent="a",
                     run_id=None, task_id=None)
    assert not d2.ok and d2.policy == OutboundPolicy.DISABLED


def test_unknown_billing_exposure_rejected():
    with pytest.raises(ValidationError):
        ApiProviderConfig.model_validate(_api_provider(billing="free_tier_unverified"))
    with pytest.raises(ValidationError):
        ApiProviderConfig.model_validate(_api_provider(billing="paid"))


def test_openmeteo_requires_noncommercial_attestation():
    with pytest.raises(ValidationError, match="NON-COMMERCIAL|attestation"):
        ApiProviderConfig.model_validate(_api_provider(
            provider="openmeteo",
            endpoints=[{"endpoint_id": "om", "scheme": "https",
                        "host": "api.open-meteo.com", "path_prefixes": ["/v1"],
                        "methods": ["GET"]}]))
    ok = ApiProviderConfig.model_validate(_api_provider(
        provider="openmeteo", non_commercial_attestation=True,
        endpoints=[{"endpoint_id": "om", "scheme": "https",
                    "host": "api.open-meteo.com", "path_prefixes": ["/v1"],
                    "methods": ["GET"]}]))
    assert ok.non_commercial_attestation is True


def test_production_api_config_rejects_http_and_loopback():
    with pytest.raises(ValidationError):
        ApiEndpointConfig.model_validate({"endpoint_id": "x", "scheme": "http",
                                          "host": "api.example.com",
                                          "path_prefixes": ["/"], "methods": ["GET"]})
    with pytest.raises(ValidationError):
        ApiEndpointConfig.model_validate({"endpoint_id": "x", "scheme": "https",
                                          "host": "127.0.0.1",
                                          "path_prefixes": ["/"], "methods": ["GET"]})
    with pytest.raises(ValidationError):
        _apis(providers=[_api_provider(outbound_policy="authorized",
                                       allowed_operations=[])])


def test_catalog_never_authorizes_api_access(tmp_path):
    # even with an ACTIVE catalog entry, the outbound authorizer only consults config
    from geosgan_hive import load_catalog
    cat = load_catalog(REPO_ROOT / "config" / "catalog.yaml")
    assert cat.provider("wikimedia").operational_status.value == "evaluation_candidate"
    store = CheckpointStore(str(tmp_path / "cp.db"))
    auth = OutboundAuthorizer(store, _apis(enabled=[]))
    d = auth.check(provider="wikimedia", endpoint_id="wikimedia-core",
                   operation_id="search", canonical_args={}, agent="a",
                   run_id=None, task_id=None)
    assert not d.ok  # catalog granted nothing
    store.close()


# ---- durable reservations / budgets / fencing / migration ----

def _store(tmp_path, **kw):
    return CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: NOW, **kw)


def _budgets():
    return {"run:r1": {"requests": 2, "wire_bytes": 10_000},
            "provider:wikimedia": {"requests": 100, "wire_bytes": 10**9},
            "endpoint:wikimedia-core": {"requests": 100, "wire_bytes": 10**9}}


def test_reserve_replay_substitution_single_winner(tmp_path):
    store = _store(tmp_path)
    kw = dict(provider="wikimedia", endpoint_id="wikimedia-core",
              operation_id="search", request_sha256=DIGEST, max_wire_bytes=1000,
              budgets=_budgets(), window_start_iso="2026-11-26")
    assert store.reserve_api_request("res1", **kw) == QuotaReservationOutcome.CREATED
    assert store.reserve_api_request("res1", **kw) == QuotaReservationOutcome.REPLAYED
    bad = dict(kw, request_sha256="b" * 64)
    with pytest.raises(CheckpointError, match="conflicting replay|substitution"):
        store.reserve_api_request("res1", **bad)
    assert store.mark_api_dispatched("res1", request_sha256=DIGEST) is True
    assert store.mark_api_dispatched("res1", request_sha256=DIGEST) is False  # single winner
    assert store.mark_api_dispatched("res1", request_sha256="c" * 64) is False
    assert store.mark_api_dispatched("missing") is False
    usage = store.api_usage_status("provider:wikimedia")
    assert {r["kind"]: r["used"] for r in usage} == {"requests": 1, "wire_bytes": 1000}
    store.close()


def _run_gen(store, run_id):
    with store._lock:
        return int(store._conn.execute(
            "SELECT lease_generation FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])


def test_budget_exhaustion_and_concurrent_exact(tmp_path):
    store = _store(tmp_path)
    store.create_run(run_id="r1", owner="w1", task="t", agent="a", model="m")
    gen = _run_gen(store, "r1")  # create_run: RUNNING, lease_owner=w1, gen 0
    kw = dict(provider="wikimedia", endpoint_id="wikimedia-core",
              operation_id="search", request_sha256=DIGEST, max_wire_bytes=100,
              budgets=_budgets(), window_start_iso="2026-11-26", run_id="r1",
              lease_owner="w1", lease_generation=gen)
    assert store.reserve_api_request("a", **kw) == QuotaReservationOutcome.CREATED
    assert store.reserve_api_request("b", **kw) == QuotaReservationOutcome.CREATED
    with pytest.raises(QuotaExhaustedError, match="run:r1"):
        store.reserve_api_request("c", **kw)  # per-run requests=2 exhausted
    errs = []

    def one(i):
        try:
            store.reserve_api_request(f"th{i}", **{**kw, "run_id": None,
                                                    "lease_owner": None,
                                                    "lease_generation": None})
        except QuotaExhaustedError:
            errs.append(i)
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    # provider budget is 100 requests; hammer it from threads, all must fit exactly
    ts = [threading.Thread(target=one, args=(i,)) for i in range(98)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not [e for e in errs if not isinstance(e, int)]  # only exhaustion allowed
    used = {r["kind"]: r["used"] for r in store.api_usage_status("provider:wikimedia")}
    assert used["requests"] == 100  # exactly the budget, never over
    store.close()


def test_stale_worker_fenced(tmp_path):
    store = _store(tmp_path)
    store.create_run(run_id="r1", owner="w1", task="t", agent="a", model="m")
    gen = _run_gen(store, "r1")
    kw = dict(provider="p", endpoint_id="e", operation_id="op",
              request_sha256=DIGEST, max_wire_bytes=10, budgets={},
              window_start_iso="2026-11-26", run_id="r1", lease_owner="w1",
              lease_generation=gen)
    assert store.reserve_api_request("res1", **kw) == QuotaReservationOutcome.CREATED
    # stale worker with a wrong (but well-formed) generation / wrong owner
    with pytest.raises(CheckpointError, match="stale worker"):
        store.reserve_api_request("res2", **{**kw, "lease_generation": gen + 7})
    with pytest.raises(CheckpointError, match="stale worker"):
        store.reserve_api_request("res3", **{**kw, "lease_owner": "w9"})
    # another worker took over the lease (owner+generation bumped in runs)
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE runs SET lease_owner='w2', lease_generation=lease_generation+1 "
            "WHERE run_id='r1'")
    with pytest.raises(CheckpointError, match="stale worker"):
        store.reserve_api_request("res4", **kw)
    assert store.mark_api_dispatched("res1", run_id="r1", lease_owner="w1",
                                     lease_generation=gen,
                                     request_sha256=DIGEST) is False  # fenced
    # the reservation stays bound to the stale worker: nobody else can dispatch it
    # (conservative: no refund, no substitution). The new owner makes its OWN
    # reservation with its own binding and dispatches that exactly once.
    kw2 = {**kw, "lease_owner": "w2", "lease_generation": gen + 1}
    assert store.reserve_api_request("res5", **kw2) == QuotaReservationOutcome.CREATED
    assert store.mark_api_dispatched("res5", run_id="r1", lease_owner="w2",
                                     lease_generation=gen + 1,
                                     request_sha256=DIGEST) is True
    assert store.mark_api_dispatched("res5", run_id="r1", lease_owner="w2",
                                     lease_generation=gen + 1,
                                     request_sha256=DIGEST) is False  # single winner
    store.close()


def test_crash_recovery_keeps_reservation_and_usage(tmp_path):
    db = str(tmp_path / "cp.db")
    store = _store(tmp_path)
    store.reserve_api_request("res-crash", provider="p", endpoint_id="e",
                              operation_id="op", request_sha256=DIGEST,
                              max_wire_bytes=500, budgets={}, window_start_iso="2026-11-26")
    store.close()
    # simulate crash after reservation, before dispatch (subprocess hard-exit)
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from datetime import UTC, datetime;"
        "from geosgan_hive.checkpoints import CheckpointStore;"
        f"s = CheckpointStore({db!r}, utcnow=lambda: datetime(2026,11,26,12,tzinfo=UTC));"
        "s.reserve_api_request('res-x', provider='p', endpoint_id='e', operation_id='op',"
        " request_sha256='" + "a" * 64 + "', max_wire_bytes=100, budgets={},"
        " window_start_iso='2026-11-26');"
        "import os; os._exit(9)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 9
    store2 = _store(tmp_path)
    rows = store2.api_usage_status("provider:p")
    assert {r["kind"]: r["used"] for r in rows} == {"requests": 2, "wire_bytes": 600}
    # conservative: crash usage is retained, never refunded
    assert store2.mark_api_dispatched("res-crash", request_sha256=DIGEST) is True
    store2.close()


def test_v8_to_v9_migration_preserves_data_and_rolls_back(tmp_path):
    db = str(tmp_path / "v8.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '8')")
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
    store = CheckpointStore(db, utcnow=lambda: NOW)  # opens and migrates v8 -> v9
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert v == "9"
    assert {"api_reservations", "api_usage", "api_audit"} <= tables
    assert store.get_reservation("d-old")["dispatch_id"] == "d-old"  # v8 data preserved
    store.close()
    # interrupted migration rolls back fully and recovers on next open
    db2 = str(tmp_path / "v8b.db")
    conn = sqlite3.connect(db2)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '8')")
    conn.commit()
    conn.close()
    import geosgan_hive.checkpoints as ckpt
    real = ckpt.CheckpointStore._migrate_8_to_9

    def boom(self):
        _orig_exec = self._conn.execute
        _orig_exec("CREATE TABLE IF NOT EXISTS api_reservations (x TEXT)")  # partial write
        raise RuntimeError("interrupted")

    ckpt.CheckpointStore._migrate_8_to_9 = boom
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            CheckpointStore(db2, utcnow=lambda: NOW)
    finally:
        ckpt.CheckpointStore._migrate_8_to_9 = real
    conn = sqlite3.connect(db2)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert v == "8" and "api_reservations" not in tables  # rolled back
    store = CheckpointStore(db2, utcnow=lambda: NOW)  # recovers
    with store._lock:
        v = store._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "9"
    store.close()


# ---- outbound authorization flow ----

class OkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "ok"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).seen.append(self.path)
        if type(self).mode == "ratelimit_once" and not type(self).seen[:-1]:
            self.send_response(429)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Length", "0")
            self.end_headers()
            type(self).mode = "ok"
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def ok_server():
    OkHandler.mode = "ok"
    OkHandler.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), OkHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()


# Production config validation rejects http/loopback endpoints, and production
# build_api_session derives the transport EndpointRules from that same validated
# config (so config and transport can never diverge in production). For offline
# tests the loopback server is injected ONLY at the test-only transport layer,
# keyed by the same endpoint_id; the ApiConfig keeps a valid https endpoint.
_TEST_EP_CONFIG = {"endpoint_id": "test-ep", "scheme": "https",
                   "host": "api.wikimedia.org", "path_prefixes": ["/api"],
                   "methods": ["GET"]}


def _session(tmp_path, port, *, policy="authorized", approvals=None, persist=None):
    store = CheckpointStore(str(tmp_path / "cp.db"),
                            persist_mode=persist or PersistMode.RESUMABLE,
                            utcnow=lambda: NOW)
    apis = _apis(providers=[_api_provider(
        outbound_policy=policy,
        allowed_operations=["search"] if policy == "authorized" else [],
        endpoints=[_TEST_EP_CONFIG])],
        # small per-request wire cap + ample byte budgets so a 429 retry can
        # reserve a SECOND time (each retry consumes a new reservation)
        max_wire_bytes_per_request=4096,
        budgets={"per_run_bytes": 10**9, "per_provider_bytes": 10**9,
                 "per_endpoint_bytes": 10**9})
    rule = EndpointRule(endpoint_id="test-ep", scheme="http", host="127.0.0.1",
                        port=port, path_prefixes=("/api",), methods=("GET",),
                        headers={"User-Agent": "hive-test/1.0"})
    client = SecureHttpClient(TransportConfig(endpoints=(rule,),
                                              limits=HttpLimits(max_retries=2),
                                              allow_loopback=True))
    auth = OutboundAuthorizer(store, apis, approvals)
    from geosgan_hive.outbound import ApiSession
    return ApiSession(client, auth, store), store, apis


def test_authorized_end_to_end_with_retry_and_audit(tmp_path, ok_server):
    OkHandler.mode = "ratelimit_once"
    session, store, apis = _session(tmp_path, ok_server)
    r = session.request(provider="wikimedia", endpoint_id="test-ep", operation_id="search",
                        method="GET", path="/api/search", params={"q": "hive"},
                        agent="researcher", run_id=None)
    assert r.status == 200
    assert OkHandler.seen[0].startswith("/api/search?q=hive")
    usage = store.api_usage_status("provider:wikimedia")
    assert {x["kind"]: x["used"] for x in usage}["requests"] == 2  # 429 retry = new reservation
    audit = store.api_audit_list(provider="wikimedia")
    assert audit and audit[0]["outcome"] == "ok" and audit[0]["status_code"] == 200
    assert "q=hive" not in json.dumps(audit)  # no private query payload in audit
    store.close()


def test_unauthorized_operation_and_unknown_provider(tmp_path, ok_server):
    session, store, _ = _session(tmp_path, ok_server)
    with pytest.raises(OutboundNotAuthorizedError):
        session.request(provider="wikimedia", endpoint_id="test-ep",
                        operation_id="delete_everything", method="GET", path="/api/x")
    with pytest.raises(OutboundNotAuthorizedError):
        session.request(provider="evil", endpoint_id="test-ep", operation_id="search",
                        method="GET", path="/api/x")
    store.close()


def test_approval_flow_preview_consume_substitution_expiry(tmp_path, ok_server):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: NOW)
    mgr = ApprovalManager(store)
    apis = _apis(providers=[_api_provider(outbound_policy="approval_required",
                                           endpoints=[_TEST_EP_CONFIG])])
    auth = OutboundAuthorizer(store, apis, mgr)
    args = {"method": "GET", "path": "/api/search",
            "params": {"q": "quarterly revenue"}, "body_sha256": "00"}
    # not approved yet -> denied
    d = auth.check(provider="wikimedia", endpoint_id="test-ep", operation_id="search",
                   canonical_args=args, agent="researcher", run_id=None, task_id=None)
    assert not d.ok and d.policy == OutboundPolicy.APPROVAL_REQUIRED
    # create approval: preview shows the ACTUAL outbound data (bounded)
    req = auth.request_approval(provider="wikimedia", endpoint_id="test-ep",
                                operation_id="search", canonical_args=args,
                                agent="researcher", run_id=None, task_id=None)
    row = store.list_approvals()[0]
    assert row["approval_id"] == req.approval_id and row["status"] == "pending"
    assert "quarterly revenue" in row["preview"]  # meaningful review of outbound data
    # substitution: different args do not match
    other = dict(args, params={"q": "other"})
    d2 = auth.check(provider="wikimedia", endpoint_id="test-ep", operation_id="search",
                    canonical_args=other, agent="researcher", run_id=None, task_id=None)
    assert not d2.ok
    # operator approves -> check passes; expiry enforced
    assert store.decide_approval(req.approval_id, decision="approved", operator="op1")
    d3 = auth.check(provider="wikimedia", endpoint_id="test-ep", operation_id="search",
                    canonical_args=args, agent="researcher", run_id=None, task_id=None)
    assert d3.ok and d3.approval_id == req.approval_id
    # consume is single-use and hash-bound
    auth.consume_approval(d3, provider="wikimedia", endpoint_id="test-ep",
                          operation_id="search",
                          canonical_args=args, agent="researcher", run_id=None,
                          task_id=None)
    with pytest.raises(OutboundNotAuthorizedError):
        auth.consume_approval(d3, provider="wikimedia", endpoint_id="test-ep",
                              operation_id="search",
                              canonical_args=args, agent="researcher", run_id=None,
                              task_id=None)  # replay
    with pytest.raises(OutboundNotAuthorizedError):
        auth.consume_approval(d3, provider="wikimedia", endpoint_id="test-ep",
                              operation_id="search",
                              canonical_args=other, agent="researcher", run_id=None,
                              task_id=None)  # substitution
    with pytest.raises(OutboundNotAuthorizedError):
        auth.consume_approval(d3, provider="wikimedia", endpoint_id="other-ep",
                              operation_id="search",
                              canonical_args=args, agent="researcher", run_id=None,
                              task_id=None)  # destination substitution
    # expired approval does not authorize
    req2 = auth.request_approval(provider="wikimedia", endpoint_id="test-ep",
                                 operation_id="search", canonical_args=args,
                                 agent="researcher", run_id=None, task_id=None)
    store.decide_approval(req2.approval_id, decision="approved", operator="op1")
    store._conn.execute("UPDATE approvals SET expires_at='2000-01-01T00:00:00.000+00:00' "
                        "WHERE approval_id=?", (req2.approval_id,))
    store._conn.commit()
    d4 = auth.check(provider="wikimedia", endpoint_id="test-ep", operation_id="search",
                    canonical_args=args, agent="researcher", run_id=None, task_id=None)
    assert not d4.ok
    store.close()


def test_metadata_only_stores_no_payload(tmp_path, ok_server):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.METADATA_ONLY,
                            utcnow=lambda: NOW)
    mgr = ApprovalManager(store)
    apis = _apis(providers=[_api_provider(outbound_policy="approval_required",
                                           endpoints=[_TEST_EP_CONFIG])])
    auth = OutboundAuthorizer(store, apis, mgr)
    args = {"method": "GET", "path": "/api/search", "params": {"q": "secret-project"},
            "body_sha256": "00"}
    auth.request_approval(provider="wikimedia", endpoint_id="test-ep",
                          operation_id="search", canonical_args=args,
                          agent="researcher", run_id=None, task_id=None)
    row = store.list_approvals()[0]
    assert row["args_json"] == ""  # metadata_only: hash only, no payload persisted
    assert row["args_sha256"]  # hash authoritative
    store.close()
