"""Offline demo of the Phase 5 (A5) SECURE PUBLIC API TRANSPORT (no API keys, no network).

Everything runs against a loopback HTTP server inside this process. The
production config still rejects http/loopback endpoints; the loopback rule
below exists ONLY in this test-only transport, exactly like the offline tests.

Shown:
1. External APIs are DISABLED by default (fail closed).
2. APPROVAL_REQUIRED: a request is denied until an operator approves, and the
   approval preview shows the ACTUAL outbound request (bounded, redacted).
3. An authorized request performs a durable atomic reservation BEFORE the
   physical HTTP call, revalidates, dispatches once, and writes bounded audit
   metadata (no query payload is persisted).
4. Budgets are enforced durably: a second request beyond the per-endpoint
   budget is rejected with QuotaExhaustedError and consumes nothing.
5. A 429 Retry-After is honored safely and consumes a NEW reservation.

Run: PYTHONPATH=src python examples/demo_secure_api.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from geosgan_hive.agent_config import ApiConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.errors import QuotaExhaustedError
from geosgan_hive.http_client import EndpointRule, HttpLimits, SecureHttpClient, TransportConfig
from geosgan_hive.outbound import (
    OutboundAuthorizer,
    OutboundNotAuthorizedError,
    build_api_session,
    build_offline_test_session,
)


class Handler(BaseHTTPRequestHandler):
    hits = []
    rate_limit_first = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        if type(self).rate_limit_first and len(type(self).hits) == 1:
            type(self).rate_limit_first = False
            self.send_response(429)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"results": [{"title": "HIVE", "source": "demo"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    # 1. default config: every external API is DISABLED
    store0 = CheckpointStore(":memory:")
    assert build_api_session(type("C", (), {"apis": None})(), store0) is None
    print("1. default config: all external APIs DISABLED (session is None)")

    # loopback "public API" for this offline demo
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 2. provider policy: approval_required (config carries a valid https endpoint;
    #    the loopback server is injected only at the test-only transport layer)
    apis = ApiConfig.model_validate({
        "enabled": ["wikimedia"],
        "max_wire_bytes_per_request": 4096,
        "budgets": {"per_endpoint_requests": 2, "per_endpoint_bytes": 10**9,
                    "per_provider_requests": 100, "per_provider_bytes": 10**9,
                    "per_run_requests": 50, "per_run_bytes": 10**9},
        "providers": [{
            "provider": "wikimedia", "billing": "public_api_no_charge",
            "outbound_policy": "approval_required", "allowed_operations": ["search"],
            "endpoints": [{"endpoint_id": "wiki", "scheme": "https",
                           "host": "api.wikimedia.org", "path_prefixes": ["/w/api"],
                           "methods": ["GET"]}],
        }],
    })
    store = CheckpointStore(":memory:", persist_mode=PersistMode.RESUMABLE)
    mgr = ApprovalManager(store)
    auth = OutboundAuthorizer(store, apis, mgr)
    rule = EndpointRule(endpoint_id="wiki", scheme="http", host="127.0.0.1", port=port,
                        path_prefixes=("/w/api",), methods=("GET",),
                        headers={"User-Agent": "GEOSGAN-HIVE-demo/0.7.0"})
    client = SecureHttpClient(TransportConfig(endpoints=(rule,),
                                              limits=HttpLimits(max_retries=1),
                                              allow_loopback=True))  # TEST-ONLY flag
    # TEST-ONLY factory: offline demo — the loopback rule replaces the https rule
    # of the same endpoint id; the public constructor/build_api_session are strict
    session = build_offline_test_session(client, auth, store)

    call = dict(provider="wikimedia", endpoint_id="wiki", operation_id="search",
                method="GET", path="/w/api/search", params={"q": "geospatial hive"},
                agent="researcher")

    # 3. denied before approval
    try:
        session.request(**call)
        raise SystemExit("expected denial")
    except OutboundNotAuthorizedError as exc:
        print(f"2. unapproved request denied: {exc}")

    # approval preview shows the ACTUAL outbound request (same canonical args
    # the session will compute: body_sha256 of the empty body)
    import hashlib
    req = auth.request_approval(provider="wikimedia", endpoint_id="wiki",
                                operation_id="search",
                                canonical_args={"method": "GET", "path": "/w/api/search",
                                                "params": {"q": "geospatial hive"},
                                                "body_sha256": hashlib.sha256(b"").hexdigest()},
                                agent="researcher", run_id=None, task_id=None)
    preview = store.list_approvals()[0]["preview"]
    assert "geospatial hive" in preview
    print("3. approval preview shows the real outbound data:")
    print("   " + preview.replace("\n", "\n   ")[:400])
    store.decide_approval(req.approval_id, decision="approved", operator="operator-1")

    # 4. authorized end-to-end: first attempt gets a 429, the retry consumes a
    #    NEW durable reservation (per-endpoint request budget is 2 -> exactly fits)
    r = session.request(**call)
    assert r.status == 200, r.status
    usage = {x["kind"]: x["used"] for x in store.api_usage_status("provider:wikimedia")}
    print(f"4. request succeeded after one 429 retry; durable usage: {usage}")
    audit = store.api_audit_list(provider="wikimedia")
    blob = json.dumps(audit)
    assert "geospatial hive" not in blob, "private query payload must not be audited"
    print(f"   audit rows: {len(audit)}, outcomes={[a['outcome'] for a in audit]}, "
          "no query payload persisted")

    # 5. budget exhausted: per-endpoint requests=2 already used by the retry.
    #    A NEW logical call needs a NEW approval (single-use), then the durable
    #    per-endpoint budget denies the reservation before any HTTP request.
    import hashlib
    args2 = {"method": "GET", "path": "/w/api/search", "params": {"q": "second"},
             "body_sha256": hashlib.sha256(b"").hexdigest()}
    req2 = auth.request_approval(provider="wikimedia", endpoint_id="wiki",
                                 operation_id="search", canonical_args=args2,
                                 agent="researcher", run_id=None, task_id=None)
    store.decide_approval(req2.approval_id, decision="approved", operator="operator-1")
    try:
        session.request(**{**call, "params": {"q": "second"}})
        raise SystemExit("expected budget exhaustion")
    except QuotaExhaustedError as exc:
        print(f"5. durable budget enforcement: {str(exc)[:100]}")
    assert len(Handler.hits) == 2  # no third physical request was made

    srv.shutdown()
    store0.close()
    store.close()
    print("\nDemo finished. Loopback only; no API keys, no external network, no cost.")


if __name__ == "__main__":
    main()
