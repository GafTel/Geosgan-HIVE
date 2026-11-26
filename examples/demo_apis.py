"""Offline demo of the Phase 5 (A6) SPECIALIZED FREE PUBLIC API ADAPTERS.

No API keys, no live external requests: both providers are faked by loopback
HTTP servers in this process (production config would use the real https
hosts; the loopback rules exist only in this test-only transport, exactly
like the offline tests).

Shown:
1. All external API tools are DISABLED by default (nothing registered).
2. wikimedia_search returns normalized, attributed results (Spanish edition).
3. crossref_search_works returns bibliographic metadata with polite pool.
4. An APPROVAL_REQUIRED provider is denied until an operator approves, and the
   consumed approval cannot be replayed.
5. Every call consumed a durable reservation (audit shows bounded metadata).

Run: PYTHONPATH=src python examples/demo_apis.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from geosgan_hive.agent_config import ApiConfig
from geosgan_hive.api_adapters import register_api_tools
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.http_client import EndpointRule, HttpLimits, SecureHttpClient, TransportConfig
from geosgan_hive.outbound import OutboundAuthorizer, build_offline_test_session
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog


class FakeProvider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "wiki"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if FakeProvider.mode == "wiki" and "/search/page" in self.path:
            body = json.dumps({"pages": [
                {"id": 1, "key": "Enjambre", "title": "Enjambre",
                 "excerpt": "<p>Colonia de abejas.</p>"}]}).encode()
        elif FakeProvider.mode == "crossref" and self.path.startswith("/works?"):
            body = json.dumps({"message": {"items": [
                {"DOI": "10.1234/hive", "title": ["Swarm intelligence"],
                 "author": [{"family": "Doe", "given": "Jane"}],
                 "published": {"date-parts": [[2025, 1, 15]]},
                 "container-title": ["Journal of Hives"], "type": "journal-article"}]}}).encode()
        else:
            body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 1. default: all external APIs disabled -> no tools registered
    store0 = CheckpointStore(":memory:")
    empty = ApiConfig.model_validate({"enabled": [], "providers": []})
    auth0 = OutboundAuthorizer(store0, empty, ApprovalManager(store0))
    rule0 = EndpointRule(endpoint_id="x", scheme="http", host="127.0.0.1", port=port,
                         path_prefixes=("/",), methods=("GET",), headers={})
    client0 = SecureHttpClient(TransportConfig(endpoints=(rule0,), allow_loopback=True))
    session0 = build_offline_test_session(client0, auth0, store0)
    cat0 = ToolCatalog()
    assert register_api_tools(cat0, session0, providers=[]) == []
    print("1. default config: all external API tools DISABLED (nothing registered)")

    # 2-3. enable wikimedia (authorized) + crossref (approval_required)
    apis = ApiConfig.model_validate({
        "enabled": ["wikimedia", "crossref"],
        "max_wire_bytes_per_request": 1_000_000,
        "budgets": {"per_endpoint_requests": 50, "per_endpoint_bytes": 10**9,
                    "per_provider_requests": 100, "per_provider_bytes": 10**9,
                    "per_run_requests": 50, "per_run_bytes": 10**9},
        "providers": [
            {"provider": "wikimedia", "billing": "public_api_no_charge",
             "outbound_policy": "authorized", "allowed_operations": ["search", "get_page"],
             "endpoints": [{"endpoint_id": "wikimedia-core", "scheme": "https",
                            "host": "api.wikimedia.org",
                            "path_prefixes": ["/core/v1/wikipedia"], "methods": ["GET"]}]},
            {"provider": "crossref", "billing": "public_api_no_charge",
             "outbound_policy": "approval_required",
             "allowed_operations": ["search_works", "get_doi"],
             "polite_mailto": "operator@example.org",
             "endpoints": [{"endpoint_id": "crossref-works", "scheme": "https",
                            "host": "api.crossref.org", "path_prefixes": ["/works"],
                            "methods": ["GET"]}]},
        ],
    })
    store = CheckpointStore(":memory:")
    rules = (
        EndpointRule(endpoint_id="wikimedia-core", scheme="http", host="127.0.0.1",
                     port=port, path_prefixes=("/core/v1/wikipedia",), methods=("GET",),
                     headers={"User-Agent": "hive-demo/0.7.0"}),
        EndpointRule(endpoint_id="crossref-works", scheme="http", host="127.0.0.1",
                     port=port, path_prefixes=("/works",), methods=("GET",),
                     headers={"User-Agent": "hive-demo/0.7.0"}),
    )
    client = SecureHttpClient(TransportConfig(endpoints=rules,
                                              limits=HttpLimits(max_retries=1),
                                              allow_loopback=True))  # TEST-ONLY flag
    mgr = ApprovalManager(store)
    auth = OutboundAuthorizer(store, apis, mgr)
    session = build_offline_test_session(client, auth, store)  # TEST-ONLY factory
    catalog = ToolCatalog()
    names = register_api_tools(catalog, session, providers=["wikimedia", "crossref"])
    print(f"2. registered tools for enabled providers: {names}")
    enabled = ["wikimedia_search", "crossref_search_works"]
    policy = build_policy(global_enabled=enabled,
                          agent_permissions={"researcher": (True, enabled)})
    gw = ToolGateway(catalog, policy, store=store, approval_manager=mgr)
    ctx = ToolExecutionContext(agent="researcher")

    FakeProvider.mode = "wiki"
    r = gw.execute("wikimedia_search", {"query": "enjambre", "language": "es"}, ctx=ctx)
    assert r.ok, r.error
    out = json.loads(r.output)
    print(f"3. wikimedia_search (es) -> {out['results'][0]['title']}: "
          f"{out['results'][0]['excerpt']!r} [{out['attribution'][:40]}...]")

    # 4. approval_required provider: denied, then approved once (no replay)
    FakeProvider.mode = "crossref"
    r = gw.execute("crossref_search_works", {"query": "swarm"}, ctx=ctx)
    print(f"4. crossref before approval -> rejected: {r.error[:80]}")
    import hashlib
    # the canonical args must match what the adapter builds; use the authorizer
    req = auth.request_approval(provider="crossref", endpoint_id="crossref-works",
                                operation_id="search_works", canonical_args={
                                    "method": "GET", "path": "/works",
                                    "params": {"mailto": "operator@example.org",
                                               "query": "swarm", "rows": 10,
                                               "select": ("DOI,title,author,published,"
                                                          "container-title,URL,license,type,"
                                                          "publisher,volume,page")},
                                    "body_sha256": hashlib.sha256(b"").hexdigest()},
                                agent="researcher", run_id=None, task_id=None)
    store.decide_approval(req.approval_id, decision="approved", operator="operator-1")
    r = gw.execute("crossref_search_works", {"query": "swarm"}, ctx=ctx)
    assert r.ok, r.error
    out = json.loads(r.output)
    print(f"   after approval -> {out['results'][0]['title']} "
          f"({out['results'][0]['doi']}, {out['results'][0]['published']})")
    r = gw.execute("crossref_search_works", {"query": "swarm"}, ctx=ctx)
    print(f"   replay after consumption -> rejected: {r.error[:60]}")

    # 5. durable accounting: every physical call was reserved + audited
    usage = {x["kind"]: x["used"] for x in store.api_usage_status("provider:crossref")}
    audit = store.api_audit_list(provider="crossref")
    print(f"5. durable usage for crossref: {usage}; audit rows: {len(audit)} "
          "(metadata only — no query payload persisted)")
    assert "swarm" not in json.dumps(audit)

    srv.shutdown()
    store0.close()
    store.close()
    print("\nDemo finished. Loopback only; no API keys, no external network, zero cost.")


if __name__ == "__main__":
    main()
