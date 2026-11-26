"""W3 (v0.8.0) offline demo: secure web retrieval + source registration.

Fully offline: a deterministic LOOPBACK HTTP server stands in for the
allowlisted web endpoint (production uses the same code path with https +
global hosts via SecureHttpClient; loopback exists only in this test transport).
No API keys, no external network, zero cost.

Shows: disabled-by-default, endpoint policy enforcement, canonical URL
validation, bounded HTML extraction (scripts/resources suppressed), source
registration with provenance + citation, truncation provenance, and
failed-retrieval-misregistration prevention.
"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from geosgan_hive.agent_config import ApiConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.errors import HiveError
from geosgan_hive.http_client import EndpointRule, HttpLimits, SecureHttpClient, TransportConfig
from geosgan_hive.outbound import OutboundAuthorizer, build_offline_test_session
from geosgan_hive.sources import SourceRegistry
from geosgan_hive.web_retrieval import WebRetrievalEngine, WebRetrievalError

PAGE = (b"<html><head><title>Demo Page</title><script>alert('x')</script></head><body>"
        b"<h1>Heading</h1><p>Visible text for the demo.</p>"
        b"<img src='/t.png'><p>Second paragraph.</p></body></html>")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits: list[str] = []
    fail = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits.append(self.path)
        if type(self).fail:
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hive-web-demo-"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        store = CheckpointStore(str(tmp / "cp.db"))
        store.create_run(run_id="demo-run", owner="demo-w", task="demo",
                         agent="researcher", model="ollama/llama3.2")
        gen = store.get("demo-run").lease_generation

        apis = ApiConfig.model_validate({
            "enabled": ["web"],
            "max_wire_bytes_per_request": 1_000_000,
            "budgets": {"per_endpoint_requests": 10, "per_endpoint_bytes": 10**7,
                        "per_provider_requests": 10, "per_provider_bytes": 10**7,
                        "per_run_requests": 10, "per_run_bytes": 10**7},
            "providers": [{"provider": "web", "billing": "public_api_no_charge",
                           "outbound_policy": "authorized", "allowed_operations": ["get_page"],
                           "endpoints": [{"endpoint_id": "web-docs", "scheme": "https",
                                          "host": "docs.example.org",
                                          "path_prefixes": ["/docs"], "methods": ["GET"]}]}],
            "web_endpoints": [{"endpoint_id": "web-docs", "allowed_path_prefixes": ["/docs"],
                               "allowed_query_params": ["q"], "max_extract_chars": 500}],
        })
        rule = EndpointRule(endpoint_id="web-docs", scheme="http", host="127.0.0.1",
                            port=port, path_prefixes=("/docs",), methods=("GET",),
                            headers={"User-Agent": "hive-demo/1.0"})
        client = SecureHttpClient(TransportConfig(
            endpoints=(rule,), limits=HttpLimits(max_retries=0), allow_loopback=True))
        auth = OutboundAuthorizer(store, apis, ApprovalManager(store))
        session = build_offline_test_session(client, auth, store)
        reg = SourceRegistry(store)
        engine = WebRetrievalEngine(session=session, registry=reg,
                                    policies={p.endpoint_id: p for p in apis.web_endpoints})

        # 1) disabled by default: no policy -> fail closed
        empty = WebRetrievalEngine(session=session, registry=reg, policies={})
        try:
            empty.fetch(endpoint_id="web-docs", url="http://x/", agent="researcher")
            raise SystemExit("FAIL: disabled engine did not fail closed")
        except WebRetrievalError as exc:
            print(f"1) disabled-by-default OK: {exc}")

        # 2) policy bypass attempts rejected before any network
        for bad in ["http://evil.org/docs", f"http://127.0.0.1:{port}/docsx",
                    f"http://127.0.0.1:{port}/docs/../etc",
                    f"http://127.0.0.1:{port}/docs?q=ok&token=secret"]:
            try:
                engine.fetch(endpoint_id="web-docs", url=bad, agent="researcher",
                             run_id="demo-run", lease_owner="demo-w", lease_generation=gen)
                raise SystemExit(f"FAIL: bad url accepted: {bad}")
            except WebRetrievalError:
                pass
        print("2) URL policy bypass attempts rejected (no request sent)")

        # 3) successful retrieval -> registered source + citation
        r = engine.fetch(endpoint_id="web-docs",
                         url=f"http://127.0.0.1:{port}/docs/guide?q=demo#frag",
                         agent="researcher", run_id="demo-run", lease_owner="demo-w",
                         lease_generation=gen)
        assert r.source_id and r.title == "Demo Page" and "Visible text" in r.text
        assert "alert('x')" not in r.text and len(Handler.hits) == 1  # no resource fetches
        print(f"3) fetched+registered source {r.source_id}")
        print(f"   citation: {reg.cite(r.source_id)}")
        d = reg.get(r.source_id)
        prov = json.loads(d["provenance_json"])
        print(f"   provenance: status={prov['http_status']} truncated={prov['truncated']} "
              f"wire={r.wire_bytes}B raw_sha={r.raw_sha256[:12]}…")
        # metadata-only: page content is NOT persisted in the DB
        assert PAGE[:20] not in open(tmp / "cp.db", "rb").read()
        print("   metadata-only OK: page content not persisted; excerpt reports unavailable")

        # 4) failed retrieval registers nothing
        Handler.fail = True
        try:
            engine.fetch(endpoint_id="web-docs", url=f"http://127.0.0.1:{port}/docs/x",
                         agent="researcher", run_id="demo-run", lease_owner="demo-w",
                         lease_generation=gen)
            raise SystemExit("FAIL: failed retrieval did not raise")
        except HiveError:
            pass
        assert len(reg.list()) == 1  # only the successful one
        print("4) failed retrieval (HTTP 500) registered NO source")

        print("Demo finished. Loopback only; no API keys, no external network, zero cost.")
        return 0
    finally:
        srv.shutdown()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
