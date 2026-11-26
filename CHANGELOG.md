# CHANGELOG

## 0.7.0 - Free API & Model Intelligence Engine (branch feat/free-api-model-intelligence)

Phase 5 (A0-A7), built on v0.6.0 (main @ 121899b):

- A0: local-inference transport hardening (sync/async parity, env restoration,
  concurrent-gateway isolation, IPv6 loopback).
- A1: informational provider/capability catalog (never grants permissions).
- A5: SecureHttpClient (stdlib): pinned numeric destinations + connected-peer
  verification, TLS hostname/SNI, SSRF destination classes, redirect/proxy
  rejection, canonical percent-encoding validation, size/decompression/total
  deadlines, bounded resolver pool, cancellation-safe semaphore; durable API
  budgets (schema v9); outbound authorization via ToolGateway/ApprovalManager.
- A6: read-only adapters: Wikimedia, Crossref, GitHub REST, World Bank,
  Open-Meteo (non-commercial attestation default false); SearXNG disabled.
  Unicode titles / DOI special chars via canonical encoding; Crossref
  references from actual reference records.
- A2: deterministic benchmark engine; suite v2.0.0 (60 cases, en+es, 5 roles);
  evidence trust tiers; trusted_runtime only via the lease-fenced recorder.
- A3: policy-first routing engine (opt-in), mandatory eligibility filters,
  deterministic ranking, durable provenance (schema v11).
- A4: cloud evidence vault (schema v12), typed evidence classes, fail-closed
  eligibility oracle; fixtures can never mint provider_enforced records.
- A7: v0.7.0 production release gate - in zero-cost mode only local (ollama)
  inference may be dispatched; non-local cloud inference denied independently
  of evidence state. Full audit + release packaging.

Zero authorized API expenditure; no live requests in dev/tests; 811-test Linux
baseline preserved and extended (see CI).

## 0.6.0 - Zero-Cost Cloud Engine (merged to main @ 121899b)

See README "Phase 4" section.
