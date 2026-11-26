"""A5: outbound authorization + durable accounting for public API requests.

Reuses the existing permission system: ToolGateway tool specs, ApprovalManager
(durable, hash-bound, expiring approvals) and CheckpointStore v9 reservations.
There is NO second permission system here.

Policy per provider (config): DISABLED (default) | APPROVAL_REQUIRED |
AUTHORIZED (only with an explicit allowed_operations list). Approvals bind run,
task, agent, tool identity+version, exact operation, canonical args digest,
destination endpoint, expiry and a unique id; consumption is single-use and
revalidated immediately before dispatch. Previews show the ACTUAL outbound
request (bounded + secret-redacted), not just a hash.

Never automatically transmits workspace files, conversation history, private
documents or credentials; external content is untrusted and grants nothing.

Billing note: provider `billing` classes in config are OPERATOR ATTESTATIONS,
not verified evidence; unknown billing exposure fails closed. HIVE-local
request/byte budgets cannot guarantee zero provider-side charges.

Authorization and transport are bound at construction: every authorized
endpoint id must map to a transport rule with the identical scheme/host/port/
path prefixes/methods/credential env/extra headers/fixed headers (see
ApiSession._verify_endpoint_binding). ApiSession.__init__ ALWAYS enforces this;
there is no public bypass. Offline loopback fixtures use the explicitly
test-only build_offline_test_session() factory; production construction
(build_api_session) never uses it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .agent_config import ApiConfig, ApiProviderConfig
from .approvals import ApprovalManager
from .checkpoints import CheckpointStore, QuotaReservationOutcome
from .errors import ConfigError, HiveError
from .http_client import (
    EndpointRule,
    HttpLimits,
    HttpResult,
    SecureHttpClient,
    TransportConfig,
    parse_retry_after,
)
from .tool_gateway import ToolExecutionContext
from .tool_spec import SideEffect, ToolSpec, args_sha256


class OutboundPolicy(StrEnum):
    DISABLED = "disabled"
    APPROVAL_REQUIRED = "approval_required"
    AUTHORIZED = "authorized"


class OutboundNotAuthorizedError(HiveError):
    """The outbound operation is not authorized (policy/approval/expiry/binding)."""


def _tool_name(provider: str, operation_id: str) -> str:
    name = f"api_{provider}_{operation_id}".replace("-", "_")
    if not name[:1].isalpha():
        name = "api_" + name
    return name[:64]


def _approval_payload(*, endpoint_id: str, operation_id: str,
                      canonical_args: dict[str, Any]) -> dict[str, Any]:
    """The exact payload whose sha256 binds an approval to one outbound operation.
    Used IDENTICALLY at request, check and consume time so the stored hash, the
    lookup hash and the consumption hash can never diverge (substitution-safe)."""
    return {"endpoint_id": endpoint_id, "operation_id": operation_id,
            "args": canonical_args}


def _approval_digest(*, endpoint_id: str, operation_id: str,
                     canonical_args: dict[str, Any]) -> str:
    return args_sha256(_approval_payload(endpoint_id=endpoint_id,
                                         operation_id=operation_id,
                                         canonical_args=canonical_args))


def canonical_request_digest(*, endpoint_id: str, method: str, path: str,
                             params: Mapping[str, Any] | None,
                             body: bytes | None) -> str:
    """One-way digest of the exact sanitized outbound request (never stored raw)."""
    payload = {
        "endpoint_id": endpoint_id,
        "method": method.upper(),
        "path": path,
        "params": {k: params[k] for k in sorted(params or {})},
        "body_sha256": hashlib.sha256(body or b"").hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class OutboundDecision:
    ok: bool
    reason: str
    approval_id: str | None = None
    policy: OutboundPolicy = OutboundPolicy.DISABLED


class OutboundAuthorizer:
    """Runtime outbound policy over the configured providers. The catalog is
    NEVER consulted here: catalog entries are informational and grant nothing."""

    def __init__(self, store: CheckpointStore, apis: ApiConfig,
                 approvals: ApprovalManager | None = None) -> None:
        self.store = store
        self.apis = apis
        self.approvals = approvals
        self._providers = {p.provider: p for p in apis.providers}

    def provider_config(self, provider: str) -> ApiProviderConfig | None:
        return self._providers.get(provider)

    def policy_for(self, provider: str) -> OutboundPolicy:
        p = self._providers.get(provider)
        if p is None or provider not in self.apis.enabled:
            return OutboundPolicy.DISABLED
        return OutboundPolicy(p.outbound_policy)

    def check(self, *, provider: str, endpoint_id: str, operation_id: str,
              canonical_args: dict[str, Any], agent: str,
              run_id: str | None, task_id: str | None) -> OutboundDecision:
        """Pre-dispatch authorization check. Raises nothing; returns a decision."""
        p = self.provider_config(provider)
        policy = self.policy_for(provider)
        if p is None:
            return OutboundDecision(False, f"unknown API provider {provider!r}", policy=policy)
        if policy == OutboundPolicy.DISABLED:
            return OutboundDecision(
                False, f"provider {provider!r} is DISABLED (external APIs default to "
                "disabled)", policy=policy)
        if not any(e.endpoint_id == endpoint_id for e in p.endpoints):
            return OutboundDecision(False, f"endpoint {endpoint_id!r} not configured for "
                                           f"{provider!r}", policy=policy)
        if policy == OutboundPolicy.AUTHORIZED:
            if operation_id not in p.allowed_operations:
                return OutboundDecision(
                    False, f"operation {operation_id!r} not in allowed_operations of "
                    f"{provider!r}", policy=policy)
            return OutboundDecision(True, "authorized by scoped policy", policy=policy)
        # APPROVAL_REQUIRED: an APPROVED, unexpired, matching approval must exist
        if self.approvals is None:
            return OutboundDecision(False, "approval manager unavailable", policy=policy)
        digest = _approval_digest(endpoint_id=endpoint_id, operation_id=operation_id,
                                  canonical_args=canonical_args)
        tool = _tool_name(provider, operation_id)
        now = self.store.now_iso()
        for row in self.store.list_approvals(run_id=run_id, status="approved"):
            # expiry is enforced HERE too (not only at consume time): an expired
            # approval never authorizes. Unparseable expiry -> fail closed.
            try:
                expires = datetime.fromisoformat(row["expires_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if expires <= datetime.fromisoformat(now):
                continue
            if (row["tool"] == tool and row["operation"] == operation_id
                    and row["target"] == endpoint_id
                    and row["args_sha256"] == digest
                    and row["status"] == "approved"
                    and row["agent"] == agent):
                return OutboundDecision(True, "approved", approval_id=row["approval_id"],
                                        policy=policy)
        return OutboundDecision(
            False, f"operation {operation_id!r} on {provider!r}/{endpoint_id!r} requires an "
            "operator approval (none approved/bound)", policy=policy)

    def revalidate(self, *, provider: str, endpoint_id: str, operation_id: str,
                   canonical_args: dict[str, Any], agent: str,
                   run_id: str | None, task_id: str | None,
                   approval_id: str | None = None) -> OutboundDecision:
        """Pre-dispatch revalidation immediately before the physical request.
        With approval_id=None this is the same as check(). With a consumed
        approval id (a retry of ONE logical call), the approval row must still
        be CONSUMED, unexpired and exactly bound (digest/tool/operation/target/
        agent) — a fresh call still requires a fresh approved row, so approvals
        can never be replayed across logical requests."""
        if approval_id is None:
            return self.check(provider=provider, endpoint_id=endpoint_id,
                              operation_id=operation_id, canonical_args=canonical_args,
                              agent=agent, run_id=run_id, task_id=task_id)
        policy = self.policy_for(provider)
        p = self.provider_config(provider)
        if p is None or policy == OutboundPolicy.DISABLED:
            return OutboundDecision(False, f"provider {provider!r} no longer enabled",
                                    policy=policy)
        if not any(e.endpoint_id == endpoint_id for e in p.endpoints):
            return OutboundDecision(False, f"endpoint {endpoint_id!r} no longer configured",
                                    policy=policy)
        if policy == OutboundPolicy.AUTHORIZED:
            if operation_id not in p.allowed_operations:
                return OutboundDecision(False, "operation no longer allowed", policy=policy)
            return OutboundDecision(True, "authorized by scoped policy", policy=policy)
        digest = _approval_digest(endpoint_id=endpoint_id, operation_id=operation_id,
                                  canonical_args=canonical_args)
        tool = _tool_name(provider, operation_id)
        now = self.store.now_iso()
        for row in self.store.list_approvals():
            if row["approval_id"] != approval_id:
                continue
            try:
                expires = datetime.fromisoformat(row["expires_at"])
            except (KeyError, TypeError, ValueError):
                return OutboundDecision(False, "approval expiry unreadable", policy=policy)
            if expires <= datetime.fromisoformat(now):
                return OutboundDecision(False, "approval expired", policy=policy)
            if (row["status"] == "consumed" and row["tool"] == tool
                    and row["operation"] == operation_id and row["target"] == endpoint_id
                    and row["args_sha256"] == digest and row["agent"] == agent):
                return OutboundDecision(True, "consumed approval still bound",
                                        approval_id=approval_id, policy=policy)
            return OutboundDecision(False, "consumed approval binding mismatch",
                                    policy=policy)
        return OutboundDecision(False, "consumed approval missing", policy=policy)

    def request_approval(self, *, provider: str, endpoint_id: str, operation_id: str,
                         canonical_args: dict[str, Any], agent: str,
                         run_id: str | None, task_id: str | None):
        """Create a PENDING approval whose preview shows the ACTUAL outbound request
        (bounded + secret-redacted). Operator decides via the existing CLI."""
        if self.approvals is None:
            raise OutboundNotAuthorizedError("approval manager unavailable")
        spec = ToolSpec(name=_tool_name(provider, operation_id),
                        description=f"outbound API call {provider}/{endpoint_id}/{operation_id}",
                        version="1", category="api", side_effect=SideEffect.READ_ONLY)
        ctx = ToolExecutionContext(agent=agent, run_id=run_id, task_id=task_id)
        return self.approvals.create(
            ctx=ctx, spec=spec,
            args=_approval_payload(endpoint_id=endpoint_id, operation_id=operation_id,
                                   canonical_args=canonical_args),
            operation=operation_id, target=endpoint_id)

    def consume_approval(self, decision: OutboundDecision, *, provider: str,
                         endpoint_id: str, operation_id: str,
                         canonical_args: dict[str, Any],
                         agent: str, run_id: str | None, task_id: str | None) -> None:
        """Single-use consumption bound to the exact args/run/agent/tool/operation
        and destination endpoint (same digest as request/check -> substitution-safe)."""
        if decision.approval_id is None:
            return
        assert self.approvals is not None
        ok = self.approvals.consume(
            decision.approval_id,
            expected_args_sha256=_approval_digest(
                endpoint_id=endpoint_id, operation_id=operation_id,
                canonical_args=canonical_args),
            expected_tool_version="1",
            expected_run_id=run_id,
            expected_agent=agent,
            expected_tool=_tool_name(provider, operation_id),
            expected_task_id=task_id,
        )
        if not ok:
            raise OutboundNotAuthorizedError(
                "approval substitution/replay rejected (consume failed)")


class ApiSession:
    """One authorized API call = policy check -> durable reservation -> revalidation
    + single-use dispatch mark -> physical request (new reservation per retry) ->
    bounded audit. Retry-After is honored safely (malformed -> ignored, capped)."""

    def __init__(self, client: SecureHttpClient, authorizer: OutboundAuthorizer,
                 store: CheckpointStore, *, clock=time.monotonic,
                 sleep=time.sleep) -> None:
        self.client = client
        self.auth = authorizer
        self.store = store
        self._clock = clock
        self._sleep = sleep
        # ALWAYS enforced — there is deliberately NO public option to skip this.
        # Authorization config and physical transport config must not be
        # independently substituted: every enabled provider endpoint must map to
        # a transport rule with the EXACT same scheme/host/port/path prefixes/
        # methods/credential env/extra header keys/fixed headers (incl. the
        # User-Agent that build_api_session prepends). Production construction
        # (build_api_session) derives both from one config and always passes.
        # Offline loopback fixtures use build_offline_test_session() instead.
        self._verify_endpoint_binding()

    def _verify_endpoint_binding(self) -> None:
        apis = self.auth.apis
        for p in apis.providers:
            if p.provider not in apis.enabled:
                continue
            for e in p.endpoints:
                rule = self.client.rule_for(e.endpoint_id)
                if rule is None:
                    raise ConfigError(
                        f"authorized endpoint {e.endpoint_id!r} (provider {p.provider!r}) "
                        "has no transport rule in the client — authorization/transport "
                        "binding mismatch")
                expected_port = e.port or (443 if e.scheme == "https" else 80)
                # fixed headers as build_api_session would set them: the shared
                # User-Agent first, then the endpoint's own headers
                expected_headers = {"User-Agent": apis.user_agent, **e.headers}
                if (rule.scheme != e.scheme or rule.host != e.host
                        or rule.default_port() != expected_port
                        or tuple(rule.path_prefixes) != tuple(e.path_prefixes)
                        or tuple(rule.methods) != tuple(m.upper() for m in e.methods)
                        or rule.auth_env != e.auth_env
                        or tuple(rule.extra_header_keys) != tuple(e.extra_header_keys)
                        or dict(rule.headers) != expected_headers):
                    raise ConfigError(
                        f"authorized endpoint {e.endpoint_id!r} does not match its transport "
                        f"rule: config={e.scheme}://{e.host}:{expected_port} "
                        f"{e.path_prefixes} {e.methods} auth_env={e.auth_env} "
                        f"extra={e.extra_header_keys} headers={expected_headers} vs "
                        f"rule={rule.scheme}://{rule.host}:{rule.default_port()} "
                        f"{rule.path_prefixes} {rule.methods} auth_env={rule.auth_env} "
                        f"extra={rule.extra_header_keys} headers={dict(rule.headers)} "
                        "— substitution rejected")

    def request(self, *, provider: str, endpoint_id: str, operation_id: str,
                method: str, path: str, params: Mapping[str, Any] | None = None,
                body: bytes | None = None, agent: str = "system",
                run_id: str | None = None, task_id: str | None = None,
                lease_owner: str | None = None, lease_generation: int | None = None,
                cancel=None) -> HttpResult:
        canonical_args = {"method": method.upper(), "path": path,
                          "params": {k: params[k] for k in sorted(params or {})},
                          "body_sha256": hashlib.sha256(body or b"").hexdigest()}
        digest = canonical_request_digest(endpoint_id=endpoint_id, method=method,
                                           path=path, params=params, body=body)
        decision = self.auth.check(provider=provider, endpoint_id=endpoint_id,
                                   operation_id=operation_id, canonical_args=canonical_args,
                                   agent=agent, run_id=run_id, task_id=task_id)
        if not decision.ok:
            self._audit(provider, endpoint_id, operation_id, "blocked", None, 0, 0, 0,
                        decision.reason, digest)
            raise OutboundNotAuthorizedError(decision.reason)
        # NOTE: an APPROVAL_REQUIRED decision is consumed only AFTER the
        # pre-dispatch revalidation below (consume is single-use, so consuming
        # first would make the revalidation check unable to find the approval).
        budgets = self._budgets(provider, endpoint_id, run_id)
        max_wire = self.auth.apis.max_wire_bytes_per_request
        attempts = 0
        max_attempts = self.client.config.limits.max_retries + 1
        consumed_approval: str | None = None  # one approval per logical call
        while True:
            attempts += 1
            # the HIVE-local accounting window is recalculated IMMEDIATELY
            # BEFORE each durable reservation, so a retry that crosses a day
            # boundary is charged to the correct window (never a provider reset)
            window = self.store.now_iso()[:10]
            reservation_id = uuid4().hex
            outcome = self.store.reserve_api_request(
                reservation_id, provider=provider, endpoint_id=endpoint_id,
                operation_id=operation_id, request_sha256=digest,
                max_wire_bytes=max_wire, budgets=budgets, window_start_iso=window,
                run_id=run_id, lease_owner=lease_owner,
                lease_generation=lease_generation)
            if outcome != QuotaReservationOutcome.CREATED:
                raise OutboundNotAuthorizedError(
                    "reservation replay is reconciliation only, never dispatch")
            # revalidate authorization + lease immediately before dispatch
            # (retries of this logical call reuse the consumed approval)
            again = self.auth.revalidate(provider=provider, endpoint_id=endpoint_id,
                                         operation_id=operation_id,
                                         canonical_args=canonical_args, agent=agent,
                                         run_id=run_id, task_id=task_id,
                                         approval_id=consumed_approval)
            if not again.ok:
                self._audit(provider, endpoint_id, operation_id, "blocked", None,
                            0, 0, 0, "revalidation failed: " + again.reason, digest)
                raise OutboundNotAuthorizedError(again.reason)
            # single-use approval consumption immediately before dispatch
            # (after revalidation, so the revalidation can still see it)
            if again.policy == OutboundPolicy.APPROVAL_REQUIRED and consumed_approval is None:
                self.auth.consume_approval(again, provider=provider,
                                           endpoint_id=endpoint_id,
                                           operation_id=operation_id,
                                           canonical_args=canonical_args, agent=agent,
                                           run_id=run_id, task_id=task_id)
                consumed_approval = again.approval_id
            if not self.store.mark_api_dispatched(
                    reservation_id, run_id=run_id, lease_owner=lease_owner,
                    lease_generation=lease_generation, request_sha256=digest):
                raise OutboundNotAuthorizedError(
                    "dispatch authorization denied (used/fenced/substituted)")
            t0 = self._clock()
            try:
                result = self.client.request(endpoint_id, method, path, params=params,
                                             body=body, cancel=cancel)
            except HiveError as exc:
                self._audit(provider, endpoint_id, operation_id, "error",
                            None, 0, 0, int((self._clock() - t0) * 1000),
                            f"{type(exc).__name__}: {exc}"[:300], digest)
                raise
            duration_ms = int((self._clock() - t0) * 1000)
            self._audit(provider, endpoint_id, operation_id, "ok", result.status,
                        len(body or b""), result.wire_bytes, duration_ms,
                        decision.policy.value, digest, reservation_id=reservation_id)
            if result.status in (429, 503) and attempts < max_attempts:
                wait = parse_retry_after(result.headers.get("Retry-After"))
                if wait is not None:
                    self._sleep(wait)
                    continue  # every retry consumes a NEW reservation
            return result

    def _budgets(self, provider: str, endpoint_id: str,
                 run_id: str | None) -> dict[str, dict[str, int]]:
        b = self.auth.apis.budgets
        out: dict[str, dict[str, int]] = {
            f"provider:{provider}": {"requests": b.per_provider_requests,
                                      "wire_bytes": b.per_provider_bytes},
            f"endpoint:{endpoint_id}": {"requests": b.per_endpoint_requests,
                                         "wire_bytes": b.per_endpoint_bytes},
        }
        if run_id is not None:
            out[f"run:{run_id}"] = {"requests": b.per_run_requests,
                                    "wire_bytes": b.per_run_bytes}
        return out

    def _audit(self, provider, endpoint_id, operation_id, outcome, status, bytes_in,
               bytes_out, duration_ms, decision, digest, reservation_id=None):
        self.store.record_api_audit(
            reservation_id=reservation_id or uuid4().hex, provider=provider,
            endpoint_id=endpoint_id, operation_id=operation_id, outcome=outcome,
            status_code=status, bytes_in=bytes_in, bytes_out=bytes_out,
            duration_ms=duration_ms, policy_decision=decision,
            provenance=digest[:16])


def build_offline_test_session(client: SecureHttpClient, authorizer: OutboundAuthorizer,
                               store: CheckpointStore, **kwargs) -> ApiSession:
    """TEST-ONLY construction for offline loopback fixtures: builds an ApiSession
    WITHOUT the endpoint-binding verification, because the test transport rule
    (loopback http) intentionally differs from the validated https config
    endpoint. This is the ONLY way to bypass _verify_endpoint_binding, it is
    explicitly named as test-only, and no production code path (in particular
    build_api_session) can reach it: ApiSession.__init__ always verifies.
    Never use in production configuration or adapters."""
    sess = ApiSession.__new__(ApiSession)
    sess.client = client
    sess.auth = authorizer
    sess.store = store
    sess._clock = kwargs.get("clock", time.monotonic)
    sess._sleep = kwargs.get("sleep", time.sleep)
    return sess


def build_api_session(hive_config, store: CheckpointStore,
                      approvals: ApprovalManager | None = None) -> ApiSession | None:
    """Build the shared secure API session from config. Returns None when no API
    provider is enabled (the default: all external APIs DISABLED). Production
    construction never sets allow_loopback and never injects resolver/ssl_context."""
    apis = getattr(hive_config, "apis", None)
    if apis is None or not apis.enabled:
        return None
    endpoints = []
    for p in apis.providers:
        if p.provider not in apis.enabled:
            continue
        for e in p.endpoints:
            headers = {"User-Agent": apis.user_agent, **e.headers}
            endpoints.append(EndpointRule(
                endpoint_id=e.endpoint_id, scheme=e.scheme, host=e.host, port=e.port,
                path_prefixes=tuple(e.path_prefixes), methods=tuple(e.methods),
                headers=headers, extra_header_keys=tuple(e.extra_header_keys),
                auth_env=e.auth_env))
    tcfg = TransportConfig(
        endpoints=tuple(endpoints),
        limits=HttpLimits(max_wire_bytes=apis.max_wire_bytes_per_request))
    client = SecureHttpClient(tcfg)
    authorizer = OutboundAuthorizer(store, apis, approvals)
    return ApiSession(client, authorizer, store)
