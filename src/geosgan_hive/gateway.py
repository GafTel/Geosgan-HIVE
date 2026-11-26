from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .cloud_gate import ZeroCostGate
from .cloud_profiles import CloudProfileRegistry
from .errors import (
    AuthenticationError as HiveAuthenticationError,
)
from .errors import (
    BadRequestError as HiveBadRequestError,
)
from .errors import (
    BudgetExceededError,
    CheckpointError,
    CloudConfigError,
    ConfigError,
    ErrorClass,
    FallbackExhaustedError,
    GatewayConnectionError,
    GatewayError,
    GatewayTimeoutError,
    HiveError,
    MissingCredentialsError,
    PermanentGatewayError,
    ProviderUnavailableError,
    RateLimitError,
    ServiceUnavailableError,
    classify_error,
)
from .errors import (
    ContextWindowExceededError as HiveContextWindowError,
)
from .errors import (
    QuotaExhaustedError as HiveQuotaError,
)
from .fallback import FallbackEngine, FallbackMode, FallbackPolicy, effective_policy
from .health import CircuitState, HealthRegistry
from .messages import Message
from .providers import Credentials, Provider, parse_model, resolve_credentials
from .retry import RetryPolicy, is_retryable
from .run_context import EventType
from .usage import UsageRecord, UsageTracker


@dataclass
class LLMResponse:
    """Full result of one model call: content plus its usage record."""

    content: str
    model: str
    usage: UsageRecord


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a Retry-After hint (seconds) from a provider exception, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None  # malformed Retry-After is ignored, never guessed
    if not math.isfinite(parsed) or parsed <= 0:
        return None  # nonfinite/negative hints are not evidence
    return min(parsed, 86400.0)  # cap excessively large hints (<= 24h) safely


def _map_litellm_error(exc: BaseException) -> GatewayError:
    """Map a LiteLLM exception to the HIVE error hierarchy."""
    if isinstance(exc, HiveError):
        return exc  # already a HIVE error (incl. terminal BudgetExceededError)
    import litellm.exceptions as litellm_exc

    if isinstance(exc, litellm_exc.ContextWindowExceededError):
        return HiveContextWindowError(str(exc))
    if isinstance(exc, litellm_exc.BudgetExceededError):
        return HiveQuotaError(str(exc))
    if isinstance(exc, litellm_exc.RateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, (litellm_exc.ServiceUnavailableError, litellm_exc.InternalServerError)):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, litellm_exc.Timeout):
        return GatewayTimeoutError(str(exc))
    if isinstance(exc, litellm_exc.APIConnectionError):
        return GatewayConnectionError(str(exc))
    if isinstance(exc, litellm_exc.AuthenticationError):
        return HiveAuthenticationError(str(exc))
    if isinstance(exc, litellm_exc.BadRequestError):
        return HiveBadRequestError(str(exc))
    if isinstance(exc, (ConnectionError, TimeoutError)):  # builtin network errors
        return GatewayConnectionError(str(exc))
    if isinstance(exc, litellm_exc.APIError):
        return PermanentGatewayError(str(exc))
    return PermanentGatewayError(f"{type(exc).__name__}: {exc}")


class ModelGateway:
    """Central gateway for every model call (sync and async) through LiteLLM.

    This is the single choke point for provider/credential resolution, retries,
    timeouts and usage/cost capture. LiteLLM's own retry mechanism is disabled
    (`num_retries=0`) so retries exist only here, in one place.
    """

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        default_timeout_seconds: float = 60.0,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Any] = asyncio.sleep,
        context_lookup: Callable[[str], Any] | None = None,
        health: HealthRegistry | None = None,
        fallback_engine: FallbackEngine | None = None,
        default_fallback: FallbackPolicy | None = None,
        cloud_registry: CloudProfileRegistry | None = None,
        cloud_adapter: Any = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        self.tracker = tracker if tracker is not None else UsageTracker()
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.default_timeout_seconds = default_timeout_seconds
        self._environ = environ
        self._sleep = sleep
        self._async_sleep = async_sleep
        self._credentials: dict[Provider, Credentials] = {}
        self._context_lookup = context_lookup
        # Phase 2.5: in-memory provider health registry + fallback engine.
        # Fallback is OFF by default (default_fallback.mode == off).
        self.health = health if health is not None else HealthRegistry()
        self.fallback_engine = (
            fallback_engine if fallback_engine is not None else FallbackEngine(self.health)
        )
        self.default_fallback = (
            default_fallback if default_fallback is not None else FallbackPolicy()
        )
        self._half_open_emitted: set[str] = set()
        # provider -> max Retry-After hint seen (seconds); used to extend the
        # persisted next_attempt_at of suspended runs (never invents reset times)
        self._retry_after_hints: dict[str, float] = {}
        # optional durable store: every physical dispatch reservation is committed
        # to SQLite BEFORE the provider request is sent (Phase 2.5 gate requirement)
        self._durable_store: Any = None
        # Phase 4 C4: central zero-cost gate in front of EVERY physical dispatch
        self._cloud_registry = cloud_registry
        self._cloud_adapter = cloud_adapter
        self._default_max_output_tokens = default_max_output_tokens
        self._cloud_gate: ZeroCostGate | None = None

    def set_durable_store(self, store: Any) -> None:
        """Wire a CheckpointStore so each physical dispatch is durably reserved
        (monotonic, atomic, before the request) for runs that have one."""
        if (store is None and self._cloud_registry is not None
                and self._cloud_registry.zero_cost):
            # zero-cost enforcement can never be silently disabled by dropping
            # the durable store
            raise ConfigError(
                "zero-cost mode requires the durable store; set_durable_store(None) "
                "is rejected while a zero-cost cloud registry is wired"
            )
        if (store is not None and self._cloud_gate is not None
                and self._cloud_gate.enabled() and store is not self._cloud_gate.store):
            # swapping the quota store would weaken enforcement
            raise ConfigError(
                "refusing to swap the durable quota store while a zero-cost gate is "
                "active — zero-cost enforcement cannot be silently weakened"
            )
        self._durable_store = store
        self._cloud_gate = None  # rebuild lazily with the new store

    def set_cloud_registry(self, registry: CloudProfileRegistry) -> None:
        """Wire the cloud profile registry (factory path). While a registry is
        wired, it can never be replaced by a different one (that would silently
        change enforcement)."""
        if (self._cloud_registry is not None and registry is not self._cloud_registry):
            raise ConfigError(
                "refusing to replace the wired cloud registry — replacing it could "
                "silently weaken zero-cost enforcement"
            )
        self._cloud_registry = registry
        self._cloud_gate = None

    def set_cloud_gate(self, gate: ZeroCostGate) -> None:
        """Wire an explicit ZeroCostGate (tests / custom wiring). An active
        zero-cost gate can only be replaced by the SAME gate object; a different
        gate/registry/adapter/store that could weaken enforcement is rejected."""
        if self._cloud_gate is not None and self._cloud_gate.enabled() \
                and gate is not self._cloud_gate:
            raise ConfigError(
                "refusing to replace an active zero-cost gate with a different gate "
                "— zero-cost enforcement cannot be silently disabled or weakened"
            )
        if (self._cloud_registry is not None and self._cloud_registry.zero_cost
                and not gate.enabled()):
            raise ConfigError(
                "refusing to replace an active zero-cost gate with an inactive one — "
                "zero-cost enforcement cannot be silently disabled"
            )
        if (self._cloud_registry is not None
                and gate.registry is not self._cloud_registry):
            raise ConfigError(
                "gate registry differs from the wired cloud registry — rejected"
            )
        if self._cloud_registry is None:
            self._cloud_registry = gate.registry
        self._cloud_gate = gate

    def cloud_gate(self) -> ZeroCostGate | None:
        """The active zero-cost gate (built lazily from registry + durable store)."""
        return self._get_cloud_gate()

    def _get_cloud_gate(self) -> ZeroCostGate | None:
        """Lazily build the zero-cost gate from the registry + durable store.
        In zero-cost mode a missing durable store fails closed at dispatch time."""
        if self._cloud_gate is not None:
            return self._cloud_gate
        if self._cloud_registry is None:
            return None
        self._cloud_gate = ZeroCostGate(
            self._cloud_registry,
            self._durable_store,
            adapter=self._cloud_adapter,
            default_max_output_tokens=self._default_max_output_tokens,
        )
        return self._cloud_gate

    def retry_after_hint(self, provider: Provider | str) -> float:
        """Largest Retry-After hint (seconds) observed for this provider, 0 if none."""
        key = provider.value if isinstance(provider, Provider) else str(provider)
        return self._retry_after_hints.get(key, 0.0)

    def _record_retry_after(self, model: str, exc: BaseException) -> float | None:
        hint = _retry_after_seconds(exc)
        if hint is not None and hint > 0:
            provider, _ = parse_model(model)
            key = provider.value
            self._retry_after_hints[key] = max(self._retry_after_hints.get(key, 0.0), hint)
        return hint

    def set_context_lookup(self, lookup: Callable[[str], Any]) -> None:
        """Wire the gateway to the run contexts (e.g. engine.context_for) so every
        dispatch can reserve its physical attempt and request slot atomically."""
        self._context_lookup = lookup

    def _run_context(self, run_id: str | None) -> Any:
        if run_id and self._context_lookup is not None:
            return self._context_lookup(run_id)
        return None

    def _reserve_physical(self, ctx: Any, run_id: str | None) -> None:
        """Reserve one physical attempt: in-memory cap check first, then the durable
        SQLite commit BEFORE the request is sent. The lease identity is the
        WORKER-OWNED binding from the execution context — never adopted by reading
        SQLite. If the durable commit fails (run no longer RUNNING / stale worker),
        the in-memory reservation is rolled back and dispatch is blocked."""
        if not ctx.reserve_physical_request():
            raise BudgetExceededError(
                f"Physical provider-request limit "
                f"({ctx.policy.max_provider_requests_per_run}) reached for this run."
            )
        store = self._durable_store
        if store is not None and run_id is not None:
            owner = getattr(ctx, "lease_owner", None)
            gen = getattr(ctx, "lease_generation", None)
            if owner is None or gen is None:
                ctx.release_physical_reservation()
                raise CheckpointError(
                    f"run {run_id!r}: worker-owned lease identity missing from the "
                    "execution context — physical dispatch fails closed (identity "
                    "is never adopted from SQLite)"
                )
            if not store.reserve_physical_durable(
                run_id, lease_owner=owner, lease_generation=gen
            ):
                ctx.release_physical_reservation()
                raise CheckpointError(
                    f"run {run_id!r} is no longer active (suspended/terminal/recovered); "
                    "dispatch blocked"
                )

    def _gate_binding(self, run_id: str | None) -> tuple[str | None, int | None]:
        """Worker-owned lease identity for run-bound dispatches, taken from the
        EXECUTION CONTEXT (never adopted by reading SQLite). Missing run/context/
        binding fails closed when the zero-cost gate is active."""
        if run_id is None:
            return None, None
        ctx = self._run_context(run_id)
        owner = getattr(ctx, "lease_owner", None) if ctx is not None else None
        gen = getattr(ctx, "lease_generation", None) if ctx is not None else None
        if owner is None or gen is None:
            raise CheckpointError(
                f"zero-cost run-bound dispatch of run {run_id!r} requires the "
                "worker-owned lease binding in the execution context (missing -> "
                "fail closed)"
            )
        return owner, gen

    def _dispatch_sync(self, kwargs: dict[str, Any], *, model: str, run_id: str | None) -> Any:
        """One physical dispatch. Zero-cost gate (when active): phase 1 reserves the
        durable quota BEFORE the slot; the request slot is acquired next and held
        ONLY around the in-flight request (never during backoff/cooldown/checkpoint
        waits); phase 2 revalidates lease+evidence+window+policy and grants the
        single-use authorization IMMEDIATELY BEFORE the LiteLLM invocation."""
        gate = self._get_cloud_gate()
        dispatch_id = ""
        owner = gen = None
        if gate is not None and gate.enabled():
            owner, gen = self._gate_binding(run_id)
            dispatch_id = gate.prepare_dispatch(
                model, kwargs, run_id=run_id, lease_owner=owner, lease_generation=gen)
        ctx = self._run_context(run_id)
        import litellm

        if ctx is None:
            if gate is not None and gate.enabled():
                gate.finalize_dispatch(model, dispatch_id, run_id=run_id,
                                       lease_owner=owner, lease_generation=gen,
                                       kwargs=kwargs)
            return litellm.completion(**kwargs)
        self._reserve_physical(ctx, run_id)
        sem = ctx.sync_semaphore()
        sem.acquire()
        try:
            if gate is not None and gate.enabled():
                gate.finalize_dispatch(model, dispatch_id, run_id=run_id,
                                       lease_owner=owner, lease_generation=gen,
                                       kwargs=kwargs)
            return litellm.completion(**kwargs)
        finally:
            sem.release()

    async def _dispatch_async(
        self, kwargs: dict[str, Any], *, model: str, run_id: str | None
    ) -> Any:
        gate = self._get_cloud_gate()
        dispatch_id = ""
        owner = gen = None
        if gate is not None and gate.enabled():
            owner, gen = self._gate_binding(run_id)
            dispatch_id = gate.prepare_dispatch(
                model, kwargs, run_id=run_id, lease_owner=owner, lease_generation=gen)
        ctx = self._run_context(run_id)
        import litellm

        if ctx is None:
            if gate is not None and gate.enabled():
                gate.finalize_dispatch(model, dispatch_id, run_id=run_id,
                                       lease_owner=owner, lease_generation=gen,
                                       kwargs=kwargs)
            return await litellm.acompletion(**kwargs)
        self._reserve_physical(ctx, run_id)
        sem = ctx.semaphore()
        await sem.acquire()
        try:
            if gate is not None and gate.enabled():
                gate.finalize_dispatch(model, dispatch_id, run_id=run_id,
                                       lease_owner=owner, lease_generation=gen,
                                       kwargs=kwargs)
            return await litellm.acompletion(**kwargs)
        finally:
            sem.release()

    def bind(
        self,
        model: str,
        *,
        agent: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> GatewayLLMClient:
        """Create an LLMClient bound to one agent's model, backed by this gateway.

        `fallback` is the agent-level fallback policy (None = gateway default);
        `needs_tools` feeds the capability gate (model must support tool calling).
        """
        parse_model(model)  # fail fast on an invalid model string
        return GatewayLLMClient(
            self,
            model=model,
            agent=agent,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            fallback=fallback,
            needs_tools=needs_tools,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> str:
        return self.complete_response(
            messages,
            model=model,
            agent=agent,
            run_id=run_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            fallback=fallback,
            needs_tools=needs_tools,
        ).content

    def complete_response(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> LLMResponse:
        policy = effective_policy(fallback, self.default_fallback)
        if policy.mode == FallbackMode.OFF or self.fallback_engine is None:
            # default path: identical to the pre-2.5 behavior (no fallback)
            return self._attempt_sync(
                messages, model=model, agent=agent, run_id=run_id,
                temperature=temperature, max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        return self._fallback_sync(
            messages, primary=model, agent=agent, run_id=run_id,
            temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds, policy=policy, needs_tools=needs_tools,
        )

    def _attempt_sync(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None,
        run_id: str | None,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
    ) -> LLMResponse:
        """One logical call on ONE model: bounded retries on transient errors only.
        Every physical dispatch reserves its attempt before the request is sent."""
        kwargs = self._build_kwargs(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._dispatch_sync(kwargs, model=model, run_id=run_id)
            except Exception as exc:
                error = _map_litellm_error(exc)
                hint = self._record_retry_after(model, exc)
                if is_retryable(error) and attempt < policy.max_attempts:
                    self._sleep(policy.delay_for_attempt(attempt, hint))
                    continue
                if not getattr(error, "terminal", False):
                    # terminal errors (budget/cancelled) never reached a provider:
                    # reserved-but-not-dispatched is not a confirmed request
                    self._record_failure(
                        model=model, agent=agent, run_id=run_id, attempts=attempt,
                        error=type(error).__name__,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                raise error from exc
            latency_ms = (time.perf_counter() - started) * 1000
            return self._to_response(
                response, model=model, agent=agent, run_id=run_id,
                latency_ms=latency_ms, attempts=attempt,
            )
        raise PermanentGatewayError("unreachable: retry loop exhausted")  # pragma: no cover

    def _fallback_sync(
        self,
        messages: Sequence[Message],
        *,
        primary: str,
        agent: str | None,
        run_id: str | None,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
        policy: FallbackPolicy,
        needs_tools: bool,
    ) -> LLMResponse:
        """Outer per-logical-call candidate loop: ordered chain, eligibility gates,
        health probes, events. Bounded: chain length x retry attempts x physical
        reservations, with `max_fallback_attempts` capping non-primary dispatches."""
        ctx = self._run_context(run_id)
        engine = self.fallback_engine
        assert engine is not None
        msg_dicts = [m.to_openai() for m in messages]
        chain = engine.chain(primary=primary, policy=policy)
        fallback_attempts = 0
        last_error: BaseException | None = None
        tried: list[str] = []
        gate = self._get_cloud_gate()
        for candidate in chain:
            if candidate != primary and fallback_attempts >= policy.max_fallback_attempts:
                break
            decision = engine.check_candidate(
                model=candidate, is_primary=(candidate == primary), primary=primary,
                policy=policy, needs_tools=needs_tools, messages=msg_dicts,
            )
            if not decision.ok:
                self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                           metadata={"model": candidate, "reason": decision.reason})
                continue
            if gate is not None and gate.enabled():
                billing_ok, billing_reason = gate.candidate_billing_ok(candidate)
                if not billing_ok:
                    if candidate == primary:
                        # the PRIMARY failing the zero-cost billing/evidence gate is
                        # a permanent configuration/billing condition — it must
                        # fail the run, never be suspended and retried
                        raise CloudConfigError(
                            f"zero-cost gate rejected the primary model {primary!r}: "
                            f"{billing_reason}"
                        )
                    # no automatic paid/unknown-cost fallback in zero-cost mode
                    self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                               metadata={"model": candidate,
                                         "reason": f"zero-cost gate: {billing_reason}"})
                    continue
            probes = self._claim_probes(candidate, ctx, agent)
            if probes is None:
                self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                           metadata={"model": candidate,
                                     "reason": "recovery probe already in flight"})
                continue
            tried.append(candidate)
            try:
                response = self._attempt_sync(
                    messages, model=candidate, agent=agent, run_id=run_id,
                    temperature=temperature, max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                error, _cls, fallback_eligible, terminal = self._handle_candidate_failure(
                    exc, candidate=candidate, probes=probes, ctx=ctx, agent=agent)
                if isinstance(error, MissingCredentialsError) and candidate != primary:
                    # a fallback candidate without credentials is skipped, not fatal
                    fallback_attempts += 1
                    last_error = error
                    continue
                if terminal or not fallback_eligible:
                    raise error from exc
                last_error = error
                if candidate != primary:
                    fallback_attempts += 1
                continue
            for key in probes:
                engine.registry.end_probe(key, success=True)
            self._record_health_success(candidate)
            if candidate != primary:
                self._emit(ctx, EventType.FALLBACK_SELECTED, agent=agent,
                           metadata={"from_model": primary, "to_model": candidate})
            return response
        if policy.mode in (FallbackMode.WAIT, FallbackMode.HYBRID):
            self._emit(ctx, EventType.TASK_SUSPENDED, agent=agent,
                       metadata={"primary": primary, "tried": tried,
                                 "reason": "no eligible model currently available"})
            raise ProviderUnavailableError(
                f"No eligible model available for this call (primary '{primary}', mode "
                f"'{policy.mode.value}', tried={tried}); the run may be suspended at a "
                "safe boundary and resumed later."
            ) from last_error
        raise FallbackExhaustedError(
            f"Fallback chain exhausted for primary '{primary}' (mode "
            f"'{policy.mode.value}', tried={tried})."
        ) from last_error

    def _emit(self, ctx: Any, event: EventType, *, agent: str | None,
              metadata: dict[str, Any] | None = None) -> None:
        """Record a structured event in the run context (never secrets/prompts)."""
        if ctx is None:
            return
        ctx.record_event(event, source_agent=agent, metadata=metadata or {})

    def _claim_probes(self, model: str, ctx: Any, agent: str | None) -> list[str] | None:
        """Claim single-flight recovery probes for the HALF_OPEN circuits of this
        candidate. Returns claimed keys ([] when nothing is HALF_OPEN), or None when
        another probe is already in flight (candidate then counts as unavailable)."""
        provider, _ = parse_model(model)
        claimed: list[str] = []
        keys = (HealthRegistry.model_key(model), HealthRegistry.provider_key(provider))
        for key in keys:
            if self.health.state(key) != CircuitState.HALF_OPEN:
                continue
            if key not in self._half_open_emitted:
                self._half_open_emitted.add(key)
                self._emit(ctx, EventType.CIRCUIT_HALF_OPEN, agent=agent, metadata={"key": key})
            if not self.health.try_begin_probe(key):
                for k in claimed:
                    self.health.release_probe(k)
                return None
            claimed.append(key)
            self._emit(ctx, EventType.RECOVERY_PROBE, agent=agent, metadata={"key": key})
        return claimed

    def _record_health_failure(self, model: str, error_class: ErrorClass, *,
                               ctx: Any, agent: str | None) -> None:
        """Record a failure with the correct scope. AUTH is credential-specific and
        never opens a provider-wide circuit; QUOTA is account-wide (provider key);
        RATE_LIMIT opens model+provider; OUTAGE/NETWORK open the provider; TIMEOUT
        opens the model. Other classes leave health untouched (conservative)."""
        if self.health is None:
            return
        provider, _ = parse_model(model)
        mkey = HealthRegistry.model_key(model)
        pkey = HealthRegistry.provider_key(provider)
        plan: list[tuple[str, bool]] = []
        if error_class == ErrorClass.AUTH:
            plan = [(mkey, False)]
        elif error_class == ErrorClass.QUOTA:
            plan = [(pkey, True), (mkey, False)]
        elif error_class == ErrorClass.RATE_LIMIT:
            plan = [(mkey, True), (pkey, True)]
        elif error_class in (ErrorClass.OUTAGE, ErrorClass.NETWORK):
            plan = [(pkey, True), (mkey, False)]
        elif error_class == ErrorClass.TIMEOUT:
            plan = [(mkey, True)]
        for key, open_circuit in plan:
            before = self.health.state(key)
            after = self.health.record_failure(key, open_circuit=open_circuit)
            if after == CircuitState.OPEN and before != CircuitState.OPEN:
                self._emit(ctx, EventType.CIRCUIT_OPENED, agent=agent, metadata={"key": key})

    def _record_health_success(self, model: str) -> None:
        if self.health is None:
            return
        provider, _ = parse_model(model)
        self.health.record_success(HealthRegistry.model_key(model))
        self.health.record_success(HealthRegistry.provider_key(provider))

    def _handle_candidate_failure(
        self, exc: BaseException, *, candidate: str, probes: list[str],
        ctx: Any, agent: str | None,
    ) -> tuple[BaseException, ErrorClass, bool, bool]:
        """Map + classify a candidate failure, end probes as failed, record health
        with the right scope and emit PROVIDER_FAILED. Returns (error, class,
        fallback_eligible, terminal)."""
        error = _map_litellm_error(exc)
        error_class, retryable, fallback_eligible, terminal = classify_error(error)
        engine = self.fallback_engine
        if engine is not None:
            for key in probes:
                engine.registry.end_probe(key, success=False)
        self._record_health_failure(candidate, error_class, ctx=ctx, agent=agent)
        self._emit(ctx, EventType.PROVIDER_FAILED, agent=agent, metadata={
            "model": candidate, "error_class": error_class.value,
            "retryable": retryable, "fallback_eligible": fallback_eligible,
            "terminal": terminal,
        })
        if terminal and error_class == ErrorClass.BUDGET:
            self._emit(ctx, EventType.BUDGET_EXHAUSTED, agent=agent,
                       metadata={"model": candidate})
        return error, error_class, fallback_eligible, terminal

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> str:
        return (
            await self.acomplete_response(
                messages,
                model=model,
                agent=agent,
                run_id=run_id,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                fallback=fallback,
                needs_tools=needs_tools,
            )
        ).content

    async def acomplete_response(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> LLMResponse:
        policy = effective_policy(fallback, self.default_fallback)
        if policy.mode == FallbackMode.OFF or self.fallback_engine is None:
            # default path: identical to the pre-2.5 behavior (no fallback)
            return await self._attempt_async(
                messages, model=model, agent=agent, run_id=run_id,
                temperature=temperature, max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        return await self._fallback_async(
            messages, primary=model, agent=agent, run_id=run_id,
            temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds, policy=policy, needs_tools=needs_tools,
        )

    async def _attempt_async(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None,
        run_id: str | None,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
    ) -> LLMResponse:
        """Async twin of `_attempt_sync`: one logical call on one model."""
        kwargs = self._build_kwargs(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = await self._dispatch_async(kwargs, model=model, run_id=run_id)
            except asyncio.CancelledError:
                self._record_failure(
                    model=model, agent=agent, run_id=run_id, attempts=attempt,
                    error="cancelled", latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise
            except Exception as exc:
                error = _map_litellm_error(exc)
                hint = self._record_retry_after(model, exc)
                if is_retryable(error) and attempt < policy.max_attempts:
                    delay = policy.delay_for_attempt(attempt, hint)
                    await self._async_sleep(delay)
                    continue
                if not getattr(error, "terminal", False):
                    # terminal errors (budget/cancelled) never reached a provider:
                    # reserved-but-not-dispatched is not a confirmed request
                    self._record_failure(
                        model=model, agent=agent, run_id=run_id, attempts=attempt,
                        error=type(error).__name__,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                raise error from exc
            latency_ms = (time.perf_counter() - started) * 1000
            return self._to_response(
                response, model=model, agent=agent, run_id=run_id,
                latency_ms=latency_ms, attempts=attempt,
            )
        raise PermanentGatewayError("unreachable: retry loop exhausted")  # pragma: no cover

    async def _fallback_async(
        self,
        messages: Sequence[Message],
        *,
        primary: str,
        agent: str | None,
        run_id: str | None,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
        policy: FallbackPolicy,
        needs_tools: bool,
    ) -> LLMResponse:
        """Async twin of `_fallback_sync` (native async; cancellation is terminal
        and never triggers retry or fallback)."""
        ctx = self._run_context(run_id)
        engine = self.fallback_engine
        assert engine is not None
        msg_dicts = [m.to_openai() for m in messages]
        chain = engine.chain(primary=primary, policy=policy)
        fallback_attempts = 0
        last_error: BaseException | None = None
        tried: list[str] = []
        gate = self._get_cloud_gate()
        for candidate in chain:
            if candidate != primary and fallback_attempts >= policy.max_fallback_attempts:
                break
            decision = engine.check_candidate(
                model=candidate, is_primary=(candidate == primary), primary=primary,
                policy=policy, needs_tools=needs_tools, messages=msg_dicts,
            )
            if not decision.ok:
                self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                           metadata={"model": candidate, "reason": decision.reason})
                continue
            if gate is not None and gate.enabled():
                billing_ok, billing_reason = gate.candidate_billing_ok(candidate)
                if not billing_ok:
                    if candidate == primary:
                        # permanent billing/config condition on the primary: fail,
                        # never suspend-and-retry unknown billing conditions
                        raise CloudConfigError(
                            f"zero-cost gate rejected the primary model {primary!r}: "
                            f"{billing_reason}"
                        )
                    # no automatic paid/unknown-cost fallback in zero-cost mode
                    self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                               metadata={"model": candidate,
                                         "reason": f"zero-cost gate: {billing_reason}"})
                    continue
            probes = self._claim_probes(candidate, ctx, agent)
            if probes is None:
                self._emit(ctx, EventType.FALLBACK_REJECTED, agent=agent,
                           metadata={"model": candidate,
                                     "reason": "recovery probe already in flight"})
                continue
            tried.append(candidate)
            try:
                response = await self._attempt_async(
                    messages, model=candidate, agent=agent, run_id=run_id,
                    temperature=temperature, max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
            except asyncio.CancelledError:
                # cancellation is terminal: no retry, no fallback, no health change
                # (a cancelled call says nothing about provider health)
                for key in probes:
                    engine.registry.end_probe(key, success=False)
                self._emit(ctx, EventType.PROVIDER_FAILED, agent=agent, metadata={
                    "model": candidate, "error_class": ErrorClass.CANCELLED.value,
                    "retryable": False, "fallback_eligible": False, "terminal": True,
                })
                raise
            except Exception as exc:
                error, _cls, fallback_eligible, terminal = self._handle_candidate_failure(
                    exc, candidate=candidate, probes=probes, ctx=ctx, agent=agent)
                if isinstance(error, MissingCredentialsError) and candidate != primary:
                    fallback_attempts += 1
                    last_error = error
                    continue
                if terminal or not fallback_eligible:
                    raise error from exc
                last_error = error
                if candidate != primary:
                    fallback_attempts += 1
                continue
            for key in probes:
                engine.registry.end_probe(key, success=True)
            self._record_health_success(candidate)
            if candidate != primary:
                self._emit(ctx, EventType.FALLBACK_SELECTED, agent=agent,
                           metadata={"from_model": primary, "to_model": candidate})
            return response
        if policy.mode in (FallbackMode.WAIT, FallbackMode.HYBRID):
            self._emit(ctx, EventType.TASK_SUSPENDED, agent=agent,
                       metadata={"primary": primary, "tried": tried,
                                 "reason": "no eligible model currently available"})
            raise ProviderUnavailableError(
                f"No eligible model available for this call (primary '{primary}', mode "
                f"'{policy.mode.value}', tried={tried}); the run may be suspended at a "
                "safe boundary and resumed later."
            ) from last_error
        raise FallbackExhaustedError(
            f"Fallback chain exhausted for primary '{primary}' (mode "
            f"'{policy.mode.value}', tried={tried})."
        ) from last_error

    def _credentials_for(self, provider: Provider) -> Credentials:
        if provider not in self._credentials:
            self._credentials[provider] = resolve_credentials(provider, self._environ)
        return self._credentials[provider]

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        provider, _ = parse_model(model)
        creds = self._credentials_for(provider)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "timeout": timeout_seconds
            if timeout_seconds is not None
            else self.default_timeout_seconds,
            "num_retries": 0,  # retries are handled exclusively by this gateway
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if creds.api_key:
            kwargs["api_key"] = creds.api_key
        if creds.api_base:
            kwargs["api_base"] = creds.api_base
        return kwargs

    def _to_response(
        self,
        response: Any,
        *,
        model: str,
        agent: str | None,
        run_id: str | None,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        provider, _ = parse_model(model)
        content = response.choices[0].message.content or ""
        usage = response.usage
        record = UsageRecord(
            agent=agent,
            model=model,
            provider=provider.value,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost_usd=self._safe_cost(response),
            latency_ms=latency_ms,
            attempts=attempts,
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            run_id=run_id,
        )
        self.tracker.record(record)
        return LLMResponse(content=content, model=model, usage=record)

    def _record_failure(
        self, *, model: str, agent: str | None, run_id: str | None,
        attempts: int, error: str, latency_ms: float,
    ) -> None:
        """Record a failed/cancelled call so physical provider requests (retry
        attempts, timeouts, cancellations) are accounted, never silently dropped.
        cost_usd stays None on failure (unknown, never 0)."""
        provider, _ = parse_model(model)
        self.tracker.record(
            UsageRecord(
                agent=agent, model=model, provider=provider.value,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_usd=None, latency_ms=latency_ms, attempts=attempts,
                timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
                run_id=run_id, error=error,
            )
        )

    @staticmethod
    def _safe_cost(response: Any) -> float | None:
        """Actual cost from LiteLLM when known; None otherwise (never 0)."""
        try:
            import litellm

            value = litellm.completion_cost(completion_response=response)
            cost = float(value)
        except Exception:
            return None
        if not math.isfinite(cost) or cost <= 0:
            return None  # unknown or free-local model: report None, not 0
        return cost


class GatewayLLMClient:
    """LLMClient bound to a single agent's model; delegates to the shared gateway."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        model: str,
        agent: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        fallback: FallbackPolicy | None = None,
        needs_tools: bool = False,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.agent = agent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback
        self.needs_tools = needs_tools

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        run_id: str | None = None,
    ) -> str:
        return self.gateway.complete(
            messages,
            model=self.model,
            agent=self.agent,
            run_id=run_id,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            fallback=self.fallback,
            needs_tools=self.needs_tools,
        )

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        run_id: str | None = None,
    ) -> str:
        return await self.gateway.acomplete(
            messages,
            model=self.model,
            agent=self.agent,
            run_id=run_id,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            fallback=self.fallback,
            needs_tools=self.needs_tools,
        )
