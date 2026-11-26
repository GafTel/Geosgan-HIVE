from __future__ import annotations

import asyncio
from enum import StrEnum

# ---------------------------------------------------------------------------
# Phase 2.5: normalized provider error taxonomy
# ---------------------------------------------------------------------------


class ErrorClass(StrEnum):
    """Normalized provider-error category (independent of the LiteLLM class)."""

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    OUTAGE = "outage"  # provider 5xx
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    CAPABILITY = "capability"  # model cannot serve this request shape
    CONTEXT_WINDOW = "context_window"
    BUDGET = "budget"  # run-level budget exhausted
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # WAIT/HYBRID suspension signal
    FALLBACK_EXHAUSTED = "fallback_exhausted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def classify_error(exc: BaseException) -> tuple[ErrorClass, bool, bool, bool]:
    """Return (error_class, retryable, fallback_eligible, terminal).

    - retryable: may be retried on the same model (transient).
    - fallback_eligible: an eligible alternative model may be tried.
    - terminal: run-level terminal condition — never retry, never fallback
      (cancellation, exhausted run budget).
    Unknown errors are classified conservative-permanent (no retry, no fallback).
    """
    if isinstance(exc, asyncio.CancelledError):
        return (ErrorClass.CANCELLED, False, False, True)
    cls = getattr(exc, "error_class", ErrorClass.UNKNOWN)
    return (
        cls,
        bool(getattr(exc, "retryable", False)),
        bool(getattr(exc, "fallback_eligible", False)),
        bool(getattr(exc, "terminal", False)),
    )


class HiveError(Exception):
    """Base error for geosgan_hive."""


class ConfigError(HiveError):
    """Invalid configuration (YAML file, env overrides, agent specs)."""


class MissingCredentialsError(ConfigError):
    """A required provider credential is missing from the environment.

    Messages must name the environment variables, never their values.
    """


class UnknownProviderError(ConfigError):
    """A model string uses a provider prefix that is not registered."""


class GatewayError(HiveError):
    """A model call through the gateway failed."""

    error_class = ErrorClass.UNKNOWN
    retryable = False
    fallback_eligible = False
    terminal = False


class TransientGatewayError(GatewayError):
    """Retryable failure (rate limit, 5xx, timeout, connection)."""

    retryable = True


class RateLimitError(TransientGatewayError):
    """The provider rate-limited the call (HTTP 429)."""

    error_class = ErrorClass.RATE_LIMIT
    fallback_eligible = True


class QuotaExhaustedError(GatewayError):
    """The provider/account quota is exhausted. Fallback-eligible, not retryable
    (we never invent quota reset timestamps)."""

    error_class = ErrorClass.QUOTA
    retryable = False
    fallback_eligible = True


class ServiceUnavailableError(TransientGatewayError):
    """The provider returned a transient server error (5xx)."""

    error_class = ErrorClass.OUTAGE
    fallback_eligible = True


class GatewayTimeoutError(TransientGatewayError):
    """The call timed out."""

    error_class = ErrorClass.TIMEOUT
    fallback_eligible = True


class GatewayConnectionError(TransientGatewayError):
    """The connection to the provider failed."""

    error_class = ErrorClass.NETWORK
    fallback_eligible = True


class PermanentGatewayError(GatewayError):
    """Non-retryable failure (authentication, bad request, ...)."""


class AuthenticationError(PermanentGatewayError):
    """The provider rejected the credentials (HTTP 401/403). Never retried,
    never fallback-eligible, and never opens a provider-wide circuit (it is
    credential-specific, not provider-wide)."""

    error_class = ErrorClass.AUTH


class BadRequestError(PermanentGatewayError):
    """The request itself is invalid (HTTP 400, ...)."""

    error_class = ErrorClass.INVALID_REQUEST


class ContextWindowExceededError(BadRequestError):
    """The prompt exceeds the model's context window. Fallback-eligible (another
    model may have a larger window), never retried on the same model."""

    error_class = ErrorClass.CONTEXT_WINDOW
    fallback_eligible = True


class CapabilityMismatchError(PermanentGatewayError):
    """The target model cannot serve this request (tools, modality, ...)."""

    error_class = ErrorClass.CAPABILITY


class FallbackExhaustedError(GatewayError):
    """The ordered fallback chain was exhausted for this logical call."""

    error_class = ErrorClass.FALLBACK_EXHAUSTED


class ProviderUnavailableError(GatewayError):
    """WAIT/HYBRID signal: no eligible model is currently available; the run may
    be suspended at a safe boundary and resumed later."""

    error_class = ErrorClass.PROVIDER_UNAVAILABLE


class BudgetExceededError(HiveError):
    """The known cost of a run reached its configured per-run budget, or a
    dispatch budget (logical/physical) was exhausted — run-terminal."""

    error_class = ErrorClass.BUDGET
    terminal = True


class CheckpointError(HiveError):
    """Checkpoint store failure (corrupt, locked, incompatible schema, ...)."""


class DelegationError(HiveError):
    """Base error for collaboration/delegation failures."""


class DelegationRejectedError(DelegationError):
    """A handoff request was rejected by policy or permissions."""




# ---------------------------------------------------------------------------
# Phase 3: tool engine errors
# ---------------------------------------------------------------------------


class ToolError(HiveError):
    """Base error for the Tool Engine. Gateway converts these into structured
    ToolResult errors (the agent loop continues); they never execute anything."""


class ToolNotFoundError(ToolError):
    """Unknown tool name — fail closed."""


class ToolPermissionError(ToolError):
    """Denied by policy / missing security context — fail closed."""


class ToolValidationError(ToolError):
    """Arguments failed schema or size validation (incl. unexpected arguments)."""


class ToolBudgetExceededError(ToolError):
    """A tool budget (run/agent/tool) was exhausted — run-terminal for tools."""

    error_class = ErrorClass.BUDGET
    terminal = True


class ToolTimeoutError(ToolError):
    """The tool execution exceeded its timeout. For mutating tools the outcome is
    UNCERTAIN (a sync thread cannot be killed); occupancy is kept until the
    underlying operation actually terminates."""


class ApprovalRequiredError(ToolError):
    """The action requires human approval; carries the pending approval id."""

    def __init__(self, message: str, *, approval_id: str = "") -> None:
        super().__init__(message)
        self.approval_id = approval_id


class ApprovalError(ToolError):
    """Approval lifecycle failure (unknown/expired/denied/replay/substitution)."""


class SandboxViolationError(ToolError):
    """A file operation violated workspace isolation (traversal, symlink escape,
    special file, denied name, outside root) — fail closed."""


# ---------------------------------------------------------------------------
# Phase 4 (C2): zero-cost cloud engine
# ---------------------------------------------------------------------------

class CloudConfigError(ConfigError):
    """Malformed/unauthorized cloud provider profile or cloud config block."""


class CostUnknownError(HiveError):
    """A cost or token usage that cannot be bounded/known — in zero-cost mode
    this fails closed BEFORE any live dispatch. Unknown is never treated as 0."""

    error_class = ErrorClass.UNKNOWN
    terminal = False


class CloudOfflineError(HiveError):
    """A strictly-offline (fake) adapter was asked to perform a live dispatch."""
