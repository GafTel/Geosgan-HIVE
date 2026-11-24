from __future__ import annotations


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


class TransientGatewayError(GatewayError):
    """Retryable failure (rate limit, 5xx, timeout, connection)."""


class RateLimitError(TransientGatewayError):
    """The provider rate-limited the call (HTTP 429)."""


class ServiceUnavailableError(TransientGatewayError):
    """The provider returned a transient server error (5xx)."""


class GatewayTimeoutError(TransientGatewayError):
    """The call timed out."""


class GatewayConnectionError(TransientGatewayError):
    """The connection to the provider failed."""


class PermanentGatewayError(GatewayError):
    """Non-retryable failure (authentication, bad request, ...)."""


class AuthenticationError(PermanentGatewayError):
    """The provider rejected the credentials (HTTP 401/403)."""


class BadRequestError(PermanentGatewayError):
    """The request itself is invalid (HTTP 400, context window, ...)."""


class BudgetExceededError(HiveError):
    """The known cost of a run reached its configured per-run budget."""
