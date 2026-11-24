from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .errors import MissingCredentialsError, UnknownProviderError


class Provider(StrEnum):
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ProviderSpec:
    provider: Provider
    litellm_prefix: str
    api_key_env_vars: tuple[str, ...] = ()
    api_base_env_var: str | None = None
    default_api_base: str | None = None
    requires_api_key: bool = True


PROVIDER_SPECS: dict[Provider, ProviderSpec] = {
    Provider.OPENAI: ProviderSpec(Provider.OPENAI, "openai/", ("OPENAI_API_KEY",)),
    Provider.MISTRAL: ProviderSpec(Provider.MISTRAL, "mistral/", ("MISTRAL_API_KEY",)),
    Provider.ANTHROPIC: ProviderSpec(Provider.ANTHROPIC, "anthropic/", ("ANTHROPIC_API_KEY",)),
    Provider.GEMINI: ProviderSpec(
        Provider.GEMINI, "gemini/", ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    ),
    Provider.OLLAMA: ProviderSpec(
        Provider.OLLAMA,
        "ollama/",
        api_base_env_var="OLLAMA_API_BASE",
        default_api_base="http://localhost:11434",
        requires_api_key=False,
    ),
}

_PREFIX_TO_PROVIDER: dict[str, Provider] = {
    "openai": Provider.OPENAI,
    "mistral": Provider.MISTRAL,
    "anthropic": Provider.ANTHROPIC,
    "gemini": Provider.GEMINI,
    "google": Provider.GEMINI,  # alias: LiteLLM also accepts gemini/ for Google models
    "ollama": Provider.OLLAMA,
}


def parse_model(model: str) -> tuple[Provider, str]:
    """Split a '<provider>/<model>' string and validate the provider prefix."""
    if not model or "/" not in model:
        raise UnknownProviderError(
            f"Invalid model {model!r}: expected '<provider>/<model>' (e.g. 'openai/gpt-4o-mini')."
        )
    prefix, name = model.split("/", 1)
    provider = _PREFIX_TO_PROVIDER.get(prefix.lower())
    if provider is None or not name:
        supported = ", ".join(sorted(_PREFIX_TO_PROVIDER))
        raise UnknownProviderError(
            f"Unknown provider prefix {prefix!r} in model {model!r}. "
            f"Supported prefixes: {supported}."
        )
    return provider, name


@dataclass(frozen=True)
class ModelCapabilities:
    """Best-effort static capability info for a model.

    `context_window=None` means UNKNOWN (never blocks a candidate); unknown is
    not treated as unlimited for cost/capability claims, only as "no data".
    """

    context_window: int | None = None
    supports_tools: bool = True


#: Small table of well-known public context windows. Everything not listed gets
#: context_window=None (unknown -> the capability gate does not reject on it).
_KNOWN_CAPABILITIES: dict[str, ModelCapabilities] = {
    "openai/gpt-4o": ModelCapabilities(context_window=128_000),
    "openai/gpt-4o-mini": ModelCapabilities(context_window=128_000),
    "openai/gpt-4-turbo": ModelCapabilities(context_window=128_000),
    "mistral/mistral-large-latest": ModelCapabilities(context_window=128_000),
    "mistral/mistral-small-latest": ModelCapabilities(context_window=128_000),
    "anthropic/claude-3-5-sonnet-latest": ModelCapabilities(context_window=200_000),
    "anthropic/claude-3-5-haiku-latest": ModelCapabilities(context_window=200_000),
    "anthropic/claude-3-opus-latest": ModelCapabilities(context_window=200_000),
    "gemini/gemini-2.0-flash": ModelCapabilities(context_window=1_048_576),
    "gemini/gemini-1.5-pro": ModelCapabilities(context_window=2_097_152),
}


def model_capabilities(model: str) -> ModelCapabilities:
    """Capability lookup with conservative defaults for unknown models."""
    return _KNOWN_CAPABILITIES.get(model, ModelCapabilities())


#: Provider-level cost tier. Ollama runs locally (no per-token billing) = free.
#: Hosted API providers = paid. Anything else = unknown (and unknown is NOT zero).
_PROVIDER_COST_TIER: dict[Provider, str] = {
    Provider.OLLAMA: "free",
    Provider.OPENAI: "paid",
    Provider.MISTRAL: "paid",
    Provider.ANTHROPIC: "paid",
    Provider.GEMINI: "paid",
}

#: Explicit per-model overrides (checked before the provider default).
_MODEL_COST_OVERRIDES: dict[str, str] = {}


def cost_tier(model: str) -> str:
    """Return 'free', 'paid' or 'unknown' for a '<provider>/<model>' string.

    Never inferred from the model NAME alone: it is derived from the provider's
    billing model plus explicit overrides. Unknown pricing is 'unknown', not 0.
    """
    provider, _ = parse_model(model)
    if model in _MODEL_COST_OVERRIDES:
        return _MODEL_COST_OVERRIDES[model]
    return _PROVIDER_COST_TIER.get(provider, "unknown")


@dataclass(frozen=True)
class Credentials:
    provider: Provider
    api_key: str | None
    api_base: str | None

    def __repr__(self) -> str:
        # Never leak the key, even in logs or tracebacks.
        key = "***" if self.api_key else None
        return (
            f"Credentials(provider={self.provider.value!r}, api_key={key!r}, "
            f"api_base={self.api_base!r})"
        )


def resolve_credentials(
    provider: Provider, environ: Mapping[str, str] | None = None
) -> Credentials:
    """Resolve credentials for a provider from the environment.

    Only providers that are actually used ever get here, so only their
    credentials are required. Ollama needs no key, only an optional base URL.
    """
    env = os.environ if environ is None else environ
    spec = PROVIDER_SPECS[provider]
    api_key: str | None = None
    for var in spec.api_key_env_vars:
        value = env.get(var)
        if value:
            api_key = value
            break
    if spec.requires_api_key and api_key is None:
        vars_list = ", ".join(spec.api_key_env_vars)
        raise MissingCredentialsError(
            f"Provider '{provider.value}' requires an API key: set one of {vars_list} "
            "in the environment (never put keys in YAML config files)."
        )
    api_base: str | None = None
    if spec.api_base_env_var is not None:
        api_base = env.get(spec.api_base_env_var) or spec.default_api_base
    elif spec.default_api_base is not None:
        api_base = spec.default_api_base
    return Credentials(provider=provider, api_key=api_key, api_base=api_base)
