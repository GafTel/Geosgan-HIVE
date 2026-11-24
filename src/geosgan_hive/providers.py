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
