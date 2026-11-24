import pytest

from geosgan_hive.errors import MissingCredentialsError, UnknownProviderError
from geosgan_hive.providers import (
    PROVIDER_SPECS,
    Provider,
    parse_model,
    resolve_credentials,
)


def test_parse_model_known_providers():
    assert parse_model("openai/gpt-4o-mini") == (Provider.OPENAI, "gpt-4o-mini")
    assert parse_model("mistral/mistral-large-latest") == (Provider.MISTRAL, "mistral-large-latest")
    assert parse_model("anthropic/claude-3-7-sonnet-latest") == (
        Provider.ANTHROPIC,
        "claude-3-7-sonnet-latest",
    )
    assert parse_model("gemini/gemini-2.0-flash") == (Provider.GEMINI, "gemini-2.0-flash")
    assert parse_model("google/gemini-2.0-flash") == (Provider.GEMINI, "gemini-2.0-flash")
    assert parse_model("ollama/llama3.2") == (Provider.OLLAMA, "llama3.2")


def test_parse_model_rejects_unknown_or_malformed():
    with pytest.raises(UnknownProviderError):
        parse_model("gpt-4o-mini")  # missing provider prefix
    with pytest.raises(UnknownProviderError):
        parse_model("cohere/command-r")
    with pytest.raises(UnknownProviderError):
        parse_model("openai/")


def test_all_five_providers_registered():
    assert set(PROVIDER_SPECS) == {
        Provider.OPENAI,
        Provider.MISTRAL,
        Provider.ANTHROPIC,
        Provider.GEMINI,
        Provider.OLLAMA,
    }


def test_resolve_credentials_reads_first_available_env_var():
    creds = resolve_credentials(
        Provider.GEMINI, {"GOOGLE_API_KEY": "google-key", "GEMINI_API_KEY": "gemini-key"}
    )
    assert creds.api_key == "gemini-key"  # first listed env var wins
    creds = resolve_credentials(Provider.GEMINI, {"GOOGLE_API_KEY": "google-key"})
    assert creds.api_key == "google-key"


def test_resolve_credentials_missing_key_error_never_contains_values():
    env = {"OPENAI_API_KEY": "", "UNRELATED_SECRET": "sk-secret-value"}
    with pytest.raises(MissingCredentialsError) as exc_info:
        resolve_credentials(Provider.OPENAI, env)
    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message  # names the required variable...
    assert "sk-secret-value" not in message  # ...but never any secret value


def test_resolve_credentials_ollama_needs_no_key():
    creds = resolve_credentials(Provider.OLLAMA, {})
    assert creds.api_key is None
    assert creds.api_base == "http://localhost:11434"  # default base
    creds = resolve_credentials(Provider.OLLAMA, {"OLLAMA_API_BASE": "http://gpu:11434"})
    assert creds.api_base == "http://gpu:11434"


def test_credentials_repr_masks_api_key():
    creds = resolve_credentials(Provider.OPENAI, {"OPENAI_API_KEY": "sk-super-secret"})
    assert "sk-super-secret" not in repr(creds)
    assert "***" in repr(creds)
