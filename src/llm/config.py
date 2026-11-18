"""Profiles from `config/llm_profiles.toml`; URLs and keys from `.env`.

A profile is `[<kind>.<name>]` with kind `chat`, `embedding` or `rerank`;
`[defaults]` fills keys every profile leaves out. Names are roles
(`default`, `judge`...), so changing a model is a one-line config edit.

Secrets never live in the file. Cloud providers (`openai`, `anthropic`,
`gemini`) use their SDK's default URL and standard key variable
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY). Self-hosted providers
(`openai_compat`, `http_rerank`) read `<KIND>_<NAME>_BASE_URL` and the
optional `<KIND>_<NAME>_API_KEY`, e.g. CHAT_JUDGE_BASE_URL.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog
from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.llm.types import LLMConfigError

logger = structlog.get_logger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_PATH = REPOSITORY_ROOT / "config" / "llm_profiles.toml"
PROFILES_PATH_ENV = "LLM_PROFILES_PATH"
SELF_HOSTED_PROVIDERS = frozenset({"openai_compat", "http_rerank"})

Kind = Literal["chat", "embedding", "rerank"]


class _Profile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    name: str
    model: str = Field(min_length=1)
    # Name the server knows the model by when it differs from `model`
    # (Cloudflare `@cf/...`); `model` stays the identity stored with results.
    served_model: str | None = Field(default=None, min_length=1)
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    # Merged into every request body as is (provider-specific features).
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def request_model(self) -> str:
        return self.served_model or self.model

    @property
    def env_prefix(self) -> str:
        return re.sub(r"[^A-Z0-9]", "_", f"{self.kind}_{self.name}".upper())


class ChatProfile(_Profile):
    provider: Literal["openai", "openai_compat", "anthropic", "gemini"]
    max_tokens: int = Field(default=4096, ge=1)
    # Only sent when set; Anthropic's current API rejects sampling parameters.
    temperature: float | None = Field(default=None, ge=0)
    # Anthropic output_config.effort, OpenAI reasoning_effort,
    # Gemini thinking_level.
    effort: str | None = None

    @model_validator(mode="after")
    def _check_temperature(self) -> ChatProfile:
        if self.provider == "anthropic" and self.temperature is not None:
            raise ValueError("anthropic takes no temperature; use effort")
        return self


class EmbeddingProfile(_Profile):
    provider: Literal["openai", "openai_compat", "gemini"]
    batch_size: int = Field(default=64, ge=1)
    # Requested vector size (Matryoshka models); None = the model's own.
    dimensions: int | None = Field(default=None, ge=1)
    # Qwen3-Embedding style: queries become "Instruct: {i}\nQuery:{q}".
    query_instruction: str | None = None


class RerankProfile(_Profile):
    provider: Literal["http_rerank"]
    # Input layout the model was trained on. "qwen3": Qwen3-Reranker is a
    # yes/no judge (vLLM serves it as a classifier) and scores bare text
    # almost at random; query and documents get its chat template. None:
    # sent as is (bge, Jina, Cohere...).
    prompt_format: Literal["qwen3"] | None = None
    # Task line of instruction-following formats (Qwen3 `<Instruct>`).
    instruction: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_instruction(self) -> RerankProfile:
        if self.instruction and not self.prompt_format:
            raise ValueError("instruction needs a prompt_format that uses it")
        return self


Profile = ChatProfile | EmbeddingProfile | RerankProfile
_PROFILE_TYPES: dict[str, type[_Profile]] = {
    "chat": ChatProfile,
    "embedding": EmbeddingProfile,
    "rerank": RerankProfile,
}


@dataclass(frozen=True)
class Endpoint:
    """Where a profile sends requests; None means the SDK default."""

    base_url: str | None
    api_key: str | None


@dataclass(frozen=True)
class Profiles:
    chat: dict[str, ChatProfile]
    embedding: dict[str, EmbeddingProfile]
    rerank: dict[str, RerankProfile]
    source: Path


def load_environment() -> None:
    """Load the repository `.env` (variables already set win). A variable set
    in the shell with another value than `.env` is logged by name: a stale
    export hides a new tunnel URL pasted into `.env`."""

    path = REPOSITORY_ROOT / ".env"
    stale = sorted(
        key
        for key, value in dotenv_values(path).items()
        if key in os.environ and value is not None and os.environ[key] != value
    )
    if stale:
        logger.warning("llm_env_differs_from_dotenv", variables=stale)
    load_dotenv(path, override=False)


def profiles_path() -> Path:
    configured = os.environ.get(PROFILES_PATH_ENV)
    return Path(configured) if configured else DEFAULT_PROFILES_PATH


def load_profiles(path: Path | None = None) -> Profiles:
    source = path or profiles_path()
    try:
        document = tomllib.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LLMConfigError(f"LLM profiles file not found: {source}") from error
    except tomllib.TOMLDecodeError as error:
        raise LLMConfigError(f"Invalid TOML in {source}: {error}") from error
    defaults = document.pop("defaults", {})
    unknown = set(document) - set(_PROFILE_TYPES)
    if unknown:
        raise LLMConfigError(f"Unknown sections in {source}: {sorted(unknown)}")
    parsed: dict[str, dict[str, Any]] = {kind: {} for kind in _PROFILE_TYPES}
    for kind, entries in document.items():
        for name, values in entries.items():
            if not isinstance(values, dict):
                raise LLMConfigError(f"[{kind}.{name}] in {source} must be a table")
            parsed[kind][name] = _parse_profile(kind, name, defaults | values)
    return Profiles(
        chat=parsed["chat"],
        embedding=parsed["embedding"],
        rerank=parsed["rerank"],
        source=source,
    )


def _parse_profile(kind: str, name: str, values: dict[str, Any]) -> _Profile:
    try:
        return _PROFILE_TYPES[kind].model_validate(
            values | {"kind": kind, "name": name}
        )
    except ValidationError as error:
        raise LLMConfigError(f"Invalid profile [{kind}.{name}]: {error}") from error


def resolve_endpoint(profile: Profile) -> Endpoint:
    """URL and key of a self-hosted profile; SDK defaults for cloud ones.

    A self-hosted profile without its own key gets the placeholder
    "unused", never None: the OpenAI SDK would otherwise fall back to
    OPENAI_API_KEY and send the real key to the self-hosted server.
    """

    if profile.provider not in SELF_HOSTED_PROVIDERS:
        return Endpoint(base_url=None, api_key=None)
    url_variable = f"{profile.env_prefix}_BASE_URL"
    base_url = os.environ.get(url_variable, "").strip()
    if not base_url:
        raise LLMConfigError(
            f"Profile [{profile.kind}.{profile.name}] needs {url_variable} in .env"
        )
    api_key = os.environ.get(f"{profile.env_prefix}_API_KEY", "").strip()
    return Endpoint(base_url=base_url.rstrip("/"), api_key=api_key or "unused")
