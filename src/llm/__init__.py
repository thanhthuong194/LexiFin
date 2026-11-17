"""Provider-agnostic LLM access: one interface whatever serves the model.

    from src.llm import Message, get_chat, get_embedder, get_reranker

    chat = get_chat("judge")            # profile [chat.judge]
    reply = chat.generate([Message.user("...")])
    vectors = get_embedder().embed_documents(["..."])

Profiles live in `config/llm_profiles.toml` (see `config.py` for the format
and which `.env` variables each provider reads).
"""

from __future__ import annotations

from pathlib import Path

from src.llm.base import ChatModel, EmbeddingModel, Reranker
from src.llm.cache import CachedChatModel
from src.llm.config import (
    ChatProfile,
    EmbeddingProfile,
    Profiles,
    RerankProfile,
    load_environment,
    load_profiles,
)
from src.llm.structured import generate_structured, strict_json_schema, tool_from_model
from src.llm.types import (
    ChatOptions,
    ChatResult,
    ImageBlock,
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMRefusalError,
    LLMResponseError,
    Message,
    OutputSchema,
    RerankResult,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)

__all__ = [
    "ChatModel",
    "ChatOptions",
    "ChatResult",
    "EmbeddingModel",
    "ImageBlock",
    "LLMConfigError",
    "LLMError",
    "LLMProviderError",
    "LLMRefusalError",
    "LLMResponseError",
    "Message",
    "OutputSchema",
    "RerankResult",
    "Reranker",
    "TextBlock",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolSpec",
    "Usage",
    "build_chat",
    "build_embedder",
    "build_reranker",
    "generate_structured",
    "get_chat",
    "get_embedder",
    "get_reranker",
    "strict_json_schema",
    "tool_from_model",
]


def _profiles(profiles: Profiles | None) -> Profiles:
    load_environment()
    return profiles or load_profiles()


def _lookup[P](table: dict[str, P], kind: str, name: str, source: Path) -> P:
    if name not in table:
        known = ", ".join(sorted(table)) or "none"
        raise LLMConfigError(f"No profile [{kind}.{name}] in {source} (have: {known})")
    return table[name]


def get_chat(
    name: str = "default",
    *,
    profiles: Profiles | None = None,
    cache_dir: Path | None = None,
) -> ChatModel:
    """The chat model of profile `[chat.<name>]`; with `cache_dir`, replies
    are cached on disk (identical requests are answered from the cache)."""

    loaded = _profiles(profiles)
    profile = _lookup(loaded.chat, "chat", name, loaded.source)
    chat = build_chat(profile)
    if cache_dir is None:
        return chat
    settings = profile.model_dump_json(exclude={"timeout_s", "max_retries"})
    return CachedChatModel(chat, cache_dir, settings)


def get_embedder(
    name: str = "default", *, profiles: Profiles | None = None
) -> EmbeddingModel:
    loaded = _profiles(profiles)
    return build_embedder(_lookup(loaded.embedding, "embedding", name, loaded.source))


def get_reranker(
    name: str = "default", *, profiles: Profiles | None = None
) -> Reranker:
    loaded = _profiles(profiles)
    return build_reranker(_lookup(loaded.rerank, "rerank", name, loaded.source))


def build_chat(profile: ChatProfile) -> ChatModel:
    """Adapter for a profile; SDKs are imported only when their provider is used."""

    if profile.provider == "anthropic":
        from src.llm.providers import anthropic

        return anthropic.AnthropicChat(profile, anthropic.create_client(profile))
    if profile.provider == "gemini":
        from src.llm.providers import gemini

        return gemini.GeminiChat(profile, gemini.create_client(profile))
    from src.llm.providers import openai_compat

    return openai_compat.OpenAIChat(profile, openai_compat.create_client(profile))


def build_embedder(profile: EmbeddingProfile) -> EmbeddingModel:
    if profile.provider == "gemini":
        from src.llm.providers import gemini

        return gemini.GeminiEmbedding(profile, gemini.create_client(profile))
    from src.llm.providers import openai_compat

    return openai_compat.OpenAIEmbedding(profile, openai_compat.create_client(profile))


def build_reranker(profile: RerankProfile) -> Reranker:
    from src.llm.providers import http_rerank

    return http_rerank.HttpReranker(profile, http_rerank.create_client(profile))
