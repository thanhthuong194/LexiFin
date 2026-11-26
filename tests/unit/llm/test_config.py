from __future__ import annotations

from pathlib import Path

import pytest

from src.llm import (
    ChatOptions,
    Message,
    build_chat,
    get_chat,
    get_embedder,
    get_reranker,
)
from src.llm.cache import CachedChatModel
from src.llm.config import (
    DEFAULT_PROFILES_PATH,
    ChatProfile,
    RerankProfile,
    load_profiles,
    resolve_endpoint,
)
from src.llm.providers.anthropic import AnthropicChat
from src.llm.providers.gemini import GeminiChat, GeminiEmbedding
from src.llm.providers.http_rerank import HttpReranker
from src.llm.providers.openai_compat import OpenAIChat, OpenAIEmbedding
from src.llm.throttle import ThrottledChatModel
from src.llm.types import LLMConfigError
from tests.llm_support import FakeChat, reply

pytestmark = pytest.mark.unit

PROFILES = """
[defaults]
timeout_s = 30
max_retries = 1

[chat.default]
provider = "anthropic"
model = "claude-opus-5-5"
effort = "high"

[chat.judge-local]
provider = "openai_compat"
model = "Qwen/Qwen3-32B"
timeout_s = 300
options = { chat_template_kwargs = { enable_thinking = false } }

[chat.gemini]
provider = "gemini"
model = "gemini-2.5-pro"

[embedding.default]
provider = "openai_compat"
model = "Qwen/Qwen3-Embedding-0.6B"
batch_size = 8

[rerank.default]
provider = "http_rerank"
model = "Qwen/Qwen3-Reranker-0.6B"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "profiles.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_profiles_merge_defaults_and_keep_role_names(tmp_path: Path) -> None:
    profiles = load_profiles(write(tmp_path, PROFILES))

    default = profiles.chat["default"]
    local = profiles.chat["judge-local"]
    assert (default.provider, default.model, default.effort) == (
        "anthropic",
        "claude-opus-5-5",
        "high",
    )
    assert (default.timeout_s, default.max_retries) == (30, 1)
    assert local.timeout_s == 300
    assert local.options == {"chat_template_kwargs": {"enable_thinking": False}}
    assert local.env_prefix == "CHAT_JUDGE_LOCAL"
    assert profiles.embedding["default"].batch_size == 8
    assert profiles.rerank["default"].model == "Qwen/Qwen3-Reranker-0.6B"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[chat.x]\nprovider='anthropic'\nmodel='m'\ntemprature=0", "temprature"),
        ("[chat.x]\nprovider='anthropic'\nmodel='m'\ntemperature=0", "temperature"),
        ("[chat.x]\nprovider='nope'\nmodel='m'", "provider"),
        ("[chat.x]\nprovider='openai'", "model"),
        ("[embeddings.x]\nprovider='openai'\nmodel='m'", "Unknown sections"),
        ("[chat]\nx = 1", "must be a table"),
        ("[chat.x\n", "Invalid TOML"),
    ],
)
def test_invalid_profiles_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(LLMConfigError, match=message):
        load_profiles(write(tmp_path, text))


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigError, match="not found"):
        load_profiles(tmp_path / "missing.toml")


def test_profiles_path_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, PROFILES)
    monkeypatch.setenv("LLM_PROFILES_PATH", str(path))

    assert load_profiles().source == path


def test_repository_profiles_file_is_valid() -> None:
    profiles = load_profiles(DEFAULT_PROFILES_PATH)

    assert {"default"} <= set(profiles.chat)
    assert {"default"} <= set(profiles.embedding)
    assert {"default"} <= set(profiles.rerank)


def test_self_hosted_endpoint_comes_from_profile_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ChatProfile(
        kind="chat", name="local", provider="openai_compat", model="m"
    )
    monkeypatch.setenv("CHAT_LOCAL_BASE_URL", "https://abc.trycloudflare.com/v1/")
    monkeypatch.setenv("CHAT_LOCAL_API_KEY", "secret")

    endpoint = resolve_endpoint(profile)

    assert endpoint.base_url == "https://abc.trycloudflare.com/v1"
    assert endpoint.api_key == "secret"


def test_self_hosted_endpoint_never_falls_back_to_the_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = RerankProfile(
        kind="rerank", name="default", provider="http_rerank", model="m"
    )
    monkeypatch.setenv("RERANK_DEFAULT_BASE_URL", "http://host:8000/v1")
    monkeypatch.delenv("RERANK_DEFAULT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")

    assert resolve_endpoint(profile).api_key == "unused"


def test_self_hosted_endpoint_without_url_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ChatProfile(
        kind="chat", name="local", provider="openai_compat", model="m"
    )
    monkeypatch.delenv("CHAT_LOCAL_BASE_URL", raising=False)

    with pytest.raises(LLMConfigError, match="CHAT_LOCAL_BASE_URL"):
        resolve_endpoint(profile)


def test_cloud_endpoint_uses_sdk_defaults() -> None:
    profile = ChatProfile(kind="chat", name="x", provider="openai", model="m")

    endpoint = resolve_endpoint(profile)

    assert (endpoint.base_url, endpoint.api_key) == (None, None)


def test_registry_builds_the_adapter_of_each_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = load_profiles(write(tmp_path, PROFILES))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("CHAT_JUDGE_LOCAL_BASE_URL", "http://host:8000/v1")
    monkeypatch.setenv("EMBEDDING_DEFAULT_BASE_URL", "http://host:8001/v1")
    monkeypatch.setenv("RERANK_DEFAULT_BASE_URL", "http://host:8002/v1")

    assert isinstance(get_chat(profiles=profiles), AnthropicChat)
    local = get_chat("judge-local", profiles=profiles)
    assert isinstance(local, OpenAIChat)
    assert local.provider == "openai_compat"
    assert isinstance(get_chat("gemini", profiles=profiles), GeminiChat)
    assert isinstance(get_embedder(profiles=profiles), OpenAIEmbedding)
    assert isinstance(get_reranker(profiles=profiles), HttpReranker)
    openai_profile = ChatProfile(kind="chat", name="o", provider="openai", model="m")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert build_chat(openai_profile).provider == "openai"


def test_unknown_profile_lists_the_known_ones(tmp_path: Path) -> None:
    profiles = load_profiles(write(tmp_path, PROFILES))

    with pytest.raises(LLMConfigError, match="have: default, gemini, judge-local"):
        get_chat("judge", profiles=profiles)


def test_registry_wraps_cached_chat_and_builds_gemini_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (
        PROFILES
        + "\n[embedding.gemini]\nprovider = 'gemini'\nmodel = 'gemini-embedding-001'\n"
    )
    profiles = load_profiles(write(tmp_path, text))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    cached = get_chat(profiles=profiles, cache_dir=tmp_path / "cache")

    assert isinstance(cached, CachedChatModel)
    assert (cached.provider, cached.model) == ("anthropic", "claude-opus-5-5")
    assert isinstance(get_embedder("gemini", profiles=profiles), GeminiEmbedding)


def test_cache_key_ignores_the_served_name(tmp_path: Path) -> None:
    base = "[chat.query]\nprovider = 'openai_compat'\nmodel = 'openai/gpt-oss-120b'\n"
    keys = []
    for served in ("", "served_model = 'gpt-oss-120b'\n"):
        profiles = load_profiles(write(tmp_path, base + served))
        cached = get_chat("query", profiles=profiles, cache_dir=tmp_path / "cache")
        assert isinstance(cached, CachedChatModel)
        keys.append(cached._key([Message.user("q")], ChatOptions()))
    assert keys[0] == keys[1]


def test_throttle_spaces_calls_and_skips_cache_hits(tmp_path: Path) -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    inner = FakeChat([reply("a"), reply("b"), reply("c")])
    throttled = ThrottledChatModel(inner, 6, clock=lambda: now[0], sleep=sleep)
    throttled.generate([Message.user("1")])
    now[0] += 4.0
    throttled.generate([Message.user("2")])
    throttled.generate([Message.user("3")])
    assert slept == [6.0, 10.0]  # 10 s apart at 6 per minute
    text = (
        "[chat.judge]\nprovider = 'openai_compat'\nmodel = 'm'\n"
        "requests_per_minute = 6\n"
    )
    profiles = load_profiles(write(tmp_path, text))
    cached = get_chat("judge", profiles=profiles, cache_dir=tmp_path / "cache")
    assert isinstance(cached, CachedChatModel)
    assert isinstance(cached._inner, ThrottledChatModel)
