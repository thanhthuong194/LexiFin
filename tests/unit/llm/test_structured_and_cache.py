from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from src.llm import (
    LLMRefusalError,
    LLMResponseError,
    Message,
    generate_structured,
    strict_json_schema,
    tool_from_model,
)
from src.llm.cache import CachedChatModel
from tests.llm_support import FakeChat, FakeEmbedding, FakeReranker, reply

pytestmark = pytest.mark.unit


class Evidence(BaseModel):
    block_id: str
    quote: str = Field(min_length=3)


class Verdict(BaseModel):
    correct: bool
    score: int = Field(default=0, ge=0, le=2)
    evidence: list[Evidence] = Field(max_length=5)
    note: str | None = None


def test_strict_schema_closes_objects_and_drops_constraints() -> None:
    schema = strict_json_schema(Verdict)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["correct", "score", "evidence", "note"]
    assert "default" not in schema["properties"]["score"]
    assert "maximum" not in schema["properties"]["score"]
    assert "maxItems" not in schema["properties"]["evidence"]
    evidence = schema["$defs"]["Evidence"]
    assert evidence["additionalProperties"] is False
    assert evidence["required"] == ["block_id", "quote"]
    assert "minLength" not in evidence["properties"]["quote"]


def test_tool_from_model_is_strict() -> None:
    tool = tool_from_model("record_verdict", "Store a verdict", Verdict)

    assert tool.strict
    assert tool.parameters == strict_json_schema(Verdict)


def test_generate_structured_validates_the_reply() -> None:
    chat = FakeChat(
        [reply('{"correct": true, "score": 2, "evidence": [], "note": null}')]
    )

    verdict, result = generate_structured(chat, [Message.user("judge")], Verdict)

    assert verdict == Verdict(correct=True, score=2, evidence=[])
    assert result.stop_reason == "end"
    options = chat.calls[0][1]
    assert options is not None
    assert options.output_schema is not None
    assert options.output_schema.name == "Verdict"


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (reply("no", stop_reason="refusal"), LLMRefusalError, "refused"),
        (reply('{"correct"', stop_reason="max_tokens"), LLMResponseError, "max_tokens"),
        (
            reply('{"correct": true, "score": 7, "evidence": []}'),
            LLMResponseError,
            "Verdict",
        ),
    ],
)
def test_generate_structured_failures(
    result: object, error: type[Exception], message: str
) -> None:
    chat = FakeChat([result])  # type: ignore[list-item]

    with pytest.raises(error, match=message):
        generate_structured(chat, [Message.user("judge")], Verdict)


def test_cache_answers_repeated_requests_from_disk(tmp_path: Path) -> None:
    inner = FakeChat(
        [reply("first"), reply("second"), reply("cut", stop_reason="max_tokens")]
    )
    cached = CachedChatModel(inner, tmp_path, settings="v1")
    question = [Message.user("q")]

    assert cached.generate(question).text == "first"
    assert cached.generate(question).text == "first"
    assert len(inner.calls) == 1
    assert (cached.provider, cached.model) == ("fake", "fake-model")

    other_settings = CachedChatModel(inner, tmp_path, settings="v2")
    assert other_settings.generate(question).text == "second"

    assert cached.generate([Message.user("other")]).text == "cut"
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_cache_ignores_corrupt_entries(tmp_path: Path) -> None:
    inner = FakeChat([reply("a"), reply("b")])
    cached = CachedChatModel(inner, tmp_path, settings="v1")
    cached.generate([Message.user("q")])
    for path in tmp_path.rglob("*.json"):
        path.write_text("{broken", encoding="utf-8")

    assert cached.generate([Message.user("q")]).text == "b"


def test_fakes_follow_the_interfaces() -> None:
    assert FakeEmbedding().embed_queries(["ab"]) == [[2.0, 0.0, 1.0]]
    assert FakeEmbedding().embed_documents(["ab"]) == [[2.0, 0.0, 0.0]]
    ranked = FakeReranker().rerank("net revenue", ["cost", "net revenue grew"], top_n=1)
    assert [r.index for r in ranked] == [1]
