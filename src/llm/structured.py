"""Typed JSON replies: a pydantic model in, a validated instance out."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm.base import ChatModel
from src.llm.types import (
    ChatOptions,
    ChatResult,
    LLMRefusalError,
    LLMResponseError,
    Message,
    OutputSchema,
    ToolSpec,
)

# Keywords the strict subsets of OpenAI/Anthropic reject; pydantic still
# enforces them when the reply is validated.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of `model` in the strict subset every provider accepts:
    every object closed (`additionalProperties: false`) with all its
    properties required, value constraints removed."""

    schema: dict[str, Any] = _strict(model.model_json_schema())
    return schema


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    strict = {
        key: (value if key == "properties" else _strict(value))
        for key, value in node.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }
    if "properties" in strict:
        strict["properties"] = {
            name: _strict(schema) for name, schema in strict["properties"].items()
        }
        strict["required"] = list(strict["properties"])
    if strict.get("type") == "object" or "properties" in strict:
        strict["additionalProperties"] = False
    return strict


def tool_from_model(name: str, description: str, model: type[BaseModel]) -> ToolSpec:
    """A strict tool whose arguments are validated by `model`."""

    return ToolSpec(
        name=name,
        description=description,
        parameters=strict_json_schema(model),
        strict=True,
    )


def generate_structured[T: BaseModel](
    chat: ChatModel,
    messages: Sequence[Message],
    schema: type[T],
    options: ChatOptions | None = None,
) -> tuple[T, ChatResult]:
    """Ask for JSON matching `schema`; raise unless the reply validates."""

    output = OutputSchema(name=schema.__name__, json_schema=strict_json_schema(schema))
    request = (options or ChatOptions()).model_copy(update={"output_schema": output})
    result = chat.generate(messages, request)
    if result.stop_reason == "refusal":
        raise LLMRefusalError(f"{chat.model} refused ({result.raw_stop_reason})")
    if result.stop_reason == "max_tokens":
        raise LLMResponseError(f"{chat.model}: reply cut at max_tokens")
    try:
        return schema.model_validate_json(result.text), result
    except ValidationError as error:
        raise LLMResponseError(
            f"{chat.model}: reply does not match {schema.__name__}: {error}"
        ) from error
