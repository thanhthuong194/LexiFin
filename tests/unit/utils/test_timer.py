from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

import src.utils.timer as timer_module


def setup_timer(
    monkeypatch,
    *,
    start_ns: int = 1_000_000_000,
    end_ns: int = 1_250_000_000,
) -> Mock:
    """Replace logger and clock with deterministic test doubles."""

    logger = Mock()
    timestamps = iter([start_ns, end_ns])

    monkeypatch.setattr(timer_module, "logger", logger)
    monkeypatch.setattr(
        timer_module,
        "perf_counter_ns",
        lambda: next(timestamps),
    )

    return logger


def test_timed_returns_result_and_logs_duration(monkeypatch):
    logger = setup_timer(monkeypatch)

    @timer_module.timed("math.add")
    def add(a: int, b: int) -> int:
        return a + b

    result = add(1, 2)

    assert result == 3

    started_call, completed_call = logger.info.call_args_list

    assert started_call.args[0] == "operation_started"
    assert started_call.kwargs["operation"] == "math.add"

    assert completed_call.args[0] == "operation_completed"
    assert completed_call.kwargs["operation"] == "math.add"
    assert completed_call.kwargs["status"] == "success"
    assert completed_call.kwargs["duration_ms"] == 250.0


def test_timed_logs_failure_and_reraises(monkeypatch):
    logger = setup_timer(monkeypatch)

    @timer_module.timed("example.fail")
    def fail() -> None:
        raise ValueError("expected error")

    with pytest.raises(ValueError, match="expected error"):
        fail()

    logger.exception.assert_called_once()

    failure_call = logger.exception.call_args

    assert failure_call.args[0] == "operation_failed"
    assert failure_call.kwargs["operation"] == "example.fail"
    assert failure_call.kwargs["status"] == "error"
    assert failure_call.kwargs["error_type"] == "ValueError"
    assert failure_call.kwargs["duration_ms"] == 250.0


def test_timed_operation_logs_code_block(monkeypatch):
    logger = setup_timer(monkeypatch)

    with timer_module.timed_operation(
        "delta.write",
        table="documents",
    ):
        result = 1 + 1

    assert result == 2

    completed_call = logger.info.call_args_list[-1]

    assert completed_call.args[0] == "operation_completed"
    assert completed_call.kwargs["operation"] == "delta.write"
    assert completed_call.kwargs["table"] == "documents"
    assert completed_call.kwargs["status"] == "success"
    assert completed_call.kwargs["duration_ms"] == 250.0


def test_timed_supports_async_function(monkeypatch):
    logger = setup_timer(monkeypatch)

    @timer_module.timed("rag.retrieve")
    async def retrieve() -> list[str]:
        return ["chunk-1"]

    result = asyncio.run(retrieve())

    assert result == ["chunk-1"]

    completed_call = logger.info.call_args_list[-1]

    assert completed_call.args[0] == "operation_completed"
    assert completed_call.kwargs["operation"] == "rag.retrieve"
    assert completed_call.kwargs["status"] == "success"
    assert completed_call.kwargs["duration_ms"] == 250.0
