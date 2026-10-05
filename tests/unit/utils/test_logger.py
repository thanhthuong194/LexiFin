from __future__ import annotations

import json

import pytest
import structlog

from src.utils.logger import (
    bind_log_context,
    clear_log_context,
    configure_logging,
    get_logger,
    log_context,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Prevent structlog state from leaking between tests."""

    structlog.reset_defaults()
    clear_log_context()

    yield

    clear_log_context()
    structlog.reset_defaults()


def read_json_events(capsys) -> list[dict]:
    """Read structured JSON events written to stdout."""

    output = capsys.readouterr().out.strip()

    return [json.loads(line) for line in output.splitlines() if line]


def test_logger_outputs_structured_json(capsys):
    configure_logging(
        service="lexifin",
        environment="test",
        json_output=True,
    )

    logger = get_logger("test_logger")

    logger.info(
        "filing_downloaded",
        ticker="AAPL",
        size_bytes=100,
    )

    [event] = read_json_events(capsys)

    assert event["event"] == "filing_downloaded"
    assert event["ticker"] == "AAPL"
    assert event["size_bytes"] == 100
    assert event["service"] == "lexifin"
    assert event["environment"] == "test"
    assert event["level"] == "info"
    assert event["logger"] == "test_logger"
    assert "timestamp" in event


def test_log_context_is_bound_and_temporary(capsys):
    configure_logging(
        environment="test",
        json_output=True,
    )

    logger = get_logger("test_logger")

    bind_log_context(run_id="run-001")

    logger.info("before_context")

    with log_context(ticker="AAPL"):
        logger.info("inside_context")

    logger.info("after_context")

    before, inside, after = read_json_events(capsys)

    assert before["run_id"] == "run-001"
    assert "ticker" not in before

    assert inside["run_id"] == "run-001"
    assert inside["ticker"] == "AAPL"

    assert after["run_id"] == "run-001"
    assert "ticker" not in after


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging(level="INVALID")
