from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.stdlib import BoundLogger
from structlog.typing import EventDict, Processor


def configure_logging(
    *,
    service: str = "lexifin",
    environment: str = "local",
    level: str = "INFO",
    json_output: bool = True,
) -> None:
    """Configure application logging once at startup."""

    log_level = getattr(logging, level.upper(), None)

    if not isinstance(log_level, int):
        # The argument is a str; an unknown level name is a bad value, and
        # callers and tests rely on ValueError.
        raise ValueError(f"Invalid log level: {level}")  # noqa: TRY004

    logging.basicConfig(
        format="%(message)s", level=log_level, stream=sys.stdout, force=True
    )

    def add_app_context(
        _: Any,
        __: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add application context to the log event dictionary."""
        event_dict.setdefault("service", service)
        event_dict.setdefault("environment", environment)
        return event_dict

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer_processors: list[Processor] = (
        [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        if json_output
        else [
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),
            )
        ]
    )

    structlog.configure(
        processors=[*shared_processors, *renderer_processors],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a structured logger."""

    return structlog.stdlib.get_logger(name)


def bind_log_context(**fields: Any) -> None:
    """Bind fields to the current execution context."""

    values = {key: value for key, value in fields.items() if value is not None}

    structlog.contextvars.bind_contextvars(**values)


def clear_log_context() -> None:
    """Clear all fields from the current execution context."""

    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Temporarily bind fields inside a code block."""

    values = {key: value for key, value in fields.items() if value is not None}

    with structlog.contextvars.bound_contextvars(**values):
        yield
