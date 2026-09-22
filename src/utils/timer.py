from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from time import perf_counter_ns
from typing import Any

from .logger import get_logger


logger = get_logger(__name__)


def _elapsed_ms(start_ns: int) -> float:
    """Calculate elapsed time in milliseconds."""

    elapsed_ns = perf_counter_ns() - start_ns
    return round(elapsed_ns / 1_000_000, 3)


@contextmanager
def timed_operation(
    operation: str,
    **fields: Any,
) -> Iterator[None]:
    """Measure and log a block of code."""

    start_ns = perf_counter_ns()

    base_fields = {
        **fields,
        "operation": operation,
    }

    logger.info(
        "operation_started",
        **base_fields,
    )

    try:
        yield

    except Exception as error:
        failure_fields = {
            **base_fields,
            "status": "error",
            "duration_ms": _elapsed_ms(start_ns),
            "error_type": type(error).__name__,
        }

        logger.exception(
            "operation_failed",
            **failure_fields,
        )

        raise

    else:
        success_fields = {
            **base_fields,
            "status": "success",
            "duration_ms": _elapsed_ms(start_ns),
        }

        logger.info(
            "operation_completed",
            **success_fields,
        )


def timed(
    operation: str,
    **fields: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure and log a synchronous or asynchronous function."""

    def decorator(
        function: Callable[..., Any],
    ) -> Callable[..., Any]:
        function_fields = {
            "function": (
                f"{function.__module__}."
                f"{function.__qualname__}"
            ),
            **fields,
        }

        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                with timed_operation(
                    operation,
                    **function_fields,
                ):
                    return await function(*args, **kwargs)

            return async_wrapper

        @wraps(function)
        def sync_wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with timed_operation(
                operation,
                **function_fields,
            ):
                return function(*args, **kwargs)

        return sync_wrapper

    return decorator