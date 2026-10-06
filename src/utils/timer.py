from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from time import perf_counter_ns
from typing import Any, TypeVar

from .logger import get_logger

logger = get_logger(__name__)
Result = TypeVar("Result")


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


class PhaseDurations:
    """Total durations of phases repeated per item, logged once instead of per item."""

    def __init__(self) -> None:
        self._elapsed_ns: Counter[str] = Counter()

    def run(self, phase: str, work: Callable[[], Result]) -> Result:
        """Run one phase of an item and add its duration to the phase total."""

        started = perf_counter_ns()
        try:
            return work()
        finally:
            self._elapsed_ns[phase] += perf_counter_ns() - started

    def milliseconds(self) -> dict[str, int]:
        """Each phase's total as a `<phase>_ms` log field."""

        return {
            f"{phase}_ms": ns // 1_000_000 for phase, ns in self._elapsed_ns.items()
        }


def timed(
    operation: str,
    **fields: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure and log a synchronous or asynchronous function."""

    def decorator(
        function: Callable[..., Any],
    ) -> Callable[..., Any]:
        function_fields = {
            "function": (f"{function.__module__}.{function.__qualname__}"),
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
