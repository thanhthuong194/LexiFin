"""Retry Qdrant calls that failed for a transient reason.

Transient: a 5xx or 429 response (`UnexpectedResponse`), a 429 with
`Retry-After` (`ResourceExhaustedResponse`, which waits as long as the server
asks), or a transport failure (`ResponseHandlingException` around a
connection or timeout error). Not transient: any other 4xx, and a response
that does not parse (`ResponseHandlingException` around a validation error).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import ValidationError
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)

WRITE_ATTEMPTS = 5
RETRY_BASE_SECONDS = 2.0
_RETRYABLE = (
    ResponseHandlingException,
    UnexpectedResponse,
    ResourceExhaustedResponse,
    ConnectionError,
    TimeoutError,
)


def transient(error: Exception) -> bool:
    if isinstance(error, UnexpectedResponse):
        status = error.status_code or 0
        return status == 429 or status >= 500
    if isinstance(error, ResponseHandlingException):
        return not isinstance(error.source, ValidationError)
    # Bare socket failures outside the REST wrapper; not other OSErrors
    # (permissions, a full disk in embedded mode).
    return isinstance(error, (ResourceExhaustedResponse, ConnectionError, TimeoutError))


def _delay(error: Exception, attempt: int) -> float:
    backoff = RETRY_BASE_SECONDS * (1 << (attempt - 1))
    if isinstance(error, ResourceExhaustedResponse):
        return max(float(error.retry_after_s), backoff)
    return backoff


def retried[T](
    action: Callable[[], T],
    what: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run a server call, retrying transient errors with backoff."""

    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            return action()
        except _RETRYABLE as error:
            if attempt == WRITE_ATTEMPTS or not transient(error):
                raise
            logger.warning(
                "qdrant_call_retry", what=what, attempt=attempt, error=str(error)[:200]
            )
            sleep(_delay(error, attempt))
    raise AssertionError("unreachable")
