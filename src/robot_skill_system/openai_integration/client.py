"""Official OpenAI client construction and retry/trace helpers."""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections.abc import Callable
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from robot_skill_system.settings import OpenAIMode, Settings

from .tool_schemas import SAFE_FUNCTION_TOOLS as SAFE_FUNCTION_TOOLS

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE_OPENAI_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class OpenAIClientFactory:
    """Construct the official SDK only when live mode is explicitly configured."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(self) -> OpenAI:
        """Create a timeout/retry-configured SDK client without logging credentials."""

        if self.settings.openai_mode is not OpenAIMode.LIVE:
            raise RuntimeError("OpenAI client creation is disabled outside live mode")
        if self.settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required when OPENAI_MODE=live")
        return OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.openai_api_timeout_seconds,
            max_retries=self.settings.openai_api_max_retries,
        )


class RetryExecutor:
    """Bound retries for transport failures, with exponential backoff and jitter."""

    def __init__(
        self,
        max_retries: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.max_retries = max_retries
        self._sleep = sleep
        self._random_value = random_value

    def call(self, operation: Callable[[], T], trace_id: str) -> tuple[T, int]:
        """Call *operation* and return its value plus number of attempts."""

        attempts = 0
        while True:
            attempts += 1
            try:
                return operation(), attempts
            except RETRYABLE_OPENAI_ERRORS as exc:
                if attempts > self.max_retries:
                    LOGGER.error(
                        "openai_request_failed trace_id=%s attempts=%d error_type=%s",
                        trace_id,
                        attempts,
                        type(exc).__name__,
                    )
                    raise
                delay_s = min(8.0, 0.25 * (2 ** (attempts - 1)))
                delay_s += delay_s * 0.25 * self._random_value()
                LOGGER.warning(
                    "openai_request_retry trace_id=%s attempt=%d error_type=%s",
                    trace_id,
                    attempts,
                    type(exc).__name__,
                )
                self._sleep(delay_s)


def new_trace_id() -> str:
    """Return an opaque request trace identifier safe to write to logs."""

    return str(uuid.uuid4())


def usage_metadata(response: object, trace_id: str, attempts: int) -> dict[str, object]:
    """Extract non-sensitive response and token metadata across SDK minor versions."""

    usage = getattr(response, "usage", None)
    result: dict[str, object] = {
        "trace_id": trace_id,
        "response_id": getattr(response, "id", None),
        "attempts": attempts,
    }
    if usage is not None:
        for output_key, attr in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            result[output_key] = getattr(usage, attr, None)
    return result
