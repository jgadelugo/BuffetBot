"""
Retry Management for GCS Operations

Implements exponential backoff retry logic with configurable attempts and delays.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base class for retryable errors"""

    pass


class NonRetryableError(Exception):
    """Base class for non-retryable errors"""

    pass


class RetryStrategy(Enum):
    """Retry strategy types"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF


class RetryManager:
    """Manages retry logic for GCS operations with exponential backoff"""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)

        # List of retryable error types (Google Cloud specific)
        self.retryable_errors = [
            "ServiceUnavailable",
            "InternalServerError",
            "BadGateway",
            "GatewayTimeout",
            "TooManyRequests",
            "RequestTimeout",
            "ConnectionError",
            "ReadTimeout",
            "Timeout",
        ]

        # List of non-retryable error types
        self.non_retryable_errors = [
            "Forbidden",
            "Unauthorized",
            "NotFound",
            "BadRequest",
            "Conflict",
        ]

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                self.logger.debug(
                    f"Executing {func.__name__}, attempt {attempt + 1}/{self.config.max_attempts}"
                )

                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success! Log if we had previous failures
                if attempt > 0:
                    self.logger.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e
                error_name = e.__class__.__name__

                # Check if this is the last attempt
                is_last_attempt = (attempt + 1) >= self.config.max_attempts

                # Determine if error is retryable
                is_retryable = self._is_retryable_error(e)

                if not is_retryable:
                    self.logger.error(
                        f"Non-retryable error in {func.__name__}: {error_name} - {str(e)}"
                    )
                    error = NonRetryableError(f"Non-retryable error: {str(e)}")
                    error.__cause__ = e
                    raise error

                if is_last_attempt:
                    self.logger.error(
                        f"Max retry attempts ({self.config.max_attempts}) exceeded for {func.__name__}"
                    )
                    break

                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)

                self.logger.warning(
                    f"Retryable error in {func.__name__} (attempt {attempt + 1}): {error_name} - {str(e)}. "
                    f"Retrying in {delay:.2f}s"
                )

                # Wait before next attempt
                await asyncio.sleep(delay)

        # All attempts failed
        error = RetryableError(f"All {self.config.max_attempts} retry attempts failed")
        error.__cause__ = last_exception
        raise error

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable"""
        error_name = error.__class__.__name__
        error_message = str(error).lower()

        # Check explicit non-retryable errors first
        for non_retryable in self.non_retryable_errors:
            if (
                non_retryable.lower() in error_name.lower()
                or non_retryable.lower() in error_message
            ):
                return False

        # Check retryable errors
        for retryable in self.retryable_errors:
            if (
                retryable.lower() in error_name.lower()
                or retryable.lower() in error_message
            ):
                return True

        # Check for common network/timeout errors
        if any(
            keyword in error_message
            for keyword in [
                "timeout",
                "connection",
                "network",
                "temporary",
                "unavailable",
                "rate limit",
                "quota",
                "throttle",
                "overloaded",
            ]
        ):
            return True

        # Default to retryable for unknown errors (conservative approach)
        self.logger.debug(
            f"Unknown error type {error_name}, treating as retryable: {str(error)}"
        )
        return True

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt"""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay_seconds * (
                self.config.exponential_base**attempt
            )
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay_seconds * (attempt + 1)
        else:  # FIXED_DELAY
            delay = self.config.base_delay_seconds

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        # Ensure delay is never negative
        return max(0, delay)

    def create_retryable_error(
        self, message: str, original_error: Exception = None
    ) -> RetryableError:
        """Create a retryable error with context"""
        if original_error:
            error = RetryableError(f"{message}: {str(original_error)}")
            error.__cause__ = original_error
            return error
        return RetryableError(message)

    def create_non_retryable_error(
        self, message: str, original_error: Exception = None
    ) -> NonRetryableError:
        """Create a non-retryable error with context"""
        if original_error:
            error = NonRetryableError(f"{message}: {str(original_error)}")
            error.__cause__ = original_error
            return error
        return NonRetryableError(message)


# Decorator for easy retry functionality
def retry_on_failure(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions"""
    retry_manager = RetryManager(config)

    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.execute_with_retry(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # Convert sync function to async for retry manager
            async def async_func():
                return func(*args, **kwargs)

            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(retry_manager.execute_with_retry(async_func))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
