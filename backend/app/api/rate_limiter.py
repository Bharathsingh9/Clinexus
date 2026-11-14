"""In-Memory Rate Limiting for Sensitive Endpoints.

Specification Section 7 (API-9):
- Settings-driven limits for login (RATE_LIMIT_LOGIN_PER_MINUTE) and analyze (RATE_LIMIT_ANALYZE_PER_MINUTE).
- Thread-safe sliding window tracking per client IP.
- Known limitation: Single-process in-memory only (GAP-10 in docs/known_gaps.md).
"""

from collections import defaultdict
from datetime import datetime, timezone
import threading
from typing import Dict, List
from fastapi import Request

from backend.app.api.errors import RateLimitedAPIError
from backend.app.core.config import settings


class InMemoryRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str, max_requests: int, window_seconds: float = 60.0) -> None:
        """Record an attempt and raise RateLimitedAPIError if limit is exceeded."""
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._records[key]
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if len(valid_timestamps) >= max_requests:
                self._records[key] = valid_timestamps
                raise RateLimitedAPIError(
                    message=f"Rate limit exceeded: maximum {max_requests} requests per {int(window_seconds)}s."
                )
            valid_timestamps.append(now)
            self._records[key] = valid_timestamps

    def reset(self) -> None:
        """Reset rate limiter state (used in testing)."""
        with self._lock:
            self._records.clear()


limiter = InMemoryRateLimiter()


def rate_limit_login(request: Request) -> None:
    client_ip = request.client.host if request.client else "127.0.0.1"
    limiter.check(f"login:{client_ip}", settings.RATE_LIMIT_LOGIN_PER_MINUTE, 60.0)


def rate_limit_analyze(request: Request) -> None:
    client_ip = request.client.host if request.client else "127.0.0.1"
    limiter.check(f"analyze:{client_ip}", settings.RATE_LIMIT_ANALYZE_PER_MINUTE, 60.0)
