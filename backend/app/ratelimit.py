"""In-memory sliding-window rate limiter.

Single process only. State is not shared between workers or restarts.
This is a protective measure, not a replacement for a gateway or WAF.
"""

from __future__ import annotations

import math
import threading
import time

from collections import deque
from typing import Callable


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        if (
            isinstance(window, bool)
            or not isinstance(window, (int, float))
            or not math.isfinite(window)
            or window <= 0
        ):
            raise ValueError("window must be a positive finite number")

        if (
            isinstance(max_keys, bool)
            or not isinstance(max_keys, int)
            or max_keys < 1
        ):
            raise ValueError("max_keys must be a positive integer")

        if not callable(clock):
            raise TypeError("clock must be callable")

        self.limit = limit
        self.window = float(window)
        self.clock = clock
        self.max_keys = max_keys

        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds).

        A request is counted only when allowed.
        """

        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")

        now = float(self.clock())

        if not math.isfinite(now):
            raise ValueError("clock must return a finite number")

        with self._lock:
            q = self._hits.get(key)

            if q is None:
                if len(self._hits) >= self.max_keys:
                    self._prune(now)

                q = self._hits.setdefault(key, deque())

            # Remove timestamps outside the active sliding window.
            while q and now - q[0] >= self.window:
                q.popleft()

            # Reject without recording the rejected request.
            if len(q) >= self.limit:
                retry_after = self.window - (now - q[0])

                return False, max(0.0, retry_after)

            q.append(now)

            return True, 0.0

    def _prune(self, now: float) -> None:
        """Remove expired keys when the key capacity is reached.

        Must be called while holding self._lock.
        """

        expired_keys = [
            key
            for key, timestamps in self._hits.items()
            if not timestamps
            or now - timestamps[-1] >= self.window
        ]

        for key in expired_keys:
            del self._hits[key]

        # Keep memory bounded if all remaining keys are active.
        # Evicting keys is a deliberate trade-off in this in-memory limiter.
        if len(self._hits) >= self.max_keys:
            remove_count = min(
                len(self._hits),
                max(1, self.max_keys // 10 + 1),
            )

            oldest_keys = list(self._hits)[:remove_count]

            for key in oldest_keys:
                del self._hits[key]