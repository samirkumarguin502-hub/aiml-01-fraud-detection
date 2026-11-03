"""In-memory sliding-window rate limiter. Single process only (state is not shared between
workers or restarts). It is a protective measure, not a replacement for a gateway/WAF."""
from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic, max_keys: int = 10_000):
        self.limit, self.window, self.clock, self.max_keys = limit, window, clock, max_keys
        self._hits: dict[str, deque] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Counts the request only if allowed."""
        now = self.clock()
        q = self._hits.get(key)
        if q is None:
            if len(self._hits) >= self.max_keys:
                self._prune(now)
            q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(0.0, self.window - (now - q[0]))
        q.append(now)
        return True, 0.0

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]
        if len(self._hits) >= self.max_keys:  # still full: drop oldest-inserted keys
            for k in list(self._hits)[: self.max_keys // 10 + 1]:
                del self._hits[k]
