"""Abuse and cost guardrails for a public deployment.

A public demo spends the owner's LLM quota on every question, so:
  * per-IP sliding-window rate limits on the expensive endpoints (runs) and on uploads;
  * a global cap on concurrently running agents (each one is a thread making LLM calls)
    lives in RunManager; a cap on datasets held in memory lives in DatasetStore.

Client IP: behind a reverse proxy (Render, Hugging Face, nginx) every request comes from
the proxy, so uvicorn must run with --proxy-headers to put the real client IP in
request.client. The Dockerfile does that. In-memory state means limits are per process,
which is right for this single-instance app; several replicas would need Redis.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from fastapi import HTTPException, Request, status


def parse_limit(spec: str) -> tuple[int, float] | None:
    """'10/600' -> (10 requests, per 600 seconds). '0' or '' disables the limit."""
    spec = spec.strip()
    if spec in ("", "0", "off"):
        return None
    count, _, seconds = spec.partition("/")
    n, window = int(count), float(seconds or 60)
    if n <= 0 or window <= 0:
        return None
    return n, window


class RateLimiter:
    """Sliding window: at most `limit` hits per `window` seconds per key."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window_s
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until the next hit would be allowed."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return q[0] + self.window - now
            q.append(now)
            self._calls += 1
            if self._calls % 1000 == 0:  # occasionally forget idle clients so memory stays bounded
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window}
            return None


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(limiter_ref: Callable[[], RateLimiter | None], what: str) -> Callable[[Request], None]:
    """FastAPI dependency factory. `limiter_ref` is looked up per request so tests can swap it."""

    def dependency(request: Request) -> None:
        limiter = limiter_ref()
        if limiter is None:
            return
        retry_after = limiter.hit(client_ip(request))
        if retry_after is not None:
            wait = math.ceil(retry_after)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many {what} from your address. This is a shared demo; please try again in about "
                f"{wait // 60} min {wait % 60} s." if wait >= 60 else f"Too many {what}. Please try again in {wait} s.",
                headers={"Retry-After": str(wait)},
            )

    return dependency
