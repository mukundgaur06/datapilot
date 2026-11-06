"""Retry wrapper for LLM calls: transient failures shouldn't sink a long batch (evals, checks).

Retries ONE chat() call, so an agent run keeps its progress:
  * 429 rate limit: wait as long as the provider says (or 30s), up to 3 times;
  * 503/504 overloaded: exponential backoff 10s, 20s, 40s;
  * daily quota exhausted: raise DailyQuotaExhausted at once, since waiting won't help.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.agent.llm import LLM, LLMRateLimited, LLMReply, LLMUnavailable

MAX_RETRIES = 3
DEFAULT_WAIT_S = 30.0  # used when the 429 doesn't say how long to wait
UNAVAILABLE_BASE_WAIT_S = 10.0  # first wait after a 503/504; doubles on each retry


class DailyQuotaExhausted(Exception):
    pass


class RetryingLLM:
    """Wraps an LLM and retries a single chat() call on 429, honouring the server's delay."""

    def __init__(
        self, inner: LLM, max_retries: int = MAX_RETRIES, sleep: Callable[[float], Any] = time.sleep,
        log: Callable[[str], Any] = print,
    ) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.sleep = sleep  # injectable, so this can be tested without real waiting
        self.log = log
        self.waited_s = 0.0  # total time spent waiting on rate limits / overload, so reports can exclude it

    @property
    def model(self) -> str:
        return str(getattr(self.inner, "model", type(self.inner).__name__))

    def user_message(self, text: str) -> Any:
        return self.inner.user_message(text)

    def tool_results_message(self, results: list[Any]) -> Any:
        return self.inner.tool_results_message(results)

    def chat(self, **kwargs: Any) -> LLMReply:
        for attempt in range(self.max_retries + 1):
            try:
                return self.inner.chat(**kwargs)
            except LLMRateLimited as e:
                if e.daily:
                    raise DailyQuotaExhausted(str(e)) from e
                if attempt == self.max_retries:
                    raise  # LLMError: the agent loop turns it into an "error" event
                wait = (e.retry_after or DEFAULT_WAIT_S) + 1
                reason = "rate limited"
            except LLMUnavailable:
                if attempt == self.max_retries:
                    raise
                wait = UNAVAILABLE_BASE_WAIT_S * 2**attempt  # exponential backoff: 10s, 20s, 40s
                reason = "model overloaded"
            self.log(f"   ({reason}; waiting {wait:.0f}s, retry {attempt + 1}/{self.max_retries})")
            self.sleep(wait)
            self.waited_s += wait
        raise AssertionError("unreachable")
