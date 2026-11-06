"""LLM adapter layer.

The agent loop only talks to the `LLM` protocol below. Everything Gemini-specific
lives in `GeminiLLM`, so we can:
  * swap providers (or move to Google's newer Interactions API) by changing one file,
  * replace the model with a scripted fake in tests (fast, free, deterministic).

The adapter owns the *message format* too: history entries are opaque to the loop.
That matters because Gemini attaches "thought signatures" to its turns, which must
be sent back unmodified. The loop simply appends whatever `raw_content` it gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings

Finish = Literal["stop", "tool_calls", "max_tokens", "blocked", "malformed"]


@dataclass
class ToolCall:
    id: str | None  # provider call id (Gemini may omit it); echoed back with the result
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    call: ToolCall
    content: str  # JSON text, already size-capped by the loop
    is_error: bool


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall]
    finish: Finish
    raw_content: Any  # provider-native assistant turn, appended to history verbatim
    thoughts: list[str] = field(default_factory=list)  # thought summaries, for the UI timeline
    finish_detail: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Provider call failed (network, quota, invalid key...). Message is user-presentable."""


class LLMConfigError(LLMError):
    """The adapter can't be built (e.g. missing API key)."""


class LLMRateLimited(LLMError):
    """HTTP 429. `retry_after` is the provider's suggested wait in seconds (if given);
    `daily` means a per-day quota is exhausted, so retrying today won't help."""

    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


class LLMUnavailable(LLMError):
    """HTTP 5xx (e.g. 503 overloaded, 504 deadline exceeded). Transient: safe to retry later."""


def _rate_limit_info(details: Any) -> tuple[float | None, bool]:
    """Read Google's RetryInfo.retryDelay ("33s") and QuotaFailure ids from a 429 body."""
    retry_after, daily = None, False
    error = details.get("error", {}) if isinstance(details, dict) else {}
    for d in error.get("details", []) or []:
        if not isinstance(d, dict):
            continue
        delay = d.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                retry_after = float(delay[:-1])
            except ValueError:
                pass
        for v in d.get("violations", []) or []:
            if isinstance(v, dict) and "PerDay" in str(v.get("quotaId", "")):
                daily = True
    return retry_after, daily


class LLM(Protocol):
    def user_message(self, text: str) -> Any: ...
    def tool_results_message(self, results: list[ToolResult]) -> Any: ...
    def chat(self, *, system: str, history: list[Any], tools: list[dict[str, Any]]) -> LLMReply: ...


_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "LANGUAGE"}
_MALFORMED = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "MALFORMED_RESPONSE"}


class GeminiLLM:
    def __init__(
        self,
        model: str | None = None,
        thinking_level: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or settings.model
        self.thinking_level = (thinking_level or settings.thinking_level).upper()
        if client is None:
            key = api_key or settings.gemini_api_key
            if not key:
                raise LLMConfigError(
                    "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                    "and put it in backend/.env"
                )
            client = genai.Client(api_key=key)
        self.client = client

    # -- message builders (the loop never constructs provider types itself)

    def user_message(self, text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    def tool_results_message(self, results: list[ToolResult]) -> types.Content:
        # All results for one model turn go back in ONE message (required for parallel calls).
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=r.call.id,
                    name=r.call.name,
                    response={"error": r.content} if r.is_error else {"output": r.content},
                )
            )
            for r in results
        ]
        return types.Content(role="user", parts=parts)

    # -- the actual model call

    def _config(self, system: str, tools: list[dict[str, Any]]) -> types.GenerateContentConfig:
        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
            )
            for t in tools
        ]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            # We run the loop ourselves; never let the SDK execute functions behind our back.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(self.thinking_level), include_thoughts=True
            ),
        )

    def chat(self, *, system: str, history: list[Any], tools: list[dict[str, Any]]) -> LLMReply:
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=history, config=self._config(system, tools)
            )
        except genai_errors.ClientError as e:
            if e.code == 429:
                retry_after, daily = _rate_limit_info(e.details)
                hint = "Daily quota used up; try again tomorrow." if daily else "Wait a minute and retry."
                raise LLMRateLimited(
                    f"Gemini rate limit / free-tier quota reached. {hint}", retry_after=retry_after, daily=daily
                ) from e
            if e.code in (401, 403):
                raise LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in backend/.env.") from e
            if e.code == 404:
                raise LLMError(f"Model {self.model!r} not found. Set DATAPILOT_MODEL to a current model.") from e
            raise LLMError(f"Gemini request error ({e.code}): {e.message}") from e
        except genai_errors.ServerError as e:
            raise LLMUnavailable(f"Gemini server error ({e.code}): the model is overloaded or timed out. Try again shortly.") from e
        return self._parse(response)

    def _parse(self, response: types.GenerateContentResponse) -> LLMReply:
        usage = {}
        if um := response.usage_metadata:
            usage = {
                "input_tokens": um.prompt_token_count or 0,
                "output_tokens": um.candidates_token_count or 0,
                "thinking_tokens": um.thoughts_token_count or 0,
            }

        if not response.candidates:
            reason = response.prompt_feedback.block_reason if response.prompt_feedback else None
            return LLMReply("", [], "blocked", None, finish_detail=str(reason or "no candidates"), usage=usage)

        candidate = response.candidates[0]
        content = candidate.content
        texts, thoughts, calls = [], [], []
        for part in (content.parts if content and content.parts else []):
            if part.thought:
                if part.text:
                    thoughts.append(part.text)
            elif part.function_call:
                fc = part.function_call
                calls.append(ToolCall(id=fc.id, name=fc.name or "", args=dict(fc.args or {})))
            elif part.text:
                texts.append(part.text)

        reason = candidate.finish_reason.name if candidate.finish_reason else "STOP"
        if calls:
            finish: Finish = "tool_calls"
        elif reason == "MAX_TOKENS":
            finish = "max_tokens"
        elif reason in _BLOCKED:
            finish = "blocked"
        elif reason in _MALFORMED:
            finish = "malformed"
        else:
            finish = "stop"

        return LLMReply(
            text="".join(texts).strip(),
            tool_calls=calls,
            finish=finish,
            raw_content=content,
            thoughts=thoughts,
            finish_detail=reason,
            usage=usage,
        )
