"""The Gemini adapter, tested against a fake client that returns real SDK response objects.

This checks our translation layer (request config in, LLMReply out) without network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.agent.llm import GeminiLLM, LLMError, LLMRateLimited, LLMUnavailable, ToolCall, ToolResult
from app.tools import registry


class FakeModels:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.kwargs = kwargs
        return self.response


def make_llm(response: types.GenerateContentResponse) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(response)
    return GeminiLLM(model="test-model", thinking_level="low", client=SimpleNamespace(models=models)), models


def response(*parts: types.Part, finish: types.FinishReason = types.FinishReason.STOP) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=120, candidates_token_count=30, thoughts_token_count=15
        ),
    )


def test_request_config_disables_auto_calling_and_declares_tools() -> None:
    llm, models = make_llm(response(types.Part(text="hi")))
    llm.chat(system="SYS", history=[llm.user_message("q")], tools=registry.schemas())

    config: types.GenerateContentConfig = models.kwargs["config"]
    assert models.kwargs["model"] == "test-model"
    assert config.system_instruction == "SYS"
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    names = [d.name for d in config.tools[0].function_declarations]
    assert names == ["correlation", "group_aggregate", "profile_dataset"]


def test_parses_function_calls_thoughts_and_usage() -> None:
    llm, _ = make_llm(
        response(
            types.Part(text="I should count labels.", thought=True),
            types.Part(function_call=types.FunctionCall(id="fc1", name="group_aggregate", args={"group_by": ["label"]})),
        )
    )
    reply = llm.chat(system="", history=[], tools=[])
    assert reply.finish == "tool_calls"
    assert reply.tool_calls == [ToolCall(id="fc1", name="group_aggregate", args={"group_by": ["label"]})]
    assert reply.thoughts == ["I should count labels."]
    assert reply.text == ""  # thoughts never leak into the answer text
    assert reply.usage == {"input_tokens": 120, "output_tokens": 30, "thinking_tokens": 15}
    assert reply.raw_content.role == "model"  # appended verbatim (keeps thought signatures)


def test_finish_reason_mapping() -> None:
    llm, _ = make_llm(response(types.Part(text="Final answer."), finish=types.FinishReason.STOP))
    assert llm.chat(system="", history=[], tools=[]).finish == "stop"
    llm, _ = make_llm(response(finish=types.FinishReason.SAFETY))
    assert llm.chat(system="", history=[], tools=[]).finish == "blocked"
    llm, _ = make_llm(response(finish=types.FinishReason.MALFORMED_FUNCTION_CALL))
    assert llm.chat(system="", history=[], tools=[]).finish == "malformed"


def test_tool_results_message_uses_output_and_error_keys() -> None:
    llm, _ = make_llm(response())
    ok = ToolResult(ToolCall("a", "profile_dataset", {}), '{"n_rows": 3}', is_error=False)
    bad = ToolResult(ToolCall("b", "correlation", {}), '{"error": "nope"}', is_error=True)
    msg = llm.tool_results_message([ok, bad])
    assert msg.role == "user" and len(msg.parts) == 2
    assert msg.parts[0].function_response.id == "a"
    assert msg.parts[0].function_response.response == {"output": '{"n_rows": 3}'}
    assert msg.parts[1].function_response.response == {"error": '{"error": "nope"}'}


# ---------------------------------------------------------------- rate limits


def _raising_llm(exc: Exception) -> GeminiLLM:
    def generate_content(**kwargs: Any) -> None:
        raise exc

    return GeminiLLM(model="m", thinking_level="low", client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


def _429(quota_id: str, delay: str | None = "33s") -> genai_errors.ClientError:
    details: list[dict[str, Any]] = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}
    ]
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": details}}
    return genai_errors.ClientError(429, body)


def test_429_per_minute_raises_rate_limited_with_retry_delay() -> None:
    llm = _raising_llm(_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))
    with pytest.raises(LLMRateLimited) as info:
        llm.chat(system="", history=[], tools=[])
    assert info.value.retry_after == 33.0
    assert info.value.daily is False
    assert isinstance(info.value, LLMError)  # existing `except LLMError` handlers still catch it


def test_429_per_day_is_flagged_daily_and_tolerates_missing_delay() -> None:
    llm = _raising_llm(_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", delay=None))
    with pytest.raises(LLMRateLimited) as info:
        llm.chat(system="", history=[], tools=[])
    assert info.value.daily is True
    assert info.value.retry_after is None


def test_5xx_raises_retryable_unavailable() -> None:
    body = {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    with pytest.raises(LLMUnavailable):
        _raising_llm(genai_errors.ServerError(503, body)).chat(system="", history=[], tools=[])
