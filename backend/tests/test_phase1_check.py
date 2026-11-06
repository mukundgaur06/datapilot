"""The acceptance-check script itself: retry policy and pass/fail logic, with no network."""

from __future__ import annotations

import pytest

from app.agent.llm import LLMRateLimited, LLMReply
from scripts.phase1_check import CASES, DailyQuotaExhausted, RetryingLLM, run_case
from tests.fakes import ScriptedLLM, call


def answer(text: str) -> LLMReply:
    return LLMReply(text=text, tool_calls=[], finish="stop", raw_content={"role": "model"})


def retrying(replies: list, max_retries: int = 3) -> tuple[RetryingLLM, list[float]]:
    waits: list[float] = []
    return RetryingLLM(ScriptedLLM(replies), max_retries=max_retries, sleep=waits.append), waits


def test_retries_only_the_failed_call_and_honours_server_delay() -> None:
    llm, waits = retrying([LLMRateLimited("429", retry_after=12), answer("ok")])
    assert llm.chat(system="", history=[], tools=[]).text == "ok"
    assert waits == [13]  # server delay + 1s margin


def test_gives_up_after_max_retries_with_an_llm_error() -> None:
    llm, waits = retrying([LLMRateLimited("429")] * 3, max_retries=2)
    with pytest.raises(LLMRateLimited):
        llm.chat(system="", history=[], tools=[])
    assert len(waits) == 2


def test_daily_quota_stops_immediately_without_waiting() -> None:
    llm, waits = retrying([LLMRateLimited("429", daily=True)])
    with pytest.raises(DailyQuotaExhausted):
        llm.chat(system="", history=[], tools=[])
    assert waits == []


LAPSED = CASES[3]  # 422 customers, 35.17%
LAPSED_CALL = call("group_aggregate", agg="count", filters=[{"column": "Days_Since_Last_Purchase", "op": ">", "value": 60}])


def test_grounded_correct_answer_passes() -> None:
    assert run_case(LAPSED, ScriptedLLM([LAPSED_CALL, answer("422 customers (35.17%).")]))


def test_ungrounded_or_wrong_answers_fail() -> None:
    assert not run_case(LAPSED, ScriptedLLM([answer("422 customers (35.17%).")]))  # no tool call
    assert not run_case(LAPSED, ScriptedLLM([LAPSED_CALL, answer("400 customers (33%).")]))


def test_failed_run_reports_only_the_failure(capsys: pytest.CaptureFixture[str]) -> None:
    assert not run_case(LAPSED, ScriptedLLM([LLMRateLimited("Gemini rate limit")]))
    out = capsys.readouterr().out
    assert "run failed: Gemini rate limit" in out
    assert "missing" not in out and "ungrounded" not in out
