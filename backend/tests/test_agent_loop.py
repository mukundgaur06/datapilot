"""Agent-loop behaviour, tested with a scripted LLM (no network)."""

from __future__ import annotations

import json

import pytest

from app.agent.llm import LLMError, LLMReply
from app.agent.loop import MAX_RESULT_CHARS, run_agent, serialize_result
from app.store import DatasetNotFound, DatasetStore
from tests.fakes import ScriptedLLM, answer, call


def run(llm: ScriptedLLM, store: DatasetStore, dataset_id: str, **kw):
    return list(run_agent("How many customers per segment?", dataset_id, llm=llm, store=store, **kw))


def test_happy_path_tool_then_answer(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([call("group_aggregate", group_by=["Segment"]), answer("A and B have 3 customers each.")])
    events = run(llm, store, dataset_id)

    assert [e.type for e in events] == ["tool_call", "tool_result", "answer"]
    assert events[1].data["is_error"] is False
    assert events[2].data["text"].startswith("A and B")
    assert events[2].data["steps"] == 2
    assert events[2].data["usage"] == {"input_tokens": 300, "output_tokens": 60}

    # The second model call must see: question, model's tool call, and the tool result with the matching id.
    history = llm.calls[1]["history"]
    assert [m["role"] for m in history] == ["user", "model", "tool"]
    result = history[2]["results"][0]
    assert result["id"] == "c1" and result["is_error"] is False
    assert json.loads(result["content"])["n_groups"] == 3


def test_system_prompt_contains_schema_but_tools_compute_numbers(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([answer("done")])
    run(llm, store, dataset_id)
    system = llm.calls[0]["system"]
    assert "<dataset_schema>" in system and "Income_k" in system
    assert {t["name"] for t in llm.calls[0]["tools"]} == {"profile_dataset", "group_aggregate", "correlation"}


def test_tool_error_is_fed_back_and_model_recovers(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM(
        [
            call("group_aggregate", call_id="bad", group_by=["segment"]),  # wrong case
            call("group_aggregate", call_id="good", group_by=["Segment"]),
            answer("Fixed it."),
        ]
    )
    events = run(llm, store, dataset_id)
    results = [e for e in events if e.type == "tool_result"]
    assert results[0].data["is_error"] is True and "did you mean 'Segment'" in results[0].data["result"]["error"]
    assert results[1].data["is_error"] is False
    assert llm.calls[1]["history"][-1]["results"][0]["is_error"] is True
    assert events[-1].type == "answer"


def test_unknown_tool_does_not_crash(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([call("delete_everything"), answer("Sorry.")])
    events = run(llm, store, dataset_id)
    assert events[1].data["is_error"] is True and "Unknown tool" in events[1].data["result"]["error"]
    assert events[-1].type == "answer"


def test_step_limit_returns_partial_answer(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([call("profile_dataset", call_id=f"c{i}") for i in range(3)])
    events = run(llm, store, dataset_id, max_steps=3)
    assert events[-1].type == "answer" and events[-1].data["partial"] is True
    assert len(llm.calls) == 3


def test_parallel_tool_calls_return_in_one_message(store: DatasetStore, dataset_id: str) -> None:
    from app.agent.llm import ToolCall

    both = LLMReply(
        text="", finish="tool_calls", raw_content={"role": "model"},
        tool_calls=[ToolCall("x", "profile_dataset", {}), ToolCall("y", "correlation", {})],
    )
    llm = ScriptedLLM([both, answer("ok")])
    run(llm, store, dataset_id)
    last = llm.calls[1]["history"][-1]
    assert [r["id"] for r in last["results"]] == ["x", "y"]


@pytest.mark.parametrize(
    ("reply", "fragment"),
    [
        (LLMReply("", [], "blocked", None, finish_detail="SAFETY"), "blocked"),
        (LLMReply("", [], "max_tokens", None), "output tokens"),
        (LLMReply("", [], "stop", {"role": "model"}), "empty answer"),
    ],
)
def test_terminal_failures_become_error_events(store, dataset_id, reply, fragment) -> None:
    events = run(ScriptedLLM([reply]), store, dataset_id)
    assert events[-1].type == "error" and fragment in events[-1].data["message"]


def test_malformed_call_gets_a_retry(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([LLMReply("", [], "malformed", None), answer("ok")])
    events = run(llm, store, dataset_id)
    assert events[-1].type == "answer"
    assert "malformed" in llm.calls[1]["history"][-1]["text"]


def test_llm_error_becomes_error_event(store: DatasetStore, dataset_id: str) -> None:
    events = run(ScriptedLLM([LLMError("quota reached")]), store, dataset_id)
    assert events == [events[0]] and events[0].type == "error" and "quota" in events[0].data["message"]


def test_unknown_dataset_raises_before_any_llm_call(store: DatasetStore) -> None:
    llm = ScriptedLLM([])
    with pytest.raises(DatasetNotFound):
        run(llm, store, "nope")
    assert llm.calls == []


def test_serialize_result_caps_size() -> None:
    text = serialize_result({"rows": ["x" * 100] * 200})
    assert len(text) < MAX_RESULT_CHARS + 200 and "truncated" in text
