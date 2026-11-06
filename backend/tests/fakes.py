"""Test doubles. ScriptedLLM plays back pre-written replies so agent tests are
deterministic, instant and free: no network, no API key, no flaky model output."""

from __future__ import annotations

from typing import Any

from app.agent.llm import LLMReply, ToolCall, ToolResult


class ScriptedLLM:
    def __init__(self, replies: list[LLMReply | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "text": text}

    def tool_results_message(self, results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "tool",
            "results": [{"id": r.call.id, "name": r.call.name, "content": r.content, "is_error": r.is_error} for r in results],
        }

    def chat(self, *, system: str, history: list[Any], tools: list[dict[str, Any]]) -> LLMReply:
        self.calls.append({"system": system, "history": list(history), "tools": tools})
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name: str, call_id: str = "c1", **args: Any) -> LLMReply:
    return LLMReply(
        text="", tool_calls=[ToolCall(id=call_id, name=name, args=args)], finish="tool_calls",
        raw_content={"role": "model", "call": name}, usage={"input_tokens": 100, "output_tokens": 10},
    )


def answer(text: str) -> LLMReply:
    return LLMReply(text=text, tool_calls=[], finish="stop", raw_content={"role": "model", "text": text},
                    usage={"input_tokens": 200, "output_tokens": 50})
