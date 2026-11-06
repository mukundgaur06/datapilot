"""The agent loop: reason -> act (call a tool) -> observe -> repeat, until the model answers.

This is the ReAct pattern, written by hand on purpose. It is a *generator* that yields
events as they happen, which gives us three things for free:
  * the CLI prints them,
  * the API (Phase 2) streams them to the browser over Server-Sent Events,
  * tests assert on them.

Safety rails: a hard step limit, size-capped tool results, and tool errors returned to
the model as observations (so it can fix its own mistakes) instead of crashing the run.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from time import perf_counter
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field

from app.agent.llm import LLM, LLMError, ToolCall, ToolResult
from app.agent.prompts import build_system_prompt
from app.config import settings
from app.store import DatasetStore
from app.store import store as default_store
from app.tools import ToolError, ToolRegistry
from app.tools import registry as default_registry

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000
EventType = Literal["thinking", "tool_call", "tool_result", "answer", "error"]


class AgentEvent(BaseModel):
    type: EventType
    step: int
    data: dict[str, Any] = Field(default_factory=dict)


def serialize_result(result: dict[str, Any]) -> str:
    """JSON for the model, capped so one big result can't flood the context window."""
    text = json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + (
            ' ...[truncated: result too large. Narrow the query, e.g. lower "limit" or add filters]'
        )
    return text


def _execute(registry: ToolRegistry, call: ToolCall, df: Any) -> tuple[dict[str, Any], bool]:
    try:
        return registry.dispatch(call.name, df, call.args), False
    except ToolError as e:
        return {"error": str(e)}, True
    except Exception:  # a bug in our tool, not the model's fault: log it, don't kill the run
        logger.exception("Tool %s crashed with args %s", call.name, call.args)
        return {"error": "The tool failed unexpectedly. Try different arguments or another tool."}, True


def run_agent(
    question: str,
    dataset_id: str,
    *,
    llm: LLM,
    store: DatasetStore = default_store,
    registry: ToolRegistry = default_registry,
    max_steps: int | None = None,
) -> Iterator[AgentEvent]:
    """Answer `question` about a stored dataset, yielding AgentEvents along the way.

    Raises DatasetNotFound before yielding anything if the dataset id is unknown.
    """
    max_steps = max_steps or settings.max_steps
    df = store.frame(dataset_id)
    system = build_system_prompt(store.schema_summary(dataset_id))
    tools = registry.schemas()
    history: list[Any] = [llm.user_message(question)]
    usage: Counter[str] = Counter()
    started = perf_counter()

    def stats(step: int) -> dict[str, Any]:
        return {"steps": step, "usage": dict(usage), "elapsed_ms": round((perf_counter() - started) * 1000)}

    for step in range(1, max_steps + 1):
        try:
            reply = llm.chat(system=system, history=history, tools=tools)
        except LLMError as e:
            yield AgentEvent(type="error", step=step, data={"message": str(e), **stats(step)})
            return
        usage.update(reply.usage)
        for thought in reply.thoughts:
            yield AgentEvent(type="thinking", step=step, data={"text": thought})
        if reply.raw_content is not None:
            history.append(reply.raw_content)  # verbatim: preserves provider metadata

        if reply.finish == "max_tokens":
            yield AgentEvent(type="error", step=step, data={"message": "The model ran out of output tokens.", **stats(step)})
            return
        if reply.finish == "blocked":
            msg = f"The model provider blocked this response ({reply.finish_detail})."
            yield AgentEvent(type="error", step=step, data={"message": msg, **stats(step)})
            return
        if reply.finish == "malformed":
            # Treat a garbled tool call like a tool error: tell the model and let it retry.
            history.append(llm.user_message("Your last tool call was malformed. Call the tool again with valid arguments."))
            continue

        if not reply.tool_calls:
            if not reply.text:
                yield AgentEvent(type="error", step=step, data={"message": "The model returned an empty answer.", **stats(step)})
                return
            yield AgentEvent(type="answer", step=step, data={"text": reply.text, "partial": False, **stats(step)})
            return

        results: list[ToolResult] = []
        for call in reply.tool_calls:
            yield AgentEvent(type="tool_call", step=step, data={"id": call.id, "name": call.name, "args": call.args})
            t0 = perf_counter()
            result, is_error = _execute(registry, call, df)
            yield AgentEvent(
                type="tool_result",
                step=step,
                data={
                    "id": call.id,
                    "name": call.name,
                    "result": result,
                    "is_error": is_error,
                    "duration_ms": round((perf_counter() - t0) * 1000, 1),
                },
            )
            results.append(ToolResult(call=call, content=serialize_result(result), is_error=is_error))
        history.append(llm.tool_results_message(results))

    yield AgentEvent(
        type="answer",
        step=max_steps,
        data={
            "text": f"I reached the step limit ({max_steps}) before finishing. The tool results above are partial findings.",
            "partial": True,
            **stats(max_steps),
        },
    )
