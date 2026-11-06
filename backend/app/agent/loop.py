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
import threading
from collections import Counter
from time import perf_counter
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field

from app.agent.llm import LLM, LLMError, ToolCall, ToolResult
from app.agent.prompts import build_system_prompt
from app.config import settings
from app.store import DatasetStore
from app.store import store as default_store
from app.ml.models import ModelStore
from app.ml.models import models as default_models
from app.tools import ToolContext, ToolError, ToolRegistry
from app.tools import registry as default_registry
from app.tracing import Tracer

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000
EventType = Literal["thinking", "tool_call", "tool_result", "answer", "error", "cancelled"]
TERMINAL_EVENTS: frozenset[str] = frozenset({"answer", "error", "cancelled"})


class AgentEvent(BaseModel):
    type: EventType
    step: int
    data: dict[str, Any] = Field(default_factory=dict)


UI_ONLY_KEY = "ui"  # tools put display-only data (chart points) here: the browser gets it, the model doesn't


def serialize_result(result: dict[str, Any]) -> str:
    """JSON for the model, without display-only data, capped so one big result can't flood the context window."""
    model_view = {k: v for k, v in result.items() if k != UI_ONLY_KEY}
    text = json.dumps(model_view, default=str, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + (
            ' ...[truncated: result too large. Narrow the query, e.g. lower "limit" or add filters]'
        )
    return text


def _execute(registry: ToolRegistry, call: ToolCall, ctx: ToolContext) -> tuple[dict[str, Any], bool]:
    try:
        return registry.dispatch(call.name, ctx, call.args), False
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
    models: ModelStore = default_models,
    max_steps: int | None = None,
    cancel: threading.Event | None = None,
    tracer: Tracer | None = None,
    prompt_version: str | None = None,
) -> Iterator[AgentEvent]:
    """Answer `question` about a stored dataset, yielding AgentEvents along the way.

    Raises DatasetNotFound before yielding anything if the dataset id is unknown.
    The last event is always terminal: answer, error, or cancelled.

    `cancel` is checked before every model call and every tool call. An in-flight
    model call can't be interrupted, but its result is discarded once cancel is set.

    `tracer` records the run (model-call latency and tokens, tool calls, outcome).
    Without one, an in-memory tracer is used, so traces are never optional internally.
    """
    tracer = tracer or Tracer(run_id="adhoc")
    ended = False
    try:
        version = prompt_version or settings.prompt_version
        for event in _steps(question, dataset_id, llm, store, registry, models, max_steps or settings.max_steps, cancel, tracer, version):
            d = event.data
            if event.type == "tool_call":
                tracer.tool_call(step=event.step, key=d["key"], name=d["name"], args=d["args"])
            elif event.type == "tool_result":
                tracer.tool_result(
                    step=event.step, key=d["key"], name=d["name"], duration_ms=d["duration_ms"],
                    is_error=d["is_error"], result=d["result"],
                )
            elif event.type in TERMINAL_EVENTS:
                status = {"answer": "done", "error": "error", "cancelled": "cancelled"}[event.type]
                tracer.end(status=status, steps=event.step, message=str(d.get("message", "")))
                ended = True
            yield event
    finally:
        if not ended:  # consumer stopped early, or an unexpected exception
            tracer.end(status="aborted", steps=0)


def _steps(
    question: str,
    dataset_id: str,
    llm: LLM,
    store: DatasetStore,
    registry: ToolRegistry,
    models: ModelStore,
    max_steps: int,
    cancel: threading.Event | None,
    tracer: Tracer,
    prompt_version: str | None,
) -> Iterator[AgentEvent]:
    ctx = ToolContext(df=store.frame(dataset_id), dataset_id=dataset_id, models=models)
    summary = store.schema_summary(dataset_id)
    system = build_system_prompt(summary, prompt_version)
    tools = registry.schemas()
    tracer.start(
        question=question,
        dataset={"name": summary["name"], "n_rows": summary["n_rows"], "n_columns": summary["n_columns"]},
        model=str(getattr(llm, "model", type(llm).__name__)),
        system_prompt=system,
        tools=[t["name"] for t in tools],
    )
    history: list[Any] = [llm.user_message(question)]
    usage: Counter[str] = Counter()
    started = perf_counter()

    def stats(step: int) -> dict[str, Any]:
        return {"steps": step, "usage": dict(usage), "elapsed_ms": round((perf_counter() - started) * 1000)}

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def cancelled_event(step: int) -> AgentEvent:
        return AgentEvent(type="cancelled", step=step, data={"message": "Run stopped by the user.", **stats(step)})

    for step in range(1, max_steps + 1):
        if cancelled():
            yield cancelled_event(step)
            return
        t_llm = perf_counter()
        try:
            reply = llm.chat(system=system, history=history, tools=tools)
        except LLMError as e:
            tracer.llm_call(step=step, duration_ms=(perf_counter() - t_llm) * 1000, usage={}, finish="exception", error=str(e))
            yield AgentEvent(type="error", step=step, data={"message": str(e), **stats(step)})
            return
        tracer.llm_call(
            step=step, duration_ms=(perf_counter() - t_llm) * 1000, usage=reply.usage, finish=reply.finish,
            n_tool_calls=len(reply.tool_calls),
        )
        usage.update(reply.usage)
        if cancelled():
            yield cancelled_event(step)
            return
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
        for i, call in enumerate(reply.tool_calls):
            if cancelled():
                yield cancelled_event(step)
                return
            key = f"{step}.{i}"  # stable pairing of call <-> result for the UI (provider ids may be missing)
            yield AgentEvent(
                type="tool_call", step=step, data={"key": key, "id": call.id, "name": call.name, "args": call.args}
            )
            t0 = perf_counter()
            result, is_error = _execute(registry, call, ctx)
            yield AgentEvent(
                type="tool_result",
                step=step,
                data={
                    "key": key,
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
