"""Tool registry: turns plain Python functions into tools an LLM can call.

Each tool = name + description + a Pydantic model for its arguments + a function.
The Pydantic model does double duty:
  1. It generates the JSON Schema the LLM sees (so docs and validation never drift apart).
  2. It validates whatever arguments the LLM sends back. LLM output is untrusted input.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, ValidationError

ToolFn = Callable[[pd.DataFrame, Any], dict[str, Any]]


class ToolError(Exception):
    """A problem the model can fix itself (wrong column, bad argument).

    The agent loop sends the message back to the model as an observation instead
    of crashing, so write messages that tell the model what to do differently.
    """


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs and drop 'title' keys so the schema is flat and portable.

    Pydantic emits $defs for nested models; not every LLM provider resolves them.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                return walk(copy.deepcopy(target))
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors(include_url=False):
        loc = ".".join(str(x) for x in err["loc"]) or "(arguments)"
        parts.append(f"{loc}: {err['msg']}")
    return "Invalid arguments: " + "; ".join(parts)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, *, name: str, description: str, args_model: type[BaseModel]) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            if name in self._tools:
                raise ValueError(f"Tool {name!r} registered twice")
            self._tools[name] = Tool(name, description, args_model, fn)
            return fn

        return decorator

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-neutral tool specs. Sorted, so the prompt prefix is stable (cache-friendly)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _inline_refs(t.args_model.model_json_schema()),
            }
            for t in sorted(self._tools.values(), key=lambda t: t.name)
        ]

    def dispatch(self, name: str, df: pd.DataFrame, raw_args: dict[str, Any] | None) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool {name!r}. Available tools: {self.names}")
        try:
            args = tool.args_model.model_validate(raw_args or {})
        except ValidationError as e:
            raise ToolError(_format_validation_error(e)) from e
        return tool.fn(df, args)


registry = ToolRegistry()
tool = registry.register
