"""Programmatic grading of one agent run against a golden case. Pure functions: unit-tested.

Why programmatic checks instead of an LLM judge: every check here is objective (a number,
a tool call, a phrase), so code grades it exactly, for free, with no judge bias. An LLM
judge is only worth its cost and noise for open-ended qualities this set doesn't test.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

# A number, optionally followed by a percent sign. Thousands separators allowed.
NUMBER = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)(\s*%)?")


def numbers_in_text(text: str) -> list[float]:
    """Every number in the text. "35.2%" also yields 0.352, so fractions and percentages both match."""
    out: list[float] = []
    for raw, pct in NUMBER.findall(text):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.append(value)
        if pct:
            out.append(value / 100)
    return out


def numbers_in(obj: Any) -> Iterator[float]:
    """Every number inside a JSON-like structure (tool results). Also parses numeric strings."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        yield from numbers_in_text(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers_in(v)


def close(x: float, target: float, tol: float) -> bool:
    """Within an absolute tolerance, or 0.1% relative (so rounding 2004.47 to 2,004 still counts)."""
    return abs(x - target) <= max(tol, abs(target) * 1e-3)


def grade(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Grade a run: {answer, status, tools_ok: [names], tool_numbers: [...]} -> per-check results.

    A case passes only if the run finished with an answer and every check passes.
    Number checks also record grounding: was the expected value present in a tool result?
    """
    answer = run.get("answer") or ""
    lower = answer.lower()
    answer_numbers = numbers_in_text(answer)
    tool_numbers = run.get("tool_numbers", [])
    tools_ok = set(run.get("tools_ok", []))

    checks = []
    for c in case["checks"]:
        kind = c["type"]
        if kind == "number":
            found = any(close(x, c["value"], c["tol"]) for x in answer_numbers)
            grounded = any(close(x, c["value"], c["tol"]) for x in tool_numbers)
            checks.append({"type": kind, "expect": c["value"], "pass": found and grounded, "found": found, "grounded": grounded})
        elif kind == "tool":
            checks.append({"type": kind, "expect": c["name"], "pass": c["name"] in tools_ok})
        elif kind == "text":
            checks.append({"type": kind, "expect": c["any"], "pass": any(a.lower() in lower for a in c["any"])})
        elif kind == "not_text":
            checks.append({"type": kind, "expect": c["value"], "pass": c["value"].lower() not in lower})
        else:
            raise ValueError(f"Unknown check type {kind!r} in case {case['id']}")

    finished = run.get("status") == "done" and not run.get("partial")
    return {"pass": finished and all(ch["pass"] for ch in checks), "finished": finished, "checks": checks}
