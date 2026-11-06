"""Phase 1 acceptance check: 5 hand-written questions against the real LLM.

    python -m scripts.phase1_check            (from backend/, needs GEMINI_API_KEY)

For each question we check two things:
  1. Correct:  every expected number appears in the final answer (within a tolerance).
  2. Grounded: every expected number also appears in some tool result, i.e. it was
     computed by pandas, not made up by the model.
Expected values were computed directly with pandas on the sample CSVs.
This is the seed of the Phase 4 golden-set evals.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.agent.llm import GeminiLLM, LLMConfigError
from app.agent.loop import run_agent
from app.store import DatasetStore

DATA = Path(__file__).resolve().parents[2] / "sample_data"
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Case:
    csv: str
    question: str
    expected: list[float]  # numbers that must appear in the answer
    words: tuple[str, ...] = ()  # strings that must appear in the answer
    tol: float = 0.01


CASES = [
    Case("sms_spam.csv", "What percentage of messages are spam?", [50]),
    Case("customer_segmentation.csv", "What is the average annual income of customers older than 40?", [83.83], tol=0.01),
    Case(
        "customer_segmentation.csv",
        "Which numeric column is most strongly correlated with Spending_Score, and how strong is it?",
        [0.77],
        words=("Website_Visits_Per_Month",),
        tol=0.005,
    ),
    Case(
        "customer_segmentation.csv",
        "How many customers haven't purchased in more than 60 days, and what share of all customers is that?",
        [422, 35.17],
    ),
    Case(
        "customer_segmentation.csv",
        "For customers with at least 10 orders in the last 90 days, what is the median number of website visits per month?",
        [17],
    ),
]


def numbers_in_text(text: str) -> list[float]:
    out = []
    for m in NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_in(obj: Any) -> Iterator[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers_in(v)


def close(x: float, target: float, tol: float) -> bool:
    # Allow the model to round (e.g. 0.7714 -> 0.77): accept within tol, absolute or relative.
    return abs(x - target) <= max(tol, abs(target) * 1e-3)


def run_case(case: Case, llm: GeminiLLM) -> bool:
    store = DatasetStore()
    dataset_id = store.load_csv(DATA / case.csv)
    answer, tool_numbers, n_calls, problems = "", [], 0, []
    for ev in run_agent(case.question, dataset_id, llm=llm, store=store):
        if ev.type == "tool_call":
            n_calls += 1
        elif ev.type == "tool_result" and not ev.data["is_error"]:
            tool_numbers.extend(numbers_in(ev.data["result"]))
        elif ev.type == "answer":
            answer = ev.data["text"]
            if ev.data.get("partial"):
                problems.append("hit the step limit")
        elif ev.type == "error":
            problems.append(f"run failed: {ev.data['message']}")

    answer_numbers = numbers_in_text(answer)
    for target in case.expected:
        if not any(close(x, target, case.tol) for x in answer_numbers):
            problems.append(f"answer is missing {target}")
        if not any(close(x, target, case.tol) for x in tool_numbers):
            problems.append(f"{target} never appeared in a tool result (ungrounded)")
    problems += [f"answer doesn't mention {w!r}" for w in case.words if w not in answer]
    if n_calls == 0:
        problems.append("no tool was called")

    status = "PASS" if not problems else "FAIL"
    print(f"[{status}] {case.question}  ({n_calls} tool calls)")
    print("   " + answer.replace("\n", "\n   ")[:600])
    for p in problems:
        print(f"   ! {p}")
    print()
    return not problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        llm = GeminiLLM()
    except LLMConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    passed = sum(run_case(c, llm) for c in CASES)
    print(f"{passed}/{len(CASES)} passed  (model: {llm.model})")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
