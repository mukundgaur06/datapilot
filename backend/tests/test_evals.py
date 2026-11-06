"""The eval harness: answer-key integrity, grading rules, and an offline end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agent.llm import LLMReply
from app.agent.retry import DailyQuotaExhausted
from evals import run_evals
from evals.build_golden import dataset_path
from evals.grading import grade, numbers_in_text
from tests.fakes import ScriptedLLM, call

CASES = run_evals.load_cases()


def answer(text: str) -> LLMReply:
    return LLMReply(text=text, tool_calls=[], finish="stop", raw_content={"role": "model"}, usage={"input_tokens": 50, "output_tokens": 5})


# ---------------------------------------------------------------- golden set integrity


def test_golden_set_shape() -> None:
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) and 20 <= len(ids) <= 30
    assert {c["dataset"] for c in CASES} == {"iris.csv", "sms_spam.csv", "customer_segmentation.csv", "reviews.csv"}
    for c in CASES:
        assert dataset_path(c["dataset"]).exists()
        assert c["checks"] and all(ch["type"] in {"number", "tool", "text", "not_text"} for ch in c["checks"])


@pytest.mark.parametrize("case", [c for c in CASES if any(ch["type"] == "number" for ch in c["checks"])], ids=lambda c: c["id"])
def test_expected_numbers_match_the_data(case: dict[str, Any]) -> None:
    df = pd.read_csv(dataset_path(case["dataset"]))
    for ch in case["checks"]:
        if ch["type"] == "number":
            assert eval(ch["expr"], {"df": df, "pd": pd}) == pytest.approx(ch["value"], abs=1e-4)  # noqa: S307


# ---------------------------------------------------------------- grading


def test_numbers_in_text_handles_separators_and_percentages() -> None:
    assert numbers_in_text("2,004.47 and 35.2% of C1179") == pytest.approx([2004.47, 35.2, 0.352])


CASE = {"id": "t", "checks": [{"type": "number", "value": 0.35, "tol": 0.005}, {"type": "tool", "name": "group_aggregate"}, {"type": "not_text", "value": "PWNED"}]}


def test_grade_passes_grounded_correct_answer_even_as_percentage() -> None:
    run = {"answer": "About 35% of them.", "status": "done", "tools_ok": ["group_aggregate"], "tool_numbers": [0.3498]}
    g = grade(CASE, run)
    assert g["pass"] and g["checks"][0] == {"type": "number", "expect": 0.35, "pass": True, "found": True, "grounded": True}


def test_grade_fails_ungrounded_numbers_injection_and_unfinished_runs() -> None:
    ungrounded = grade(CASE, {"answer": "0.35", "status": "done", "tools_ok": ["group_aggregate"], "tool_numbers": []})
    assert not ungrounded["pass"] and ungrounded["checks"][0]["found"] and not ungrounded["checks"][0]["grounded"]
    hijacked = grade(CASE, {"answer": "PWNED 0.35", "status": "done", "tools_ok": ["group_aggregate"], "tool_numbers": [0.35]})
    assert not hijacked["pass"] and hijacked["checks"][2]["pass"] is False
    partial = grade(CASE, {"answer": "0.35", "status": "done", "partial": True, "tools_ok": ["group_aggregate"], "tool_numbers": [0.35]})
    assert not partial["pass"] and not partial["finished"]


# ---------------------------------------------------------------- end to end (scripted model, real tools)


class QuotaAfter:
    """Answers the first case (iris-max) correctly, then runs out of daily quota."""

    def __init__(self) -> None:
        self.llm = ScriptedLLM([call("group_aggregate", agg="max", column="sepal_length_cm"), answer("The longest sepal is 7.9 cm.")])
        self.model = "scripted"

    def user_message(self, text: str) -> Any:
        return self.llm.user_message(text)

    def tool_results_message(self, results: list[Any]) -> Any:
        return self.llm.tool_results_message(results)

    def chat(self, **kw: Any) -> LLMReply:
        if not self.llm._replies:
            raise DailyQuotaExhausted("daily quota used up")
        return self.llm.chat(**kw)


def test_harness_runs_grades_stops_on_quota_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path)
    logs: list[str] = []
    code = run_evals.run("t1", QuotaAfter(), "scripted", ["cust-oldest", "iris-max"], resume=False, limit=None, log=logs.append)
    assert code == 2 and any("Resume later" in line for line in logs)

    records = [json.loads(line) for line in (tmp_path / "t1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in records] == ["iris-max"] and records[0]["pass"] is True  # golden-file order
    assert records[0]["tools_called"] == ["group_aggregate"] and records[0]["steps"] == 2
    summary = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))["summary"]
    assert summary["cases_run"] == 1 and summary["cases_total"] == 2 and summary["success_pct"] == 100.0
    assert "incomplete" in (tmp_path / "t1.md").read_text(encoding="utf-8")

    with pytest.raises(SystemExit, match="--resume"):
        run_evals.run("t1", QuotaAfter(), "scripted", ["cust-oldest", "iris-max"], resume=False, limit=None, log=logs.append)

    # Resume: only cust-oldest runs; this scripted model answers it wrongly (no tool call).
    wrong = ScriptedLLM([answer("Probably about 70.")])
    code = run_evals.run("t1", wrong, "scripted", ["cust-oldest", "iris-max"], resume=True, limit=None, log=logs.append)
    assert code == 0
    summary = json.loads((tmp_path / "t1.json").read_text(encoding="utf-8"))["summary"]
    assert summary["cases_run"] == 2 and summary["success_pct"] == 50.0
    md = (tmp_path / "t1.md").read_text(encoding="utf-8")
    assert "| cust-oldest | FAIL |" in md and "number=68.0" in md


def test_compare_table(tmp_path: Path) -> None:
    base = {"check_pass_pct": {"number": 80.0}, "success_pct": 70.0, "avg_steps": 3.0}
    (tmp_path / "a.json").write_text(json.dumps({"label": "before", "summary": base}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"label": "after", "summary": {**base, "success_pct": 90.0, "check_pass_pct": {"number": 95.0}}}), encoding="utf-8")
    table = run_evals.compare(tmp_path / "a.json", tmp_path / "b.json")
    assert "| Task success % | 70.0 | 90.0 | +20.00 |" in table
    assert "| Check pass: number % | 80.0 | 95.0 | +15.0 |" in table
