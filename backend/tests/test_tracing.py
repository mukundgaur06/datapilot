"""Traces: what gets recorded, that it's written as JSONL, and the API endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.loop import run_agent
from app.main import app, get_llm
from app.store import DatasetStore
from app.tracing import Tracer
from tests.fakes import ScriptedLLM, answer, call


def test_trace_records_model_calls_tools_and_outcome(tmp_path: Path, store: DatasetStore, dataset_id: str) -> None:
    tracer = Tracer("r1", tmp_path / "r1.jsonl")
    llm = ScriptedLLM([call("group_aggregate", group_by=["Segment"]), call("nope", call_id="c2"), answer("done")])
    list(run_agent("How many per segment?", dataset_id, llm=llm, store=store, tracer=tracer))

    kinds = [r["kind"] for r in tracer.records]
    assert kinds == ["run_start", "llm_call", "tool_call", "tool_result", "llm_call", "tool_call", "tool_result", "llm_call", "run_end"]
    start = tracer.records[0]
    assert start["question"] == "How many per segment?" and start["dataset"]["n_rows"] == 8
    assert "Segment" not in json.dumps(start["system_prompt"])  # prompt stored as hash + length only
    assert tracer.records[1]["usage"] == {"input_tokens": 100, "output_tokens": 10} and tracer.records[1]["n_tool_calls"] == 1
    assert all(r["t_ms"] >= 0 for r in tracer.records)

    end = tracer.records[-1]
    assert end["status"] == "done" and end["steps"] == 3
    assert end["summary"]["llm_calls"] == 3 and end["summary"]["tool_calls"] == 2 and end["summary"]["tool_errors"] == 1
    assert end["summary"]["usage"] == {"input_tokens": 400, "output_tokens": 70}

    lines = (tmp_path / "r1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == kinds


def test_tool_result_preview_drops_ui_data_and_is_capped(store: DatasetStore, dataset_id: str) -> None:
    tracer = Tracer("r2")
    llm = ScriptedLLM([call("make_chart", kind="scatter", x="Income_k", y="Spend"), answer("ok")])
    list(run_agent("q", dataset_id, llm=llm, store=store, tracer=tracer))
    result = next(r for r in tracer.records if r["kind"] == "tool_result")
    assert '"ui"' not in result["result_preview"] and len(result["result_preview"]) <= 2000


def test_consumer_stopping_early_still_closes_the_trace(store: DatasetStore, dataset_id: str) -> None:
    tracer = Tracer("r3")
    events = run_agent("q", dataset_id, llm=ScriptedLLM([call("profile_dataset"), answer("ok")]), store=store, tracer=tracer)
    next(events)  # read one event, then abandon the generator
    events.close()
    assert tracer.records[-1]["kind"] == "run_end" and tracer.records[-1]["status"] == "aborted"


def test_trace_endpoint() -> None:
    client = TestClient(app)
    dataset_id = client.post("/datasets", files={"file": ("d.csv", b"a,b\n1,2\n3,4\n", "text/csv")}).json()["dataset_id"]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM([call("profile_dataset"), answer("ok")])
    try:
        run_id = client.post("/runs", json={"dataset_id": dataset_id, "question": "q"}).json()["run_id"]
        with client.stream("GET", f"/runs/{run_id}/stream") as r:
            r.read()  # wait for the run to finish
        body = client.get(f"/runs/{run_id}/trace").json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "done"
    assert body["summary"]["llm_calls"] == 2 and body["summary"]["tool_calls"] == 1
    assert body["records"][-1]["kind"] == "run_end"
    assert client.get("/runs/missing/trace").status_code == 404
