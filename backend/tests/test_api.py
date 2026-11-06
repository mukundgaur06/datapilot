from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, get_llm
from tests.fakes import ScriptedLLM, answer, call

client = TestClient(app)
CSV = b"label,length\nspam,40\nham,12\nham,20\nspam,55\n"


def upload(content: bytes = CSV, name: str = "sms.csv"):
    return client.post("/datasets", files={"file": (name, content, "text/csv")})


def test_health() -> None:
    assert client.get("/health").json()["ok"] is True


def test_upload_and_fetch_schema() -> None:
    r = upload()
    assert r.status_code == 201
    body = r.json()
    assert body["summary"]["n_rows"] == 4
    assert client.get(f"/datasets/{body['dataset_id']}").status_code == 200
    assert client.get("/datasets/missing").status_code == 404


def test_upload_validation() -> None:
    assert upload(name="notes.txt").status_code == 415
    assert upload(content=b"").status_code == 422


def test_ask_runs_agent_with_injected_llm() -> None:
    dataset_id = upload().json()["dataset_id"]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(
        [call("group_aggregate", group_by=["label"]), answer("50% of messages are spam.")]
    )
    try:
        r = client.post("/ask", json={"dataset_id": dataset_id, "question": "What share is spam?"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    events = r.json()["events"]
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer"]
    assert {row["label"]: row["pct_of_rows"] for row in events[1]["data"]["result"]["rows"]} == {"spam": 50.0, "ham": 50.0}


def test_ask_unknown_dataset_404() -> None:
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM([])
    try:
        r = client.post("/ask", json={"dataset_id": "nope", "question": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 404
