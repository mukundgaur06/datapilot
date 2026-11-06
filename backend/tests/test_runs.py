"""Background runs, cancellation and the SSE stream (no network, scripted LLM)."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.llm import LLMReply
from app.agent.loop import run_agent
from app.main import app, get_llm
from app.runs import RunManager, runs
from app.store import DatasetNotFound, DatasetStore
from tests.fakes import ScriptedLLM, answer, call

client = TestClient(app)
CSV = b"label,length\nspam,40\nham,12\nham,20\nspam,55\n"


class BlockingLLM(ScriptedLLM):
    """Blocks inside chat() until released: simulates a slow model call so we can cancel mid-run."""

    def __init__(self, replies: list[LLMReply]) -> None:
        super().__init__(replies)
        self.entered = threading.Event()
        self.release = threading.Event()

    def chat(self, **kwargs: Any) -> LLMReply:
        self.entered.set()
        assert self.release.wait(5), "test never released the LLM"
        return super().chat(**kwargs)


def parse_sse(text: str) -> list[dict[str, Any]]:
    messages = []
    for block in text.strip().split("\n\n"):
        msg: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            msg[key] = value
        if "event" in msg:
            msg["data"] = json.loads(msg["data"])
            messages.append(msg)
    return messages


def stream(run_id: str, **headers: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/runs/{run_id}/stream", headers=headers) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse_sse(r.read().decode())


@pytest.fixture
def api_dataset_id() -> str:
    """A dataset uploaded through the API (lives in the app's global store)."""
    return client.post("/datasets", files={"file": ("sms.csv", CSV, "text/csv")}).json()["dataset_id"]


@pytest.fixture
def use_llm():
    def install(llm: Any) -> Any:
        app.dependency_overrides[get_llm] = lambda: llm
        return llm

    yield install
    app.dependency_overrides.clear()


# ---------------------------------------------------------------- loop-level cancellation


def test_cancel_before_start_makes_no_model_call(store: DatasetStore, dataset_id: str) -> None:
    llm, cancel = ScriptedLLM([]), threading.Event()
    cancel.set()
    events = list(run_agent("q", dataset_id, llm=llm, store=store, cancel=cancel))
    assert [e.type for e in events] == ["cancelled"]
    assert llm.calls == []


def test_cancel_during_model_call_discards_its_reply(store: DatasetStore, dataset_id: str) -> None:
    cancel = threading.Event()

    class CancelsWhileThinking(ScriptedLLM):
        def chat(self, **kwargs: Any) -> LLMReply:
            cancel.set()  # the user clicks Stop while the model is still working
            return super().chat(**kwargs)

    events = list(run_agent("q", dataset_id, llm=CancelsWhileThinking([call("profile_dataset")]), store=store, cancel=cancel))
    assert [e.type for e in events] == ["cancelled"]  # the tool call it asked for never runs


def test_tool_events_share_a_stable_key(store: DatasetStore, dataset_id: str) -> None:
    llm = ScriptedLLM([call("profile_dataset", call_id=None), answer("ok")])
    events = list(run_agent("q", dataset_id, llm=llm, store=store))
    assert events[0].data["key"] == events[1].data["key"] == "1.0"


# ---------------------------------------------------------------- run manager


def test_run_manager_rejects_unknown_dataset_synchronously() -> None:
    with pytest.raises(DatasetNotFound):
        RunManager(store=DatasetStore()).start("q", "nope", ScriptedLLM([]))


def test_run_manager_evicts_only_finished_runs(store: DatasetStore, dataset_id: str) -> None:
    manager = RunManager(store=store, max_runs=2)
    blocked = BlockingLLM([answer("late")])
    first = manager.start("q", dataset_id, blocked)  # stays running
    done = [manager.start("q", dataset_id, ScriptedLLM([answer("ok")])) for _ in range(3)]
    for r in done:
        r.wait_for_events(0, timeout=5)
    manager.start("q", dataset_id, ScriptedLLM([answer("ok")]))
    assert manager.get(first.id) is first  # never evicted while running
    blocked.release.set()


def test_crashing_llm_still_ends_the_run(store: DatasetStore, dataset_id: str) -> None:
    run = RunManager(store=store).start("q", dataset_id, ScriptedLLM([RuntimeError("boom")]))
    run.wait_for_events(0, timeout=5)
    assert run.status == "error"
    assert run.events[-1].data["message"] == "Internal error while running the agent."


# ---------------------------------------------------------------- HTTP + SSE


def test_run_streams_events_then_end(api_dataset_id: str, use_llm) -> None:
    use_llm(ScriptedLLM([call("group_aggregate", group_by=["label"]), answer("50% are spam.")]))
    r = client.post("/runs", json={"dataset_id": api_dataset_id, "question": "Share of spam?"})
    assert r.status_code == 201
    run_id = r.json()["run_id"]

    messages = stream(run_id)
    assert [m["event"] for m in messages] == ["tool_call", "tool_result", "answer", "end"]
    assert [m.get("id") for m in messages] == ["0", "1", "2", None]
    assert messages[2]["data"]["data"]["text"] == "50% are spam."
    assert messages[-1]["data"] == {"status": "done"}

    snapshot = client.get(f"/runs/{run_id}").json()
    assert snapshot["status"] == "done" and len(snapshot["events"]) == 3


def test_reconnect_with_last_event_id_replays_only_missed_events(api_dataset_id: str, use_llm) -> None:
    use_llm(ScriptedLLM([call("group_aggregate", group_by=["label"]), answer("done")]))
    run_id = client.post("/runs", json={"dataset_id": api_dataset_id, "question": "q"}).json()["run_id"]
    stream(run_id)  # let it finish

    resumed = stream(run_id, **{"Last-Event-ID": "0"})
    assert [m["event"] for m in resumed] == ["tool_result", "answer", "end"]
    assert resumed[0]["id"] == "1"


def test_cancel_stops_a_running_agent(api_dataset_id: str, use_llm) -> None:
    llm = use_llm(BlockingLLM([call("profile_dataset"), answer("never sent")]))
    run_id = client.post("/runs", json={"dataset_id": api_dataset_id, "question": "q"}).json()["run_id"]
    assert llm.entered.wait(5)

    r = client.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 202
    llm.release.set()

    messages = stream(run_id)
    assert [m["event"] for m in messages] == ["cancelled", "end"]
    assert messages[-1]["data"] == {"status": "cancelled"}


def test_run_endpoints_404(use_llm) -> None:
    use_llm(ScriptedLLM([]))
    assert client.post("/runs", json={"dataset_id": "nope", "question": "q"}).status_code == 404
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/stream").status_code == 404
    assert client.post("/runs/nope/cancel").status_code == 404


def test_global_manager_is_used_by_api(api_dataset_id: str, use_llm) -> None:
    use_llm(ScriptedLLM([answer("hi")]))
    run_id = client.post("/runs", json={"dataset_id": api_dataset_id, "question": "q"}).json()["run_id"]
    assert runs.get(run_id).question == "q"
