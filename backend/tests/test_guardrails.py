"""Guardrails for public deployment: rate limits, concurrency cap, memory cap, headers, static serving."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.guardrails import RateLimiter, parse_limit
from app.runs import RunManager, TooBusy, runs
from app.store import DatasetNotFound, DatasetStore
from tests.fakes import ScriptedLLM, answer
from tests.test_runs import BlockingLLM

client = TestClient(main.app)
CSV = b"a,b\n1,2\n3,4\n"


def test_parse_limit() -> None:
    assert parse_limit("10/600") == (10, 600.0)
    assert parse_limit("5") == (5, 60.0)
    assert parse_limit("0") is None and parse_limit("") is None and parse_limit("off") is None


def test_sliding_window_blocks_then_frees() -> None:
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.hit("ip") is None and limiter.hit("ip") is None
    assert limiter.hit("ip") == pytest.approx(60)  # 3rd hit within the window: wait until the 1st expires
    assert limiter.hit("other-ip") is None  # limits are per client
    now[0] = 30
    assert limiter.hit("ip") == pytest.approx(30)
    now[0] = 60.01
    assert limiter.hit("ip") is None


def test_upload_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(main.limiters, "uploads", RateLimiter(2, 3600))
    codes = [client.post("/datasets", files={"file": ("d.csv", CSV, "text/csv")}).status_code for _ in range(3)]
    assert codes == [201, 201, 429]
    r = client.post("/datasets", files={"file": ("d.csv", CSV, "text/csv")})
    assert int(r.headers["Retry-After"]) > 3500 and "shared demo" in r.json()["detail"]


def test_run_rate_limit_applies_to_runs_and_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = client.post("/datasets", files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    monkeypatch.setitem(main.limiters, "runs", RateLimiter(1, 600))
    main.app.dependency_overrides[main.get_llm] = lambda: ScriptedLLM([answer("ok")])
    try:
        assert client.post("/runs", json={"dataset_id": dataset_id, "question": "q"}).status_code == 201
        assert client.post("/ask", json={"dataset_id": dataset_id, "question": "q"}).status_code == 429
    finally:
        main.app.dependency_overrides.clear()


def test_concurrency_cap(store: DatasetStore, dataset_id: str) -> None:
    manager = RunManager(store=store, max_active=1)
    blocked = BlockingLLM([answer("late")])
    manager.start("q", dataset_id, blocked)
    with pytest.raises(TooBusy):
        manager.start("q", dataset_id, ScriptedLLM([answer("ok")]))
    blocked.release.set()


def test_api_returns_503_when_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = client.post("/datasets", files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    monkeypatch.setattr(runs, "max_active", 0)
    main.app.dependency_overrides[main.get_llm] = lambda: ScriptedLLM([])
    try:
        r = client.post("/runs", json={"dataset_id": dataset_id, "question": "q"})
    finally:
        main.app.dependency_overrides.clear()
    assert r.status_code == 503 and r.headers["Retry-After"] == "30"


def test_dataset_store_evicts_least_recently_used() -> None:
    s = DatasetStore(max_datasets=2)
    a = s.add(pd.DataFrame({"x": [1]}), "a")
    b = s.add(pd.DataFrame({"x": [2]}), "b")
    s.get(a)  # touch a, so b is now the least recently used
    c = s.add(pd.DataFrame({"x": [3]}), "c")
    assert s.get(a).name == "a" and s.get(c).name == "c"
    with pytest.raises(DatasetNotFound):
        s.get(b)


def test_security_headers() -> None:
    h = client.get("/health").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"


def test_serves_built_frontend_without_shadowing_the_api(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>DataPilot</title>", encoding="utf-8")
    code = (
        "import json; from fastapi.testclient import TestClient; from app.main import app; c = TestClient(app); "
        "print(json.dumps([c.get('/').text, c.get('/health').json()['ok'], c.get('/datasets/nope').status_code]))"
    )
    env = {**os.environ, "DATAPILOT_STATIC_DIR": str(tmp_path), "DATAPILOT_TRACING": "0"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1])
    page, health_ok, missing = json.loads(out.stdout.strip().splitlines()[-1])
    assert "<title>DataPilot</title>" in page and health_ok is True and missing == 404
