"""FastAPI app: upload a CSV, ask questions about it.

`POST /ask` is non-streaming (returns all events at once). The UI uses runs instead:
`POST /runs` starts the agent in the background, `GET /runs/{id}/stream` streams its
events over Server-Sent Events, and `POST /runs/{id}/cancel` stops it.

Most handlers are plain `def` (not `async def`): pandas and the Gemini SDK are
blocking, so FastAPI runs them in its threadpool instead of freezing the event loop.
The SSE endpoint is async because it mostly waits, holding a connection open.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, AsyncIterator

import anyio
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.llm import LLM, GeminiLLM, LLMConfigError
from app.agent.loop import AgentEvent, run_agent
from app.config import settings
from app.runs import Run, RunNotFound, RunStatus, runs
from app.store import DatasetNotFound, InvalidDataset, store

SSE_KEEPALIVE_S = 15.0
SSE_RETRY_MS = 2000

app = FastAPI(title="DataPilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class DatasetOut(BaseModel):
    dataset_id: str
    summary: dict[str, Any]


class AskIn(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=2000)


class AskOut(BaseModel):
    events: list[AgentEvent]


@lru_cache(maxsize=1)
def _gemini() -> GeminiLLM:
    return GeminiLLM()


def get_llm() -> LLM:
    """Dependency, so tests can swap in a fake via app.dependency_overrides."""
    try:
        return _gemini()
    except LLMConfigError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": settings.model, "llm_configured": bool(settings.gemini_api_key)}


@app.post("/datasets", status_code=status.HTTP_201_CREATED, response_model=DatasetOut)
def upload_dataset(file: UploadFile = File(...)) -> DatasetOut:
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only .csv files are supported.")
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # read at most limit+1 bytes: never load a huge file into memory
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds {settings.max_upload_mb} MB.")
    try:
        dataset_id = store.load_csv(data, name=file.filename or "upload.csv")
    except InvalidDataset as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    return DatasetOut(dataset_id=dataset_id, summary=store.schema_summary(dataset_id))


@app.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str) -> DatasetOut:
    try:
        return DatasetOut(dataset_id=dataset_id, summary=store.schema_summary(dataset_id))
    except DatasetNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Dataset not found.") from None


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, llm: LLM = Depends(get_llm)) -> AskOut:
    """Non-streaming: runs the whole agent, then returns every event. Handy for scripts and curl."""
    try:
        return AskOut(events=list(run_agent(body.question, body.dataset_id, llm=llm)))
    except DatasetNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Dataset not found. Upload it first.") from None


# ---------------------------------------------------------------- streaming runs (Phase 2)


class RunOut(BaseModel):
    run_id: str
    dataset_id: str
    question: str
    status: RunStatus
    events: list[AgentEvent] = Field(default_factory=list)


def _run_out(run: Run, with_events: bool = True) -> RunOut:
    return RunOut(
        run_id=run.id, dataset_id=run.dataset_id, question=run.question, status=run.status,
        events=list(run.events) if with_events else [],
    )


def _get_run(run_id: str) -> Run:
    try:
        return runs.get(run_id)
    except RunNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Run not found.") from None


@app.post("/runs", status_code=status.HTTP_201_CREATED, response_model=RunOut)
def create_run(body: AskIn, llm: LLM = Depends(get_llm)) -> RunOut:
    """Start the agent in the background; watch it with GET /runs/{id}/stream."""
    try:
        run = runs.start(body.question, body.dataset_id, llm)
    except DatasetNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Dataset not found. Upload it first.") from None
    return _run_out(run, with_events=False)


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str) -> RunOut:
    return _run_out(_get_run(run_id))


@app.get("/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict[str, Any]:
    """The run's trace: model calls (latency, tokens), tool calls, outcome, with timings for a waterfall."""
    run = _get_run(run_id)
    tracer = run.tracer
    if tracer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No trace for this run.")
    return {
        "run_id": run.id, "status": run.status, "saved": tracer.path is not None,
        "records": list(tracer.records), "summary": tracer.summary(),
    }


@app.post("/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED, response_model=RunOut)
def cancel_run(run_id: str) -> RunOut:
    """Ask the agent to stop. It stops at its next checkpoint and emits a 'cancelled' event."""
    run = _get_run(run_id)
    run.cancel.set()
    return _run_out(run, with_events=False)


def _sse(event: str, data: Any, event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


@app.get("/runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    last_event_id: int | None = Header(default=None),
    after: int | None = Query(default=None, ge=-1, description="Resume after this event id (alternative to the header)."),
) -> StreamingResponse:
    """Server-Sent Events: one SSE message per AgentEvent, then a final `end` message.

    Each message's `id` is the event's index, so a reconnecting EventSource (which sends
    Last-Event-ID automatically) only receives what it missed.
    """
    run = _get_run(run_id)
    resume_after = last_event_id if last_event_id is not None else after
    cursor = 0 if resume_after is None else max(resume_after + 1, 0)

    async def events() -> AsyncIterator[str]:
        nonlocal cursor
        yield f"retry: {SSE_RETRY_MS}\n\n"  # tells EventSource how long to wait before reconnecting
        while True:
            # wait_for_events blocks, so run it in a worker thread, not on the event loop.
            new, finished = await anyio.to_thread.run_sync(run.wait_for_events, cursor, SSE_KEEPALIVE_S)
            for event in new:
                yield _sse(event.type, event.model_dump(), event_id=cursor)
                cursor += 1
            if finished and cursor >= len(run.events):
                yield _sse("end", {"status": run.status})
                return
            if not new:
                yield ": keep-alive\n\n"  # comment line: stops proxies from closing an idle connection

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
