"""FastAPI app: upload a CSV, ask questions about it.

Phase 1 endpoints are non-streaming (`POST /ask` returns all events at once).
Phase 2 adds `GET /runs/{id}/stream` over Server-Sent Events using the same run_agent().

Handlers are plain `def` (not `async def`): pandas and the Gemini SDK call here are
blocking, so FastAPI runs them in its threadpool instead of freezing the event loop.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.llm import LLM, GeminiLLM, LLMConfigError
from app.agent.loop import AgentEvent, run_agent
from app.config import settings
from app.store import DatasetNotFound, InvalidDataset, store

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
    try:
        return AskOut(events=list(run_agent(body.question, body.dataset_id, llm=llm)))
    except DatasetNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Dataset not found. Upload it first.") from None
