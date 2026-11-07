"""Central settings, read once from environment variables (and backend/.env).

Keeping every knob here means no magic strings scattered around the code, and
secrets (the API key) never get hardcoded or committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    model: str
    thinking_level: str  # MINIMAL | LOW | MEDIUM | HIGH: an eval-tunable knob (Phase 4)
    max_steps: int  # hard cap on agent iterations: protects cost and avoids infinite loops
    max_upload_mb: int
    cors_origins: tuple[str, ...]
    trace_dir: Path | None  # None = traces kept in memory only
    prompt_version: str | None  # None = latest; pin one (e.g. "v1") for A/B evals
    # Guardrails for public deployments (see app/guardrails.py)
    rate_limit_runs: str  # "N/seconds" per client IP for questions; "0" disables
    rate_limit_uploads: str
    max_active_runs: int  # agents running at the same time, across all users
    max_datasets: int  # uploads kept in memory; least recently used are dropped
    static_dir: Path | None  # built frontend to serve from "/" (single-container deploy)


def _load() -> Settings:
    origins = os.getenv("DATAPILOT_CORS_ORIGINS", "http://localhost:5173")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
        model=os.getenv("DATAPILOT_MODEL", "gemini-3.5-flash-lite"),
        thinking_level=os.getenv("DATAPILOT_THINKING_LEVEL", "LOW").upper(),
        max_steps=int(os.getenv("DATAPILOT_MAX_STEPS", "8")),
        max_upload_mb=int(os.getenv("DATAPILOT_MAX_UPLOAD_MB", "10")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        prompt_version=os.getenv("DATAPILOT_PROMPT_VERSION") or None,
        rate_limit_runs=os.getenv("DATAPILOT_RATE_LIMIT_RUNS", "10/600"),
        rate_limit_uploads=os.getenv("DATAPILOT_RATE_LIMIT_UPLOADS", "20/3600"),
        max_active_runs=int(os.getenv("DATAPILOT_MAX_ACTIVE_RUNS", "4")),
        max_datasets=int(os.getenv("DATAPILOT_MAX_DATASETS", "50")),
        static_dir=Path(os.environ["DATAPILOT_STATIC_DIR"]) if os.getenv("DATAPILOT_STATIC_DIR") else None,
        trace_dir=None
        if os.getenv("DATAPILOT_TRACING", "1") == "0"
        else Path(os.getenv("DATAPILOT_TRACE_DIR") or BACKEND_DIR / "traces"),
    )


settings = _load()
