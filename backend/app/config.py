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


def _load() -> Settings:
    origins = os.getenv("DATAPILOT_CORS_ORIGINS", "http://localhost:5173")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
        model=os.getenv("DATAPILOT_MODEL", "gemini-3.5-flash-lite"),
        thinking_level=os.getenv("DATAPILOT_THINKING_LEVEL", "LOW").upper(),
        max_steps=int(os.getenv("DATAPILOT_MAX_STEPS", "8")),
        max_upload_mb=int(os.getenv("DATAPILOT_MAX_UPLOAD_MB", "10")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )


settings = _load()
