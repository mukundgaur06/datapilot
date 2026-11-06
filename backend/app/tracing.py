"""Per-run traces: one JSONL file per agent run, plus an in-memory copy for the UI.

What a trace answers: where did the time go (model vs tools), how many tokens did it
cost, which tools ran with which arguments, and what failed. Every record carries
`t_ms`, the time since the run started, so the UI can draw a waterfall.

Privacy: tool results are stored as a size-capped preview (display-only "ui" data
dropped), and the system prompt, which contains example cell values, is stored
only as a hash and length. Traces still contain the user's question and tool
arguments, so the trace directory is git-ignored and should be treated as user data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 2000


class Tracer:
    def __init__(self, run_id: str, path: Path | None = None) -> None:
        self.run_id = run_id
        self.path = path
        self.records: list[dict[str, Any]] = []
        self._t0 = perf_counter()
        self._lock = threading.Lock()
        self._file = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._file = path.open("a", encoding="utf-8")
            except OSError:  # tracing must never break a run
                logger.exception("Could not open trace file %s", path)

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._t0) * 1000, 1)

    def _write(self, kind: str, **data: Any) -> None:
        record = {"t_ms": self.elapsed_ms(), "kind": kind, **data}
        with self._lock:
            self.records.append(record)
            if self._file is not None:
                try:
                    self._file.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
                    self._file.flush()
                except OSError:
                    logger.exception("Could not write trace record")

    # -- what the agent loop reports

    def start(self, *, question: str, dataset: dict[str, Any], model: str, system_prompt: str, tools: list[str]) -> None:
        self._write(
            "run_start",
            run_id=self.run_id,
            at=datetime.now(timezone.utc).isoformat(),
            question=question,
            dataset=dataset,
            model=model,
            system_prompt={"sha256": hashlib.sha256(system_prompt.encode()).hexdigest()[:16], "chars": len(system_prompt)},
            tools=tools,
        )

    def llm_call(self, *, step: int, duration_ms: float, usage: dict[str, int], finish: str, n_tool_calls: int = 0, error: str | None = None) -> None:
        self._write("llm_call", step=step, duration_ms=round(duration_ms, 1), usage=usage, finish=finish, n_tool_calls=n_tool_calls, error=error)

    def tool_call(self, *, step: int, key: str, name: str, args: dict[str, Any]) -> None:
        self._write("tool_call", step=step, key=key, name=name, args=args)

    def tool_result(self, *, step: int, key: str, name: str, duration_ms: float, is_error: bool, result: dict[str, Any]) -> None:
        text = json.dumps({k: v for k, v in result.items() if k != "ui"}, default=str, ensure_ascii=False)
        self._write(
            "tool_result", step=step, key=key, name=name, duration_ms=duration_ms, is_error=is_error,
            result_chars=len(text), result_preview=text[:PREVIEW_CHARS],
        )

    def end(self, *, status: str, steps: int, message: str = "") -> None:
        self._write("run_end", status=status, steps=steps, message=message[:500], summary=self.summary())
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    # -- aggregates (used by run_end, the UI and the eval harness)

    def summary(self) -> dict[str, Any]:
        llm = [r for r in self.records if r["kind"] == "llm_call"]
        tools = [r for r in self.records if r["kind"] == "tool_result"]
        usage: dict[str, int] = {}
        for r in llm:
            for k, v in (r.get("usage") or {}).items():
                usage[k] = usage.get(k, 0) + int(v or 0)
        return {
            "llm_calls": len(llm),
            "llm_ms": round(sum(r["duration_ms"] for r in llm), 1),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for r in tools if r["is_error"]),
            "tool_ms": round(sum(r["duration_ms"] for r in tools), 1),
            "total_ms": self.elapsed_ms(),
            "usage": usage,
        }


def trace_path(trace_dir: Path, run_id: str) -> Path:
    return trace_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}" / f"{run_id}.jsonl"
