"""Agent runs that execute in the background and can be watched while they happen.

Why a background thread instead of running the agent inside the HTTP request?
  * The run outlives any single connection: if the browser's SSE connection drops,
    EventSource reconnects (sending Last-Event-ID) and we replay what it missed.
  * Stopping a run is an explicit action (POST /runs/{id}/cancel), not a side
    effect of a network blip.
The agent itself is blocking code (pandas + the Gemini SDK), so a plain thread fits;
the async SSE endpoint waits on it without blocking the event loop.

State is in memory, like the dataset store: fine for one process, lost on restart.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.agent.llm import LLM
from app.agent.loop import TERMINAL_EVENTS, AgentEvent, run_agent
from app.store import DatasetStore
from app.store import store as default_store

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "done", "error", "cancelled"]
_STATUS_FOR_EVENT: dict[str, RunStatus] = {"answer": "done", "error": "error", "cancelled": "cancelled"}


class RunNotFound(KeyError):
    """No run with that id (finished long ago and evicted, or never existed)."""


@dataclass
class Run:
    id: str
    dataset_id: str
    question: str
    status: RunStatus = "running"
    events: list[AgentEvent] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def append(self, event: AgentEvent) -> None:
        with self._cond:
            self.events.append(event)
            if event.type in TERMINAL_EVENTS:
                self.status = _STATUS_FOR_EVENT[event.type]
            self._cond.notify_all()

    def wait_for_events(self, cursor: int, timeout: float) -> tuple[list[AgentEvent], bool]:
        """Block until there are events after `cursor` or the run ends (or timeout).

        Returns (new events, finished). An empty list with finished=False means timeout.
        """
        with self._cond:
            self._cond.wait_for(lambda: len(self.events) > cursor or self.finished, timeout=timeout)
            return self.events[cursor:], self.finished


class RunManager:
    def __init__(self, store: DatasetStore = default_store, max_runs: int = 200) -> None:
        self.store = store
        self.max_runs = max_runs
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, question: str, dataset_id: str, llm: LLM) -> Run:
        """Create a run and start it in a background thread.

        Raises DatasetNotFound synchronously, so the API can answer 404 instead of
        starting a run that fails immediately.
        """
        self.store.get(dataset_id)
        run = Run(id=uuid.uuid4().hex[:12], dataset_id=dataset_id, question=question)
        with self._lock:
            self._runs[run.id] = run
            self._evict()
        threading.Thread(target=self._execute, args=(run, llm), name=f"run-{run.id}", daemon=True).start()
        return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise RunNotFound(run_id)
        return run

    def _evict(self) -> None:
        # Drop the oldest *finished* runs beyond the cap; never drop one that's still running.
        excess = len(self._runs) - self.max_runs
        for run_id in [rid for rid, r in self._runs.items() if r.finished][: max(excess, 0)]:
            del self._runs[run_id]

    def _execute(self, run: Run, llm: LLM) -> None:
        try:
            for event in run_agent(run.question, run.dataset_id, llm=llm, store=self.store, cancel=run.cancel):
                run.append(event)
        except Exception:  # a bug on our side: never leave a run stuck in "running"
            logger.exception("Run %s crashed", run.id)
        if not run.finished:
            run.append(AgentEvent(type="error", step=0, data={"message": "Internal error while running the agent."}))


runs = RunManager()
