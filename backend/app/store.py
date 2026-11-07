"""In-memory dataset store: maps a dataset id to a pandas DataFrame.

Phase 1 keeps everything in memory (lost on restart). Later this can move to
MongoDB/S3 without touching the tools, because tools only ever receive a DataFrame.
"""

from __future__ import annotations

import io
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MAX_EXAMPLE_CHARS = 40


class DatasetNotFound(KeyError):
    """No dataset with that id (expired, never uploaded, or typo)."""


class InvalidDataset(ValueError):
    """The upload could not be turned into a usable table."""


@dataclass
class Dataset:
    id: str
    name: str
    df: pd.DataFrame
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def semantic_type(s: pd.Series) -> str:
    """Classify a column the way an analyst would, not the way pandas stores it.

    numeric | boolean | datetime | categorical | text | id | empty
    'id' matters later: ID columns are useless (and leaky) as ML features.
    """
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    n_unique = non_null.nunique()
    avg_len = non_null.astype(str).str.len().mean()
    if n_unique > 0.5 * len(non_null):
        if avg_len > 30:
            return "text"
        return "id" if n_unique == len(non_null) else "categorical"
    return "text" if avg_len > 50 else "categorical"


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_EXAMPLE_CHARS else text[: MAX_EXAMPLE_CHARS - 1] + "…"


class DatasetStore:
    """In memory, capped: beyond `max_datasets` the least recently used upload is dropped."""

    def __init__(self, max_datasets: int = 50) -> None:
        self._datasets: OrderedDict[str, Dataset] = OrderedDict()
        self._lock = threading.Lock()
        self.max_datasets = max_datasets

    def add(self, df: pd.DataFrame, name: str = "dataset") -> str:
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise InvalidDataset("The file has no rows or no columns.")
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if df.columns.duplicated().any():
            dupes = sorted(set(df.columns[df.columns.duplicated()]))
            raise InvalidDataset(f"Column names must be unique; duplicated: {dupes}")
        dataset_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._datasets[dataset_id] = Dataset(id=dataset_id, name=name, df=df)
            while len(self._datasets) > self.max_datasets:
                self._datasets.popitem(last=False)
        return dataset_id

    def load_csv(self, source: str | Path | bytes, name: str | None = None) -> str:
        if isinstance(source, bytes):
            reader_input: Any = io.BytesIO(source)
            name = name or "upload.csv"
        else:
            reader_input = Path(source)
            name = name or reader_input.name
        try:
            df = pd.read_csv(reader_input)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise InvalidDataset(f"Could not parse CSV: {e}") from e
        return self.add(df, name)

    def get(self, dataset_id: str) -> Dataset:
        with self._lock:
            ds = self._datasets.get(dataset_id)
            if ds is not None:
                self._datasets.move_to_end(dataset_id)  # recently used: keep it longest
        if ds is None:
            raise DatasetNotFound(dataset_id)
        return ds

    def frame(self, dataset_id: str) -> pd.DataFrame:
        return self.get(dataset_id).df

    def schema_summary(self, dataset_id: str) -> dict[str, Any]:
        """Compact description of the table: what the LLM sees *instead of* raw rows."""
        ds = self.get(dataset_id)
        df = ds.df
        columns = []
        for name in df.columns:
            s = df[name]
            examples = [_short(v) for v in s.dropna().drop_duplicates().head(3)]
            columns.append(
                {
                    "name": name,
                    "dtype": str(s.dtype),
                    "semantic_type": semantic_type(s),
                    "missing_pct": round(float(s.isna().mean() * 100), 2),
                    "n_unique": int(s.nunique()),
                    "examples": examples,
                }
            )
        return {"name": ds.name, "n_rows": int(df.shape[0]), "n_columns": int(df.shape[1]), "columns": columns}


def _default_store() -> DatasetStore:
    from app.config import settings  # local import: keeps this module importable without config

    return DatasetStore(max_datasets=settings.max_datasets)


store = _default_store()
