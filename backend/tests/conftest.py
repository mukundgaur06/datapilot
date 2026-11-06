from __future__ import annotations

import os

# Before any app import: tests keep traces in memory (no files in backend/traces).
os.environ["DATAPILOT_TRACING"] = "0"

import pandas as pd
import pytest

from app.store import DatasetStore


@pytest.fixture
def customers() -> pd.DataFrame:
    """Small, hand-checkable table. Income and spend are perfectly correlated on purpose."""
    return pd.DataFrame(
        {
            "Customer_ID": [f"C{i:03d}" for i in range(1, 9)],
            "Segment": ["A", "A", "B", "B", "B", "C", "C", "A"],
            "Age": [22, 35, 41, 29, 52, 33, 61, 27],
            "Income_k": [30.0, 50.0, 70.0, 40.0, 90.0, 55.0, 100.0, 35.0],
            "Spend": [60.0, 100.0, 140.0, 80.0, 180.0, 110.0, 200.0, 70.0],
        }
    )


@pytest.fixture
def store(customers: pd.DataFrame) -> DatasetStore:
    return DatasetStore()


@pytest.fixture
def dataset_id(store: DatasetStore, customers: pd.DataFrame) -> str:
    return store.add(customers, "customers.csv")
