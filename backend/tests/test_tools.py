from __future__ import annotations

import pandas as pd
import pytest

from app.store import DatasetStore, InvalidDataset, semantic_type
from app.tools import ToolError, registry

# ---------------------------------------------------------------- store


def test_semantic_types(customers: pd.DataFrame) -> None:
    assert semantic_type(customers["Customer_ID"]) == "id"
    assert semantic_type(customers["Segment"]) == "categorical"
    assert semantic_type(customers["Age"]) == "numeric"
    long_text = pd.Series([f"message number {i} with plenty of words in it" for i in range(10)])
    assert semantic_type(long_text) == "text"


def test_schema_summary_has_no_full_rows(store: DatasetStore, dataset_id: str) -> None:
    summary = store.schema_summary(dataset_id)
    assert summary["n_rows"] == 8 and summary["n_columns"] == 5
    assert all(len(c["examples"]) <= 3 for c in summary["columns"])


def test_rejects_duplicate_columns_and_empty() -> None:
    s = DatasetStore()
    with pytest.raises(InvalidDataset, match="unique"):
        s.add(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(InvalidDataset):
        s.load_csv(b"")


def test_csv_duplicate_headers_are_renamed_by_pandas() -> None:
    # pandas' read_csv de-duplicates headers itself ("a", "a.1"), so CSV uploads still load.
    s = DatasetStore()
    assert list(s.frame(s.load_csv(b"a,a\n1,2\n")).columns) == ["a", "a.1"]


# ---------------------------------------------------------------- registry


def test_schemas_are_flat_json_schema() -> None:
    specs = registry.schemas()
    assert [s["name"] for s in specs] == ["correlation", "group_aggregate", "profile_dataset"]
    for spec in specs:
        params = spec["parameters"]
        assert params["type"] == "object"
        assert params.get("additionalProperties") is False
        assert "$defs" not in str(params) and "$ref" not in str(params)
        assert spec["description"]


def test_unknown_tool_and_bad_args_become_tool_errors(customers: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="Unknown tool"):
        registry.dispatch("drop_table", customers, {})
    with pytest.raises(ToolError, match="Invalid arguments"):
        registry.dispatch("group_aggregate", customers, {"agg": "average"})
    with pytest.raises(ToolError, match="Invalid arguments"):
        registry.dispatch("profile_dataset", customers, {"surprise": 1})


# ---------------------------------------------------------------- profile_dataset


def test_profile(customers: pd.DataFrame) -> None:
    out = registry.dispatch("profile_dataset", customers, {})
    assert out["n_rows"] == 8 and out["duplicate_rows"] == 0
    age = next(c for c in out["columns"] if c["name"] == "Age")
    assert age["mean"] == 37.5 and age["min"] == 22 and age["max"] == 61
    seg = next(c for c in out["columns"] if c["name"] == "Segment")
    assert seg["top_values"][0] == {"value": "A", "count": 3} or seg["top_values"][0]["count"] == 3


# ---------------------------------------------------------------- group_aggregate


def test_count_by_group_with_percentages(customers: pd.DataFrame) -> None:
    out = registry.dispatch("group_aggregate", customers, {"group_by": ["Segment"], "agg": "count"})
    rows = {r["Segment"]: r for r in out["rows"]}
    assert rows["A"]["value"] == 3 and rows["B"]["value"] == 3 and rows["C"]["value"] == 2
    assert rows["C"]["pct_of_rows"] == 25.0
    assert sum(r["pct_of_rows"] for r in out["rows"]) == pytest.approx(100)


def test_mean_by_group_sorted_desc(customers: pd.DataFrame) -> None:
    out = registry.dispatch("group_aggregate", customers, {"group_by": ["Segment"], "column": "Income_k", "agg": "mean"})
    assert [r["Segment"] for r in out["rows"]] == ["C", "B", "A"]
    assert out["rows"][0]["value"] == pytest.approx(77.5)


def test_filters_and_whole_table_aggregate(customers: pd.DataFrame) -> None:
    out = registry.dispatch(
        "group_aggregate", customers,
        {"agg": "count", "filters": [{"column": "Age", "op": ">", "value": 30}]},
    )
    assert out["value"] == 5 and out["pct_of_rows"] == 62.5
    out = registry.dispatch(
        "group_aggregate", customers,
        {"column": "Spend", "agg": "sum", "filters": [{"column": "Segment", "op": "in", "value": ["A", "C"]}]},
    )
    assert out["value"] == 60 + 100 + 70 + 110 + 200
    out = registry.dispatch(
        "group_aggregate", customers, {"agg": "count", "filters": [{"column": "Age", "op": ">", "value": "50"}]}
    )
    assert out["value"] == 2  # string "50" coerced to a number for a numeric column


def test_no_matching_rows(customers: pd.DataFrame) -> None:
    out = registry.dispatch("group_aggregate", customers, {"agg": "count", "filters": [{"column": "Age", "op": ">", "value": 100}]})
    assert out["n_rows_matched"] == 0 and out["rows"] == []


def test_helpful_errors(customers: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="did you mean 'Age'"):
        registry.dispatch("group_aggregate", customers, {"group_by": ["age"]})
    with pytest.raises(ToolError, match="not numeric"):
        registry.dispatch("group_aggregate", customers, {"column": "Segment", "agg": "mean"})
    with pytest.raises(ToolError, match="needs a 'column'"):
        registry.dispatch("group_aggregate", customers, {"agg": "mean"})
    with pytest.raises(ToolError, match="only works on numeric"):
        registry.dispatch("group_aggregate", customers, {"filters": [{"column": "Segment", "op": ">", "value": "A"}]})


def test_limit_truncates(customers: pd.DataFrame) -> None:
    out = registry.dispatch("group_aggregate", customers, {"group_by": ["Customer_ID"], "limit": 3})
    assert len(out["rows"]) == 3 and out["n_groups"] == 8 and out["truncated"] is True


# ---------------------------------------------------------------- correlation


def test_correlation_finds_planted_relationship(customers: pd.DataFrame) -> None:
    out = registry.dispatch("correlation", customers, {})
    top = out["pairs"][0]
    assert {top["a"], top["b"]} == {"Income_k", "Spend"} and top["r"] == pytest.approx(1.0)


def test_correlation_with_target(customers: pd.DataFrame) -> None:
    out = registry.dispatch("correlation", customers, {"target": "Age", "method": "spearman"})
    assert all("Age" in (p["a"], p["b"]) for p in out["pairs"])
    assert out["n_pairs"] == 2


def test_correlation_rejects_text_columns(customers: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="not numeric"):
        registry.dispatch("correlation", customers, {"columns": ["Segment", "Age"]})
