"""make_chart: data computed by pandas, column types validated per chart kind, ui data hidden from the model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.agent.loop import serialize_result
from app.tools import ToolError, registry


def chart(df: pd.DataFrame, **args):
    return registry.dispatch("make_chart", df, args)


def test_bar_mean_by_category(customers: pd.DataFrame) -> None:
    out = chart(customers, kind="bar", x="Segment", y="Income_k", agg="mean")
    spec = out["ui"]
    assert spec["title"] == "Mean of Income_k by Segment"
    assert spec["data"][0] == {"x": "C", "y": 77.5}  # sorted largest first
    assert out["summary"]["n_groups"] == 3


def test_bar_counts_without_y(customers: pd.DataFrame) -> None:
    spec = chart(customers, kind="bar", x="Segment")["ui"]
    assert {d["x"]: d["y"] for d in spec["data"]} == {"A": 3, "B": 3, "C": 2}
    assert spec["y_label"] == "Rows"


def test_histogram_counts_every_value(customers: pd.DataFrame) -> None:
    spec = chart(customers, kind="histogram", x="Age", bins=5)["ui"]
    assert len(spec["data"]) == 5 and sum(d["y"] for d in spec["data"]) == len(customers)


def test_scatter_samples_and_folds_colours_into_other() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "g": rng.choice(list("pqrst"), 2000)})
    out = chart(df, kind="scatter", x="a", y="b", color="g")
    spec = out["ui"]
    assert spec["sampled"] is True
    assert sum(len(s["points"]) for s in spec["series"]) == 500
    assert len(spec["series"]) == 4 and spec["series"][-1]["name"] == "Other"  # 3 named groups + Other
    assert out["summary"]["n_points"] == 2000


def test_line_parses_date_strings_and_resamples() -> None:
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    df = pd.DataFrame({"day": days.strftime("%Y-%m-%d"), "sales": range(60)})
    spec = chart(df, kind="line", x="day", y="sales", agg="sum")["ui"]
    assert spec["x_type"] == "date" and len(spec["data"]) == 60
    assert spec["data"][0] == {"x": "2026-01-01", "y": 0}


def test_ui_data_is_hidden_from_the_model(customers: pd.DataFrame) -> None:
    out = chart(customers, kind="scatter", x="Income_k", y="Spend")
    sent = json.loads(serialize_result(out))
    assert "ui" not in sent and sent["summary"]["pearson_r"] == 1


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"kind": "scatter", "x": "Segment", "y": "Age"}, "numeric"),
        ({"kind": "histogram", "x": "Segment"}, "numeric column"),
        ({"kind": "line", "x": "Segment", "y": "Age"}, "ordered x"),
        ({"kind": "bar", "x": "Segment", "y": "Customer_ID"}, "y must be numeric"),
        ({"kind": "bar", "x": "Segment", "color": "Segment"}, "only supported for scatter"),
        ({"kind": "bar", "x": "segment"}, "did you mean"),
        ({"kind": "bar", "x": "Segment", "filters": [{"column": "Age", "op": ">", "value": 99}]}, "No rows"),
    ],
)
def test_invalid_charts_explain_why(customers: pd.DataFrame, args: dict, fragment: str) -> None:
    with pytest.raises(ToolError, match=fragment):
        chart(customers, **args)


def test_bar_rejects_continuous_x() -> None:
    df = pd.DataFrame({"v": np.arange(100, dtype=float)})
    with pytest.raises(ToolError, match="histogram"):
        chart(df, kind="bar", x="v")
