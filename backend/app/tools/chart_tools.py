"""make_chart: the agent asks for a chart; the backend computes and validates its data.

The LLM never invents chart data. It picks a kind and columns; pandas computes the
points; column types are checked against the chart kind. The point data goes under the
"ui" key, which the agent loop shows to the browser but strips before the result is sent
back to the model (the model gets a compact summary instead, saving tokens).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from app.tools.data_tools import Filter, _Args, _is_numeric, apply_filters, num, require_columns
from app.tools.registry import ToolError, tool

MAX_SCATTER_POINTS = 500
MAX_LINE_POINTS = 200
MAX_COLOR_GROUPS = 3  # the validated palette keeps scatter series distinguishable up to 3; the rest fold into "Other"

ChartAgg = Literal["count", "mean", "median", "sum", "min", "max"]


class MakeChartArgs(_Args):
    kind: Literal["bar", "line", "scatter", "histogram"] = Field(
        description=(
            "bar: a value per category (x categorical). line: a trend along an ordered numeric/date x. "
            "scatter: relationship between two numeric columns. histogram: distribution of one numeric column."
        )
    )
    x: str = Field(description="Column for the x axis (for histogram: the column to bin).")
    y: str | None = Field(default=None, description="Numeric column for the y axis. Omit for counts (bar/line) and histograms.")
    agg: ChartAgg = Field(default="mean", description="How to combine y values per x (bar/line). Ignored without y.")
    color: str | None = Field(default=None, description="Scatter only: a categorical column to colour points by.")
    filters: list[Filter] = Field(default_factory=list, description="Optional row filters (AND) applied first.")
    bins: int = Field(default=20, ge=5, le=50, description="Histogram bin count.")
    limit: int = Field(default=15, ge=3, le=30, description="Bar charts: max categories shown (largest first).")
    title: str | None = Field(default=None, description="Short chart title. Optional; a default is generated.")


def _as_datetime(s: pd.Series) -> pd.Series | None:
    """Parse a text column as dates if (nearly) all of it parses; else None."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if _is_numeric(s):
        return None
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    return parsed if parsed.notna().mean() >= 0.9 else None


def _label(agg: str, y: str | None) -> str:
    return "Rows" if y is None or agg == "count" else f"{agg.capitalize()} of {y}"


def _bar(data: pd.DataFrame, a: MakeChartArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    s = data[a.x]
    if _is_numeric(s) and s.nunique() > 50:
        raise ToolError(f"'{a.x}' has {s.nunique()} distinct numbers; use kind='histogram' (distribution) or 'line' (trend) instead.")
    grouped = data.groupby(a.x, dropna=False)
    values = grouped.size() if a.y is None or a.agg == "count" else grouped[a.y].agg(a.agg)
    values = values.sort_values(ascending=False)
    n_groups = len(values)
    rows = [{"x": "(missing)" if pd.isna(k) else str(k), "y": num(v)} for k, v in values.head(a.limit).items()]
    summary = {"n_groups": n_groups, "shown": len(rows), "values": rows[:10]}
    return {"data": rows, "truncated": n_groups > a.limit}, summary


def _line(data: pd.DataFrame, a: MakeChartArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    x = data[a.x]
    dates = _as_datetime(x)
    if dates is None and not _is_numeric(x):
        raise ToolError(f"Line charts need an ordered x (numbers or dates); '{a.x}' is {x.dtype}. Use kind='bar' for categories.")
    frame = pd.DataFrame({"x": dates if dates is not None else x})
    frame["y"] = 1 if a.y is None else data[a.y]
    frame = frame.dropna()
    if dates is not None:
        span = (frame["x"].max() - frame["x"].min()).days if len(frame) else 0
        freq = "D" if span <= 120 else "W" if span <= 3 * 365 else "MS"
        grouped = frame.set_index("x")["y"].resample(freq)
        series = grouped.count() if a.y is None or a.agg == "count" else grouped.agg(a.agg)
        series = series.dropna()
        fmt = lambda k: k.strftime("%Y-%m-%d")  # noqa: E731
    else:
        if frame["x"].nunique() > MAX_LINE_POINTS:  # too many distinct x: bin into equal-width buckets (midpoints)
            cuts = pd.cut(frame["x"], MAX_LINE_POINTS // 4)
            frame["x"] = cuts.map(lambda iv: iv.mid).astype(float)
        grouped = frame.groupby("x")["y"]
        series = grouped.count() if a.y is None or a.agg == "count" else grouped.agg(a.agg)
        fmt = num
    series = series.sort_index().tail(MAX_LINE_POINTS)
    rows = [{"x": fmt(k), "y": num(v)} for k, v in series.items()]
    if not rows:
        raise ToolError("No data points to plot after filtering.")
    ys = [r["y"] for r in rows if r["y"] is not None]
    summary = {"n_points": len(rows), "first": rows[0], "last": rows[-1], "min_y": min(ys), "max_y": max(ys)}
    return {"data": rows, "x_type": "date" if dates is not None else "number"}, summary


def _scatter(data: pd.DataFrame, a: MakeChartArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    if a.y is None:
        raise ToolError("Scatter charts need both x and y (numeric columns).")
    for c in (a.x, a.y):
        if not _is_numeric(data[c]):
            raise ToolError(f"Scatter needs numeric columns; '{c}' is {data[c].dtype}.")
    cols = [a.x, a.y] + ([a.color] if a.color else [])
    frame = data[cols].dropna(subset=[a.x, a.y])
    n_total = len(frame)
    r = frame[a.x].corr(frame[a.y]) if n_total > 2 else None
    if n_total > MAX_SCATTER_POINTS:
        frame = frame.sample(MAX_SCATTER_POINTS, random_state=0)
    series = []
    if a.color:
        top = data[a.color].value_counts().head(MAX_COLOR_GROUPS).index.tolist()
        groups = frame[a.color].where(frame[a.color].isin(top), other="Other").fillna("Other").astype(str)
        for name in [str(t) for t in top] + (["Other"] if (groups == "Other").any() else []):
            part = frame[groups == name]
            series.append({"name": name, "points": [[num(x), num(y)] for x, y in zip(part[a.x], part[a.y])]})
    else:
        series.append({"name": a.y, "points": [[num(x), num(y)] for x, y in zip(frame[a.x], frame[a.y])]})
    summary = {
        "n_points": n_total,
        "plotted": len(frame),
        "pearson_r": num(r),
        "x_range": [num(frame[a.x].min()), num(frame[a.x].max())],
        "y_range": [num(frame[a.y].min()), num(frame[a.y].max())],
    }
    return {"series": series, "sampled": n_total > MAX_SCATTER_POINTS}, summary


def _histogram(data: pd.DataFrame, a: MakeChartArgs) -> tuple[dict[str, Any], dict[str, Any]]:
    s = data[a.x].dropna()
    if not _is_numeric(s):
        raise ToolError(f"Histograms need a numeric column; '{a.x}' is {s.dtype}. Use kind='bar' to count categories.")
    if s.empty:
        raise ToolError("No values to plot after filtering.")
    counts, edges = np.histogram(s, bins=a.bins)
    rows = [{"x0": num(lo), "x1": num(hi), "y": int(c)} for lo, hi, c in zip(edges[:-1], edges[1:], counts)]
    summary = {
        "n_values": int(len(s)),
        "min": num(s.min()), "median": num(s.median()), "max": num(s.max()),
        "peak_bin": max(rows, key=lambda r: r["y"]),
    }
    return {"data": rows}, summary


_BUILDERS = {"bar": _bar, "line": _line, "scatter": _scatter, "histogram": _histogram}


@tool(
    name="make_chart",
    description=(
        "Create a chart for the user (it appears in their artifacts panel). The backend computes the data from the "
        "dataset, so pick the kind and columns, never supply numbers. Returns a compact summary of what was plotted. "
        "Use it whenever a visual would make the answer clearer: comparisons across groups, trends, distributions, "
        "or relationships between two numeric columns."
    ),
    args_model=MakeChartArgs,
)
def make_chart(df: pd.DataFrame, args: MakeChartArgs) -> dict[str, Any]:
    require_columns(df, [c for c in (args.x, args.y, args.color) if c])
    if args.y and not _is_numeric(df[args.y]):
        raise ToolError(f"y must be numeric; '{args.y}' is {df[args.y].dtype}. Omit y to count rows instead.")
    if args.color and args.kind != "scatter":
        raise ToolError("'color' is only supported for scatter charts.")
    data = apply_filters(df, args.filters)
    if data.empty:
        raise ToolError("No rows match the filters.")
    ui, summary = _BUILDERS[args.kind](data, args)

    y_label = "Rows" if args.kind == "histogram" else (args.y or "Rows") if args.kind == "scatter" else _label(args.agg, args.y)
    default_title = {
        "bar": f"{y_label} by {args.x}",
        "line": f"{y_label} over {args.x}",
        "scatter": f"{args.y} vs {args.x}",
        "histogram": f"Distribution of {args.x}",
    }[args.kind]
    spec = {
        "kind": args.kind,
        "title": args.title or default_title,
        "x_label": args.x,
        "y_label": y_label,
        "n_rows_used": int(len(data)),
        **ui,
    }
    return {"chart": args.kind, "title": spec["title"], "rows_used": int(len(data)), "summary": summary, "ui": spec}
