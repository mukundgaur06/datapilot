"""Deterministic pandas tools the agent can call.

Design rules (the "deterministic core" of the system):
  * Pure functions of (DataFrame, validated args) -> small JSON-able dict. Easy to unit test.
  * Results are bounded (limits, top-k) so they never blow up the LLM's context window.
  * Errors raise ToolError with a hint the model can act on (e.g. "did you mean 'Age'?").
"""

from __future__ import annotations

import difflib
import json
import math
import operator
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolError, tool


class _Args(BaseModel):
    # extra="forbid" -> "additionalProperties": false: hallucinated arguments are rejected loudly.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- helpers


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return
    available = [str(c) for c in df.columns]
    hints = []
    for c in missing:
        close = difflib.get_close_matches(c, available, n=3, cutoff=0.5)
        hints.append(f"{c!r}" + (f" (did you mean {', '.join(repr(x) for x in close)}?)" if close else ""))
    raise ToolError(
        f"Unknown column(s): {', '.join(hints)}. Column names are case-sensitive. "
        f"Available columns: {available[:50]}"
    )


def require_numeric(df: pd.DataFrame, column: str, purpose: str) -> None:
    if not _is_numeric(df[column]):
        raise ToolError(f"Column {column!r} is {df[column].dtype}, not numeric, so it can't be used for {purpose}.")


def num(x: Any) -> float | int | None:
    """Make numpy scalars JSON-safe and readable (4 decimals, NaN -> None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 4)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", double_precision=4, date_format="iso"))


def _short(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------- filters

FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
_COMPARE = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


class Filter(_Args):
    column: str = Field(description="Column to filter on (exact, case-sensitive name).")
    op: FilterOp = Field(description="Comparison. 'in' takes a list; 'contains' is a case-insensitive substring match.")
    value: bool | float | str | list[float | str] = Field(description="Value to compare against.")


def _coerce(value: Any, s: pd.Series, column: str) -> Any:
    """Match the filter value's type to the column's type."""
    if _is_numeric(s):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ToolError(f"Column {column!r} is numeric but the filter value {value!r} is not a number.") from None
        return value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value if isinstance(value, bool) else str(value)


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    if not filters:
        return df
    require_columns(df, [f.column for f in filters])
    mask = pd.Series(True, index=df.index)
    for f in filters:
        s = df[f.column]
        if f.op == "in":
            if not isinstance(f.value, list):
                raise ToolError("Filter op 'in' needs a list value, e.g. [\"a\", \"b\"].")
            m = s.isin([_coerce(v, s, f.column) for v in f.value])
        elif isinstance(f.value, list):
            raise ToolError(f"Filter op {f.op!r} needs a single value, not a list. Use op 'in' for lists.")
        elif f.op == "contains":
            m = s.astype(str).str.contains(str(f.value), case=False, regex=False, na=False)
        else:
            if f.op in (">", ">=", "<", "<=") and not _is_numeric(s):
                raise ToolError(f"Filter {f.op!r} only works on numeric columns; {f.column!r} is {s.dtype}.")
            target = _coerce(f.value, s, f.column)
            m = _COMPARE[f.op](s if _is_numeric(s) or isinstance(target, bool) else s.astype(str), target)
        mask &= m.fillna(False).astype(bool)
    return df[mask]


# ---------------------------------------------------------------- tool 1: profile_dataset


class ProfileArgs(_Args):
    pass


@tool(
    name="profile_dataset",
    description=(
        "Overview of the whole dataset: row/column counts, duplicate rows, and per-column statistics "
        "(missing values; mean/std/min/quartiles/max for numeric columns; most frequent values for other columns). "
        "Use it for a first look or for data-quality questions."
    ),
    args_model=ProfileArgs,
)
def profile_dataset(df: pd.DataFrame, args: ProfileArgs) -> dict[str, Any]:
    columns = []
    for name in df.columns:
        s = df[name]
        info: dict[str, Any] = {"name": str(name), "dtype": str(s.dtype), "missing": int(s.isna().sum())}
        if _is_numeric(s):
            d = s.describe()
            info.update(
                mean=num(d.get("mean")), std=num(d.get("std")), min=num(d.get("min")),
                p25=num(d.get("25%")), median=num(d.get("50%")), p75=num(d.get("75%")), max=num(d.get("max")),
            )
        else:
            vc = s.value_counts(dropna=True).head(5)
            info["n_unique"] = int(s.nunique())
            info["top_values"] = [{"value": _short(v), "count": int(c)} for v, c in vc.items()]
        columns.append(info)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


# ---------------------------------------------------------------- tool 2: group_aggregate

Agg = Literal["count", "mean", "median", "sum", "min", "max", "std", "nunique"]
_NUMERIC_AGGS = {"mean", "median", "sum", "min", "max", "std"}


class GroupAggregateArgs(_Args):
    group_by: list[str] = Field(
        default_factory=list, max_length=2,
        description="0-2 columns to group by. Empty list = one aggregate over all (filtered) rows.",
    )
    column: str | None = Field(
        default=None, description="Column to aggregate. Optional for 'count' (omit it to count rows)."
    )
    agg: Agg = Field(default="count", description="Aggregation to apply.")
    filters: list[Filter] = Field(
        default_factory=list, description="Optional row filters, combined with AND, applied before grouping."
    )
    sort: Literal["desc", "asc", "none"] = Field(default="desc", description="Sort groups by the aggregated value.")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of groups to return.")


@tool(
    name="group_aggregate",
    description=(
        "Filter rows, optionally group them, and aggregate: counts, averages, sums, etc. "
        "Examples: share of each label (agg='count', group_by=['label']); average income by segment; "
        "number of customers older than 30 (agg='count', filters=[{column:'Age', op:'>', value:30}]). "
        "Counts include 'pct_of_rows' so you never need to compute percentages yourself."
    ),
    args_model=GroupAggregateArgs,
)
def group_aggregate(df: pd.DataFrame, args: GroupAggregateArgs) -> dict[str, Any]:
    if args.agg != "count" and not args.column:
        raise ToolError(f"agg={args.agg!r} needs a 'column' to aggregate.")
    require_columns(df, args.group_by + ([args.column] if args.column else []))
    if args.agg in _NUMERIC_AGGS:
        require_numeric(df, args.column, f"agg={args.agg!r}")  # type: ignore[arg-type]

    data = apply_filters(df, args.filters)
    n = len(data)
    base = {"n_rows_matched": n, "n_rows_total": int(len(df)), "agg": args.agg, "column": args.column}
    if n == 0:
        return {**base, "rows": [], "note": "No rows match the filters."}

    if not args.group_by:
        if args.agg == "count":
            value = int(data[args.column].count()) if args.column else n
            return {**base, "value": value, "pct_of_rows": round(value / len(df) * 100, 2)}
        return {**base, "value": num(data[args.column].agg(args.agg))}

    grouped = data.groupby(args.group_by, dropna=False)
    if args.agg == "count":
        series = grouped[args.column].count() if args.column else grouped.size()
    else:
        series = grouped[args.column].agg(args.agg)
    out = series.rename("value").reset_index()
    if args.agg == "count":
        out["pct_of_rows"] = (out["value"] / n * 100).round(2)
    if args.sort != "none":
        out = out.sort_values("value", ascending=args.sort == "asc", kind="stable")
    n_groups = len(out)
    return {
        **base,
        "group_by": args.group_by,
        "n_groups": n_groups,
        "truncated": n_groups > args.limit,
        "rows": records(out.head(args.limit)),
    }


# ---------------------------------------------------------------- tool 3: correlation


class CorrelationArgs(_Args):
    columns: list[str] = Field(
        default_factory=list, description="Numeric columns to include. Empty list = all numeric columns."
    )
    target: str | None = Field(default=None, description="If set, only return correlations involving this column.")
    method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="'pearson' for linear relationships; 'spearman' for monotonic/ranked ones."
    )
    top_k: int = Field(default=10, ge=1, le=30, description="Number of strongest pairs to return.")


@tool(
    name="correlation",
    description=(
        "Correlation between numeric columns, returned as the strongest pairs sorted by |r|. "
        "Use 'target' to ask what is most related to one specific column."
    ),
    args_model=CorrelationArgs,
)
def correlation(df: pd.DataFrame, args: CorrelationArgs) -> dict[str, Any]:
    if args.columns:
        require_columns(df, args.columns)
        for c in args.columns:
            require_numeric(df, c, "correlation")
        cols = list(dict.fromkeys(args.columns))
    else:
        cols = [str(c) for c in df.columns if _is_numeric(df[c])]
    if args.target:
        require_columns(df, [args.target])
        require_numeric(df, args.target, "correlation")
        if args.target not in cols:
            cols.append(args.target)
    if len(cols) < 2:
        raise ToolError(f"Need at least two numeric columns to correlate; found {cols}.")

    matrix = df[cols].corr(method=args.method)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if args.target and args.target not in (a, b):
                continue
            r = matrix.loc[a, b]
            if pd.notna(r):
                pairs.append({"a": a, "b": b, "r": round(float(r), 4)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return {
        "method": args.method,
        "n_pairs": len(pairs),
        "pairs": pairs[: args.top_k],
        "note": "|r| < 0.3 is weak, 0.3-0.7 moderate, > 0.7 strong. Correlation does not imply causation.",
    }
