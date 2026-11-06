"""Train and honestly evaluate a model on a DataFrame.

The rules that make the numbers trustworthy:
  * All preprocessing lives INSIDE the sklearn Pipeline, so cross-validation fits the
    imputer/scaler/encoder/TF-IDF on the training folds only (no train-test contamination).
  * Every score is a k-fold cross-validated mean ± std, compared with a dummy baseline
    evaluated on the same folds. "90% accuracy" means nothing without the baseline.
  * Leakage guards: ID-like and constant columns are dropped (with the reason reported),
    and features almost perfectly correlated with the target trigger a warning.
  * Caveats are generated from the data (small n, imbalance, duplicates, suspiciously
    perfect scores), not left to the LLM's imagination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app.store import semantic_type

Task = Literal["classification", "regression"]
ModelKind = Literal["auto", "linear", "random_forest"]

MAX_TRAIN_ROWS = 20_000  # keeps a CV run to seconds on the free tier of any host
MAX_CLASSES = 20
MAX_FOLDS = 5
LEAKAGE_R = 0.95
RANDOM_STATE = 0


class TrainingError(ValueError):
    """The request can't produce a sensible model. Message tells the agent what to change."""


# ---------------------------------------------------------------- feature planning


@dataclass
class FeaturePlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical + self.text


def _looks_like_numeric_id(name: str, s: pd.Series) -> bool:
    name = name.lower()
    named_like_id = name == "id" or name.endswith(("_id", " id", "id_")) or name.startswith("id_")
    return named_like_id and pd.api.types.is_integer_dtype(s) and s.nunique() == len(s.dropna())


def plan_features(df: pd.DataFrame, target: str, features: list[str] | None) -> FeaturePlan:
    plan = FeaturePlan()
    candidates = [c for c in (features or list(df.columns)) if c != target]
    for col in candidates:
        s = df[col]
        kind = semantic_type(s)
        if kind == "empty":
            plan.dropped.append({"column": col, "reason": "all values are missing"})
        elif s.nunique(dropna=True) <= 1:
            plan.dropped.append({"column": col, "reason": "constant (one value), carries no information"})
        elif kind == "id" or _looks_like_numeric_id(col, s):
            plan.dropped.append({"column": col, "reason": "ID-like (a unique value per row): it can't generalise and can leak row identity"})
        elif kind == "datetime":
            plan.dropped.append({"column": col, "reason": "dates aren't supported as features yet"})
        elif kind == "numeric":
            plan.numeric.append(col)
        elif kind == "text":
            plan.text.append(col)
        else:  # categorical, boolean
            plan.categorical.append(col)
    if not plan.columns:
        raise TrainingError(f"No usable feature columns left after dropping: {plan.dropped}. Pass different 'features'.")

    # Leakage check: a feature (almost) perfectly correlated with the target is usually
    # computed from it, or only known after the fact.
    y = df[target]
    y_num = y if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) else None
    if y_num is None and y.nunique(dropna=True) == 2:
        y_num = (y == sorted(y.dropna().unique(), key=str)[1]).astype(float).where(y.notna())
    if y_num is not None:
        for col in plan.numeric:
            r = df[col].corr(y_num)
            if pd.notna(r) and abs(r) > LEAKAGE_R:
                plan.warnings.append(
                    f"'{col}' is almost perfectly correlated with the target (r = {r:.3f}). If it is derived from "
                    f"'{target}' or only known afterwards, that's leakage: retrain without it."
                )
    return plan


# ---------------------------------------------------------------- task & pipeline


def detect_task(y: pd.Series, requested: Literal["auto", "classification", "regression"]) -> Task:
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if requested == "regression":
        if not numeric:
            raise TrainingError(f"Regression needs a numeric target; '{y.name}' is {y.dtype}. Use task='classification'.")
        return "regression"
    if requested == "classification":
        task: Task = "classification"
    else:
        task = "regression" if numeric and y.nunique() > 10 else "classification"
    if task == "classification" and y.nunique() > MAX_CLASSES:
        raise TrainingError(
            f"'{y.name}' has {y.nunique()} distinct values, too many classes. "
            + ("Use task='regression'." if numeric else "Pick a target with fewer categories.")
        )
    return task


# Module-level functions (not lambdas) so the pipeline stays picklable later.
def _as_float(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(pd.to_numeric, errors="coerce").astype(float)


def _as_category_strings(X: pd.DataFrame) -> pd.DataFrame:
    # Mixed bool/str/NaN columns break OneHotEncoder; make every value a string, missing explicit.
    return X.apply(lambda s: s.map(lambda v: "(missing)" if pd.isna(v) else str(v))).astype(object)


def _as_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def build_preprocessor(plan: FeaturePlan) -> ColumnTransformer:
    parts: list[tuple[str, Any, Any]] = []
    if plan.numeric:
        parts.append((
            "num",
            Pipeline([("float", FunctionTransformer(_as_float, feature_names_out="one-to-one")), ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            plan.numeric,
        ))
    if plan.categorical:
        parts.append((
            "cat",
            Pipeline([
                ("strings", FunctionTransformer(_as_category_strings, feature_names_out="one-to-one")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, max_categories=20)),
            ]),
            plan.categorical,
        ))
    for col in plan.text:  # a single column name (not a list) hands TF-IDF the 1-D Series it expects
        parts.append((
            f"text:{col}",
            Pipeline([
                ("fill", FunctionTransformer(_as_text, feature_names_out="one-to-one")),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)),
            ]),
            col,
        ))
    return ColumnTransformer(parts)


def make_estimator(task: Task, kind: ModelKind, has_text: bool) -> tuple[str, BaseEstimator]:
    # Linear models handle sparse TF-IDF well and are interpretable; forests capture
    # non-linear interactions in tabular data. 'auto' picks by that rule.
    resolved = kind if kind != "auto" else ("linear" if has_text else "random_forest")
    if task == "classification":
        if resolved == "linear":
            return "logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced")
        return "random_forest", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )
    if resolved == "linear":
        return "ridge_regression", Ridge(alpha=1.0)
    return "random_forest", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- evaluation


@dataclass
class Scoring:
    scorers: dict[str, Any]
    primary: str  # headline metric
    compare_on: str  # metric used for "beats the baseline" (one where the baseline is not degenerate)
    positive_class: Any = None


def pick_positive_class(y: pd.Series, requested: str | None = None) -> Any:
    """The class precision/recall/F1 are reported for.

    Explicit choice wins. Otherwise the rarer class (spam, churn, fraud are usually rare);
    on a tie, the class that sorts last, matching the usual 0/1, no/yes, ham/spam convention.
    """
    classes = sorted(y.dropna().unique().tolist(), key=str)
    if requested is not None:
        match = [c for c in classes if str(c) == str(requested)]
        if not match:
            raise TrainingError(f"positive_class {requested!r} is not one of the target's classes: {[str(c) for c in classes]}")
        return match[0]
    counts = y.value_counts()
    rarest = counts[counts == counts.min()].index.tolist()
    return sorted(rarest, key=str)[-1]


def make_scoring(task: Task, classes: list[Any], y: pd.Series, positive_class: str | None = None) -> Scoring:
    if task == "regression":
        return Scoring(
            {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            primary="r2",
            compare_on="r2",
        )
    if len(classes) == 2:
        positive = pick_positive_class(y, positive_class)
        kw = {"pos_label": positive, "zero_division": 0}
        return Scoring(
            {
                "f1": make_scorer(f1_score, **kw),
                "precision": make_scorer(precision_score, **kw),
                "recall": make_scorer(recall_score, **kw),
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
            },
            primary="f1",
            compare_on="roc_auc",
            positive_class=positive,
        )
    kw = {"average": "macro", "zero_division": 0}
    return Scoring(
        {
            "f1_macro": make_scorer(f1_score, **kw),
            "precision_macro": make_scorer(precision_score, **kw),
            "recall_macro": make_scorer(recall_score, **kw),
            "accuracy": "accuracy",
            "roc_auc_ovr": "roc_auc_ovr",
        },
        primary="f1_macro",
        compare_on="roc_auc_ovr",
    )


def _summarise(cv_result: dict[str, np.ndarray], names: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for name in names:
        scores = cv_result[f"test_{name}"]
        if name in ("mae", "rmse"):
            scores = -scores  # sklearn maximises, so error metrics come back negated
        out[name] = {"mean": round(float(np.nanmean(scores)), 4), "std": round(float(np.nanstd(scores)), 4)}
    return out


@dataclass
class TrainingResult:
    pipeline: Pipeline
    task: Task
    target: str
    model_name: str
    plan: FeaturePlan
    classes: list[Any]
    scoring: Scoring
    n_rows: int
    report: dict[str, Any]


def train(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    task: Literal["auto", "classification", "regression"] = "auto",
    model: ModelKind = "auto",
    positive_class: str | None = None,
) -> TrainingResult:
    if target not in df.columns:
        raise TrainingError(f"Unknown target column {target!r}.")
    notes: list[str] = []
    data = df[df[target].notna()]
    if len(data) < len(df):
        notes.append(f"Dropped {len(df) - len(data)} rows where '{target}' is missing.")
    if len(data) > MAX_TRAIN_ROWS:
        data = data.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
        notes.append(f"Trained on a random sample of {MAX_TRAIN_ROWS:,} rows to keep training fast.")
    if len(data) < 20:
        raise TrainingError(f"Only {len(data)} rows with a target value; need at least 20 to train and validate a model.")

    y = data[target]
    resolved_task = detect_task(y, task)
    plan = plan_features(data, target, features)
    X = data[plan.columns]
    model_name, estimator = make_estimator(resolved_task, model, has_text=bool(plan.text))
    pipeline = Pipeline([("prep", build_preprocessor(plan)), ("model", estimator)])

    classes: list[Any] = []
    if resolved_task == "classification":
        counts = y.value_counts()
        classes = sorted(counts.index.tolist(), key=str)
        n_folds = min(MAX_FOLDS, int(counts.min()))
        if n_folds < 2:
            raise TrainingError(f"Class {counts.idxmin()!r} has a single example; every class needs at least 2.")
        cv: Any = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
        baseline: BaseEstimator = DummyClassifier(strategy="prior")
    else:
        n_folds = MAX_FOLDS
        cv = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
        baseline = DummyRegressor(strategy="mean")

    scoring = make_scoring(resolved_task, classes, y, positive_class)
    names = list(scoring.scorers)
    # return_estimator/indices let us reuse each fold's fitted model for out-of-fold
    # predictions (confusion matrix) instead of fitting every fold a second time.
    model_cv = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring.scorers, error_score="raise", return_estimator=True, return_indices=True
    )
    base_cv = cross_validate(baseline, X, y, cv=cv, scoring=scoring.scorers, error_score="raise")
    metrics, base_metrics = _summarise(model_cv, names), _summarise(base_cv, names)

    # "Beats the baseline" is judged on a metric where the baseline isn't degenerate: a dummy
    # classifier scores ROC-AUC 0.5 (F1 would be 0, which anything "beats"); a mean-predictor
    # scores R² ≈ 0. The model must win by more than its fold-to-fold noise.
    primary = scoring.primary
    compare = scoring.compare_on
    margin = metrics[compare]["mean"] - base_metrics[compare]["mean"]
    beats = margin > max(metrics[compare]["std"], 0.02)

    report: dict[str, Any] = {
        "task": resolved_task,
        "target": target,
        "model": model_name,
        "n_rows_used": int(len(data)),
        "features": {"numeric": plan.numeric, "categorical": plan.categorical, "text": plan.text},
        "dropped_features": plan.dropped,
        "validation": f"{n_folds}-fold {'stratified ' if resolved_task == 'classification' else ''}cross-validation",
        "primary_metric": primary,
        "metrics": metrics,
        "baseline": {"model": "always predicts the most common class" if resolved_task == "classification" else "always predicts the mean", "metrics": base_metrics},
        "beats_baseline": bool(beats),
        "baseline_compared_on": compare,
    }

    if resolved_task == "classification":
        counts = y.value_counts()
        report["class_balance"] = [
            {"class": str(c), "count": int(n), "pct": round(n / len(y) * 100, 1)} for c, n in counts.items()
        ]
        if scoring.positive_class is not None:
            report["positive_class"] = str(scoring.positive_class)
            report["metrics_note"] = f"precision, recall and F1 are for class '{scoring.positive_class}'"
        if len(classes) <= 10:
            oof = pd.Series(index=y.index, dtype=object)
            for est, test_idx in zip(model_cv["estimator"], model_cv["indices"]["test"]):
                oof.iloc[test_idx] = est.predict(X.iloc[test_idx])
            report["confusion_matrix"] = {
                "labels": [str(c) for c in classes],
                "matrix": confusion_matrix(y, oof, labels=classes).tolist(),
                "note": "rows = actual, columns = predicted; out-of-fold predictions",
            }
    else:
        report["target_summary"] = {"mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4)}

    report["caveats"] = notes + plan.warnings + _caveats(report, X, y, resolved_task, plan)
    pipeline.fit(X, y)  # final model on all rows; the CV numbers above estimate how it generalises
    return TrainingResult(pipeline, resolved_task, target, model_name, plan, classes, scoring, int(len(data)), report)


def templated_share(text: pd.Series, n_words: int = 5) -> float:
    """Share of rows whose first few (normalised) words also open another row.

    High values mean templated or near-duplicate text, which inflates cross-validated
    scores: near-copies land in both training and test folds.
    """
    words = text.fillna("").astype(str).str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.split()
    opening = words.map(lambda w: " ".join(w[:n_words]) if len(w) >= n_words else None).dropna()
    if opening.empty:
        return 0.0
    return float(opening.duplicated(keep=False).sum() / len(text))


def _caveats(report: dict[str, Any], X: pd.DataFrame, y: pd.Series, task: Task, plan: FeaturePlan) -> list[str]:
    out = []
    n = len(y)
    primary = report["primary_metric"]
    m = report["metrics"][primary]
    if n < 200:
        out.append(f"Only {n} rows: scores vary noticeably between folds (± {m['std']:.3f}), so treat them as rough.")
    if task == "classification":
        minority = min(report["class_balance"], key=lambda c: c["count"])
        if minority["pct"] < 20:
            out.append(
                f"Imbalanced classes: '{minority['class']}' is only {minority['pct']}% of rows, so accuracy is misleading. "
                "Judge the model by recall, precision and F1. Training used balanced class weights."
            )
    dupes = int(X.assign(_y=y.values).duplicated().sum())
    if dupes:
        out.append(f"{dupes} duplicate rows: copies can land in both training and test folds and inflate the scores.")
    for col in plan.text:
        share = templated_share(X[col])
        if share >= 0.3:
            out.append(
                f"{share:.0%} of '{col}' values share their opening words with another row (templated or near-duplicate "
                "text). Near-copies end up in both training and test folds, so these scores likely overstate how the "
                "model would do on real, varied text."
            )
    if plan.text:
        out.append(
            f"The text model only knows words that appear in these {n} rows; new wording or topics it hasn't seen "
            "will be handled poorly, and the way people write changes over time."
        )
    best = max(m["mean"], report["metrics"].get("roc_auc", report["metrics"].get("roc_auc_ovr", {})).get("mean", 0))
    if best >= 0.98:
        out.append(
            f"Near-perfect scores ({primary} {m['mean']:.3f}"
            + (f", ROC-AUC {report['metrics']['roc_auc']['mean']:.3f}" if "roc_auc" in report["metrics"] else "")
            + ") are a warning sign, not a triumph: check that no feature leaks the answer and that rows aren't "
            "templated. Real-world data is rarely this easy."
        )
    if not report["beats_baseline"]:
        out.append(
            f"The model does not clearly beat the baseline on {report['baseline_compared_on']}, so these features may "
            "not be predictive of the target."
        )
    out.append("Cross-validation estimates performance on data like this sample; it is not proof the model works on future or different data.")
    return out
