"""ML tools: honest evaluation, leakage guards, explanations, scoping. Deterministic (fixed seeds)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.loop import serialize_result
from app.ml.models import ModelStore
from app.ml.training import TrainingError, train
from app.tools import ToolContext, ToolError, registry

SPAM_CSV = Path(__file__).resolve().parents[2] / "sample_data" / "sms_spam.csv"


@pytest.fixture
def churn() -> pd.DataFrame:
    """Tabular classification: churn depends on tenure and plan; customer_id is pure noise."""
    rng = np.random.default_rng(0)
    n = 300
    tenure = rng.integers(1, 60, n)
    plan = rng.choice(["basic", "pro", "team"], n)
    churned = (tenure < 12) | ((plan == "basic") & (rng.random(n) < 0.3))
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "tenure_months": tenure,
            "plan": plan,
            "monthly_spend": rng.normal(50, 10, n).round(2),
            "churned": np.where(churned, "yes", "no"),
        }
    )


@pytest.fixture
def ctx(churn: pd.DataFrame) -> ToolContext:
    return ToolContext(df=churn, dataset_id="ds1", models=ModelStore())


# ---------------------------------------------------------------- training core


def test_classification_drops_ids_and_beats_baseline(churn: pd.DataFrame) -> None:
    r = train(churn, "churned")
    rep = r.report
    assert rep["task"] == "classification" and rep["model"] == "random_forest"
    assert {"column": "customer_id", "reason": rep["dropped_features"][0]["reason"]} in rep["dropped_features"]
    assert "customer_id" not in r.plan.columns
    assert rep["validation"] == "5-fold stratified cross-validation"
    assert set(rep["metrics"]) == {"f1", "precision", "recall", "accuracy", "roc_auc"}
    assert rep["metrics"]["roc_auc"]["mean"] > 0.85
    assert rep["baseline"]["metrics"]["f1"]["mean"] == 0  # 'most common class' never predicts the positive class
    assert rep["beats_baseline"] is True
    assert rep["positive_class"] == "yes"  # the rarer class
    cm = rep["confusion_matrix"]
    assert cm["labels"] == ["no", "yes"] and sum(map(sum, cm["matrix"])) == len(churn)


def test_regression_reports_error_metrics_as_positive_numbers() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 200)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=200), "y": 3 * x + rng.normal(0, 1, 200)})
    rep = train(df, "y", model="linear").report
    assert rep["task"] == "regression" and rep["model"] == "ridge_regression"
    assert rep["metrics"]["r2"]["mean"] > 0.9
    assert rep["metrics"]["mae"]["mean"] > 0 and rep["metrics"]["rmse"]["mean"] >= rep["metrics"]["mae"]["mean"]
    assert rep["beats_baseline"]


def test_leakage_warning_for_near_perfect_feature(churn: pd.DataFrame) -> None:
    df = churn.assign(refund_issued=(churn["churned"] == "yes").astype(int))  # only known after churn: leakage
    caveats = " ".join(train(df, "churned").report["caveats"])
    assert "refund_issued" in caveats and "leakage" in caveats


def test_imbalance_and_small_sample_caveats() -> None:
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({"a": rng.normal(size=n), "label": ["rare"] * 12 + ["common"] * (n - 12)})
    caveats = " ".join(train(df, "label").report["caveats"])
    assert "Imbalanced classes" in caveats and "Only 120 rows" in caveats
    assert "does not clearly beat the baseline" in caveats  # 'a' is pure noise


@pytest.mark.parametrize(
    "df, target, fragment",
    [
        (pd.DataFrame({"a": range(30), "b": range(30)}), "nope", "Unknown target"),
        (pd.DataFrame({"a": range(10), "y": [0, 1] * 5}), "y", "at least 20"),
        (pd.DataFrame({"a": range(40), "y": [f"c{i}" for i in range(40)]}), "y", "too many classes"),
        (pd.DataFrame({"a": range(30), "y": ["x"] * 29 + ["z"]}), "y", "single example"),
        (pd.DataFrame({"id": [f"r{i}" for i in range(30)], "y": [0, 1] * 15}), "y", "No usable feature"),
    ],
)
def test_training_errors_are_actionable(df: pd.DataFrame, target: str, fragment: str) -> None:
    with pytest.raises(TrainingError, match=fragment):
        train(df, target)


# ---------------------------------------------------------------- tools


def test_train_then_explain_then_predict_text_model() -> None:
    df = pd.read_csv(SPAM_CSV)
    ctx = ToolContext(df=df, dataset_id="spam", models=ModelStore())
    trained = registry.dispatch("train_model", ctx, {"target": "label"})
    assert trained["model"] == "logistic_regression" and trained["features"]["text"] == ["message"]
    assert trained["beats_baseline"] and trained["metrics"]["f1"]["mean"] > 0.9

    imp = registry.dispatch("feature_importance", ctx, {"model_id": trained["model_id"]})
    assert imp["features"][0]["feature"] == "message"
    toward = imp["top_terms"]["toward"]
    assert set(toward) == {"ham", "spam"} and len(toward["spam"]) == 10
    assert all(t["weight"] > 0 for t in toward["spam"])

    out = registry.dispatch(
        "predict", ctx, {"model_id": trained["model_id"], "rows": [{"message": "Claim your free prize now"}, {"message": "See you at lunch?"}]}
    )
    preds = out["predictions"]
    assert {p["prediction"] for p in preds} <= {"spam", "ham"}
    assert all(abs(sum(p["probabilities"].values()) - 1) < 1e-6 for p in preds)


def test_models_are_scoped_to_their_dataset(ctx: ToolContext) -> None:
    model_id = registry.dispatch("train_model", ctx, {"target": "churned"})["model_id"]
    other = ToolContext(df=ctx.df, dataset_id="someone-else", models=ctx.models)
    with pytest.raises(ToolError, match="Unknown model_id"):
        registry.dispatch("feature_importance", other, {"model_id": model_id})


def test_predict_rejects_unknown_features_and_fills_missing(ctx: ToolContext) -> None:
    model_id = registry.dispatch("train_model", ctx, {"target": "churned"})["model_id"]
    with pytest.raises(ToolError, match="Unknown feature"):
        registry.dispatch("predict", ctx, {"model_id": model_id, "rows": [{"favourite_colour": "red"}]})
    out = registry.dispatch("predict", ctx, {"model_id": model_id, "rows": [{"tenure_months": 2}]})
    assert "plan" in out["note"] and out["predictions"][0]["prediction"] in {"yes", "no"}


def test_feature_importance_ranks_the_real_driver_first(ctx: ToolContext) -> None:
    model_id = registry.dispatch("train_model", ctx, {"target": "churned"})["model_id"]
    imp = registry.dispatch("feature_importance", ctx, {"model_id": model_id})
    assert imp["features"][0]["feature"] == "tenure_months"
    assert imp["top_terms"] is None  # no text features


def test_ml_tools_need_a_model_store(churn: pd.DataFrame) -> None:
    with pytest.raises(ToolError, match="not available"):
        registry.dispatch("train_model", churn, {"target": "churned"})  # bare DataFrame: no context


def test_train_model_validates_columns(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="did you mean"):
        registry.dispatch("train_model", ctx, {"target": "Churned"})


def test_result_for_model_fits_context_budget(ctx: ToolContext) -> None:
    result = registry.dispatch("train_model", ctx, {"target": "churned"})
    assert "truncated" not in serialize_result(result)


# ---------------------------------------------------------------- positive class & text caveats


def test_positive_class_on_a_tie_is_the_one_that_sorts_last() -> None:
    from app.ml.training import pick_positive_class

    assert pick_positive_class(pd.Series(["ham", "spam"] * 5)) == "spam"
    assert pick_positive_class(pd.Series(["no", "yes"] * 5)) == "yes"
    assert pick_positive_class(pd.Series(["a"] * 8 + ["b"] * 2)) == "b"  # rarer class wins without a tie
    assert pick_positive_class(pd.Series(["no", "yes"] * 5), requested="no") == "no"
    with pytest.raises(TrainingError, match="not one of"):
        pick_positive_class(pd.Series(["no", "yes"]), requested="maybe")


def test_spam_report_names_the_positive_class_and_flags_templated_text() -> None:
    rep = train(pd.read_csv(SPAM_CSV), "label").report
    assert rep["positive_class"] == "spam" and "for class 'spam'" in rep["metrics_note"]
    caveats = " ".join(rep["caveats"])
    assert "templated or near-duplicate" in caveats
    assert "Near-perfect scores" in caveats
    assert "only knows words" in caveats


def test_templated_share() -> None:
    from app.ml.training import templated_share

    varied = pd.Series(["the cat sat on a mat today", "dogs run fast in the park daily", "rain falls softly on old roofs"])
    assert templated_share(varied) == 0
    templ = pd.Series([f"Congratulations! You have been selected for prize {i}" for i in range(4)] + ["something else entirely here ok"])
    assert templated_share(templ) == 0.8
