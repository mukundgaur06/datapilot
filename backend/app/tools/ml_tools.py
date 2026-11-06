"""ML tools: train a model, explain it, use it. Thin wrappers over app.ml (where the logic lives)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.ml import models as ml_models
from app.ml.training import TrainingError, train
from app.tools.data_tools import _Args, require_columns
from app.tools.registry import ToolContext, ToolError, tool


def _get_model(ctx: ToolContext, model_id: str) -> ml_models.TrainedModel:
    try:
        return ctx.models.get(ctx.dataset_id, model_id)
    except ml_models.ModelNotFound:
        known = ctx.models.for_dataset(ctx.dataset_id)
        raise ToolError(
            f"Unknown model_id {model_id!r}. Models trained on this dataset: {known or 'none yet (call train_model first)'}"
        ) from None


class TrainModelArgs(_Args):
    target: str = Field(description="Column to predict (exact name).")
    features: list[str] = Field(
        default_factory=list,
        description="Columns to use as inputs. Empty = all other columns (ID-like and constant columns are dropped automatically).",
    )
    task: Literal["auto", "classification", "regression"] = Field(
        default="auto", description="'auto' picks regression for numeric targets with many values, else classification."
    )
    model: Literal["auto", "linear", "random_forest"] = Field(
        default="auto", description="'auto': linear for text features, random forest otherwise."
    )
    positive_class: str | None = Field(
        default=None,
        description="Binary classification: the class of interest (e.g. 'spam', 'yes'), which precision/recall/F1 "
        "are reported for. Default: the rarer class.",
    )


@tool(
    name="train_model",
    description=(
        "Train a machine-learning model that predicts `target` from other columns, and evaluate it honestly with k-fold "
        "cross-validation against a dummy baseline. Handles numbers, categories and free text. Returns a model_id, "
        "cross-validated metrics (mean ± std), the baseline's metrics, a confusion matrix (classification), dropped "
        "features with reasons, and data-driven caveats. Use it for 'build a model / classifier / predictor' or 'can X "
        "predict Y' questions. Follow up with feature_importance to explain what drives the model."
    ),
    args_model=TrainModelArgs,
    needs_context=True,
)
def train_model(ctx: ToolContext, args: TrainModelArgs) -> dict[str, Any]:
    require_columns(ctx.df, [args.target, *args.features])
    try:
        result = train(ctx.df, args.target, args.features or None, args.task, args.model, args.positive_class)
    except TrainingError as e:
        raise ToolError(str(e)) from None
    model = ctx.models.add(ctx.dataset_id, result)
    return {"model_id": model.id, **result.report}


class FeatureImportanceArgs(_Args):
    model_id: str = Field(description="Id returned by train_model.")
    top_k: int = Field(default=10, ge=3, le=20, description="How many features (and text terms) to return.")


@tool(
    name="feature_importance",
    description=(
        "Explain what a trained model relies on: permutation importance of each input column (measured on held-out "
        "data), plus, for text columns, the words that push predictions toward each class. Use after train_model."
    ),
    args_model=FeatureImportanceArgs,
    needs_context=True,
)
def feature_importance(ctx: ToolContext, args: FeatureImportanceArgs) -> dict[str, Any]:
    model = _get_model(ctx, args.model_id)
    return ml_models.feature_importance(model, ctx.df, args.top_k)


class PredictArgs(_Args):
    model_id: str = Field(description="Id returned by train_model.")
    rows: list[dict[str, Any]] = Field(
        min_length=1, max_length=20,
        description="1-20 new rows to score, as objects mapping feature column -> value. Omitted features count as missing.",
    )


@tool(
    name="predict",
    description="Use a trained model to predict the target for new rows the user describes (e.g. 'is this message spam?').",
    args_model=PredictArgs,
    needs_context=True,
)
def predict(ctx: ToolContext, args: PredictArgs) -> dict[str, Any]:
    model = _get_model(ctx, args.model_id)
    try:
        return ml_models.predict(model, args.rows)
    except ValueError as e:
        raise ToolError(str(e)) from None
