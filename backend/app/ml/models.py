"""Trained-model storage, feature importance and prediction.

Models live in memory, scoped to the dataset they were trained on: looking a model up
with a different dataset id fails exactly like an unknown id (no cross-dataset access).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from app.ml.training import RANDOM_STATE, TrainingResult


class ModelNotFound(KeyError):
    pass


@dataclass
class TrainedModel:
    id: str
    dataset_id: str
    result: TrainingResult
    importance: dict[str, Any] | None = field(default=None, repr=False)  # cached: it's the slow part


class ModelStore:
    def __init__(self, max_models: int = 50) -> None:
        self._models: OrderedDict[str, TrainedModel] = OrderedDict()
        self._lock = threading.Lock()
        self.max_models = max_models

    def add(self, dataset_id: str, result: TrainingResult) -> TrainedModel:
        model = TrainedModel(id=f"m_{uuid.uuid4().hex[:8]}", dataset_id=dataset_id, result=result)
        with self._lock:
            self._models[model.id] = model
            while len(self._models) > self.max_models:
                self._models.popitem(last=False)
        return model

    def get(self, dataset_id: str, model_id: str) -> TrainedModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None or model.dataset_id != dataset_id:
            raise ModelNotFound(model_id)
        return model

    def for_dataset(self, dataset_id: str) -> list[str]:
        with self._lock:
            return [m.id for m in self._models.values() if m.dataset_id == dataset_id]


models = ModelStore()


# ---------------------------------------------------------------- feature importance


def _top_terms(result: TrainingResult, top_k: int) -> dict[str, Any] | None:
    """For text features: which words push predictions where. Only meaningful for linear models."""
    if not result.plan.text:
        return None
    prep = result.pipeline.named_steps["prep"]
    estimator = result.pipeline.named_steps["model"]
    names = np.array(prep.get_feature_names_out())
    is_text = np.array([n.startswith("text:") for n in names])
    terms = np.array([n.split("__", 1)[1] for n in names])

    coef = getattr(estimator, "coef_", None)
    if coef is None:  # random forest: magnitude only, no direction
        imp = estimator.feature_importances_
        order = [i for i in np.argsort(imp)[::-1] if is_text[i]][:top_k]
        return {"kind": "magnitude", "terms": [{"term": terms[i], "weight": round(float(imp[i]), 4)} for i in order]}

    coef = np.atleast_2d(coef)
    if result.task == "regression":
        w = coef[0]
        idx = [i for i in np.argsort(w) if is_text[i]]
        return {
            "kind": "direction",
            "increase": [{"term": terms[i], "weight": round(float(w[i]), 4)} for i in idx[::-1][:top_k]],
            "decrease": [{"term": terms[i], "weight": round(float(w[i]), 4)} for i in idx[:top_k]],
        }
    estimator_classes = [str(c) for c in estimator.classes_]
    if coef.shape[0] == 1:  # binary: positive weight pushes toward classes_[1]
        w = coef[0]
        idx = [i for i in np.argsort(w) if is_text[i]]
        return {
            "kind": "direction",
            "toward": {
                estimator_classes[1]: [{"term": terms[i], "weight": round(float(w[i]), 4)} for i in idx[::-1][:top_k]],
                estimator_classes[0]: [{"term": terms[i], "weight": round(float(-w[i]), 4)} for i in idx[:top_k]],
            },
        }
    return {
        "kind": "direction",
        "toward": {
            cls: [{"term": terms[i], "weight": round(float(row[i]), 4)} for i in [j for j in np.argsort(row)[::-1] if is_text[j]][: max(top_k // 2, 5)]]
            for cls, row in zip(estimator_classes, coef)
        },
    }


def feature_importance(model: TrainedModel, df: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
    """Permutation importance on a held-out 25%: how much the score drops when one input column is shuffled."""
    if model.importance is not None:
        return model.importance
    r = model.result
    data = df[df[r.target].notna()]
    X, y = data[r.plan.columns], data[r.target]
    stratify = y if r.task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=stratify, random_state=RANDOM_STATE)
    fitted = clone(r.pipeline).fit(X_tr, y_tr)
    scorer = r.scoring.scorers[r.scoring.primary]
    pi = permutation_importance(fitted, X_te, y_te, scoring=scorer, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    columns = sorted(
        (
            {"feature": col, "importance": round(float(m), 4), "std": round(float(s), 4)}
            for col, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )
    model.importance = {
        "model_id": model.id,
        "method": f"permutation importance on a held-out 25% ({len(X_te)} rows), metric = {r.scoring.primary}",
        "features": columns[:top_k],
        "top_terms": _top_terms(r, top_k),
        "note": "Importance = how much the score drops when the column is shuffled. It shows what the model relies on, "
        "not what causes the outcome.",
    }
    return model.importance


# ---------------------------------------------------------------- prediction


def predict(model: TrainedModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    r = model.result
    expected = r.plan.columns
    unknown = sorted({k for row in rows for k in row} - set(expected))
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}. This model uses: {expected}")
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in expected} for row in rows], columns=expected)
    missing = sorted({c for row in rows for c in expected if c not in row})
    out: dict[str, Any] = {"model_id": model.id, "target": r.target, "predictions": []}
    if r.task == "classification":
        proba = model.result.pipeline.predict_proba(X)
        classes = [str(c) for c in r.pipeline.named_steps["model"].classes_]
        for p in proba:
            ranked = sorted(zip(classes, p), key=lambda t: t[1], reverse=True)[:3]
            out["predictions"].append(
                {"prediction": ranked[0][0], "probabilities": {c: round(float(v), 4) for c, v in ranked}}
            )
    else:
        out["predictions"] = [{"prediction": round(float(v), 4)} for v in r.pipeline.predict(X)]
    if missing:
        out["note"] = f"Features not given were treated as missing (numbers: training median; categories: a '(missing)' category; text: empty): {missing}"
    return out
