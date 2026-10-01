"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .baselines import GRIDS as BASELINE_GRIDS, make_model as make_baseline
from .metrics import msm, rmse
from .tuning import fit_predict as fit_proposed
from .tuning import kfold_indices
from .utils import FeatureScaler, TargetScale

PROPOSED = {"FLADSVR", "FLADTSVR"}
METHODS = [
    "FLADSVR", "FLADTSVR", "HaoChiangFSVR", "FuzzyMean", "FuzzyMedian", "FuzzyNW",
    "ComponentRidge", "ComponentLAD", "ComponentRBF_SVR",
]
ROBUSTNESS_METHODS = [
    "FLADSVR", "FLADTSVR", "HaoChiangFSVR", "FuzzyMean", "FuzzyMedian", "FuzzyNW",
    "ComponentRidge", "ComponentLAD", "ComponentRBF_SVR",
]


def fit_predict(method, parameters, x_train, y_train, x_test):
    if method in PROPOSED:
        prediction, model, _, _ = fit_proposed(
            method, parameters, x_train, y_train, x_test
        )
        return prediction, model
    feature_scale = FeatureScaler().fit(x_train)
    if method == "HaoChiangFSVR":
        symmetric = np.column_stack((y_train[:, 0], (y_train[:, 1]+y_train[:, 2])/2))
        target_scale = TargetScale().fit(symmetric)
        model = make_baseline(method, parameters).fit(
            feature_scale.transform(x_train), target_scale.transform(symmetric)
        )
        restored = target_scale.inverse(model.predict(feature_scale.transform(x_test)))
        prediction = np.column_stack((restored[:, 0], restored[:, 1], restored[:, 1]))
        prediction[:, 1:] = np.maximum(prediction[:, 1:], 0.0)
        return prediction, model
    target_scale = TargetScale().fit(y_train)
    model = make_baseline(method, parameters).fit(
        feature_scale.transform(x_train), target_scale.transform(y_train)
    )
    prediction = target_scale.inverse(model.predict(feature_scale.transform(x_test)))
    prediction[:, 1:] = np.maximum(prediction[:, 1:], 0.0)
    return prediction, model


def tune_baseline(method, features, targets, seed, folds=3):
    splits = list(kfold_indices(len(features), min(folds, len(features)), seed, True))
    records = []
    for candidate_id, parameters in enumerate(BASELINE_GRIDS[method]):
        prediction = np.empty_like(targets)
        for train, validation in splits:
            prediction[validation] = fit_predict(
                method, parameters, features[train], targets[train], features[validation]
            )[0]
        records.append({
            "candidate_id": candidate_id,
            "params": parameters,
            "inner_rmse": rmse(targets, prediction),
            "inner_msm": msm(targets, prediction),
        })
    best = min(records, key=lambda row: (row["inner_rmse"], -row["inner_msm"], row["candidate_id"]))
    return dict(best["params"]), records


def clean_outer_parameters(repository_root):
    path = Path(repository_root) / "configs" / "clean_outer_parameters.json"
    return json.loads(path.read_text(encoding="utf-8"))


def robust_scale(values, axis=0):
    values = np.asarray(values, float)
    median = np.median(values, axis=axis)
    mad = np.median(np.abs(values - median), axis=axis) * 1.4826
    iqr = (np.quantile(values, 0.75, axis=axis) - np.quantile(values, 0.25, axis=axis)) / 1.349
    standard = np.std(values, axis=axis)
    return np.where(mad > 1e-12, mad, np.where(iqr > 1e-12, iqr, np.where(standard > 1e-12, standard, 1.0)))


def contaminate(features, targets, kind, rate, seed):
    features = np.array(features, float, copy=True)
    targets = np.array(targets, float, copy=True)
    random = np.random.default_rng(seed)
    if rate == 0:
        return features, targets, [], {}
    count = max(1, int(round(rate * len(features))))
    indices = np.sort(random.choice(len(features), count, replace=False))
    if kind == "response_center":
        scale = float(robust_scale(targets[:, 0]))
        delta = random.choice([-1.0, 1.0], count) * 6 * scale
        targets[indices, 0] += delta
        detail = {"delta_center": delta.tolist(), "scale": scale}
    elif kind == "fuzzy_spread":
        factors = random.choice([0.25, 4.0], size=(count, 2))
        targets[indices, 1:] *= factors
        detail = {"spread_factors": factors.tolist()}
    elif kind == "leverage":
        scale = np.asarray(robust_scale(features, axis=0))
        delta = random.choice([-1.0, 1.0], size=(count, features.shape[1])) * 6 * scale
        features[indices] += delta
        detail = {"feature_delta": delta.tolist(), "scale": scale.tolist()}
    else:
        raise ValueError(kind)
    return features, targets, indices.tolist(), detail


def model_diagnostics(model):
    reasons, histories = [], []
    if hasattr(model, "stopping_reasons_"):
        stopping = model.stopping_reasons_
        if isinstance(stopping, dict):
            for side in ("lower", "upper"):
                reasons.extend(stopping[side])
                histories.extend(model.history_[side])
        else:
            reasons.extend(stopping)
            histories.extend(model.history_)
    if not histories:
        return {"converged": True, "iterations": 0, "stopping_reasons": "not_iterative"}
    last = [history[-1] for history in histories]
    return {
        "converged": all(reason == "residual_tolerance" for reason in reasons),
        "iterations": max(len(history) for history in histories),
        "final_objective": float(np.mean([row["objective"] for row in last])),
        "final_primal": float(max(row["primal_residual"] for row in last)),
        "final_dual": float(max(row["dual_residual"] for row in last)),
        "final_parameter_change": float(max(row["parameter_change"] for row in last)),
        "stopping_reasons": "|".join(reasons),
    }
