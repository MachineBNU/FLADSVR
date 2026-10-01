#!/usr/bin/env python3
"""Run the clean-data evaluation with fold-local preprocessing."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEED
from src.datasets import load_example, load_public_dataset
from src.evaluation import outer_splits
from src.experiments import METHODS, PROPOSED, fit_predict, tune_baseline
from src.metrics import msm, rmse
from src.tuning import kfold_indices, tune


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quick_check(output):
    features = np.linspace(-1.5, 1.5, 9)[:, None]
    targets = np.column_stack((
        0.4 + 0.8 * features[:, 0],
        0.25 + 0.04 * (features[:, 0] + 0.3) ** 2,
        0.35 + 0.03 * (features[:, 0] - 0.2) ** 2,
    ))
    rows = []
    parameters = {
        "FLADSVR": {"gamma": 5.0, "h": 0.8, "mu": 1.0},
        "FLADTSVR": {"h": 0.8, "lambda_reg": 0.05, "epsilon1": 0.12,
                      "epsilon2": 0.18, "tau": 1.0, "c_lower": 4.0, "c_upper": 4.0},
        "HaoChiangFSVR": {"H": 0.5, "K_tradeoff": 25.0, "P": 15.0, "h": 0.8},
    }
    for method in ("FLADSVR", "FLADTSVR", "HaoChiangFSVR"):
        prediction, _ = fit_predict(method, parameters[method], features, targets, features)
        rows.append({
            "mode": "quick_check", "method": method, "n": len(targets),
            "msm": msm(targets, prediction), "rmse": rmse(targets, prediction),
            "finite": bool(np.isfinite(prediction).all()),
            "valid_spreads": bool((prediction[:, 1:] >= 0).all()),
        })
    write_csv(output / "quick_check_summary.csv", rows)
    return rows


def evaluate_dataset(name, features, targets, splits, dataset_seed, methods, legacy=False):
    prediction_rows, tuning_rows, summaries = [], [], []
    for method in methods:
        prediction = np.empty_like(targets)
        parameter_strings = []
        start = time.perf_counter()
        for fold, (train, test) in enumerate(splits):
            if legacy:
                method_offset = 0 if method == "FLADSVR" else (1 if method == "FLADTSVR" else 17)
            else:
                method_offset = 0
            seed = dataset_seed + 100 * fold + method_offset
            if method in PROPOSED:
                parameters, trace = tune(method, features[train], targets[train], seed, 3)
            else:
                parameters, trace = tune_baseline(method, features[train], targets[train], seed, 3)
            fold_prediction, _ = fit_predict(
                method, parameters, features[train], targets[train], features[test]
            )
            prediction[test] = fold_prediction
            parameter_strings.append(json.dumps(parameters, sort_keys=True))
            for record in trace:
                tuning_rows.append({
                    "dataset": name, "method": method, "outer_fold": fold,
                    **{**record, "params": json.dumps(record["params"], sort_keys=True)},
                })
            for local, sample in enumerate(test):
                prediction_rows.append({
                    "dataset": name, "method": method, "outer_fold": fold,
                    "sample_id": int(sample), "selected_hyperparameters": json.dumps(parameters, sort_keys=True),
                    "predicted_center": fold_prediction[local, 0],
                    "predicted_left_spread": fold_prediction[local, 1],
                    "predicted_right_spread": fold_prediction[local, 2],
                    "true_center": targets[sample, 0], "true_left_spread": targets[sample, 1],
                    "true_right_spread": targets[sample, 2],
                })
        summaries.append({
            "dataset": name, "method": method, "n": len(targets), "features": features.shape[1],
            "msm": msm(targets, prediction), "rmse": rmse(targets, prediction),
            "runtime_seconds": time.perf_counter() - start,
            "selected_parameter_distribution": json.dumps(dict(Counter(parameter_strings)), sort_keys=True),
        })
        print(f"DONE {name} {method}", flush=True)
    return prediction_rows, tuning_rows, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-check", action="store_true",
                        help="run a small generated-data check")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "generated" / "clean")
    args = parser.parse_args()
    if args.quick_check:
        print(json.dumps(quick_check(args.output), indent=2))
        return
    all_predictions, all_tuning, all_summaries = [], [], []
    for example in range(1, 6):
        features, targets = load_example(example)
        result = evaluate_dataset(
            f"Example {example}", features, targets,
            list(outer_splits(example, len(features))), SEED + 1000 * example, args.methods, legacy=True,
        )
        all_predictions += result[0]; all_tuning += result[1]; all_summaries += result[2]
    for dataset_index, name in enumerate(("Diabetes", "EnergyEfficiency", "ConcreteStrength")):
        features, targets, _ = load_public_dataset(name)
        splits = list(kfold_indices(len(features), 3, 20260920 + dataset_index, True))
        result = evaluate_dataset(
            name, features, targets, splits, 20260920 + 1000 * dataset_index, args.methods,
        )
        all_predictions += result[0]; all_tuning += result[1]; all_summaries += result[2]
    write_csv(args.output / "clean_raw_predictions.csv", all_predictions)
    write_csv(args.output / "clean_inner_tuning.csv", all_tuning)
    write_csv(args.output / "clean_summary.csv", all_summaries)


if __name__ == "__main__":
    main()
