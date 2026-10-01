#!/usr/bin/env python3
"""Run the paired training-only contamination experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import load_example
from src.evaluation import outer_splits
from src.experiments import ROBUSTNESS_METHODS, clean_outer_parameters, contaminate, fit_predict, model_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, choices=(1, 4, 5))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--methods", nargs="+", choices=ROBUSTNESS_METHODS,
                        default=ROBUSTNESS_METHODS)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "generated" / "robustness")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / "experimental_settings.json").read_text())
    parameters = clean_outer_parameters(ROOT)
    hao_parameters = json.loads(
        (ROOT / "configs" / "hao_chiang_outer_parameters.json").read_text()
    )
    predictions, realizations, fits = [], [], []
    for example in config["contamination_examples"]:
        if args.example is not None and example != args.example:
            continue
        features, targets = load_example(example)
        for fold, (train, test) in enumerate(outer_splits(example, len(features))):
            if args.fold is not None and fold != args.fold:
                continue
            for kind in config["contamination_types"]:
                for rate in config["contamination_rates"]:
                    for seed in config["contamination_seeds"]:
                        contaminated_x, contaminated_y, indices, detail = contaminate(
                            features[train], targets[train], kind, rate,
                            seed + 10000 * example + 100 * fold,
                        )
                        realization = f"E{example}_F{fold}_{kind}_R{rate:.2f}_S{seed}"
                        realizations.append({
                            "realization_id": realization, "example": example, "outer_fold": fold,
                            "type": kind, "rate": rate, "seed": seed,
                            "contaminated_local_indices": json.dumps(indices), "detail": json.dumps(detail),
                        })
                        for method in args.methods:
                            source = hao_parameters if method == "HaoChiangFSVR" else parameters
                            selected = source[str(example)][method][str(fold)]
                            predicted, model = fit_predict(
                                method, selected, contaminated_x, contaminated_y, features[test]
                            )
                            fits.append({
                                "realization_id": realization, "method": method,
                                "parameters": json.dumps(selected, sort_keys=True), **model_diagnostics(model),
                            })
                            for local, sample in enumerate(test):
                                predictions.append({
                                    "realization_id": realization, "example": example, "outer_fold": fold,
                                    "type": kind, "rate": rate, "seed": seed, "method": method,
                                    "sample_id": int(sample), "predicted_center": predicted[local, 0],
                                    "predicted_left_spread": predicted[local, 1],
                                    "predicted_right_spread": predicted[local, 2],
                                    "true_center": targets[sample, 0], "true_left_spread": targets[sample, 1],
                                    "true_right_spread": targets[sample, 2],
                                })
            print(f"DONE Example {example} fold {fold}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("contamination_realizations.csv", realizations),
                           ("robustness_fit_diagnostics.csv", fits),
                           ("robustness_predictions.csv", predictions)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
