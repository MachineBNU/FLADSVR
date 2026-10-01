#!/usr/bin/env python3
"""Run the mu/epsilon sensitivity analysis."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import load_example
from src.evaluation import outer_splits
from src.experiments import clean_outer_parameters, fit_predict, model_diagnostics


def main():
    config = json.loads((ROOT / "configs" / "experimental_settings.json").read_text())
    base_parameters = clean_outer_parameters(ROOT)
    output = ROOT / "results" / "generated" / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    predictions, diagnostics = [], []
    for example in config["sensitivity_examples"]:
        features, targets = load_example(example)
        for method in ("FLADSVR", "FLADTSVR"):
            for fold, (train, test) in enumerate(outer_splits(example, len(features))):
                base = base_parameters[str(example)][method][str(fold)]
                for factor in config["sensitivity_factors"]:
                    varied_names = ("mu",) if method == "FLADSVR" else ("epsilon1", "epsilon2", "joint")
                    for varied in varied_names:
                        parameters = dict(base)
                        if varied == "mu": parameters["mu"] = base["mu"] * factor
                        if varied in ("epsilon1", "joint"): parameters["epsilon1"] = base["epsilon1"] * factor
                        if varied in ("epsilon2", "joint"): parameters["epsilon2"] = base["epsilon2"] * factor
                        predicted, model = fit_predict(
                            method, parameters, features[train], targets[train], features[test]
                        )
                        scenario = f"E{example}_{method}_F{fold}_{varied}_X{factor}"
                        diagnostics.append({
                            "scenario_id": scenario, "example": example, "method": method,
                            "outer_fold": fold, "varied": varied, "factor": factor,
                            "parameters": json.dumps(parameters, sort_keys=True), **model_diagnostics(model),
                        })
                        for local, sample in enumerate(test):
                            predictions.append({
                                "scenario_id": scenario, "example": example, "method": method,
                                "outer_fold": fold, "varied": varied, "factor": factor,
                                "sample_id": int(sample), "predicted_center": predicted[local, 0],
                                "predicted_left_spread": predicted[local, 1],
                                "predicted_right_spread": predicted[local, 2],
                                "true_center": targets[sample, 0], "true_left_spread": targets[sample, 1],
                                "true_right_spread": targets[sample, 2],
                            })
            print(f"DONE Example {example} {method}", flush=True)
    for filename, rows in (("sensitivity_predictions.csv", predictions),
                           ("sensitivity_fit_diagnostics.csv", diagnostics)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
