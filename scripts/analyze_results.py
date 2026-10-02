#!/usr/bin/env python3
"""Compute clean-data ranks and statistical tests."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LEARNING_METHODS = [
    "FLADSVR", "FLADTSVR", "HaoChiangFSVR", "FuzzyNW",
    "ComponentRBF_SVR", "ComponentLAD", "ComponentRidge",
]
REFERENCE_PREDICTORS = ["FuzzyMean", "FuzzyMedian"]
ALL_PREDICTORS = LEARNING_METHODS + REFERENCE_PREDICTORS
PAIRS = [("FLADSVR", "FuzzyNW"), ("FLADSVR", "ComponentRBF_SVR"),
         ("FLADTSVR", "FuzzyNW"), ("FLADSVR", "FLADTSVR"),
         ("FLADSVR", "HaoChiangFSVR"), ("FLADTSVR", "HaoChiangFSVR")]


def holm(values):
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
        default=ROOT / "results" / "reported_results" / "final_clean_result_matrix.csv")
    parser.add_argument("--robustness-input", type=Path,
        default=ROOT / "results" / "reported_results" / "robustness_dataset_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "generated" / "statistics")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    observed = set(zip(data["dataset"], data["method"]))
    datasets = list(dict.fromkeys(data["dataset"]))
    expected = {(dataset, method) for dataset in datasets for method in ALL_PREDICTORS}
    if len(datasets) != 8 or observed != expected:
        raise RuntimeError("The clean-data result matrix is incomplete")
    rank_rows, friedman_rows, pair_rows, nemenyi_rows = [], [], [], []
    for metric, higher in (("msm", True), ("rmse", False)):
        pivot = data.pivot(index="dataset", columns="method", values=metric)[LEARNING_METHODS]
        ranks = pivot.rank(axis=1, ascending=not higher, method="average")
        for dataset in ranks.index:
            for method in LEARNING_METHODS:
                rank_rows.append({"metric": metric.upper(), "dataset": dataset,
                                  "method": method, "rank": ranks.loc[dataset, method]})
        statistic, p_value = friedmanchisquare(*[pivot[method] for method in LEARNING_METHODS])
        friedman_rows.append({"metric": metric.upper(), "datasets": 8, "methods": 7,
                              "statistic": statistic, "p_value": p_value})
        average = ranks.mean()
        method_count = len(LEARNING_METHODS)
        standard_error = math.sqrt(method_count * (method_count + 1) / (6 * len(datasets)))
        critical = (
            studentized_range.ppf(0.95, method_count, np.inf)
            / math.sqrt(2)
            * standard_error
        )
        for i, first in enumerate(LEARNING_METHODS):
            for second in LEARNING_METHODS[i + 1:]:
                difference = abs(average[first] - average[second])
                q_range = difference / standard_error * math.sqrt(2)
                nemenyi_rows.append({"metric": metric.upper(), "first_method": first,
                                     "second_method": second, "average_rank_difference": difference,
                                     "critical_difference_0.05": critical,
                                     "p_value": float(studentized_range.sf(q_range, method_count, np.inf)),
                                     "significant_0.05": difference > critical})
        family = []
        for first, second in PAIRS:
            improvement = (pivot[first] - pivot[second] if higher else pivot[second] - pivot[first]).to_numpy()
            nonzero = improvement[np.abs(improvement) > 1e-12]
            raw = 1.0 if len(nonzero) == 0 else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            ranks_abs = rankdata(np.abs(nonzero)) if len(nonzero) else np.array([])
            effect = 0.0 if len(nonzero) == 0 else float(
                (ranks_abs[nonzero > 0].sum() - ranks_abs[nonzero < 0].sum()) / ranks_abs.sum()
            )
            family.append({"metric": metric.upper(), "first_method": first, "second_method": second,
                           "raw_p": raw, "wins": int((improvement > 1e-12).sum()),
                           "ties": int((np.abs(improvement) <= 1e-12).sum()),
                           "losses": int((improvement < -1e-12).sum()), "rank_biserial": effect})
        adjusted = holm([row["raw_p"] for row in family])
        for row, value in zip(family, adjusted):
            row["holm_adjusted_p"] = value; row["significant_0.05"] = value < 0.05
            pair_rows.append(row)
    pd.DataFrame(rank_rows).to_csv(args.output / "clean_dataset_ranks.csv", index=False)
    pd.DataFrame(friedman_rows).to_csv(args.output / "clean_friedman_tests.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output / "clean_wilcoxon_holm.csv", index=False)
    pd.DataFrame(nemenyi_rows).to_csv(args.output / "clean_nemenyi.csv", index=False)

    robustness = pd.read_csv(args.robustness_input)
    robustness = robustness[robustness["method"].isin(LEARNING_METHODS)]
    endpoints = ["msm_degradation_auc", "rmse_degradation_auc"]
    keys = ["example", "type", "method"]
    if robustness.groupby(keys)[endpoints].nunique().to_numpy().max() != 1:
        raise RuntimeError("Robustness AUC values are inconsistent across rates")
    auc = robustness.drop_duplicates(keys)
    robustness_rows = []
    for contamination_type in sorted(auc["type"].unique()):
        block = auc[auc["type"] == contamination_type]
        for endpoint in endpoints:
            pivot = block.pivot(index="example", columns="method", values=endpoint)[LEARNING_METHODS]
            statistic, p_value = friedmanchisquare(
                *[pivot[method] for method in LEARNING_METHODS]
            )
            robustness_rows.append({
                "contamination_type": contamination_type,
                "endpoint": endpoint,
                "datasets": len(pivot),
                "methods": len(LEARNING_METHODS),
                "friedman_statistic": statistic,
                "p_value": p_value,
                "posthoc_justified": p_value < 0.05,
            })
    pd.DataFrame(robustness_rows).to_csv(
        args.output / "robustness_friedman_tests.csv", index=False
    )


if __name__ == "__main__":
    main()
