#!/usr/bin/env python3
"""Evaluate the linear example in Table I of Hao and Chiang (2008)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hao_chiang import _intercepts, linear_kernel, solve_dual


def main():
    x = np.arange(1.0, 9.0)[:, None] / 10.0
    center = np.array([2.25, 2.875, 2.5, 4.25, 4.0, 5.25, 7.5, 8.5])
    spread = np.array([0.75, 0.875, 1.0, 1.75, 1.5, 1.25, 2.0, 1.5])
    a1, a2, kx, ka, iterations = solve_dual(
        x, center, spread, 0.5, 25.0, 15.0, linear_kernel
    )
    center_no = kx @ (a1 - a2)
    spread_no = 0.5 / 25.0 * (ka @ (a1 + a2))
    intercept, recovery = _intercepts(
        a1, a2, center_no, spread_no, center, spread, 0.5, 25.0, 15.0
    )
    result = {
        "center_slope": float((a1 - a2) @ x[:, 0]),
        "center_intercept": float(intercept[0]),
        "spread_slope": float(0.5 / 25.0 * ((a1 + a2) @ x[:, 0])),
        "spread_intercept": float(intercept[1]),
        "intercept_recovery": recovery,
        "solver_iterations": iterations,
        "reported_coefficients": [8.0, 1.1525, 0.58, 2.931],
    }
    output = ROOT / "results" / "generated" / "hao_chiang_example.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
