# FLADSVR and FLADTSVR

This repository contains Python implementations of two nonlinear regression
models for triangular fuzzy responses:

- **FLADSVR**, a kernel least absolute deviation support vector regression
  model optimized by Split--Bregman iterations;
- **FLADTSVR**, an anchored twin formulation with separate lower and upper
  objectives optimized by ADMM.

The empirical study accompanying *Two Fuzzy Least Absolute Deviation Support
Vector Regression Models* compares these models with fuzzy-regression and
support-vector baselines on eight regression tasks. Triangular fuzzy numbers
are stored as `(center, left_spread, right_spread)`.

## Setup

Python 3.10--3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Conda users may instead run:

```bash
conda env create -f environment.yml
conda activate fuzzy-lad-regression
```

## Data

Examples 1--5 are native fuzzy-response datasets. Example 6 uses the
scikit-learn Diabetes data. Examples 7 and 8 use the full UCI Energy Efficiency
and Concrete Compressive Strength datasets, with 768 and 1,030 observations,
respectively. Heating load is the target for Energy Efficiency.

The crisp targets in Examples 6--8 are mapped to triangular fuzzy responses by

```text
center       = y
left_spread  = 0.08 * max(abs(y), 1)
right_spread = 0.12 * max(abs(y), 1)
```

The native fuzzy-response datasets are not redistributed. File locations,
public sources, target choices, and preparation instructions are given in
[data/data_sources.md](data/data_sources.md).

## Experiments

Run commands from the repository root. Output files are written under
`results/generated/`.

Clean-data evaluation:

```bash
python scripts/run_clean_experiments.py
```

To run only the proposed methods:

```bash
python scripts/run_clean_experiments.py --methods FLADSVR FLADTSVR
```

Robustness experiments:

```bash
python scripts/run_robustness_experiments.py
```

Parameter sensitivity:

```bash
python scripts/run_sensitivity_analysis.py
```

Dataset-level statistical analysis:

```bash
python scripts/analyze_results.py
```

The clean-data comparison includes FLADSVR, FLADTSVR, Hao--Chiang fuzzy SVR,
FuzzyNW, component RBF-SVR, component LAD, component ridge, fuzzy mean, and
fuzzy median. Hyperparameters are selected on the inner training folds using
RMSE, with MSM used only to resolve exact ties.

The Hao--Chiang implementation can also be evaluated on the linear example
from its source article:

```bash
python scripts/run_hao_chiang_example.py
```

## Metrics

- **MSM:** membership-function intersection over union (higher is better).
- **RMSE:** `sqrt(mean(center_error^2 + left_spread_error^2 + right_spread_error^2))`
  (lower is better).

Both metrics are calculated from the same held-out predictions.

## Results and project structure

The numerical results reported in the article are available in
`results/reported_results/`. Runtime varies with hardware, while metric values
are reported at manuscript precision.

```text
src/       model, metric, data, tuning, and evaluation code
configs/   experimental settings and selected parameters
scripts/   experiment and analysis entry points
tests/     unit tests
data/      dataset source and preparation information
results/   reported tables and experiment outputs
```

A small generated-data run is available for checking the installation:

```bash
python scripts/run_clean_experiments.py --quick-check
```

The test suite can be run with:

```bash
python -m unittest discover -s tests -v
```

## License

The code is released under the MIT License. Dataset licenses remain those of
their original providers.
