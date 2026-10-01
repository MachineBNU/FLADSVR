# Dataset download and preparation

## Native fuzzy-response datasets

Obtain Examples 1--5 from the sources cited in the article or from the authors,
preserve their workbook structure, and use the filenames below.

| Filename | n | p |
|---|---:|---:|
| `data/native/example1.xlsx` | 30 | 1 |
| `data/native/example2.xlsx` | 30 | 5 |
| `data/native/example3.xlsx` | 8 | 1 |
| `data/native/example4.xlsx` | 184 | 1 |
| `data/native/example5.xlsx` | 25 | 18 |

The data loaders and feature representations are documented in
`src/datasets.py`. Fuzzy outputs use center and left/right spreads, not
endpoints.

## Diabetes

`sklearn.datasets.load_diabetes` supplies all 442 cases and 10 features. The bundled disease-progression target is used.

## Energy Efficiency

1. Download the Energy Efficiency dataset from the UCI Machine Learning Repository, DOI [10.24432/C51307](https://doi.org/10.24432/C51307).
2. Place the original workbook at `data/public/ENB2012_data.xlsx`.
3. The first eight columns are features.
4. Use **Y1 heating load** as the target.
5. Use all 768 observations.

## Concrete Compressive Strength

1. Download the Concrete Compressive Strength dataset from UCI, DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67).
2. Save the CSV representation as `data/public/Concrete_Data.csv`, with eight feature columns followed by compressive strength.
3. Use all 1,030 observations.

## Synthetic fuzzy-response rule

For Diabetes, Energy, and Concrete only:

```text
center       = y
left_spread  = 0.08 * max(abs(y), 1)
right_spread = 0.12 * max(abs(y), 1)
```

The transformation is deterministic and depends only on each target value.
These three datasets are synthetically fuzzified crisp regression tasks.
