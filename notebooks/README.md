# Exploration notebooks — Lab 1 (k-NN)

Purely exploratory helpers for reasoning about/justifying the results
from the C++ `KNNClassifier`/`KNNRegression` (the graded implementation,
untouched by anything here). `scikit-learn` is used only as a fast
reference for sweeps we can't easily do by hand in the app UI — nothing
here replaces or grades as the lab's actual algorithm implementation.

## Setup

```bash
cd notebooks
uv sync
uv run jupyter lab
```

Then open either notebook and select the `lab1-exploration` kernel.

## What's here

- **`iris_exploration.ipynb`** — why Class 1 (Setosa) is always perfectly
  classified regardless of `k` (pairplot, correlation), and a `k` sweep
  (accuracy vs `k`, log scale) for the Task 4 discussion.
- **`boston_housing_exploration.ipynb`** — feature-scale comparison
  backing the Euclidean-vs-Manhattan discussion (Task 3/6), and a
  quantified comparison of raw vs. `StandardScaler`-scaled features
  across both metrics and several `k` values, to check whether scaling
  or metric choice matters more.

## A caveat on comparing numbers to the C++ app

`DataPreprocessor::splitDataset` shuffles with `std::random_shuffle` and
no fixed seed, so every C++ run gets a different train/test split — the
exact R²/accuracy won't match these notebooks (which use a fixed
`random_state=42`) run to run. Expect the same *pattern*, not identical
numbers.
