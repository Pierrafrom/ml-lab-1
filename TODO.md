# TODO

Labs 1 to 4 are done and merged into `main`. The next assignment is the
Clustering lab (`KMeans`, `FuzzyCMeans`), which has no assignment sheet yet.

## Done

- **Lab 1 — k-NN** (`Lab1_kNN.pdf`): `KNNClassifier`, `KNNRegression`,
  `SimilarityFunctions` (Euclidean + Manhattan).
- **Lab 2 — Decision Trees** (`Lab2_Decision_Tree.pdf`):
  `DecisionTreeClassification`, `DecisionTreeRegression`,
  `EntropyFunctions`. Defaults tuned to their empirically best
  hyperparameters (see `notebooks/iris_exploration.ipynb` and
  `notebooks/boston_housing_exploration.ipynb`).
- **Lab 3 — Logistic Regression** (`Lab3_Logistics Regression.pdf`):
  `LogisticRegression` (One-vs-Rest, gradient descent per class). Defaults
  tuned (`learning_rate=0.001`, `nb_rounds=1000`).
- **Lab 4 — Linear Regression** (`Lab4_Linear_Regression.pdf`):
  - Task 1: `LinearRegression::fit(trainData, trainLabels)` — Matrix Form
    (normal equation `θ̂ = (XᵀX)⁻¹Xᵀy`, Eigen).
  - Task 2: overloaded `fit(trainData, trainLabels, learning_rate, nb_rounds)`
    — per-example gradient descent on standardized features. One shared
    `predict()` serves both (both store their result in `m_coefficients`;
    it standardizes the test data only after a gradient descent fit).
  - Task 3: discussion (no code) — Matrix Form is exact, needs no tuning or
    standardization and is cheaper at this size (d=13); gradient descent
    needs standardization plus tuning, reaches the same optimum, and wins
    when d is large or no closed form exists.
  - Task 4: sweep in section 7 of `notebooks/boston_housing_exploration.ipynb`.
    Best: `learning_rate=0.0005`, `nb_rounds=100` (test R² ≈ 0.66, Matrix
    Form ≈ 0.67). Without standardization nothing converges.
  - The Regression tab lists both variants ("Linear Regression (Matrix
    Form)" / "(Gradient Descent)") so each can be demonstrated.

## Next (no assignment sheet yet — do not start)

`KMeans`, `FuzzyCMeans` (Clustering). Branches `feat/kmeans` (in progress:
`KMeans::predict` done) and `feat/fuzzy-cmeans` (empty placeholder) exist.
