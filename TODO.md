# TODO — Lab 4: Linear Regression

Source: [`Lab4_Linear_Regression.pdf`](Lab4_Linear_Regression.pdf) (course:
DVA262). **Lab 4 is the current assignment** — Linear Regression only, one
part (regression, `BostonHousing.csv`). Labs 1-3 are done (see below).
Clustering (`KMeans`, `FuzzyCMeans`) still belongs to a future lab with no
assignment sheet yet.

## Files

`Regression/LinearRegression.cpp`/`.h`. Currently only one `fit()`/
`predict()` pair exists (Matrix Form, both still `//TODO`) — Task 2 requires
**adding a second, overloaded `fit()`/`predict()` pair** for Gradient
Descent, not just filling in a stub. `m_coefficients` (`Eigen::VectorXd`)
is the only member declared so far; Task 2 will need `learning_rate`/
`num_epochs` added too (same pattern as `LogisticRegression`).

## Tasks

| Task | What | Done |
|---|---|---|
| 1 | Implement `fit()`/`predict()` using the **Matrix Form** (normal equation `θ̂ = (XᵀX)⁻¹Xᵀy`, via Eigen) | [ ] |
| 2 | Add an **overloaded** `fit()`/`predict()` pair using **Gradient Descent** — same structure as `LogisticRegression::fit()` (bias trick, weighted sum, gradient, update loop), but no sigmoid: `h(x) = θᵀx` directly, not `σ(θᵀx)` | [ ] |
| 3 | Discussion (no code): differences between Matrix Form and Gradient Descent — which is faster, which predicts better. Be ready to discuss with the lab assistant | [ ] |
| 4 | Same hyperparameter sweep pattern as Lab 2/3: try different `learning_rate`/`num_epochs` for the Gradient Descent version, find the best combo (see `notebooks/` for how we did this for Lab 2/3) | [ ] |

**Open question, not yet decided**: unlike Lab 1 (naturally split into
Classification/Regression), Lab 4 has a single part — no obvious way to
split Task 1 (Matrix Form) and Task 2 (Gradient Descent) between two
people without one blocking the other on the same file. Decide together
before starting (e.g. one person does Task 1 while the other reads up on
Task 2, or split by branch and merge).

**Also open**: `runLinearRegression()` currently calls a single `fit()`/
`predict()`. Once two overloaded pairs exist, `MainForm.h` will need some
way to let the user pick Matrix Form vs Gradient Descent (the PDF says
"you may need to edit some parts of the code... to display the results
correctly") — not designed yet.

## Demonstration

Both Task 1 (Matrix Form) and Task 2 (Gradient Descent) must be
demonstrated, plus the Task 3/4 discussion.

---

## Done

- **Lab 1 — k-NN** (`Lab1_kNN.pdf`): `KNNClassifier`, `KNNRegression`,
  `SimilarityFunctions` (Euclidean + Manhattan). Merged into `main`.
- **Lab 2 — Decision Trees** (`Lab2_Decision_Tree.pdf`):
  `DecisionTreeClassification`, `DecisionTreeRegression`,
  `EntropyFunctions`. Defaults tuned to their empirically best
  hyperparameters (see `notebooks/iris_exploration.ipynb` and
  `notebooks/boston_housing_exploration.ipynb`). Merged into `main`.
- **Lab 3 — Logistic Regression** (`Lab3_Logistics Regression.pdf`):
  `LogisticRegression` (One-vs-Rest, gradient descent per class).
  Defaults tuned (`learning_rate=0.001`, `nb_rounds=1000`). Merged into
  `main`.

## Future labs (no assignment sheet yet — do not start)

`KMeans`, `FuzzyCMeans` (Clustering). Branches `feat/kmeans`,
`feat/fuzzy-cmeans` already exist, kept up to date with `main`. Nothing to
implement until the actual lab PDF arrives.
