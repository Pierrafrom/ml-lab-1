# TODO — Lab 1 task breakdown

Single source of truth for who implements what — update the checkbox when
a branch is merged into `main`, don't track progress anywhere else.

Each algorithm has its own branch (created from `main`, already pushed).
Workflow: checkout your branch → implement `fit`/`predict` (see the
`/* Implement the following */` comments already in each `.cpp`) → test via
the app (F5, load the matching dataset, run the algorithm, check the
metrics panel) → commit → push → open a PR into `main` → the other person
reviews before merging.

## Pierre — KNN + Decision Trees (same logic, applied twice each)

| # | Algorithm | Branch | Dataset to test with | Done |
|---|---|---|---|---|
| 1 | KNN Classifier | `feat/knn-classifier` | `Iris.csv` | [ ] |
| 2 | KNN Regression | `feat/knn-regression` | `BostonHousing.csv` | [ ] |
| 3 | Decision Tree Classification | `feat/decision-tree-classification` | `Iris.csv` | [ ] |
| 4 | Decision Tree Regression | `feat/decision-tree-regression` | `BostonHousing.csv` | [ ] |

## Maceo — Linear/Logistic Regression + Clustering

| # | Algorithm | Branch | Dataset to test with | Done |
|---|---|---|---|---|
| 5 | Linear Regression | `feat/linear-regression` | `BostonHousing.csv` | [ ] |
| 6 | Logistic Regression | `feat/logistic-regression` | `Iris.csv` | [ ] |
| 7 | KMeans | `feat/kmeans` | `Iris.csv` | [ ] |
| 8 | Fuzzy C-Means | `feat/fuzzy-cmeans` | `Iris.csv` | [ ] |

## Suggested order within each person's list

Top to bottom in the tables above — each pair shares most of its logic
(KNN classifier → KNN regression only changes the last step; decision
tree classification → regression only changes the split criterion and
leaf value). Do the first of a pair, then adapt for the second.

## Reference

- Algorithm-by-algorithm guidance (what each `fit`/`predict` needs to do, without the implementation itself) — ask Claude Code, it has the full lab context in `CLAUDE.md`.
- `ML_Lab_Project.pdf` — the actual assignment.
- Utilities already implemented, reuse them rather than reimplementing: `SimilarityFunctions` (distances, for KNN), `EntropyFunctions` (entropy, for Decision Tree Classification's split), `DataPreprocessor::normalizeDataset`/`scaleDataset` (call in `KMeans::fit` before clustering), `Node` (tree node), `Metrics`, `PCADimensionalityReduction`.
