# TODO — Lab 1 task breakdown

Single source of truth for who implements what — update the checkbox when
a branch is merged into `main`, don't track progress anywhere else.

Each algorithm has its own branch (created from `main`, already pushed).
Workflow: checkout your branch → implement `fit`/`predict` (see the
`/* Implement the following */` comments already in each `.cpp`) → test via
the app (F5, load the matching dataset, run the algorithm, check the
metrics panel) → commit → push → open a PR into `main` → the other person
reviews before merging.

## Now — KNN + Decision Trees

| # | Algorithm | Branch | Owner | Dataset to test with | Done |
|---|---|---|---|---|---|
| 1 | KNN Classifier | `feat/knn-classifier` | Pierre | `Iris.csv` | [ ] |
| 2 | KNN Regression | `feat/knn-regression` | Maceo | `BostonHousing.csv` | [ ] |
| 3 | Decision Tree Classification | `feat/decision-tree-classification` | Maceo | `Iris.csv` | [ ] |
| 4 | Decision Tree Regression | `feat/decision-tree-regression` | Pierre | `BostonHousing.csv` | [ ] |

Each pair (1+2, 3+4) shares most of its logic (KNN classifier → KNN
regression only changes the last step; decision tree classification →
regression only changes the split criterion and leaf value) — since the
two halves of each pair now belong to different people, compare notes
with each other once you've both got your half working, it'll make
finishing the other half faster.

## Later — deprioritized for now

Not started yet, revisit once the four above are done.

| # | Algorithm | Branch | Owner (tentative) | Dataset to test with | Done |
|---|---|---|---|---|---|
| 5 | Linear Regression | `feat/linear-regression` | Maceo | `BostonHousing.csv` | [ ] |
| 6 | Logistic Regression | `feat/logistic-regression` | Maceo | `Iris.csv` | [ ] |
| 7 | KMeans | `feat/kmeans` | Maceo | `Iris.csv` | [ ] |
| 8 | Fuzzy C-Means | `feat/fuzzy-cmeans` | Maceo | `Iris.csv` | [ ] |

## Reference

- Algorithm-by-algorithm guidance (what each `fit`/`predict` needs to do, without the implementation itself) — ask Claude Code, it has the full lab context in `CLAUDE.md`.
- `ML_Lab_Project.pdf` — the actual assignment.
- Utilities already implemented, reuse them rather than reimplementing: `SimilarityFunctions` (distances, for KNN), `EntropyFunctions` (entropy, for Decision Tree Classification's split), `DataPreprocessor::normalizeDataset`/`scaleDataset` (call in `KMeans::fit` before clustering), `Node` (tree node), `Metrics`, `PCADimensionalityReduction`.
