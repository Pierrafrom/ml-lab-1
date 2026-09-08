# TODO — Lab 1: k-NN

Source: [`Lab1_kNN.pdf`](Lab1_kNN.pdf) (course: DVA262). This is the actual
Lab 1 assignment — **k-NN only**, split into Part 1 (Classification) and
Part 2 (Regression). Everything else in the skeleton (Decision Trees,
Linear/Logistic Regression, Clustering) belongs to **future labs with no
assignment sheet yet** — see the bottom of this file, don't work on those
now.

## ⚠️ Dependency — read before starting

`KNNRegression::predict()` (Maceo's Task 5) needs a working distance
function to run at all, and `SimilarityFunctions.cpp` is shared by both of
you. **Pierre should implement `euclideanDistance()` (Task 2) first and
push/merge it to `main` before Maceo starts Task 5** — otherwise Maceo is
blocked on an empty function. Everything else can happen in parallel.

## Part 1 — Classification (Pierre), branch `feat/knn-classifier`

Files: `Classification/KNNClassifier.cpp`, `Utils/SimilarityFunctions.cpp`.
`fit()` is already implemented (just stores the training data) — nothing
to do there.

| Task | What | Done |
|---|---|---|
| 2 (do this first — see dependency above) | Implement `SimilarityFunctions::euclideanDistance()` | [ ] |
| 1 | Implement `KNNClassifier::predict()` (loop test points → distance to every training point → majority vote among the k nearest → check `fit()` was called first) | [ ] |
| 3 | Implement **one more** distance function in `SimilarityFunctions.cpp` (Manhattan is the simplest second choice), use it in `predict()` instead of Euclidean, compare the results | [ ] |
| 4 | Try different values of `k` (set in `MainForm.cpp` where `KNNClassifier` is constructed) — note what breaks at the extremes (too small = noisy, too large = degenerates), be ready to discuss with the lab assistant | [ ] |

Test with `Iris.csv`, Classification tab.

## Part 2 — Regression (Maceo), branch `feat/knn-regression`

File: `Regression/KNNRegression.cpp`. `fit()` already implemented. No
changes needed in `SimilarityFunctions.cpp` if Pierre's Task 2/3 are
already merged into `main` — reuse those, don't reimplement.

| Task | What | Done |
|---|---|---|
| 5 | Implement `KNNRegression::predict()` (same nearest-neighbor search as classification, but **average** the k neighbors' target values instead of voting) | [ ] |
| 6 | Re-run using the Task 3 distance function (not just Euclidean) and show the result | [ ] |
| 7 | Try different values of `k` (set in `MainForm.cpp` where `KNNRegression` is constructed), same discussion as Task 4 but for regression | [ ] |

Test with `BostonHousing.csv`, Regression tab.

## Optional — extra credit ("if you want to learn some more")

Implement every remaining function in `SimilarityFunctions.cpp` (Hamming,
Jaccard, Cosine, Minkowski), test all of them for both classification and
regression, compare which performs best. Not required to pass the lab —
only if there's time left after Tasks 1–7.

## Demonstration

Both parts must be demonstrated to a lab assistant together — plan to show
both `feat/knn-classifier` and `feat/knn-regression` merged into `main`
before booking a demo slot, not one half only.

---

## Future labs (same skeleton, no assignment sheet yet — do not start)

Branches already exist for when the next lab statements arrive:
`feat/decision-tree-classification`, `feat/decision-tree-regression`,
`feat/linear-regression`, `feat/logistic-regression`, `feat/kmeans`,
`feat/fuzzy-cmeans`. Nothing to implement on these until we get the actual
lab PDF for them, the same way `Lab1_kNN.pdf` clarified this one.
