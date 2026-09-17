# Lab 1 report — k-NN Classification & Regression

Date: 2026-09-10, DVA262. Findings from Task 4 and Task 7 (k value
experimentation) — the task code itself lives in the actual source files
and `TODO.md`, not duplicated here.

## Task 4 — k-NN Classification (Iris)

Confusion matrices for k = 1, 5, 11 (≈ number of features × 2-3), 50:

| k=1 | k=5 |
|---|---|
| ![classification k=1](assets/lab1/lab1_KNN_k1.jpeg) | ![classification k=5](assets/lab1/lab1_KNN_k5.jpeg) |

| k=11 | k=50 |
|---|---|
| ![classification k=11](assets/lab1/lab1_KNN_k11.jpeg) | ![classification k=50](assets/lab1/lab1_KNN_k50.jpeg) |

**Conclusion: k=5 gave the best results** on this run.

## Task 7 — k-NN Regression (Boston Housing)

| k=1 (train) | k=3 (train) |
|---|---|
| ![regression k=1 train](assets/lab1/k1_training.jpeg) | ![regression k=3 train](assets/lab1/k3_trainingSet.jpeg) |

| k=5 | k=3 (test) |
|---|---|
| ![regression k=5](assets/lab1/k5.jpeg) | ![regression k=3 test](assets/lab1/k3_test.jpeg) |

R² recap across k values:

![R² vs k recap](assets/lab1/recap_k_rSquare.jpeg)
