# ML Lab — Machine Learning Algorithms from Scratch

[![C++](https://img.shields.io/badge/C%2B%2B-C%2B%2FCLI-00599C?logo=cplusplus)](MachineLearningLab/MachineLearningLab.sln)
[![Eigen](https://img.shields.io/badge/Eigen-3.4.0-informational)](AdditionalLibraries/eigen-3.4.0)
[![WinForms](https://img.shields.io/badge/UI-WinForms-5C2D91?logo=windows)](MachineLearningLab)

Mälardalen University (DVA262) ML course: implementing classification,
regression, and clustering algorithms **from scratch in C++** (no
scikit-learn) on top of a provided WinForms lab skeleton that handles data
loading, train/test splitting, and metrics — see `ML_Lab_Project.pdf` for
the full assignment. Pair project with Maceo.

## Quick start

1. **Clone** (don't download a zip — see "gotchas" below for why):
   ```powershell
   git clone <this-repo-url>
   ```
2. **Open** `MachineLearningLab/MachineLearningLab.sln` in Visual Studio 2026.
3. **First time only** — install the required component if the build fails
   with a toolset/CLI error:
   - Visual Studio Installer → *Modify* → check **"Desktop development with
     C++"** workload.
   - Under *Individual Components*, search **"CLI"** → check **"C++/CLI
     support (latest MSVC)"**.
4. **Build**: `Ctrl+Shift+B`. **Run**: `F5`.
5. In the app: *Browse* → pick `Iris.csv` (Classification/Clustering tab) or
   `BostonHousing.csv` (Regression tab) → set the train/test split → pick an
   algorithm → *Run*.

## What's implemented

The UI, data loading, and evaluation metrics are provided by the skeleton;
each lab implements the `fit`/`predict` logic of specific algorithms.

| Algorithm | Status |
|---|---|
| k-NN (classification + regression) | ✅ implemented, incl. Euclidean/Manhattan distance |
| Decision Trees (classification + regression) | ✅ implemented, hyperparameters tuned via notebook sweeps |
| Logistic Regression | ⬜ not yet assigned |
| Linear Regression | ⬜ not yet assigned |
| K-Means / Fuzzy C-Means | ⬜ not yet assigned |

Every algorithm class follows the same shape — `fit(X_train, y_train)`,
`predict(X_test)`, and a `runXxx(filePath, trainingRatio)` orchestration
method already wired to the UI. Full breakdown of what's done vs. pending
per file is in `CLAUDE.md`; hyperparameter exploration for the completed
algorithms lives in `notebooks/`.

## Known gotchas already fixed in this repo

- **Eigen path**: the skeleton originally hardcoded an absolute path
  (`C:\ML Lab\...`) that only existed on the original author's machine —
  now a relative path, works regardless of where you clone this.
- **Platform toolset**: the skeleton shipped targeting `v143` (VS2022).
  Visual Studio 2026 doesn't have that toolset — already retargeted to
  `v145` here. If you ever start from a *different* copy of the skeleton
  and hit `MSB8020`, that's why.
- **`MSB3821` on `MainForm.resx`**: happens if you get the code from a
  downloaded/extracted zip instead of `git clone` — Windows marks
  zip-extracted files as "from the internet" and MSBuild refuses to
  process them. Fix: select all files in the extracted folder →
  Properties → check "Unblock". Cloning with git never triggers this.

## Docs

- [`TODO.md`](TODO.md) — current task breakdown and ownership.
- [`CLAUDE.md`](CLAUDE.md) — architecture notes for Claude Code.
- [`.github/copilot-instructions.md`](.github/copilot-instructions.md) — same for GitHub Copilot.
- [`ML_Lab_Project.pdf`](ML_Lab_Project.pdf) — the full skeleton assignment (all labs).
- [`Lab1_kNN.pdf`](Lab1_kNN.pdf) — the Lab 1 statement.
- [`docs/lab1-report.md`](docs/lab1-report.md) — Lab 1 findings (k value sweeps, screenshots).
