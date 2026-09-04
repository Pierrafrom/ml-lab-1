# ML Lab 1 — Machine Learning Algorithms from Scratch

Mälardalen University (dva262) ML lab: implementing classification,
regression, and clustering algorithms from scratch in C++, on top of a
provided WinForms lab skeleton (see `ML_Lab_Project.pdf` for the full
assignment).

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

## What's already done vs. what we implement

Full breakdown in `CLAUDE.md`. Short version: the UI, data loading, and
evaluation metrics are already provided. We implement `fit`/`predict` for:

- **Classification**: `DecisionTreeClassification`, `KNNClassifier`, `LogisticRegression`
- **Regression**: `DecisionTreeRegression`, `KNNRegression`, `LinearRegression`
- **Clustering**: `KMeans`, `FuzzyCMeans`

Every algorithm class follows the same shape — `fit(X_train, y_train)`,
`predict(X_test)`, and a `runXxx(filePath, trainingRatio)` orchestration
method already wired to the UI. Look at any already-implemented method in
`DataUtils/` or `Evaluation/` for the coding pattern this codebase uses.

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

- [`CLAUDE.md`](CLAUDE.md) — architecture notes for Claude Code.
- [`.github/copilot-instructions.md`](.github/copilot-instructions.md) — same for GitHub Copilot.
- [`ML_Lab_Project.pdf`](ML_Lab_Project.pdf) — the actual lab assignment.
