# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine Learning lab project for Mälardalen University's ML course
(DVA262). A C++/CLI WinForms application (`MachineLearningLab`) provides
the UI (data loading, train/test split, algorithm selection, result
visualization) shared across every lab in the course; each lab's task is
to implement the **training and prediction logic** of specific ML
algorithms from scratch — the UI glue, data loading, and evaluation
metrics are already provided.

`ML_Lab_Project.pdf` describes the whole skeleton (all algorithm classes
it contains, across every lab). **Lab 1** (`Lab1_kNN.pdf`) covered
**k-NN only** (`KNNClassifier` + `KNNRegression` + `SimilarityFunctions`)
and is done. **Lab 2** (`Lab2_Decision_Tree.pdf`) is the current
assignment — **Decision Trees only**
(`DecisionTreeClassification` + `DecisionTreeRegression`). Linear/Logistic
Regression and Clustering still belong to a future lab with no assignment
sheet yet — don't implement those until a new lab PDF shows up, even
though the classes and branches already exist.

## Build

This is a Visual Studio solution (`MachineLearningLab/MachineLearningLab.sln`), not a CMake project — no `vcpkg`/`CMakePresets` involved.

- **Primary**: open `MachineLearningLab.sln` in Visual Studio 2026, `Ctrl+Shift+B` to build, `F5` to run.
- **Command line**: from a "Developer PowerShell for VS 2026" (has MSBuild on PATH):
  ```powershell
  MSBuild.exe MachineLearningLab\MachineLearningLab.sln /p:Configuration=Debug /p:Platform=x64
  ```
- Platform toolset is `v145` (VS 2026's native toolset — retargeted from the
  `v143` the skeleton originally shipped with; VS 2026 does not ship v143,
  building an untouched copy fails with MSB8020).
- Requires the **"Desktop development with C++"** workload with the
  **"C++/CLI support"** individual component — plain "Desktop development
  with C++" alone is not enough, `<CLRSupport>true</CLRSupport>` needs the
  CLI component specifically.
- No automated test suite — validation is manual: run the app, load a
  dataset, run an algorithm, check the confusion matrix / metrics panel
  against expected results, demonstrate to the lab assistant.

## Architecture

Every ML algorithm class follows the same three-method shape:

```cpp
class SomeAlgorithm {
public:
    void fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train);
    std::vector<double> predict(const std::vector<std::vector<double>>& X_test);
    std::tuple<...> runSomeAlgorithm(const std::string& filePath, int trainingRatio);
};
```

- `fit`/`predict` are the actual algorithm — **this is what each lab implements**, currently stubbed/incomplete (see the `warning C4101: variable locale non référencée` on unused `DataPreprocessor` locals across most algorithm files — a straightforward signal of what's not wired up yet).
- `runXxx(filePath, trainingRatio)` is the orchestration entry point `MainForm.cpp` calls: load the CSV via `DataLoader`, split train/test by `trainingRatio`, call `fit`+`predict`, compute metrics via `Metrics`, return everything the UI needs to render (accuracy, confusion matrix, recall/precision/F1, etc. — the exact tuple shape varies per algorithm, see each header).
- Public interfaces use plain `std::vector<std::vector<double>>`/`std::vector<double>` even though implementations may use Eigen internally (e.g. `LinearRegression` stores `Eigen::VectorXd m_coefficients`) — keep that boundary when implementing a new algorithm rather than leaking Eigen types into the public signature.

**Already implemented — read for the pattern, don't reimplement**:
- `DataUtils/DataLoader`, `DataUtils/DataPreprocessor` — CSV loading and preprocessing.
- `Evaluation/Metrics`, `Evaluation/KFoldCrossValidation` — all accuracy/precision/recall/F1/confusion-matrix/regression-error/clustering-index functions.
- `Utils/PCADimensionalityReduction` — required as a preprocessing step in the Clustering lab specifically (per the lab PDF).
- `MainForm.cpp/.h/.resx` — the WinForms UI; C++/CLI (`/clr`), not parseable by clangd/clang-tidy — edit via Visual Studio's designer, not by hand unless necessary.

**Done (Lab 1)**: `KNNClassifier::predict()`, `KNNRegression::predict()`,
`SimilarityFunctions::euclideanDistance()`/`manhattanDistance()`.

**To implement now (Lab 2 — see `TODO.md` for the exact task numbers)**:
- `DecisionTreeClassification`: `EntropyFunctions::entropy()` (both
  overloads), `growTree()` (done — see `feat/decision-tree-classification`),
  `informationGain()`, `mostCommonlLabel()`, `predict()`, `traverseTree()`.
- `DecisionTreeRegression`: the analogous set, with a variance/MSE split
  criterion instead of entropy and a mean instead of a majority vote.

**Not part of the current lab — skeletons only, don't touch until assigned**:
`LogisticRegression`, `LinearRegression`, `KMeans`, `FuzzyCMeans`.

## Datasets

- `Iris.csv` — 150 rows, 4 numeric features, 3 classes. Used for Classification and Clustering.
- `BostonHousing.csv` — 506 rows, 13 features, continuous target (`MEDV`). Used for Regression.

## Dependencies

- **Eigen 3.4.0**, vendored at `AdditionalLibraries/eigen-3.4.0/` (trimmed to `Eigen/` only — `unsupported/`, bench/test/doc/ci/demos/lapack/blas were all dropped, none of it is `#include`d by this project). Referenced via `$(ProjectDir)..\..\AdditionalLibraries\eigen-3.4.0` in the `.vcxproj` (relative, so it works regardless of where each teammate clones the repo — the skeleton originally hardcoded an absolute path that only existed on the original author's machine).
