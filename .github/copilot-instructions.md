# Copilot Instructions — ML Lab 1

## Project summary

Machine Learning lab (Mälardalen University, `dva262`) — implementing ML
algorithms from scratch in C++, inside a shared WinForms skeleton used
across every lab of the course. The UI, data loading, and evaluation
metrics are already implemented.

**Current assignment (see `TODO.md` and `Lab1_kNN.pdf`): Lab 1 is k-NN
only** — `KNNClassifier::predict()`, `KNNRegression::predict()`, and two
distance functions in `SimilarityFunctions.cpp`. Every other algorithm
class described below (Decision Trees, Linear/Logistic Regression,
Clustering) is scaffolding for a **future lab with no assignment sheet
yet** — don't suggest implementing those until told otherwise.

## Tech stack

- Language: C++/CLI (`<CLRSupport>true</CLRSupport>`), MSVC platform toolset v145, targeting .NET WinForms for the UI.
- Key dependencies: **Eigen 3.4.0** (vendored, `AdditionalLibraries/eigen-3.4.0/`) for linear algebra — used internally by algorithm implementations, never exposed in public method signatures.
- Tools: no CMake/vcpkg here — this is a plain Visual Studio `.sln`/`.vcxproj`, built via MSBuild or the VS IDE. No automated test framework; validation is running the app and checking results against the metrics panel.

## Code conventions

- **Public algorithm interfaces stay in plain STL types** (`std::vector<std::vector<double>>` for feature matrices, `std::vector<double>` for labels/targets) even when the implementation uses Eigen internally — this matches every existing algorithm class (`LinearRegression`, `LogisticRegression`, etc.) and keeps the UI layer decoupled from the linear-algebra library.
  - Preferred:
    ```cpp
    void fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train);
    ```
  - Avoid:
    ```cpp
    void fit(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train); // leaks Eigen into the public API
    ```
- **Every algorithm class follows the same three-method shape**: `fit(X_train, y_train)`, `predict(X_test)` (both to implement), and `runXxx(filePath, trainingRatio)` — the orchestration method `MainForm.cpp` calls, which loads the CSV via `DataLoader`, splits by `trainingRatio`, calls `fit`+`predict`, and returns a tuple of everything the UI renders (accuracy, confusion matrix, recall/precision/F1, etc. — shape varies per algorithm, match the existing tuple signature in that class's header, don't redesign it).
- **Reuse `DataUtils::DataPreprocessor`, `Evaluation::Metrics`, and `Evaluation::KFoldCrossValidation`** rather than reimplementing preprocessing or scoring — they're already provided and used by `MainForm`.
- **`Utils/PCADimensionalityReduction` is a required preprocessing step for the Clustering lab specifically** (per the lab statement), not optional.
- **`MainForm.cpp/.h/.resx` is C++/CLI UI glue** — edit it through the Visual Studio WinForms designer, not by hand; standard C++ tooling (clangd, clang-tidy) cannot parse `/clr` syntax (`ref class`, `gcnew`, `^`), so don't expect useful completions/diagnostics there. The algorithm classes under `Classification/`, `Regression/`, `Clustering/` are plain standard C++ — normal tooling works fine on those.

## Project structure

```
MachineLearningLab/MachineLearningLab/src/
├── Classification/   # KNNClassifier — Lab 1, to implement. DecisionTreeClassification, LogisticRegression — future lab, don't touch yet
├── Regression/        # KNNRegression — Lab 1, to implement. DecisionTreeRegression, LinearRegression — future lab, don't touch yet
├── Clustering/         # KMeans, FuzzyCMeans — future lab, don't touch yet
├── DataUtils/           # DataLoader, DataPreprocessor — already implemented
├── Evaluation/          # Metrics, KFoldCrossValidation — already implemented
├── Utils/                # PCADimensionalityReduction (implemented), EntropyFunctions, SimilarityFunctions, Node
└── MainForm.cpp/.h/.resx # WinForms UI (C++/CLI) — already implemented
```

## Tests

No automated test suite. Validation is manual: run the app (`F5` in Visual
Studio), load `Iris.csv` or `BostonHousing.csv`, pick an algorithm, run it,
and check the confusion matrix / accuracy / F1 panel — this is also how
results get demonstrated to the lab assistant.
