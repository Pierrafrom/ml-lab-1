# ML Lab1 - k-NN classification & regression

Date: 10/09/2026 13:15 (UTC+2) → 17:00
Matiere: DVA262
Status: Acquis
Type: CS

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

# k-NN Classification

### TASK1:

**Consigne:** Implement the following:
	--- Loop through each test data point
	--- Calculate Euclidean distance between test data point and each training data point
	--- Loop through the labels and their counts to find the most frequent label
	--- Check if predicted label is valid

```cpp
for (const auto& test_point : X_test) {
	std::vector<std::pair<double, double>> distances; // Pair of distance and corresponding label
	// Calculate distances to all training points
	for (size_t i = 0; i < X_train_.size(); ++i) {
		double dist = SimilarityFunctions::euclideanDistance(test_point, X_train_[i]);
		// double dist = SimilarityFunctions::minkowskiDistance(test_point, X_train_[i], 2); // Using Minkowski distance with p=2 (Euclidean)
		distances.emplace_back(dist, y_train_[i]);
	}
	// Sort distances to find the k nearest neighbors
	std::sort(distances.begin(), distances.end(),
		[](const std::pair<double, double>& a, const std::pair<double, double>& b) {
			return a.first < b.first; // Sort by distance
		});
	// Count the frequency of labels among the k nearest neighbors
	std::unordered_map<double, int> label_count;
	for (int j = 0; j < k_ && j < distances.size(); ++j) {
		label_count[distances[j].second]++;
	}
	// Find the label with the maximum count
	double predicted_label = -1;
	int max_count = 0;
	for (const auto& pair : label_count) {
		if (pair.second > max_count) {
			max_count = pair.second;
			predicted_label = pair.first;
		}
	}
	// Check if predicted label is valid
	if (predicted_label == -1) {
		throw std::runtime_error("Error: Unable to determine predicted label.");
	}
	y_pred.push_back(predicted_label); // Store the predicted label
}

return y_pred; // Return vector of predicted labels for all test data points
```

### TASK2:

**Consignes:** 

You must complete `eulideanDistance()` function to compute the similarity
between instances.

```cpp
/// euclideanDistance function: Calculates the Euclidean distance between two vectors.///
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Euclidean Distance
	for (size_t i = 0; i < a.size(); ++i) {
		dist += std::pow(a[i] - b[i], 2);
	}
	// we avoid using sqrt for performance reasons
	// dist = std::sqrt(dist); 

	return dist;
}
```

### TASK3

> 
> 
> 
> **Consignes:** 
> 
> You should at least implement one other distance function
> apart from the eulideanDistance() in the SimilarityFunctions.cpp class
> 

```cpp
/// manhattanDistance function: Calculates the Manhattan distance between two vectors.///
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Manhattan Distance
	for (size_t i = 0; i < a.size(); ++i) {
		dist += std::abs(a[i] - b[i]);
	}

	return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.///
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Minkowski Distance
	for (size_t i = 0; i < a.size(); ++i) {
		dist += std::pow(std::abs(a[i] - b[i]), p);
	}
	dist = std::pow(dist, 1.0 / p);

	return dist;
}
```

### TASK4

> **Consigne:**
> 
> 
> What value of k have you used? Try different values for k. What is the
> largest and smallest value of k you can choose?
> 

![classification - k = 1](lab1_KNN_k1.jpeg)

classification - k = 1

![classification - k = 5](lab1_KNN_k5.jpeg)

classification - k = 5

![classification - k = 11 (nb of features)](lab1_KNN_k11.jpeg)

classification - k = 11 (nb of features)

![classification - k=50](lab1_KNN_k50.jpeg)

classification - k=50

⇒ k = 5 donne les meilleurs résultats

# k-NN Regression

### TASK5

> **Consignes**
> 
> 
> You must complete the predict() function.
> 

```cpp
/* Implement the following:
	--- Loop through each test data point
	--- Calculate Euclidean distance between test data point and each training data point
	--- Loop through the labels and their counts to find the most frequent label
	--- Store sum of y_train values for k-nearest neighbors
	--- Calculate average of y_train values for k-nearest neighbors
*/

for (const auto& test_point : X_test) {
	std::vector<std::pair<double, double>> distances; // Pair of distance and corresponding label
	// Calculate distances to all training points
	for (size_t i = 0; i < X_train_.size(); ++i) {
		// double dist = SimilarityFunctions::euclideanDistance(test_point, X_train_[i]);
		double dist = SimilarityFunctions::manhattanDistance(test_point, X_train_[i]);
		distances.emplace_back(dist, y_train_[i]);
	}
	// Sort distances to find the k nearest neighbors
	std::sort(distances.begin(), distances.end(),
		[](const std::pair<double, double>& a, const std::pair<double, double>& b) {
			return a.first < b.first; // Sort by distance
		});
	// Calculate the average of the labels among the k nearest neighbors
	double sum = 0.0;
	for (int j = 0; j < k_; ++j) {
		sum += distances[j].second; // Sum the labels of the k nearest neighbors
	}
	double average = sum / k_; // Calculate average
	y_pred.push_back(average); // Store the predicted value
}
```

### TASK6

> 
> 
> 
> If you have already completed Task 2 and Task 3, you are good to go with
> the SimilarityFunctions.cpp class. Use the similarity function that you implemented
> in Task 3 and show the result.
> 

### TASK7

> 
> 
> 
> What value of k have you used? Try different values for k. What is the
> largest and smallest value of k you can choose?
> 

![k1_training.jpeg](k1_training.jpeg)

![k3_trainingSet.jpeg](k3_trainingSet.jpeg)

![k5.jpeg](k5.jpeg)

![k3_test.jpeg](k3_test.jpeg)

![recap_k_rSquare.jpeg](recap_k_rSquare.jpeg)