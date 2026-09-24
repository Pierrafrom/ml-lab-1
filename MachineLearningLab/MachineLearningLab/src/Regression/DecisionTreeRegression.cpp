#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	predictions.reserve(X.size());

	// One traversal per test sample, always starting back at the tree's root.
	for (std::vector<double>& x : X) {
		predictions.push_back(traverseTree(x, root));
	}

	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {


	int split_idx = -1;
	double split_thresh = 0.0;

	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- Find the best split threshold for the current feature.
		--- grow the children that result from the split
	*/

	int n_samples = static_cast<int>(X.size());

	// Stopping criteria checked before searching for a split: depth/sample-count
	// limits must be enforced unconditionally or the recursion never ends.
	if (depth >= max_depth || n_samples < min_samples_split) {
		return new Node(0, 0.0, nullptr, nullptr, mean(y));
	}

	double best_gain = -1.0;

	// Parent MSE, computed once per node
	double y_mean = mean(y);
	double parent_mse = 0.0;
	for (double value : y) {
		parent_mse += (value - y_mean) * (value - y_mean);
	}
	parent_mse /= n_samples;

	// Loop through candidate features and potential split thresholds: same
	// exhaustive search as DecisionTreeClassification
	for (int feat_idx = 0; feat_idx < n_feats; ++feat_idx) {
		std::vector<double> X_column;
		X_column.reserve(n_samples);
		for (int i = 0; i < n_samples; ++i) {
			X_column.push_back(X[i][feat_idx]);
		}

		std::set<double> thresholds(X_column.begin(), X_column.end());
		for (auto it = thresholds.begin(); it != std::prev(thresholds.end()); ++it) {
			double threshold = *it;
			// meanSquaredError returns the split's own (weighted, child) MSE --
			// the gain is computed here.
			double split_mse = meanSquaredError(y, X_column, threshold);
			double gain = parent_mse - split_mse;
			if (gain > best_gain) {
				best_gain = gain;
				split_idx = feat_idx;
				split_thresh = threshold;
			}
		}
	}

	// No candidate reduced the parent's MSE (e.g. every remaining sample is
	// identical across all features) -- nothing left to split on.
	if (split_idx == -1) {
		return new Node(0, 0.0, nullptr, nullptr, mean(y));
	}

	// Partition the samples using the winning (feature, threshold) pair --
	// same "<= goes left" convention traverseTree must use at prediction time.
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;
	for (int i = 0; i < n_samples; ++i) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		} else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	Node* left = growTree(X_left, y_left, depth + 1); // grow the left tree
	Node* right = growTree(X_right, y_right, depth + 1); // grow the right tree
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double mse = 0.0;

	// Calculate the mse: generate the split first, same "<= goes left"
	// convention growTree partitions with.
	std::vector<double> y_left, y_right;
	for (int i = 0; i < static_cast<int>(X_column.size()); ++i) {
		if (X_column[i] <= split_thresh) {
			y_left.push_back(y[i]);
		} else {
			y_right.push_back(y[i]);
		}
	}

	// Weighted average of each side's own MSE -- MSE of a set is its own
	// variance: the average squared distance to its own mean.
	int n_samples = static_cast<int>(y.size());

	double left_mean = mean(y_left);
	double left_mse = 0.0;
	for (double value : y_left) {
		left_mse += (value - left_mean) * (value - left_mean);
	}
	left_mse /= y_left.size();

	double right_mean = mean(y_right);
	double right_mse = 0.0;
	for (double value : y_right) {
		right_mse += (value - right_mean) * (value - right_mean);
	}
	right_mse /= y_right.size();

	mse = (static_cast<double>(y_left.size()) / n_samples) * left_mse +
		(static_cast<double>(y_right.size()) / n_samples) * right_mse;

	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;

	// calculate the mean
	double sum = 0.0;
	for (double value : values) {
		sum += value;
	}
	meanValue = sum / static_cast<double>(values.size());

	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/

	// Leaf node: no decision left to make, its value IS the prediction.
	if (node->isLeafNode()) {
		return node->value;
	}

	// Same "<= goes left" convention growTree used to build the split --
	// must match exactly, or predictions silently diverge from training.
	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	return traverseTree(x, node->right);
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

