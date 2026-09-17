#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	predictions.reserve(X.size());

	// One traversal per test sample, always starting back at the tree's root.
	for (std::vector<double>& x : X) {
		predictions.push_back(traverseTree(x, root));
	}

	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	/* Implement the following:
	--- define stopping criteria
	--- Loop through candidate features and potential split thresholds.
	--- greedily select the best split according to information gain
	---grow the children that result from the split
	*/

	int n_samples = static_cast<int>(X.size());
	std::unordered_set<double> unique_labels(y.begin(), y.end());

	// Stopping criteria checked before searching for a split: a pure node (one
	// label left) has nothing to gain from splitting, and depth/sample-count
	// limits must be enforced unconditionally or the recursion never ends.
	if (depth >= max_depth || unique_labels.size() == 1 || n_samples < min_samples_split) {
		return new Node(0, 0.0, nullptr, nullptr, mostCommonlLabel(y));
	}

	double best_gain = -1.0; // set the best gain to -1
	int split_idx = -1; // split index
	double split_thresh = 0.0; // split threshold

	// Loop through candidate features and potential split thresholds: only
	// values actually present in the column can change which samples land on
	// each side, so those are the only thresholds worth testing.
	for (int feat_idx = 0; feat_idx < n_feats; ++feat_idx) {
		std::vector<double> X_column;
		X_column.reserve(n_samples);
		for (int i = 0; i < n_samples; ++i) {
			X_column.push_back(X[i][feat_idx]);
		}

		std::set<double> thresholds(X_column.begin(), X_column.end());
		for (double threshold : thresholds) {
			double gain = informationGain(y, X_column, threshold);
			if (gain > best_gain) {
				best_gain = gain;
				split_idx = feat_idx;
				split_thresh = threshold;
			}
		}
	}

	// No candidate improved on the parent's entropy (e.g. every remaining
	// sample is identical across all features) -- nothing left to split on.
	if (split_idx == -1) {
		return new Node(0, 0.0, nullptr, nullptr, mostCommonlLabel(y));
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
	Node* right = growTree(X_right, y_right, depth + 1);  // grow the right tree
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/
	double ig = 0.0;

	// Generate split: collect indices (not copies of y) on each side of this
	// threshold, using the same "<= goes left" convention growTree partitions
	// with, so entropy(y, idxs) can read straight from the shared y vector.
	std::vector<int> left_idxs, right_idxs;
	for (int i = 0; i < static_cast<int>(X_column.size()); ++i) {
		if (X_column[i] <= split_thresh) {
			left_idxs.push_back(i);
		} else {
			right_idxs.push_back(i);
		}
	}

	// A degenerate split (every sample on one side, e.g. testing the column's
	// max value) separates nothing -- treat it as zero gain rather than
	// computing entropy on an empty set (division by zero in total_samples).
	if (left_idxs.empty() || right_idxs.empty()) {
		return 0.0;
	}

	// Weighted avg. of the loss for the children: a branch's entropy only
	// counts for as much as the fraction of the node's samples it holds.
	int n_samples = static_cast<int>(y.size());
	double child_entropy =
		(static_cast<double>(left_idxs.size()) / n_samples) * EntropyFunctions::entropy(y, left_idxs) +
		(static_cast<double>(right_idxs.size()) / n_samples) * EntropyFunctions::entropy(y, right_idxs);

	// Information gain is the difference in loss before vs. after split.
	ig = parent_entropy - child_entropy;

	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {
	double most_common = 0.0;

	// Count occurrences per label -- same counting pattern as
	// EntropyFunctions::entropy, just tracking the argmax afterward instead
	// of summing a probability-weighted formula.
	std::unordered_map<double, int> label_counts;
	for (double label : y) {
		label_counts[label]++;
	}

	int best_count = 0;
	for (const auto& entry : label_counts) {
		if (entry.second > best_count) {
			best_count = entry.second;
			most_common = entry.first;
		}
	}

	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

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


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
	DataPreprocessor DataPreprocessor;
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

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}