#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	if (data.empty() || numClusters_ <= 0 || numClusters_ > static_cast<int>(data.size())) {
		return;
	}

	// centroides de depart : k points tires au hasard
	std::vector<int> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::random_device rd;
	std::shuffle(indices.begin(), indices.end(), std::mt19937(rd()));

	centroids_.clear();
	for (int i = 0; i < numClusters_; ++i) {
		centroids_.push_back(data[indices[i]]);
	}

	for (int iteration = 0; iteration < maxIterations_; ++iteration) {
		// newCentroids accumule les sommes des points de chaque cluster
		std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0.0));
		std::vector<int> clusterCounts(numClusters_, 0);

		for (const auto& point : data) {
			double minDistance = (std::numeric_limits<double>::max)();
			int closestCentroid = 0;

			for (size_t i = 0; i < centroids_.size(); ++i) {
				double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);
				if (distance < minDistance) {
					minDistance = distance;
					closestCentroid = static_cast<int>(i);
				}
			}

			for (size_t j = 0; j < point.size(); ++j) {
				newCentroids[closestCentroid][j] += point[j];
			}
			clusterCounts[closestCentroid]++;
		}

		// somme / effectif = moyenne du cluster
		bool hasConverged = true;
		for (int i = 0; i < numClusters_; ++i) {
			if (clusterCounts[i] > 0) {
				for (size_t j = 0; j < newCentroids[i].size(); ++j) {
					newCentroids[i][j] /= clusterCounts[i];
				}

				double centroidChange = SimilarityFunctions::euclideanDistance(centroids_[i], newCentroids[i]);
				if (centroidChange > 1e-6) {
					hasConverged = false;
				}
			}
			else {
				// cluster vide : on garde l'ancien centroide (sinon il partirait a l'origine)
				newCentroids[i] = centroids_[i];
			}
		}

		centroids_ = newCentroids;

		if (hasConverged) {
			break;
		}
	}
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());

	for (const auto& point : data) {
		double minDistance = (std::numeric_limits<double>::max)();
		int closestCentroid = 0;

		for (size_t i = 0; i < centroids_.size(); ++i) {
			double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);
			if (distance < minDistance) {
				minDistance = distance;
				closestCentroid = static_cast<int>(i);
			}
		}

		labels.push_back(closestCentroid);
	}

	return labels;
}

/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
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

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);

		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}