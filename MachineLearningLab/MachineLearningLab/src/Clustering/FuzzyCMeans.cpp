#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	if (data.empty() || numClusters_ <= 0) {
		return;
	}
	// l'exposant 1 / (fuzziness - 1) n'existe pas si fuzziness <= 1
	if (fuzziness_ <= 1.0) {
		throw std::invalid_argument("Fuzziness must be strictly greater than 1.");
	}

	// appartenances aleatoires au depart, centroides deduits de celles-ci
	initializeMembershipMatrix(static_cast<int>(data.size()));
	centroids_.clear();
	centroids_ = updateCentroids(data);

	for (int iteration = 0; iteration < maxIterations_; ++iteration) {
		updateMembershipMatrix(data, centroids_);
		std::vector<std::vector<double>> newCentroids = updateCentroids(data);

		bool hasConverged = true;
		for (int i = 0; i < numClusters_; ++i) {
			double centroidChange = SimilarityFunctions::euclideanDistance(centroids_[i], newCentroids[i]);
			if (centroidChange > 1e-6) {
				hasConverged = false;
			}
		}

		centroids_ = newCentroids;

		if (hasConverged) {
			break;
		}
	}
}


// computeMemberships function: Computes the membership of one point to every cluster (the values sum up to 1).//
std::vector<double> FuzzyCMeans::computeMemberships(const std::vector<double>& point, const std::vector<std::vector<double>>& centroids) const {
	std::vector<double> memberships(centroids.size(), 0.0);

	// euclideanDistance renvoie le carre de la distance
	std::vector<double> distances(centroids.size());
	double minDistance = (std::numeric_limits<double>::max)();
	for (size_t j = 0; j < centroids.size(); ++j) {
		distances[j] = SimilarityFunctions::euclideanDistance(point, centroids[j]);
		minDistance = (std::min)(minDistance, distances[j]);
	}

	// point exactement sur un centroide : la formule diviserait par 0,
	// il appartient entierement a ce(s) centroide(s)
	if (minDistance == 0.0) {
		int numClosest = 0;
		for (size_t j = 0; j < centroids.size(); ++j) {
			if (distances[j] == 0.0) {
				++numClosest;
			}
		}
		for (size_t j = 0; j < centroids.size(); ++j) {
			memberships[j] = (distances[j] == 0.0) ? 1.0 / numClosest : 0.0;
		}
		return memberships;
	}

	// (d_min / d_j)^(1 / (fuzziness - 1)) : distances au carre, donc l'exposant du cours 2 / (m - 1) devient 1 / (m - 1)
	// diviser par d_min garde toutes les valeurs <= 1 (pas de depassement)
	double exponent = 1.0 / (fuzziness_ - 1.0);
	double sum = 0.0;
	for (size_t j = 0; j < centroids.size(); ++j) {
		memberships[j] = std::pow(minDistance / distances[j], exponent);
		sum += memberships[j];
	}

	for (size_t j = 0; j < memberships.size(); ++j) {
		memberships[j] /= sum;
	}

	return memberships;
}


// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));

	std::random_device rd;
	std::mt19937 generator(rd());
	// borne basse > 0 pour que la somme d'une ligne ne soit jamais nulle
	std::uniform_real_distribution<double> distribution(1e-6, 1.0);

	for (int i = 0; i < numDataPoints; ++i) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] = distribution(generator);
			sum += membershipMatrix_[i][j];
		}

		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sum;
		}
	}
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& centroids) {
	for (size_t i = 0; i < data.size(); ++i) {
		membershipMatrix_[i] = computeMemberships(data[i], centroids);
	}
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {
	std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0.0));

	for (int j = 0; j < numClusters_; ++j) {
		double weightSum = 0.0;

		// moyenne des points ponderee par (appartenance ^ fuzziness)
		for (size_t i = 0; i < data.size(); ++i) {
			double weight = std::pow(membershipMatrix_[i][j], fuzziness_);
			weightSum += weight;

			for (size_t k = 0; k < data[i].size(); ++k) {
				newCentroids[j][k] += weight * data[i][k];
			}
		}

		if (weightSum > 0.0) {
			for (size_t k = 0; k < newCentroids[j].size(); ++k) {
				newCentroids[j][k] /= weightSum;
			}
		}
		else if (j < static_cast<int>(centroids_.size())) {
			// plus aucun poids sur ce cluster : on garde l'ancien centroide
			newCentroids[j] = centroids_[j];
		}
	}

	return newCentroids;
}


// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels

	for (const auto& point : data) {
		std::vector<double> memberships = computeMemberships(point, centroids_);

		// cluster avec l'appartenance la plus elevee
		double maxMembership = -1.0;
		int bestCluster = 0;
		for (size_t j = 0; j < memberships.size(); ++j) {
			if (memberships[j] > maxMembership) {
				maxMembership = memberships[j];
				bestCluster = static_cast<int>(j);
			}
		}

		labels.push_back(bestCluster);
	}

	return labels; // Return the labels vector
}


/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
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