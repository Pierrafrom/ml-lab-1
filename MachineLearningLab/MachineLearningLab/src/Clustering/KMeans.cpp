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
	// Étape 0 : Vérifier que le dataset n'est pas vide
	if (data.empty() || numClusters_ <= 0 || numClusters_ > static_cast<int>(data.size())) {
		return;
	}

	// Étape 1 : Initialiser les centroïdes aléatoirement
	// Créer un vecteur d'indices [0, 1, 2, ..., N-1]
	std::vector<int> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);

	// Mélanger les indices aléatoirement
	std::random_device rd;
	std::shuffle(indices.begin(), indices.end(), std::mt19937(rd()));

	// Sélectionner les numClusters_ premiers indices (qui sont maintenant aléatoires)
	centroids_.clear();
	for (int i = 0; i < numClusters_; ++i) {
		centroids_.push_back(data[indices[i]]);
	}

	// Étape 2 : Itérer pour trouver les centroïdes optimaux
	for (int iteration = 0; iteration < maxIterations_; ++iteration) {
		// Étape 2a : Initialiser les nouveaux centroïdes et les compteurs
		std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0.0));
		std::vector<int> clusterCounts(numClusters_, 0);

		// Etape 2b : Assigner chaque point de donnees au centroide le plus proche
		for (const auto& point : data) {
			// Initialiser la distance minimale et le centroide le plus proche
			double minDistance = (std::numeric_limits<double>::max)();
			int closestCentroid = 0;

			// Iterer a travers tous les centroïdes
			for (size_t i = 0; i < centroids_.size(); ++i) {
				// Calculer la distance euclidienne entre le point et le centroide actuel
				double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);

				// Mettre a jour le centroide le plus proche si une distance plus petite est trouvee
				if (distance < minDistance) {
					minDistance = distance;
					closestCentroid = static_cast<int>(i);
				}
			}

			// Ajouter le point au cluster correspondant (pour recalculer le centroide)
			for (size_t j = 0; j < point.size(); ++j) {
				newCentroids[closestCentroid][j] += point[j];
			}
			clusterCounts[closestCentroid]++;
		}

		// Etape 2c : Recalculer les centroïdes comme la moyenne des points de chaque cluster
		bool hasConverged = true;
		for (int i = 0; i < numClusters_; ++i) {
			if (clusterCounts[i] > 0) {
				// Calculer la moyenne des points du cluster
				for (size_t j = 0; j < newCentroids[i].size(); ++j) {
					newCentroids[i][j] /= clusterCounts[i];
				}

				// Verifier la convergence : comparer anciens et nouveaux centroïdes
				// Calcule la distance entre l'ANCIEN centroide et le NOUVEAU centroide
				double centroidChange = SimilarityFunctions::euclideanDistance(centroids_[i], newCentroids[i]);

				// Si cette distance est GRANDE (> 1e-6), le centroide a beaucoup change
				// Donc l'algorithme n'a pas converged -> continue
				if (centroidChange > 1e-6) {
					hasConverged = false;
				}
			}
			else {
				// Cluster vide : on garde l'ancien centroide (sinon il partirait a l'origine)
				newCentroids[i] = centroids_[i];
			}
		}

		// Etape 2d : Mettre a jour les centroïdes
		centroids_ = newCentroids;

		// Etape 2e : Verifier la convergence et arreter si atteinte
		// Si TOUS les centroïdes ont bouge de moins de 1e-6, on a trouve la stabilite
		if (hasConverged) {
			break;  // Les centroïdes ne changent plus, arreter l'iteration
		}
	}
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size()); // un label par point de données

	// Pour chaque point de données
	for (const auto& point : data) {
		// Etape 1 : Initialiser la distance minimale et le centroide le plus proche
		double minDistance = (std::numeric_limits<double>::max)();
		int closestCentroid = 0; // par défaut, le premier centroide est le plus proche

		// Etape 2 : Itérer à travers tous les centroïdes
		for (size_t i = 0; i < centroids_.size(); ++i) {
			// Étape 3 : Calculer la distance euclidienne entre le point et le centroïde actuel
			double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);

			// Étape 4 : Mettre à jour le centroïde le plus proche si une distance plus petite est trouvée
			if (distance < minDistance) {
				minDistance = distance;
				closestCentroid = static_cast<int>(i);
			}
		}

		// Étape 5 : Ajouter l'étiquette du centroïde le plus proche au vecteur de résultats
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