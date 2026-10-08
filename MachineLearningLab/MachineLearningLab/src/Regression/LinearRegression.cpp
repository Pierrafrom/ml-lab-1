#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///

// Best gradient descent hyperparameters found on Boston Housing with standardized features
// (see the Lab 4 section of notebooks/boston_housing_exploration.ipynb): test R2 ~0.66, same as Matrix Form.
static constexpr double GRADIENT_DESCENT_LEARNING_RATE = 0.0005;
static constexpr int GRADIENT_DESCENT_ROUNDS = 100;


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/
	
	if (trainData.size() != trainLabels.size()) {
		throw std::invalid_argument("Size of trainData and trainLabels must match.");
	}

	// Coefficients are learned on raw features: drop any standardization left by a previous gradient descent fit,
	// otherwise predict() would standardize the test data and misuse these coefficients.
	m_featureMean.resize(0);
	m_featureSD.resize(0);

    // création de la matrice avec une colonne de biais à 1
	Eigen::MatrixXd X(static_cast<int>(trainData.size()), static_cast<int>(trainData[0].size() + 1));
	for (size_t i = 0; i < trainData.size(); ++i) {
		X(i, 0) = 1.0; // Bias term
		for (size_t j = 0; j < trainData[i].size(); ++j) {
			X(i, j + 1) = trainData[i][j];
		}
	}

	Eigen::VectorXd y(static_cast<int>(trainLabels.size()));
	for (size_t i = 0; i < trainLabels.size(); ++i) {
		y(i) = trainLabels[i];
	}

	// calcul coeff avec moindres carrés
	m_coefficients = (X.transpose() * X).inverse() * X.transpose() * y;

}

// Function to fit with the GradienDescent method
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int nb_rounds) {
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("Size of trainData and trainLabels must match.");
    }

    int num_features = trainData[0].size();
    int n = trainData.size();
    m_featureMean = Eigen::VectorXd(num_features); //init the size at 13
    m_featureSD = Eigen::VectorXd(num_features); 
    
    // Calculation of the mean for each features
    for (int j = 0; j < num_features; j++) {
        double somme = 0.0;
        for (int i = 0; i < n; i++) {
            somme += trainData[i][j];
        }
        m_featureMean[j] = somme / n;
    }

    // Calculation of the SD for each features
    for (int j = 0; j < num_features; j++) {
        double somme_square = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = trainData[i][j] - m_featureMean[j];
            somme_square += diff * diff;
        }
        double sd = std::sqrt(somme_square / n);
        if (sd == 0) { sd = 1.0; }  // if the column doesn't vary
        m_featureSD[j] = sd;
    }

    Eigen::VectorXd weights(num_features + 1);
	for (int i = 0; i < num_features + 1; i++) {
		weights[i] = 0.0; // Initialize weights to zero
	}    

    for (int round = 0; round < nb_rounds; round++) {

        for (size_t i = 0; i < trainData.size(); i++) {

            // Add the bias term in front of the features
            std::vector<double> x;
            x.push_back(1.0);
            for (int j = 0; j < num_features; j++) {
                double normalized = (trainData[i][j] - m_featureMean[j]) / m_featureSD[j];
                x.push_back(normalized);
            }
            double prediction = 0.0;
            for (int j = 0; j < num_features + 1; j++) {
                prediction = prediction + weights[j] * x[j];    
            }
            
            double target = trainLabels[i];
            double error = prediction - target;

            // Gradient descent update
            for (int j = 0; j < num_features + 1; j++) {
                weights[j] -= learning_rate * error * x[j];
            }
        }
    }

	m_coefficients = weights;
 
}

// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// Shared by both fit() overloads: they all store their result in m_coefficients.
	// Gradient descent coefficients live in standardized space, so test data is standardized the same way.
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	if (m_coefficients.size() == 0) {
		throw std::runtime_error("Model has not been fitted yet.");
	}

    bool isNormalized = (m_featureSD.size() > 0); // Are we using the fit with gradient descent ?

	Eigen::MatrixXd X(static_cast <int>(testData.size()), static_cast <int>(testData[0].size() + 1));
	for (size_t i = 0; i < testData.size(); ++i) {
		X(i, 0) = 1.0; // Bias term
		for (size_t j = 0; j < testData[i].size(); ++j) {
            if (isNormalized) {
                X(i, j + 1) = (testData[i][j] - m_featureMean[j]) / m_featureSD[j];
            }
            else {
                X(i, j + 1) = testData[i][j];
            }

		}
	}

	Eigen::VectorXd predictions = X * m_coefficients;


	std::vector<double> result;
	for (int i = 0; i < predictions.size(); ++i) {
		result.push_back(predictions(i));
	}

    return result;
}



/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio, Method method) {
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
        if (method == Method::MatrixForm) {
            fit(trainData, trainLabels);
        }
        else {
            fit(trainData, trainLabels, GRADIENT_DESCENT_LEARNING_RATE, GRADIENT_DESCENT_ROUNDS);
        }

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