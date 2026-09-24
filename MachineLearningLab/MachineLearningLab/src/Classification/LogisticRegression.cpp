#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 
#include <cstdlib>

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int nb_rounds)
    : learning_rate(learning_rate), nb_rounds(nb_rounds) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = X_train[0].size();

    // List the different labels found in the dataset (1, 2 and 3 for Iris)
    class_labels.clear();
    for (size_t i = 0; i < y_train.size(); i++) {
        bool already_found = false;
        for (size_t k = 0; k < class_labels.size(); k++) {
            if (class_labels[k] == y_train[i]) {
                already_found = true;
            }
        }
        if (!already_found) {
            class_labels.push_back(y_train[i]);
        }
    }

    int num_classes = class_labels.size();

    // One row of weights per class: one weight per feature, plus one for the bias
    weights.clear();
    for (int c = 0; c < num_classes; c++) {
        std::vector<double> row;
        for (int j = 0; j < num_features + 1; j++) {
            double random_weight = ((double)rand() / RAND_MAX - 0.5) * 0.02;
            row.push_back(random_weight);
        }
        weights.push_back(row);
    }

    // Train one binary classifier per class
    for (int c = 0; c < num_classes; c++) {

        for (int round = 0; round < nb_rounds; round++) {

            for (size_t i = 0; i < X_train.size(); i++) {

                // Add the bias term in front of the features
                std::vector<double> x;
                x.push_back(1.0);
                for (int j = 0; j < num_features; j++) {
                    x.push_back(X_train[i][j]);
                }

                // The answer is 1 for the current class, 0 for the others
                double target = 0.0;
                if (y_train[i] == class_labels[c]) {
                    target = 1.0;
                }

                // Weighted sum of the features
                double z = 0.0;
                for (int j = 0; j < num_features + 1; j++) {
                    z += weights[c][j] * x[j];
                }

                double prediction = sigmoid(z);
                double error = prediction - target;

                // Gradient descent update
                for (int j = 0; j < num_features + 1; j++) {
                    weights[c][j] -= learning_rate * error * x[j];
                }
            }
        }
    }
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    for (size_t i = 0; i < X_test.size(); i++) {

        // Add the bias term, exactly like in fit()
        std::vector<double> x;
        x.push_back(1.0);
        for (size_t j = 0; j < X_test[i].size(); j++) {
            x.push_back(X_test[i][j]);
        }

        // One score per class
        std::vector<double> scores;
        for (size_t c = 0; c < weights.size(); c++) {
            double z = 0.0;
            for (size_t j = 0; j < x.size(); j++) {
                z += weights[c][j] * x[j];
            }
            scores.push_back(z);
        }

        // Keep the class with the highest score
        size_t best_class = 0;
        for (size_t c = 1; c < scores.size(); c++) {
            if (scores[c] > scores[best_class]) {
                best_class = c;
            }
        }

        predictions.push_back(class_labels[best_class]);
    }

    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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
        fit(trainData, trainLabels);

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