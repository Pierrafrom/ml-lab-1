#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <Eigen/Core>

										/// LinearRegression class definition ///

class LinearRegression {
public:
    // Which of the two fit() overloads runLinearRegression() uses.
    enum class Method { MatrixForm, GradientDescent };

    void fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels);
	void fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int nb_rounds);
    std::vector<double> predict(const std::vector<std::vector<double>>& testData);
    std::tuple<double, double, double, double, double, double,
        std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>>
        runLinearRegression(const std::string& filePath, int trainingRatio, Method method = Method::GradientDescent);

private:

    Eigen::VectorXd m_coefficients; // Store the coefficients for future predictions
    Eigen::VectorXd m_featureMean;  // mean of the 13 features
    Eigen::VectorXd m_featureSD;    // sd of the 13 features

};

#endif // LINEARREGRESSION_H
