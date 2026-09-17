#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;

	// Guard against an empty set: nothing prevents a future caller from
	// passing one, and total_samples would otherwise divide by zero below.
	if (total_samples == 0) {
		return 0.0;
	}

	// Convert labels to unique integers and count their occurrences: labels
	// here are already numeric (0/1/2 for classes), so the map can be keyed
	// directly on the label value -- no separate int-conversion pass needed.
	for (double label : y) {
		label_map[label]++;
	}
	for (const auto& entry : label_map) {
		hist.push_back(entry.second);
	}

	// Compute the probability and entropy: H = -Sum(p_c * log2(p_c)) over
	// every class present in this set of labels.
	for (double count : hist) {
		double p = count / static_cast<double>(total_samples);
		entropy -= p * std::log2(p);
	}

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;

	// Guard against an empty index list -- see the same guard above.
	if (total_samples == 0) {
		return 0.0;
	}

	// Convert labels to unique integers and count their occurrences: read
	// through the index list into the shared y vector rather than a copy of
	// it -- see DecisionTreeClassification::informationGain for why this
	// overload exists (avoids copying labels for every threshold candidate).
	for (int idx : idxs) {
		label_map[y[idx]]++;
	}
	for (const auto& entry : label_map) {
		hist.push_back(entry.second);
	}

	// Compute the probability and entropy: same formula as the full-vector
	// overload, just normalized over this subset's own total.
	for (double count : hist) {
		double p = count / static_cast<double>(total_samples);
		entropy -= p * std::log2(p);
	}

	return entropy;
}


