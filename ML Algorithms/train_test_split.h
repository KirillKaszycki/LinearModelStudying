#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

template <typename T>
void train_test_split(const std::vector<std::vector<T>>& features,
                      const std::vector<T>& target,
                      std::vector<std::vector<T>>& train_features,
                      std::vector<T>& train_target,
                      std::vector<std::vector<T>>& test_features,
                      std::vector<T>& test_target,
                      double test_size = 0.2) {
    if (features.size() != target.size() || test_size < 0.0 || test_size > 1.0) {
        std::cerr << "Invalid input data or test_size value" << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    const size_t total_samples = features.size();
    const size_t num_test_samples = static_cast<size_t>(total_samples * test_size);

    std::vector<size_t> indices(total_samples);
    for (size_t i = 0; i < total_samples; ++i) {
        indices[i] = i;
    }

    std::shuffle(indices.begin(), indices.end(), gen);

    train_features.clear();
    train_target.clear();
    test_features.clear();
    test_target.clear();

    for (size_t i = 0; i < total_samples; ++i) {
        if (i < num_test_samples) {
            test_features.push_back(features[indices[i]]);
            test_target.push_back(target[indices[i]]);
        } else {
            train_features.push_back(features[indices[i]]);
            train_target.push_back(target[indices[i]]);
        }
    }
}
