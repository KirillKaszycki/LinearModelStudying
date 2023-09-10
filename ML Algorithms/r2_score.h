#include <iostream>
#include <vector>

class R2ScoreCalculator {
public:
    R2ScoreCalculator(const std::vector<double>& true_values, const std::vector<double>& predicted_values) {
        if (true_values.size() != predicted_values.size()) {
            std::cerr << "Error: The number of true values and predicted values must be the same." << std::endl;
            return;
        }

        true_values_ = true_values;
        predicted_values_ = predicted_values;
    }

    double CalculateR2Score() const {
        double mean_true = CalculateMean(true_values_);
        double ss_total = 0.0;  
        double ss_residual = 0.0;  

        for (size_t i = 0; i < true_values_.size(); ++i) {
            ss_total += (true_values_[i] - mean_true) * (true_values_[i] - mean_true);
            ss_residual += (true_values_[i] - predicted_values_[i]) * (true_values_[i] - predicted_values_[i]);
        }

        // Calculation R2
        if (ss_total == 0.0) {
            return 1.0;  // If ss_total = 0, R2 = 1
        } else {
            return 1.0 - (ss_residual / ss_total);
        }
    }

private:
    std::vector<double> true_values_;
    std::vector<double> predicted_values_;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
    static double CalculateMean(const std::vector<double>& values) {
        double sum = 0.0;
        for (const double& val : values) {
            sum += val;
        }
        return sum / values.size();
    }
#pragma clang diagnostic pop
};
