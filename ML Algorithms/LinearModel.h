#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include <iostream>
#include <vector>

class LinearRegression{
    private:
    std::vector<double> w;
    double w0;

    public:
    void fit(const std::vector<std::vector<double>>& train_features, 
             const std::vector<double>& train_target){        
        size_t num_samples = train_features.size();
        size_t num_features = train_features[0].size();


        // Initializing of X matrix and y vector
        std::vector<std::vector<double>> X(num_samples, std::vector<double>(num_features + 1, 1.0));
        for(size_t i = 0; i < num_samples; ++i){
            for(size_t j = 0; j < num_features; ++j){
                X[i][j + 1] = train_features[i][j];
            }
        }

        std::vector<double> y = train_target;

        // Calculating w and w0 using matrix operations
        std::vector<std::vector<double>> X_transpose(num_features + 1, std::vector<double>(num_samples));
        for(int i = 0; i < num_features + 1; ++i){
            for(int j = 0; j < num_samples; ++j){
                X_transpose[i][j] = X[j][i];
            }
        }

        std::vector<std::vector<double>> XTX(num_features + 1, std::vector<double>(num_features + 1));
        for(int i = 0; i < num_features; ++i){
            for(int j = 0; j < num_features; ++j){
                for(int k = 0; k < num_samples; ++k){
                    XTX[i][j] += X_transpose[i][k] * X[k][j];
                }
            }
        }

        std::vector<double> XTX_inv_y(num_features + 1);
        for(int i = 0; i < num_features; ++i){
            for(int j = 0; j < num_samples; ++j){
                XTX_inv_y[i] += X_transpose[i][j] * y[j];
            }
        }

        // Compute w and w0
        std::vector<std::vector<double>> XTX_inv(num_features + 1, std::vector<double>(num_features + 1));
        for(int i = 0; i < num_features + 1; ++i){
            XTX_inv[i][i] = 1.0;
        }

        // Gaussian elimination to find the inverse of XTX
        for(int i = 0; i < num_features + 1; ++i){
            double pivot = XTX[i][i];
            for(int j = 0; j < num_features+ 1; ++j){
                XTX[i][j] /= pivot;
                XTX_inv[i][j] /= pivot;
            }

            for(int k = 0; k < num_features + 1; ++k){
                if (k != i){
                    double factor = XTX[k][i];
                    for(int j = 0; j < num_features+ 1; ++j){
                        XTX[k][j] -= factor * XTX[i][j];
                        XTX_inv[k][j] -= factor * XTX_inv[i][j];
                    }
                }
            }
        }

        // Compute w0 and w
        w0 = XTX_inv_y[0];
        w.assign(XTX_inv_y.begin() + 1, XTX_inv_y.end());
    }

    double predict(const std::vector<std::vector<double>>& test_features) {
        double result = w0;
        for (size_t i = 0; i < w.size(); ++i) {
            result += w[i] * test_features[0][i];
        }
        return result;
    }

};

#endif
