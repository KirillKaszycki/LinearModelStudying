#include <iostream>
#include <vector>
#include "LinearModel.h"
#include "train_test_split.h"
#include "r2_score.h"


int main(){
    std::vector<std::vector<double>> features = {
    {6.0, 7.0, 8.0, 9.0},
    {13.0, 14.0, 15.0, 16.0},
    {9.0, 10.0, 11.0, 12.0},
    {8.0, 9.0, 10.0, 11.0},
    {24.0, 25.0, 26.0, 27.0},
    {1.0, 2.0, 3.0, 4.0},
    {3.0, 4.0, 5.0, 6.0},
    {27.0, 28.0, 29.0, 30.0},
    {7.0, 8.0, 9.0, 10.0},
    {4.0, 5.0, 6.0, 7.0},
    {26.0, 27.0, 28.0, 29.0},
    {30.0, 31.0, 32.0, 33.0},
    {2.0, 3.0, 4.0, 5.0},
    {11.0, 12.0, 13.0, 14.0},
    {12.0, 13.0, 14.0, 15.0},
    {21.0, 22.0, 23.0, 24.0},
    {14.0, 15.0, 16.0, 17.0},
    {15.0, 16.0, 17.0, 18.0},
    {28.0, 29.0, 30.0, 31.0},
    {29.0, 30.0, 31.0, 32.0},
    {22.0, 23.0, 24.0, 25.0},
    {20.0, 21.0, 22.0, 23.0},
    {16.0, 17.0, 18.0, 19.0},
    {19.0, 20.0, 21.0, 22.0},
    {25.0, 26.0, 27.0, 28.0},
    {10.0, 11.0, 12.0, 13.0},
    {23.0, 24.0, 25.0, 26.0},
    {17.0, 18.0, 19.0, 20.0},
    {18.0, 19.0, 20.0, 21.0},
    {5.0, 6.0, 7.0, 8.0},
    {4.0, 5.0, 6.0, 7.0}
    };

    std::vector<double> target = {
    5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
    25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0,
    45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 63.0
    };

    LinearRegression model;
    std::vector<std::vector<double>> train_features;
    std::vector<double> train_target;
    std::vector<std::vector<double>> test_features;
    std::vector<double> test_target;

    train_test_split(features, target, train_features, train_target, test_features, test_target);

    model.fit(train_features, train_target);
    std::vector<double> pred_target;
    pred_target.push_back(model.predict(test_features));

    R2ScoreCalculator calculator(test_target, pred_target);
    double r2_score = calculator.CalculateR2Score();

    std::cout << "R2 Score: " << r2_score << std::endl;

    return 0;
}