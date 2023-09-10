#include <iostream>
#include <vector>
#include "LinearModel.h"
#include "r2_score.h"
#include "data.h"


int main(){

    LinearRegression model;
    model.fit(train_features, train_target);

    std::vector<double> pred_target;


    R2ScoreCalculator calculator(test_target, pred_target);
    double r2_score = calculator.CalculateR2Score();

    std::cout << "R2 Score: " << r2_score << std::endl;

    return 0;
}