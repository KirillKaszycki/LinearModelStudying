cmake_minimum_required(VERSION 3.24)
project(ML_Algorithms)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ML_Algorithms
        LinearModel.h
        main.cpp
        r2_score.h
        train_test_split.h)
