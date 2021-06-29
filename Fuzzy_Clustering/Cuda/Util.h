#ifndef UTIL
#define UTIL

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include<random>
#include <ctime>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../inc/common/book.h"

using namespace std;


class Util{
    public:
        double distance(vector<double> x1, vector<double> x2);
        double Round(double c);
        vector<double> scalar_multiply(double c, vector<double> x);
        vector<double> vector_addition(vector<double> x1, vector<double> x2);
};

__device__ double Util::Round(double c){
    return round(c*100000)/100000;
}

__device__ vector<double> Util::scalar_multiply(double c, vector<double> x){
    for(int i = 0; i < x.size(); i++){
        x.at(i) = Round(x.at(i)*c);
    }
    return x;
}

___device_ double Util::distance(vector<double> x1, vector<double> x2){
    double sum = 0;
    for(int i = 0; i < x1.size(); i++){
        sum += pow((x1.at(i) - x2.at(i)),2);
    }

    return Round(sqrt(sum));
}

__device__ vector<double> Util::vector_addition(vector<double> x1, vector<double> x2){
    vector<double> vec;
    for(int i = 0; i < x1.size(); i++){
        vec.push_back(Round(Round(x1.at(i) + x2.at(i))));
    }
    return vec;
}

#endif