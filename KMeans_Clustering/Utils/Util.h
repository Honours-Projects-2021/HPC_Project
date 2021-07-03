#ifndef UTIL
#define UTIL

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include<random>
#include <ctime>

using namespace std;


class Util{
    public:
        double Round(double c);
        double distance(vector<double> x1, vector<double> x2);
        vector<double> scalar_multiply(double c, vector<double> x);
        vector<double> vector_addition(vector<double> x1, vector<double> x2);


        double Fdistance(vector<double> x1, vector<double> x2 , int idx , int len);
        vector<double> Fscalar_multiply(double c, vector<double> x, int idx, int len);
        vector<double> Fvector_addition(vector<double> x1, vector<double> x2, int idx, int len);
};

double Util::Round(double c){
    // return round(c*100000)/100000;
    return c;
}

vector<double> Util::scalar_multiply(double c, vector<double> x){
    for(int i = 0; i < x.size(); i++){
        x.at(i) = x.at(i)*c;
    }
    return x;
}

double Util::distance(vector<double> x1, vector<double> x2){
    double sum = 0;
    for(int i = 0; i < x1.size(); i++){
        sum += pow((x1.at(i) - x2.at(i)),2);
    }

    return Round(sqrt(sum));
}

vector<double> Util::vector_addition(vector<double> x1, vector<double> x2){
    vector<double> vec;
    for(int i = 0; i < x1.size(); i++){
        vec.push_back((Round(x1.at(i) + x2.at(i))));
    }
    return vec;
}

/*================================================================================*/

#endif