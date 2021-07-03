#ifndef FUZZY
#define FUZZY

#include <cmath>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "../Utils/Util.h"
#include "Data.h"
using namespace std;

class Fuzzy{

    private:
        int r = 100000; // round value
        int numClusters; 
        double fMeasure; 
        int numDataPoints;
        int numFeatures;
        vector<vector<double>> clusters;
        vector<vector<double>> weights=Data("../Utils/weights.csv").getData();
        vector<vector<double>> data = Data("../Utils/wine-clustering.csv").getData();
        

    public:
        Fuzzy(vector<vector<double>> d, int c , double m);
        Fuzzy(vector<vector<double>> d , vector<vector<double>> w, int c , double m);

        void init_weights();
        void init_centroids();
        void compute_centroids();
        void compute_weights(vector<vector<double>> *W);
        void display_weights(int n);
        void display_data(int n);
        void display_centroids();
        void run_fuzzy_c_means(int epochs);

};


// Constructors
Fuzzy::Fuzzy(vector<vector<double>> d, int c, double m){
    data = d;
    numClusters = c;
    fMeasure = m;
    numDataPoints = d.size();
    srand(time(0));
    init_weights();
    init_centroids();
}

Fuzzy::Fuzzy(vector<vector<double>> d,vector<vector<double>> w, int c, double m){
    data = d; // Set the dataset
    weights = w; // Set the weights
    init_centroids(); // initialize centroids
    numClusters = c; // number of clusters
    fMeasure = m; // fuzzy measure usually equal to 2
    numDataPoints = d.size(); // number of data point
    numFeatures = d.at(0).size();
}


// Initializes the weights randomly
void Fuzzy::init_weights(){
    for(int i = 0; i < numDataPoints; i++){
        vector<double> vec;
        for(int j = 0; j < numClusters; j++){
            vec.push_back(round(((double) rand()/(RAND_MAX))*r)/r);
        }
        weights.push_back(vec);
    }
}

// Initializes the centroids to zero values
void Fuzzy::init_centroids(){
    clusters.clear();
    for(int i = 0; i < numClusters; i++){
        vector<double> cluster;    
        for(int j = 0; j < data.at(0).size(); j++){
            cluster.push_back(0);
        }
        clusters.push_back(cluster);
    }

    
}

// Computations
void Fuzzy::compute_centroids(){
    init_centroids();
    for(int i = 0; i < numClusters; i++){
        // double w = pow(weights.at(0).at(i),fMeasure);
        // vector<double> cluster = data.at(0);
        data = Data("../Utils/wine-clustering.csv").getData();
        double denominator = 0;
        // double w = 0;

        for(int x = 0; x < numDataPoints; x++){
            double w = pow(weights[x][i],fMeasure);
            denominator += w;
            // cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
            
            for(int k = 0; k < numFeatures; ++k){
                clusters[i][k] += w*data[x][k];
            }

        }

        for(int k = 0; k < numFeatures; ++k){
            clusters[i][k] = clusters[i][k]*(1/denominator);
        }
        // display_centroids();
        // cout<<endl<<"===========================================================================================";
        // cout<<endl<<"==========================================================================================="<<endl;

    }
}

// computes the weights for  a vector W
void Fuzzy::compute_weights(vector<vector<double>> *W){

    for(int i = 0; i < numDataPoints; i++){
            vector<double> vec;
        for(int j = 0; j < numClusters; j++){
            double w = 0;
            for(int k = 0; k < numClusters; k++){
                double numerator = Util().distance(data.at(i), clusters.at(j));
                double denominator = Util().distance(data.at(i), clusters.at(k));
                
                w += pow((numerator/denominator) , (2/(fMeasure-1)));
            }
            // cout<<weights.at(i).at(j)<< " "<<1/w << "\n";
            vec.push_back((double)(1/w));
        }
            W->at(i) = vec;
    }
}


// Display the weights for the first n data points
void Fuzzy::display_weights(int n){
    for(int i = 0; i < n; i++){
        
        for(int j = 0; j < weights.at(0).size(); j++){
            printf("%.5f ",weights.at(i).at(j));
        }
        cout<<"\t\t----------- Data point number :\t\t"<<i+1<<endl;
    }
}

// Display the first n data points
void Fuzzy::display_data(int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < data.at(0).size(); j++){
            printf("%.5f ",data.at(i).at(j));
        }
        cout<<"\t\t----------- Data point number :\t\t"<<i+1<<endl;
    }
}

// Neatly displays the centroids
void Fuzzy::display_centroids(){
    for(int i = 0; i < numClusters; i++){
        cout << "cluster "<<i+1<<" [ ";
        for(int j = 0; j < clusters.at(0).size(); j++){
            if(j != clusters.at(i).size()-1)
                printf("%f, ",clusters[i][j]); 
            else
                printf("%f ]\n",clusters.at(i).at(j));

        }
    }
}


// Run algorithm
void Fuzzy::run_fuzzy_c_means(int epochs){
    for(int i = 0; i < epochs; i++){
        compute_centroids();
        compute_weights(&weights);
    // display_centroids();
    // cout<<endl<<"===========================================================================================";
    // cout<<endl<<"==========================================================================================="<<endl;

    }
}

#endif