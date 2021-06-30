#ifndef FUZZY
#define FUZZY

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include<random>
#include <ctime>
#include "../Utils/Util.h"
using namespace std;

class Fuzzy{

    private:
        int r = 100000; // round value
        int numClusters; 
        double fMeasure; 
        int numDataPoints;
        vector<vector<double>> clusters;
        vector<vector<double>> weights;
        vector<vector<double>> data;
        

    public:
        Fuzzy(vector<vector<double>> d, int c , double m);
        Fuzzy(vector<vector<double>> d , vector<vector<double>> w, int c , double m);

        void init_weights();
        void init_centroids();
        void compute_centroids(vector<vector<double>> *Cent );
        void compute_weights(vector<vector<double>> *W);
        void display_weights();
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
    numClusters = c; // number of clusters
    fMeasure = m; // fuzzy measure usually equal to 2
    numDataPoints = d.size(); // number of data point
    srand(time(0)); // Random seed
    init_centroids(); // initialize centroids
}


// Initializations
void Fuzzy::init_weights(){
    for(int i = 0; i < numDataPoints; i++){
        vector<double> vec;
        for(int j = 0; j < numClusters; j++){
            vec.push_back(round(((double) rand()/(RAND_MAX))*r)/r);
        }
        weights.push_back(vec);
    }
}

void Fuzzy::init_centroids(){

    for(int i = 0; i < numClusters; i++){
        vector<double> cluster = data.at(0);    
        clusters.push_back(Util().scalar_multiply(0 , cluster));
    }

}


// Computations
void Fuzzy::compute_centroids(  vector<vector<double>> *Cent  ){

    for(int i = 0; i < numClusters; i++){
        double denominator = pow(weights.at(0).at(i),fMeasure);
        vector<double> cluster = data.at(0);
        

        for(int x = 1; x < data.size(); x++){
            double w = pow(weights.at(x).at(i),fMeasure);
            denominator += w;
            cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
        }

        Cent->at(i) = Util().scalar_multiply((double)(1/denominator) , cluster);
    }
}

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


// Displays
void Fuzzy::display_weights(){
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < weights.at(0).size(); j++){
            printf("%.5f ",weights.at(i).at(j));
        }
        cout<<"\t\t----------- Data point number :\t\t"<<i+1<<endl;
    }
}

void Fuzzy::display_centroids(){
    for(int i = 0; i < numClusters; i++){
        cout << "cluster "<<i+1<<" [ ";
        for(int j = 0; j < clusters.at(0).size(); j++){
            if(j != clusters.at(i).size()-1)
                cout << clusters.at(i).at(j) << ", "; 
            else
                cout << clusters.at(i).at(j) << " ]" << endl; 

        }
    }
}


// Run algorithm
void Fuzzy::run_fuzzy_c_means(int epochs){
    for(int i = 0; i < epochs; i++){
        compute_centroids(&clusters);
        compute_weights(&weights);

        // cout<<endl<<"= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="<<endl<<endl;
        // display_centroids();
    }
}



#endif