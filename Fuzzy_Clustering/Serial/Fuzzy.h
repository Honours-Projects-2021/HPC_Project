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
        vector<vector<double>> clusters;
        vector<vector<double>> weights=Data("../Utils/weights.csv").getData();
        vector<vector<double>> data = Data("../Utils/wine-clustering.csv").getData();

        vector<double> clus;
        

    public:
        Fuzzy(vector<vector<double>> d, int c , double m);
        Fuzzy(vector<vector<double>> d , vector<vector<double>> w, int c , double m);

        void init_weights();
        void init_centroids();
        void compute_centroids(vector<vector<double>> Cent );
        void compute_weights(vector<vector<double>> *W);
        void display_weights();
        void display_data();
        void display_centroids();
        void run_fuzzy_c_means(int epochs);

        void c();

};


// Constructors
Fuzzy::Fuzzy(vector<vector<double>> d, int c, double m){
    // data = d;
    numClusters = c;
    fMeasure = m;
    numDataPoints = d.size();
    srand(time(0));
    init_weights();
    init_centroids();
}

Fuzzy::Fuzzy(vector<vector<double>> d,vector<vector<double>> w, int c, double m){
    // data = d; // Set the dataset
    // weights = w; // Set the weights
    init_centroids(); // initialize centroids
    numClusters = c; // number of clusters
    fMeasure = m; // fuzzy measure usually equal to 2
    numDataPoints = d.size(); // number of data point
    srand(time(0)); // Random seed
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
        vector<double> cluster;    
        for(int j = 0; j < data.at(0).size(); j++){
            cluster.push_back(0);
        }
        clusters.push_back(cluster);
    }

    for(int i = 0; i < 3*13; i++){
        clus.push_back(0);
    }
}

void Fuzzy:: c(){
    init_centroids();
    vector<double> d = Data("../Utils/wine-clustering.csv").getFlat();
    vector<double> we = Data("../Utils/weights.csv").getFlat();

    for(int i = 0; i < numClusters; i++){
         double denominator = 0;
        
        for(int x = 0; x < numDataPoints; x++){
            double w = pow(we[x*numClusters + i],fMeasure);
            denominator = denominator + w;

            for(int k = i*13; k < (i+1)*13; k++){
                clus[k] = clus[k] + w*d[x*13 + k];
            }
            // cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
        }


        for(int k = i*13; k < (i+1)*13; k++){
            clus[k] = clus[k]*(1/denominator);
        }
    }

    for(int i =0; i < 3*13; i++){
        printf("%f --- " , clus[i]);
        if((i+1)%13 == 0)
            printf("\n");
    }   
}

// Computations
void Fuzzy::compute_centroids(  vector<vector<double>> Cent  ){
    // init_weights();
    for(int i = 0; i < numClusters; i++){
        // double w = pow(weights.at(0).at(i),fMeasure);
        vector<double> cluster = data.at(0);
        
        double denominator = 0;
        double w = 0;

        for(int x = 0; x < data.size(); x++){
            w = pow(weights.at(x).at(i),fMeasure);
            denominator = denominator + w;
            // cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
            
            for(int k = 0; k < data.at(0).size(); k++){
                clusters[i][k] += w*data[x][k];
            }

        }

        for(int k = 0; k < data.at(0).size(); k++){
            clusters[i][k] *= (1/denominator);
        }
        // Cent->at(i) = Util().scalar_multiply((double)(1/denominator) , cluster);
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
void Fuzzy::display_data(){
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < data.at(0).size(); j++){
            printf("%.5f ",data.at(i).at(j));
        }
        cout<<"\t\t----------- Data point number :\t\t"<<i+1<<endl;
    }
}

void Fuzzy::display_centroids(){
    for(int i = 0; i < numClusters; i++){
        cout << "cluster "<<i+1<<" [ ";
        for(int j = 0; j < clusters.at(0).size(); j++){
            if(j != clusters.at(i).size()-1)
                // cout << clusters.at(i).at(j) << ", ";
                printf("%f, ",clusters[i][j]); 
            else
                // cout << clusters.at(i).at(j) << " ]" << endl; 
                printf("%f ]\n",clusters.at(i).at(j));

        }
    }
}


// Run algorithm
void Fuzzy::run_fuzzy_c_means(int epochs){
    for(int i = 0; i < epochs; i++){
        
        compute_centroids(weights);
        compute_weights(&weights);
    }
}



#endif