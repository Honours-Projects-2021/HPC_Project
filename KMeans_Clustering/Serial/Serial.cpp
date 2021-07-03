#include <bits/types/clock_t.h>
#include <bits/types/time_t.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include<ctime>
#include <stdio.h>
#include "Data.h"
#include "../Utils/Util.h"

#define K 3
#define EPOCHS  10

using namespace std;

void initClusters(Data d, vector<vector<double>> *clusters){
    srand(time(0));
    for(int i = 0; i < K; i++){
        int r = rand()%d.getNumRows();
        clusters->push_back(d.getData().at(r));
    }
}

void displayClusters(vector<vector<double>> clusters){
    for(int i = 0; i < clusters.size(); i++){
        cout<<"cluster " << i+1<<" [ ";
        for(int j = 0; j < clusters.at(i).size(); j++){
            if(j != clusters.at(i).size()-1)
                printf("%f, ",clusters.at(i).at(j));
            else
                printf("%f ]\n",clusters.at(i).at(j));
        }
        // printf("\n");
    }
    printf("\n");
}

void initDistances(Data d, vector<vector<double>> *distances){
    for(int i = 0; i < d.getData().size(); i++){
        vector<double> vec;
        for(int j = 0; j < K; j++){
            vec.push_back(0);
        }
        distances->push_back(vec);
    }
}

void displayDistance(vector<vector<double>> distances){
    for(int i = 0; i < distances.size(); i++){
        for(int j = 0; j < distances.at(0).size(); j ++){
            printf("%f   ",distances.at(i).at(j));
        }
        printf("\n");
    }
    printf("\n");

}
 
void initAssigns(vector<int> *assigns, int rows){
    for(int i = 0; i < rows; i++){
        assigns->push_back(0);
    }
}

void displayAssigns(vector<int> assigns){
    for(int i = 0; i < assigns.size(); i ++){
        printf("%d  " , assigns.at(i));
    }
    printf("\n\n");
}


int main(int argc, char **argv){

    // Import the data set
    Data d = Data("../Utils/wine-clustering.csv");
    vector<vector<double>> data = d.getData(); // get the 2 dimensional vector presentation of the data
    vector<vector<double>> clusters;    // This will store the clusters 
    vector<vector<double>> distances;   // Store distances of each data point and a cluster
    vector<int> assigns;    // every occurance will describe the cluster a datapoint belongs to

    int rows = d.getNumRows();

    // Initialize the vectors
    initClusters(d, &clusters);
    initDistances(d, &distances);
    initAssigns(&assigns, rows);

    // displayClusters(clusters);

    clock_t start = clock();
    // Run for the algorithm from here for a number of epochs
    for(int z = 0; z < EPOCHS; z++){

        // THIS forloop handles calculations of Distances for every data point
        for(int i = 0; i < rows; i++){
            int max = 1000000;
            
            for(int j = 0; j < K; j++){ // For every cluster in the set of all clusters
                double dist = Util().distance(data.at(i), clusters.at(j)); // distance between current datapoint and the clusters
                if(dist < max){ // condition for assigning a data point to a cluster
                    max = dist;
                    assigns.at(i) = j;
                }
                // update the distance matrix for the current datapoint  and the clusters
                distances.at(i).at(j) = dist;
            }
        }

        // This section recalculates the centroids
        for(int i = 0; i < K; i++){
            // Assign the value 0 for the current cluster. This allows us to use the += operator without problems
            clusters.at(i) = Util().scalar_multiply(0, clusters.at(i));
            double occurances = 0; // Controls the averaging 

            // Loop through the entire data set to find data points that belong to a current cluster
            for(int j = 0; j < rows; j++){
                if(assigns[j] == i){ // Controls the condition of a data point being found
                    clusters.at(i) = Util().vector_addition(clusters.at(i), data.at(j)); // Do vector addition between the current datapoint and cluster
                    occurances ++; // increment the number of occurance for averaging
                }
            }
            occurances = 1/occurances;
            clusters.at(i) = Util().scalar_multiply(occurances, clusters.at(i)); // average the occurances
        }

    }

    clock_t end = clock() - start;
    double timelapsed = ((double)end)/CLOCKS_PER_SEC;
    displayClusters(clusters);
    printf("The serial time for K Means in  miliseconds is %fms for %d epochs\n", timelapsed*1000,EPOCHS );



    return 0;
}