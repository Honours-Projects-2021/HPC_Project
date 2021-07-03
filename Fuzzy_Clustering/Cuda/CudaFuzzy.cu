#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include<random>
#include <ctime>

#include "./inc/common/book.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Data.h"

#define NUMCENTS 3
#define FMEASURE 3
#define EPOCHS  500

using namespace std;

// This function rounds off to 5 decimal places
__device__ double Round(double c){
    return round(c*100000)/100000;
}

void displayCentroids(double* cent, int NumFeatures){
    for(int i = 0; i < NUMCENTS; i++){
        cout << "cluster "<<i+1<<" [ ";
        for(int j = 0; j < NumFeatures; j++){
            if(j != NumFeatures-1)
                printf("%f, ",cent[i*NUMCENTS +j]); 
            else
                printf("%f ]\n",cent[i*NUMCENTS +j]);

        }
    }
}

void displayWeights(double* weights, int n){
    for(int i = 0; i < n; i++){
        cout << "weight for data-point "<<i+1<<" : [ ";
        for(int j = 0; j < NUMCENTS; j++){
            if(j != NUMCENTS-1)
                printf("%f, ",weights[i*NUMCENTS +j]); 
            else
                printf("%f ]\n",weights[i*NUMCENTS +j]);

        }
    }

}


__global__ void computeCentroids(double *data, double *weights, double *centroids, int numCents, int dRows, int dCols , int fMeasure){
   
    int i = threadIdx.x ; // centroid number
    for(int k = i*dCols; k < (i+1)*dCols; k++){
        centroids[k] = 0;
    }

    double denominator = 0;
    
    for(int x = 0; x < dRows; x++){
        double w = pow(weights[x*numCents + i],fMeasure);
        denominator = denominator + w;

        for(int k = i*dCols; k < (i+1)*dCols; k++){
            centroids[k] = centroids[k] + w*data[x*dCols + k];
        }
    }


    for(int k = i*dCols; k < (i+1)*dCols; k++){
        centroids[k] = centroids[k]*(1/denominator);
    }
}


__global__ void computeWeights(double *data, double *weights, double *centroids, int numCents,int dRows, int dCols , int fMeasure){

    int i = threadIdx.x ; // datapoint
    int j = blockIdx.x; // cluster

    double w = 0; // the weight for index [i,j] of the weight matrix
    if(i < dRows)

    // loops for through every centroids
    for(int k = 0; k < numCents; k++){
        double numerator = 0;
        double denominator = 0;

        // calculates the distances between the data points and centroids as described in the formula
        for(int l = 0; l < dCols; l++){
            double dPoint = data[i*dCols+l]; // the data point we are dealing with
            numerator += pow((dPoint - centroids[j*dCols+l]),2);
            denominator += pow((dPoint - centroids[k*dCols+l]),2);
            
        }
        numerator = sqrt(numerator); // top distance
        denominator = sqrt(denominator); // bottom distance
        w += pow((numerator/denominator),(2/(fMeasure-1))); // add to w
    }

    if(i < dRows)
    weights[i*numCents + j] = 1/w; // Put the weight into the weight matrix
}

// sets every value in the array to zero
void initCentroids(double *cents, int numCols){
    for(int i = 0; i < numCols*NUMCENTS; i++ ){
        cents[i] = 0.0;
    }
}

// creates an array equivalent of the given vector
void initWeights(vector<double> flatWeights, double *weights){
    for(int i = 0; i < flatWeights.size(); i++){
        weights[i] = flatWeights.at(i);
    }
}
int main(){

    // Import in the data
    Data d = Data("../Utils/wine-clustering.csv"); // The complete datasets of 13 columns and 178 records
    Data w = Data("../Utils/weights.csv");  // The initialized weight matrix

    // Attributes of the dataset and the weight matrix
    int dataRows = d.getNumRows();
    int dataColumns = d.getNumCols();
    int weightRows = w.getNumRows();
    int weightCols = w.getNumCols();

    // Size variables for cudaMalloc funcntions
    int centSize = NUMCENTS*dataColumns*sizeof(double); // Number of centroids
    int dataSize = dataRows*dataColumns*sizeof(double); // Number of data points
    int weightSize = dataRows*NUMCENTS*sizeof(double);  // Size of the weight matrix


    // Flattened dataset and weights to make it easy for us to work with
    vector<double> flatData = d.getFlat(); // flattened data
    vector<double> flatWeights = w.getFlat(); // flattened weights

    // Create centroids arrays and weights arrays
    double *centroids = (double*) malloc(NUMCENTS*dataColumns*sizeof(double)); // flattened centroids
    double *weights = (double*) malloc(dataRows*dataColumns*sizeof(double)); // array equivalent of our vector weights array


    initWeights(flatWeights , weights); // converts vector into a normal array
    initCentroids(centroids , dataColumns); // initialize all array values to 0's

    // data for device declaration
    double *deviceWeights, *deviceData , *deviceCentroids;

    // allocate space in the device
    checkCudaErrors(cudaMalloc((void**) &deviceData, dataSize));
    checkCudaErrors(cudaMalloc((void**) &deviceWeights, weightSize));
    checkCudaErrors(cudaMalloc((void**) &deviceCentroids, centSize));

    // copying memory
    checkCudaErrors(cudaMemcpy(deviceData , &flatData[0] , dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceWeights , weights , weightSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceCentroids , centroids, centSize, cudaMemcpyHostToDevice));

    // Number of blocks and threads for calcualting the centroids
    dim3 Cblock(1);
    dim3 Cthreads(NUMCENTS);
    
    // Number of blocks and threads for calculating the weights
    dim3 Wblock(NUMCENTS);
    dim3 Wthreads(dataRows);


    StopWatchInterface *se_timer = NULL;
    sdkCreateTimer(&se_timer);
    sdkStartTimer(&se_timer);


    // Run the algorithm for the number of epochs
    for(int i = 0; i < EPOCHS; i++){
        // Compute the new centroids
        computeCentroids<<<Cblock,Cthreads>>>(deviceData, deviceWeights, deviceCentroids, NUMCENTS, dataRows, dataColumns , FMEASURE);
        // Compute the new weights
        computeWeights<<<Wblock,Wthreads>>>(deviceData, deviceWeights, deviceCentroids, weightCols, dataRows, dataColumns , FMEASURE);
    }
    

    sdkStopTimer(&se_timer);
    printf("Processing time for Cuda Parallel: %f (ms)\n", sdkGetTimerValue(&se_timer));
    sdkDeleteTimer(&se_timer);

    // copy the results into the respective arrays
    checkCudaErrors(cudaMemcpy(centroids, deviceCentroids ,  centSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(weights, deviceWeights ,  weightSize, cudaMemcpyDeviceToHost));


    // for(int i = 0; i < 100; i++){
    //     for(int j = 0; j < 3; j++){
    //         printf("%.5f " ,weights[i*NUMCENTS + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    
    displayCentroids(centroids,dataColumns);
    // free acquried Device memory
    checkCudaErrors(cudaFree(deviceCentroids));
    checkCudaErrors(cudaFree(deviceWeights));
    checkCudaErrors(cudaFree(deviceData));

    // Free acquired Host memory
    free(centroids);
    free(weights);


    return 0;
}




