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
#include "Data.h"

#define NUMCENTS 3
#define FMEASURE 2

using namespace std;


__device__ double Round(double c){
    return round(c*100000)/100000;
}

__global__ void computeCentroids(double *data, double *weights, double *centroids, int numCents, int dRows, int dCols , int fMeasure){
   
    int i = threadIdx.x + blockIdx.x*blockDim.x; // centroid number

    double denominator = 0;
    
    for(int x = 0; x < dRows; x++){
        double w = weights[x*numCents + i]*weights[x*numCents + i];
        denominator = denominator + w;

        for(int k = i*dCols; k < (i+1)*dCols; k++){
            centroids[k] = centroids[k] + w*data[x*dCols + k];
        }
        // cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
    }


    for(int k = i*dCols; k < (i+1)*dCols; k++){
        centroids[k] = centroids[k]*(1/denominator);
    }


}

void initCentroids(double *cents, int numCols){
    for(int i = 0; i < numCols*NUMCENTS; i++ ){
        cents[i] = 0;
    }
}

int main(){


    Data d = Data("../Utils/wine-clustering.csv");
    Data w = Data("../Utils/weights.csv");

    int dataRows = d.getNumRows();
    int dataColumns = d.getNumCols();
    int weightRows = w.getNumRows();
    int weightCols = w.getNumCols();

    int centSize = NUMCENTS*dataColumns*sizeof(double);
    int dataSize = dataRows*dataColumns*sizeof(double);
    int weightSize = dataRows*NUMCENTS*sizeof(double);



    vector<double> flatData = d.getFlat(); // flattened data
    vector<double> flatWeights = w.getFlat(); // flattened weights
    double *centroids = (double*) malloc(NUMCENTS*dataColumns*sizeof(double)); // flattened centroids
    initCentroids(centroids , dataColumns);
   

    // data for device
    double *deviceWeights, *deviceData , *deviceCentroids;

    // allocate space in the device
    checkCudaErrors(cudaMalloc((void**) &deviceData, dataSize));
    checkCudaErrors(cudaMalloc((void**) &deviceWeights, weightSize));
    checkCudaErrors(cudaMalloc((void**) &deviceCentroids, centSize));

    // copying memory
    checkCudaErrors(cudaMemcpy(deviceData , &flatData[0] , dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceWeights , &flatWeights[0] , weightSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceCentroids , centroids, centSize, cudaMemcpyHostToDevice));

    // Compute the centroids

    dim3 block(1);
    dim3 threads(NUMCENTS);

    computeCentroids<<<threads,block>>>(deviceData, deviceWeights, deviceCentroids, NUMCENTS, dataRows, dataColumns , FMEASURE);

    // checkCudaErrors(cudaMemcpy(flatWeights , deviceWeights ,  weightSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(centroids, deviceCentroids ,  centSize, cudaMemcpyDeviceToHost));

    
    for(int i = 0; i < NUMCENTS*dataColumns; i++){
        printf("%.5f    ",centroids[i]);
    }
    printf("\n");
    
    
    checkCudaErrors(cudaFree(deviceCentroids));
    checkCudaErrors(cudaFree(deviceWeights));
    checkCudaErrors(cudaFree(deviceData));






    
        
    
   




    return 0;
}




