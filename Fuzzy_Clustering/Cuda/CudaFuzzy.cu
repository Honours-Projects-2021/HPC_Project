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
        // cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
    }


    for(int k = i*dCols; k < (i+1)*dCols; k++){
        centroids[k] = centroids[k]*(1/denominator);
    }


}


__global__ void computeWeights(double *data, double *weights, double *centroids, int numCents,int dRows, int dCols , int fMeasure){

    int i = threadIdx.x ; // datapoint
    int j = blockIdx.x; // cluster

    double w = 0;
    if(i < dRows)
    for(int k = 0; k < numCents; k++){
        double numerator = 0;
        double denominator = 0;

        for(int l = 0; l < dCols; l++){
            double dPoint = data[i*dCols+l];
            numerator += pow((dPoint - centroids[j*dCols+l]),2);
            denominator += pow((dPoint - centroids[k*dCols+l]),2);
            
        }
        numerator = sqrt(numerator);
        denominator = sqrt(denominator);
        w += pow((numerator/denominator),(2/(fMeasure-1)));
    }
    if(i < dRows)
    weights[i*numCents + j] = 1/w;
}

void initCentroids(double *cents, int numCols){
    for(int i = 0; i < numCols*NUMCENTS; i++ ){
        cents[i] = 0.0;
    }
}

void initWeights(vector<double> flatWeights, double *weights){
    for(int i = 0; i < flatWeights.size(); i++){
        weights[i] = flatWeights.at(i);
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
    double *weights = (double*) malloc(dataRows*dataColumns*sizeof(double));
    initWeights(flatWeights , weights);
    initCentroids(centroids , dataColumns);
   
    // for(int i = 0; i < 15; i++){
    //     printf("%.5f    ",weights[i]);
    // }
    // printf("\n");

    // data for device
    double *deviceWeights, *deviceData , *deviceCentroids;

    // allocate space in the device
    checkCudaErrors(cudaMalloc((void**) &deviceData, dataSize));
    checkCudaErrors(cudaMalloc((void**) &deviceWeights, weightSize));
    checkCudaErrors(cudaMalloc((void**) &deviceCentroids, centSize));

    // copying memory
    checkCudaErrors(cudaMemcpy(deviceData , &flatData[0] , dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceWeights , weights , weightSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceCentroids , centroids, centSize, cudaMemcpyHostToDevice));

    // Compute the centroids

    dim3 Cblock(1);
    dim3 Cthreads(NUMCENTS);

    dim3 Wblock(NUMCENTS);
    dim3 Wthreads(dataRows);

    for(int i = 0; i < 50; i++){
        computeCentroids<<<Cblock,Cthreads>>>(deviceData, deviceWeights, deviceCentroids, NUMCENTS, dataRows, dataColumns , FMEASURE);
        computeWeights<<<Wblock,Wthreads>>>(deviceData, deviceWeights, deviceCentroids, weightCols, dataRows, dataColumns , FMEASURE);
    }
    
    checkCudaErrors(cudaMemcpy(centroids, deviceCentroids ,  centSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(weights, deviceWeights ,  weightSize, cudaMemcpyDeviceToHost));

    
    // for(int i = 175*weightCols; i < dataRows*weightCols; i = i+1){
    //     printf("%.5f    %d\n",weights[i],i-177);
    // }
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            printf("%.5f " ,weights[i*NUMCENTS + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    checkCudaErrors(cudaFree(deviceCentroids));
    checkCudaErrors(cudaFree(deviceWeights));
    checkCudaErrors(cudaFree(deviceData));






    
        
    
   




    return 0;
}




