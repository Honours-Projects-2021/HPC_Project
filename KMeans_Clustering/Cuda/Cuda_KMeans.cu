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

#define K 3 // NUMBER OF CLUSTERS
#define EPOCHS  500 // NUMBER OF ITERATIONS

using namespace std;

// This function rounds off to 5 decimal places
__device__ double Round(double c){
    return round(c*100000)/100000;
}

void displayCentroids(double* cent, int NumFeatures){
    for(int i = 0; i < K; i++){
        for(int j = 0; j < NumFeatures; j++){
            printf("%f   ",cent[i*NumFeatures +j]); 
        }
        printf("\n");
    }
}

// sets every value in the array to zero
void initCentroids(vector<double> data,double *cents, int numCols,int numRows){
    srand(time(0));
    for(int i = 0; i < K ; i++){
        int r = rand()%numRows;

        for(int j = 0; j < numCols; j++){
            cents[i*numCols + j] = data[i*numCols +j]; 
        }
    }
}

void initDistances(double *distances, int rows){
    for(int i = 0; i < K*rows; i++){
        distances[i] = 0;
    }
}

void displayDistances(double *distances , int rows){
    for(int i = 0; i < rows*K; i++){
        if(i%K == 0 && i != 0 )
        printf(" -------- %d\n",i/4 + 1);
        printf("%f  ", distances[i]);

    }
    printf("\n");
}



__global__ void calcDistances( double *clusters, double *distances, double *data,int *assigns , int cols){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double max = 10000000;
    for(int l = 0; l < K; l++){
        distances[i*K + l] = 0;

        for(int j = 0; j < cols; j++){
            distances[i*K + l] += pow(data[i*cols +j] - clusters[l*cols +j],2);
        }

        distances[i*K + l] = sqrt(distances[i*K + l]);

        if(distances[i*K + l] < max){
            max = distances[i*K + l] ;
            assigns[i] = l;
        } 
    }
}

__global__ void calcCentroids(int *assigns, double *clusters, double *data, int cols, int rows){
    int c_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int occurances = 0; // starts at one because the cluster itself is a point


    // reset the clusters to zeros
    for(int i = 0; i < cols; i++){
        clusters[c_idx*cols + i] = 0;
    }

    // calculate the occurances 
    for(int i = 0; i < rows; i++){
        if(assigns[i] == c_idx){
            occurances ++;
            for(int j = 0; j< cols; j++){
                clusters[c_idx*cols + j] += data[i*cols +j];
            }
        }
    }

    // Get the avarage of the clusters
    for(int i = 0; i < cols; i++){
        clusters[c_idx*cols + i] = clusters[c_idx*cols+i]/ occurances;
    }
}

int main(){

    // Import in the data
    Data d = Data("../Utils/wine-clustering.csv"); // The complete datasets of 13 columns and 178 records
    int cols = d.getNumCols();
    int rows = d.getNumRows();
    vector<double> data = d.getFlat();
    // TODO: calculate distances
    double *clusters = (double*) malloc(K*cols*sizeof(double));
    initCentroids(data,clusters, cols, rows);
    displayCentroids(clusters , cols);


    double * distances = (double*) malloc(K*rows*sizeof(double));
    initDistances(distances,rows);
    // displayDistances(distances,rows);

    int *clusterAssigns = (int*) malloc(rows*sizeof(int));


    // Create Data variables for device
    double *dev_clusters, *dev_distances, *dev_data;
    int *dev_Assigns;
    checkCudaErrors(cudaMalloc((void**)&dev_clusters,K*cols*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_distances,K*rows*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_Assigns,rows*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_data,cols*rows*sizeof(double)));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(dev_data,&data[0],cols*rows*sizeof(double),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_clusters,&clusters[0],K*cols*sizeof(double),cudaMemcpyHostToDevice));


    dim3 dis_threads(rows);
    dim3 dis_blocks(K);
    dim3 clust_threads(K);

    StopWatchInterface *se_timer = NULL;
    sdkCreateTimer(&se_timer);
    sdkStartTimer(&se_timer);

    for(int i = 0; i < EPOCHS; i++){
        calcDistances<<<1 , dis_threads>>>(dev_clusters,dev_distances,dev_data,dev_Assigns,cols);
        calcCentroids<<<1 , clust_threads>>>(dev_Assigns, dev_clusters, dev_data, cols,rows);
    }

    sdkStopTimer(&se_timer);
    printf("The Cuda Parallel time to run K Means in second is  %f (ms) for %d epochs\n", sdkGetTimerValue(&se_timer),EPOCHS);
    sdkDeleteTimer(&se_timer);

    checkCudaErrors(cudaMemcpy(clusterAssigns,dev_Assigns,rows*sizeof(int),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(distances,dev_distances,K*rows*sizeof(double),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(clusters,dev_clusters,K*cols*sizeof(double),cudaMemcpyDeviceToHost));

    printf("\n = ============================================================== = \n\n");
    displayCentroids(clusters , cols);



    free(clusters);
    free(distances);
    free(clusterAssigns);
    cudaFree(dev_Assigns);
    cudaFree(dev_clusters);
    cudaFree(dev_distances);
    return 0;
}




