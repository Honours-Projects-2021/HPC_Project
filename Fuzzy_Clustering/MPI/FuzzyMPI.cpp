#include <iostream>
#include <stdio.h>
#include <vector>
#include "Data.h"
#include <mpi.h>
#include <cmath>
#include "../Utils/Util.h"

#define MASTER 0
#define CENTROIDS 3
#define FMEASURE 3
#define EPOCHS 10
using namespace std;

void init_centroids(vector<double> *clusters ,int size ){

    for(int i = 0; i < CENTROIDS*size; i++){    
        clusters->push_back(0);
    }

}

void displayCentroids(vector<double> cent, int NumFeatures);
void displayWeights(vector<double> weights, int n);


int main(int argc, char *argv[]){

    Data d = Data("../Utils/wine-clustering.csv");
    Data w = Data("../Utils/weights.csv");
    vector<double> data = d.getFlat();
    vector<double> weights = w.getFlat();

    
    int numRecords = d.getNumRows();
    int numFeatures = d.getNumCols();

    vector<double> centroids;
    init_centroids(&centroids, numFeatures);

    int numProc, rank, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];


    MPI_Init( &argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Get_processor_name( hostname , &len);
    MPI_Status status;
    MPI_Request request;
    // Calculating the centroids

    int myrows = numRecords/numProc;
    int chunksize = myrows;
    int idx = rank*myrows;
    if(rank == numProc-1){
        chunksize += numRecords - myrows*numProc;
    }

    
    clock_t start = clock();

    // Running the algorithm here... Time from here
    for(int z = 0; z < EPOCHS; z++){
        if(rank < CENTROIDS){
            
           

            for(int i = rank; i < CENTROIDS; ++i){
                 for(int k = i*numFeatures; k < (i+1)*numFeatures; k++){
                    centroids[k] = 0;
                }
                

                double denominator = 0;
            
                for(int x = 0; x < numRecords; x++){
                    double w = pow(weights[x*CENTROIDS+rank],FMEASURE);
                    denominator += w;

                    for(int k = i*numFeatures; k < (i+1)*numFeatures; k++){
                        centroids[k] = (centroids[k] + w*data[x*numFeatures +k]);
                    }                
                }

                for(int k = i*numFeatures; k < (i+1)*numFeatures; k++){
                    centroids[k] = (centroids[k]*(1/denominator));
                }
            
            }

            if(rank != MASTER)
                MPI_Send( &(centroids[rank*numFeatures]) , numFeatures , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
            else if (numProc > 1){
                int index = (numProc > 2)?CENTROIDS:numProc;
                for(int i = 1; i < index; i++){
                    MPI_Recv( &(centroids[i*numFeatures]) , numFeatures , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
                }
            }
        }
        MPI_Barrier( MPI_COMM_WORLD);
        MPI_Bcast( &(centroids[0]) , CENTROIDS*numFeatures , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);    
        
        /*=========================================================================================================================*/
        /*=========================================================================================================================*/
        /*=========================================================================================================================*/
        

        // printf("ID: %d, my number of rows is %d, my index is %d, my chuncksize is %d\n",rank, myrows,idx,chunksize);

        for(int i = idx; i < idx + chunksize; i++){
            
            for(int j = 0; j < CENTROIDS; j++){
                double w = 0;
                for(int k = 0; k < CENTROIDS; k++){
                    double numerator = 0;
                    double denominator = 0;

                    for(int l = 0; l < numFeatures; l++){
                        numerator += pow(data[i*numFeatures+l]-centroids[j*numFeatures+l],2);
                        denominator += pow(data[i*numFeatures+l]-centroids[k*numFeatures+l],2);
                    }
                    numerator = sqrt(numerator);
                    denominator = sqrt(denominator);
                    w += pow((numerator/denominator),2/(FMEASURE-1));
                }
                weights[i*CENTROIDS+j] = 1/w;
            }
        }

        if(rank != MASTER)
                MPI_Send( &(weights[idx*w.getNumCols()]) , chunksize*w.getNumCols() , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
        else if(numProc>1){
            for(int i = 1; i < numProc-1; i++){
                MPI_Recv( &(weights[i*myrows*w.getNumCols()]) , chunksize*w.getNumCols() , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
            } 
            MPI_Recv( &(weights[(numProc-1)*myrows*w.getNumCols()]) , (chunksize+w.getNumRows() - myrows*numProc)*w.getNumCols() , MPI_DOUBLE , numProc-1 , 1 , MPI_COMM_WORLD , &status);
        }

        MPI_Bcast( &(weights[0]) , CENTROIDS*numRecords , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);    
        MPI_Barrier( MPI_COMM_WORLD);
        
    }

    // End timer here 
    if(rank == 0){
        displayCentroids(centroids,numFeatures);
        clock_t end = clock() - start;
        double timelapsed = ((double)end)/CLOCKS_PER_SEC;
        printf("\nThe MPI parallel time for Fuzzy C Means in  miliseconds is %fms for %d epochs\n", timelapsed*1000,EPOCHS );
    }
    MPI_Finalize();
    return 0;
}

void displayCentroids(vector<double> cent, int NumFeatures){
    for(int i = 0; i < CENTROIDS; i++){
        cout << "cluster "<<i+1<<" [ ";
        for(int j = 0; j < NumFeatures; j++){
            if(j != NumFeatures-1)
                printf("%f, ",cent[i*NumFeatures +j]); 
            else
                printf("%f ]\n",cent[i*NumFeatures +j]);

        }
    }
}

void displayWeights(vector<double> weights, int n){
    for(int i = 0; i < n; i++){
        cout << "weight for data-point "<<i+1<<" : [ ";
        for(int j = 0; j < CENTROIDS; j++){
            if(j != CENTROIDS-1)
                printf("%f, ",weights[i*CENTROIDS +j]); 
            else
                printf("%f ]\n",weights[i*CENTROIDS +j]);

        }
    }

}
