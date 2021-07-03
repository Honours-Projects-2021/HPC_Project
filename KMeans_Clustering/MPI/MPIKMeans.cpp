#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "Data.h"
#include <mpi.h>
#include <cmath>
#include "../Utils/Util.h"

#define MASTER 0
#define K 3
#define EPOCHS 200
using namespace std;

void displayAssigns(int * assigns, int rows){
    for(int i = 0; i < rows; i++){
        printf("%d --- ", assigns[i]);
    }
}

void displayCentroids(double* cent, int NumFeatures){
    for(int i = 0; i < K; i++){
        for(int j = 0; j < NumFeatures; j++){
            printf("%.3f   ",cent[i*NumFeatures +j]); 
        }
        printf("\n");
    }
}

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


int main(int argc, char *argv[]){

    Data d = Data("../Utils/wine-clustering.csv");
    vector<double> data = d.getFlat(); // one dimensional representation of the data in vector form
    int rows = d.getNumRows();
    int cols = d.getNumCols();

    // clusters, distances and assigns
    double* clusters = (double*) malloc(K*cols*sizeof(double));  // K x m is the size
    double* distances = (double*) malloc(rows*K*sizeof(double)); // n x K is the size
    int* assigns =(int*) malloc(rows*sizeof(int));      // n x 1 is the size 

    for(int i = 0; i < rows*K; i++){
        distances[i]=0;
    }
    // displayDistances(distances , rows);
    initCentroids(data, clusters,cols,rows);
    // displayCentroids(clusters,cols);
  
    int numProc, rank;

    MPI_Init( &argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Status status,status2;
    MPI_Request request;

    int myrows = rows/numProc;
    int chunksize = myrows;
    int idx = rank*myrows;
    if(rank == numProc-1){
        chunksize += rows - myrows*numProc;
    }

    // printf("ID: %d, my number of rows is %d, my index is %d, my chuncksize is %d\n",rank, myrows,idx,chunksize);
    for(int z = 0; z < EPOCHS; z++){
        
        for(int i = idx; i < idx+chunksize; i++){
            double sum = 100000;
            for(int j = 0; j < K; j++){
                
                distances[i*K+j] = 0;
                for(int k = 0; k < cols; k++){
                    distances[i*K+j] += pow(data[i*cols+k]-clusters[j*cols +k],2);
                }

                distances[i*K+j] = sqrt(distances[i*K+j]);

                if(distances[i*K+j] < sum){
                    assigns[i] = j;
                    sum = distances[i*K+j];
                }
            }
        }

        if(rank != MASTER){
            MPI_Send( &distances[idx*K] , chunksize*K , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
            MPI_Send( &assigns[idx] , chunksize, MPI_INT, MASTER , 2 , MPI_COMM_WORLD);
        }
        else if(numProc>1){
            for(int i = 1; i < numProc-1; i++){
                MPI_Recv( &distances[i*myrows*K] , chunksize*K, MPI_DOUBLE , i , 1, MPI_COMM_WORLD , &status);
                MPI_Recv( &assigns[i*myrows] , chunksize , MPI_INT , i , 2 , MPI_COMM_WORLD , &status2);

            } 

            MPI_Recv( &distances[(numProc-1)*myrows*K] , (rows+rows - myrows*numProc)*K , MPI_DOUBLE , numProc-1 , 1 , MPI_COMM_WORLD , &status);
            MPI_Recv( &assigns[(numProc-1)*myrows] , (rows + rows - myrows*numProc) , MPI_INT , numProc-1 , 2 , MPI_COMM_WORLD , &status2);

        }

        

        MPI_Bcast( &distances[0] , K*rows , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);  
        MPI_Bcast( &assigns[0] , rows , MPI_INT , MASTER , MPI_COMM_WORLD);  
        // MPI_Barrier( MPI_COMM_WORLD);

        if(rank < K){

            // Each i corresponds to the each cluster
            for(int c_idx = rank; c_idx < K; c_idx ++){
                int occurances = 0;

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

            if(rank != MASTER)
                MPI_Send( &(clusters[rank*cols]) , cols , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
            else if (numProc > 1){
                int index = (numProc > 2)?K:numProc;
                for(int i = 1; i < index; i++){
                    MPI_Recv( &(clusters[i*cols]) , cols , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
                }
            }

        }

        MPI_Bcast( &clusters[0] , cols*K , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);
    }

    if(rank == MASTER){
        // displayAssigns(assigns, rows);

        printf("\n");
        displayCentroids(clusters,cols);

        // displayDistances(distances,rows);
    }
    
    MPI_Finalize();

    free(clusters);
    free(distances);
    free(assigns);
    return 0;
}



