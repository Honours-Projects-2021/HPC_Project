#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "Data.h"
#include <mpi.h>
#include <cmath>
#include "../Utils/Util.h"

#define MASTER 0
#define CENTROIDS 3
#define FMEASURE 2

using namespace std;

void init_centroids(vector<double> *clusters ,int size ){

    for(int i = 0; i < CENTROIDS*size; i++){    
        clusters->push_back(0);
    }

}


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
    if(rank < CENTROIDS){
        for(int i = rank; i < CENTROIDS; i++){
          /*  double denominator = pow(weights.at(0).at(i),FMEASURE);

            vector<double> cluster = data.at(0);
        
            for(int x = 1; x < data.size(); x++){
                double w = pow(weights.at(x).at(i),FMEASURE);
                denominator += w;
                cluster = Util().vector_addition(cluster, Util().scalar_multiply(w,data.at(x)));
            }

            centroids.at(i) = Util().scalar_multiply((double)(1/denominator) , cluster);
          */

            double denominator = 0;
            
            for(int x = 0; x < numRecords; x++){
                double w = pow(weights[x*CENTROIDS+i],FMEASURE);
                denominator += w;

                for(int k = i*numFeatures; k < (i+1)*numFeatures; k++){
                    centroids[k] = centroids[k] + w*data[x*numFeatures +k];
                    // centroids[k] += 0;

                }                
            }

            for(int k = i*numFeatures; k < (i+1)*numFeatures; k++){
                centroids[k] = centroids[k]*(1/denominator);
            }
        
        }
        if(rank != MASTER)
            MPI_Send( &(centroids[rank*numFeatures]) , numFeatures , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
        else {
            int index = (numProc > 2)?CENTROIDS:numProc;
            for(int i = 1; i < index; i++){
                MPI_Recv( &(centroids[i*numFeatures]) , numFeatures , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
            }
        }
    }
    MPI_Bcast( &(centroids[0]) , CENTROIDS*numFeatures , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);    
    MPI_Barrier( MPI_COMM_WORLD);
    
/*=========================================================================================================================*/
/*=========================================================================================================================*/
/*=========================================================================================================================*/
    int myrows = d.getNumRows()/numProc;
    int chunksize = myrows;
    int idx = rank*myrows;
    if(rank == numProc-1){
        chunksize += d.getNumRows() - myrows*numProc;
    }

    printf("ID: %d, my number of rows is %d, my index is %d, my chuncksize is %d\n",rank, myrows,idx,chunksize);

    for(int i = idx; i < idx + chunksize; i++){
        
        for(int j = 0; j < CENTROIDS; j++){
            double w = 0;
            for(int k = 0; k < CENTROIDS; k++){
                double numerator = 0;
                double denominator = 0;

                for(int l = 0; l < d.getNumCols(); l++){
                    numerator += pow(data[i*d.getNumCols()+l]-centroids[j*d.getNumCols()+l],2);
                    denominator += pow(data[i*d.getNumCols()+l]-centroids[k*d.getNumCols()+l],2);
                }
                numerator = sqrt(numerator);
                denominator = sqrt(denominator);
                w += pow((numerator/denominator),2/(FMEASURE-1));
            }
            weights[i*CENTROIDS+j] = 1/w;
        }
    }

    
    MPI_Barrier( MPI_COMM_WORLD);

    if(rank != MASTER)
            MPI_Send( &(weights[idx]) , chunksize , MPI_DOUBLE , MASTER , 1 , MPI_COMM_WORLD);
    else {
        for(int i = 1; i < numProc-1; i++){
            MPI_Recv( &(weights[i*myrows]) , chunksize , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
        } 
        MPI_Recv( &(weights[(numProc-1)*myrows]) , chunksize+d.getNumRows() - myrows*numProc , MPI_DOUBLE , numProc-1 , 1 , MPI_COMM_WORLD , &status);
    }

    MPI_Bcast( &(weights[0]) , CENTROIDS*d.getNumRows() , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);    
    
    if(rank == MASTER)
        for(int i = 0; i < 100*CENTROIDS; i++){
                printf("  %.3f --- " , weights[i]);
            if((i+1)%CENTROIDS == 0)
                printf("             %d\n",i/3);
        }

    // printf("Hello from task %d on %s!\n",rank,hostname);
    // if(rank == 0){
    //     // printf("MASTER: Number of MPI tasks is: %d\n",numProc);
    // }
    MPI_Finalize();

    //

    return 0;
}