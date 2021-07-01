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
        else if(numProc > 1){
            for(int i = 1; i < CENTROIDS; i++){
                MPI_Recv( &(centroids[i*numFeatures]) , numFeatures , MPI_DOUBLE , i , 1 , MPI_COMM_WORLD , &status);
            }
        }
    }
    MPI_Bcast( &(centroids[0]) , CENTROIDS*numFeatures , MPI_DOUBLE , MASTER , MPI_COMM_WORLD);    
    MPI_Barrier( MPI_COMM_WORLD);
    
    if(rank == numProc-1)
        for(int i =0; i < 3*numFeatures; i++){
                printf("%f --- " , centroids[i]);
            if((i+1)%numFeatures == 0)
                printf("\n");
        }
    // printf("Hello from task %d on %s!\n",rank,hostname);
    // if(rank == 0){
    //     // printf("MASTER: Number of MPI tasks is: %d\n",numProc);
    // }
    MPI_Finalize();

    //

    return 0;
}