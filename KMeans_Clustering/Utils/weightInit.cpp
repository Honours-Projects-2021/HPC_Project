#include<iostream>
#include<random>
#include<fstream>


using namespace std;

int main(){

    fstream weights;
    weights.open("./weights.csv",ios::out);
    weights << "These are the initial weights" << endl;
    
    int r = 100000; // round value
    int centroids = 3;
    int numDataPoints = 178;

    for(int i = 0; i < numDataPoints; i++){
        vector<double> vec;
        for(int j = 0; j < centroids; j++){
            if(j != centroids-1)
                weights << round(((double) rand()/(RAND_MAX))*r)/r<<",";
            else
                weights << round(((double) rand()/(RAND_MAX))*r)/r;

        }
        if(i != 178 -1)
            weights << endl;
    }



    return 0;
}