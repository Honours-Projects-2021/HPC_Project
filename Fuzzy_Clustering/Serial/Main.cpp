#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include<random>
#include <ctime>

#include "Data.h"
#include "../Utils/Util.h"
#include "Fuzzy.h"

#define EPOCHS 500
#define CLASSES 3
#define FMEASURE 3

using namespace std;

int main(){

    // Import the dataset and the randomly initialized weights
    Data d = Data("../Utils/wine-clustering.csv");
    Data w = Data("../Utils/weights.csv");
    d.getFlat();
    // create fuzzy means
    Fuzzy f = Fuzzy(d.getData(),w.getData(),CLASSES,FMEASURE);
    clock_t start = clock();

    f.run_fuzzy_c_means(EPOCHS); // run the algorithm for a number of epochs
    
    clock_t end = clock() - start;
    double timelapsed = ((double)end)/CLOCKS_PER_SEC;
    printf("The serial time for Fuzzy C Means in  miliseconds is %fms for %d epochs\n", timelapsed*1000,EPOCHS );

    // d.displayFlat();
    return 0;
}