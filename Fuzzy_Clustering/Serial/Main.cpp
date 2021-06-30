#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include<random>
#include <ctime>

#include "Data.h"
#include "../Utils/Util.h"
#include "Fuzzy.h"

using namespace std;

int main(){

    Data d = Data("../Utils/wine-clustering.csv");
    Data w = Data("../Utils/weights.csv");
    // w.display();
    // d.display();
    Fuzzy f = Fuzzy(d.getData(),w.getData(),3,2);


    // f.display_weights();
    f.run_fuzzy_c_means(50);
    // cout<<endl<<"==========================================================================================="<<endl;
    f.display_weights();
    cout<<endl<<"===========================================================================================";
    cout<<endl<<"==========================================================================================="<<endl;

    return 0;
}