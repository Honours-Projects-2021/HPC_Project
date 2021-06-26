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

    Data d = Data();
    // d.display();
    Fuzzy f = Fuzzy(Data().getData(),3,2);


    f.display_weights();
    cout<<endl<<"==========================================================================================="<<endl;
    f.display_weights();
    return 0;
}