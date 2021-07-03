#ifndef DATA
#define DATA

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

class Data{

    private:
        int numRows;
        int numCols;
        string location;
        vector<vector<double>> data;

        void setNumCols();
        void setNumRows();
        void split(vector<double> &vec , string str);
        void read();
        vector<double>flatD;

    public:
        Data(string loc);
        void display();
        int getNumRows();
        int getNumCols();
        void flat();
        vector<vector<double>> getData();
        vector<double> getFlat();
        void displayFlat();





};

Data::Data(string loc){
    location = loc;
    read();
    setNumCols();
    setNumRows();
}

void Data::flat(){
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            flatD.push_back(data.at(i).at(j));
        }
    }
   
}

vector<double> Data::getFlat(){
    flat();
    return flatD;
}

void Data::displayFlat(){
    for(int i = 0; i < numCols*numRows; i++){
        if(i%numCols == 0){
            printf("\n");
        }
        printf("%f ", flatD[i]);
    }
}

void Data::display(){
    int n = numRows;
    int m = numCols;

    for(int i = 0; i < n; i++){
        vector<double> vec = data.at(i);
        for(int j = 0; j < m; j++){
            cout<< vec.at(j) << " ";
        }
        cout<< endl;
    }
}

void Data::read(){

    fstream inputFile;
    inputFile.open(location,ios::in);

    if(!inputFile){
        cout<<"can't find anything here\n";
        exit;
    }
    
    int controller = 0;

    string line;
    while(true){

        

        vector<double> vec;
        getline(inputFile , line);
       
        if(controller == 0) {
            controller ++;
            continue;
        }

        split(vec,line);
        data.push_back(vec);
         if(inputFile.eof()){
            break;
        }
    }

    inputFile.close();
}

void Data::setNumCols(){
    numCols = data.at(0).size();
}

void Data::setNumRows(){
    numRows = data.size();
}
int Data::getNumCols(){
    return numCols ;
}

int Data::getNumRows(){
    return numRows = data.size();
}
void Data::split(vector<double> &vec, string str){

    int start = 0;
    string del = ",";
    int end = str.find(del);
    while(end != -1){
        vec.push_back(stod(str.substr(start, end-start)));
        start = end + del.size();
        end = str.find(del,start);
    }
    vec.push_back(stod(str.substr(start, end-start)));
}

vector<vector<double>> Data::getData(){
    return data;
}

#endif