#ifndef DATA
#define DATA

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
        vector<double> flatD;
        

        void setNumCols();
        void setNumRows();
        void split(vector<double> &vec , string str);
        void read();


    public:
        Data(string loc);
        void display();
        vector<vector<double>> getData();
        int getNumRows();
        int getNumCols();
        void flat();
        vector<double> getFlat();


};

void Data::flat(){
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            flatD.push_back(data.at(i).at(j));
        }
    }
   
}

vector<double> Data::getFlat(){
    return flatD;
}

Data::Data(string loc){
    location = loc;
    read();
    setNumCols();
    setNumRows();
    flat();
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
    
    int controller = 0;

    string line;
    while(true){

        

        vector<double> vec;
        getline(inputFile , line);
       
        if(controller == 0 || line == "" || line == " ") {
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

int Data::getNumCols(){
    return numCols;
}


int Data::getNumRows(){
    return numRows;
}
#endif