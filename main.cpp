#include <EigenIncl.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <fstream>
#include <ctime>
#include <LoadMNIST.h>
#include <time.h>
#include <Mushroom.h>
#include <Sat.h>
#include <HELM.h>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;
int randomImage = 5;


//puts a vector of labels into a matrix of labels for classification, parameter numOfClasses is used to define the number of labels/classes
MatrixXd prepareOutputData(VectorXd label,int numOfClasses){
    MatrixXd labelM = MatrixXd::Zero(label.size(),numOfClasses) - MatrixXd::Ones(label.size(),numOfClasses);


    for(int i = 0 ; i<label.size(); i++){
		/*for MNIST dataset is
		labelM(i,label(i)) = 1;*/
        labelM(i,label(i)-1) = 1;
    }
    

    return labelM;
}


MatrixXd prepareData(vector<vector<double>> arr){
    long c = 0;
    VectorXd input(arr.size()*arr[0].size());
    for(int i = 0; i<arr.size(); i++){
        for(int j = 0; j<arr[0].size(); j++){
            input(c) = arr[i][j];
            c++;
        }
    }

    MatrixXd IN = MatrixXd::Map(input.data(),arr[0].size(),arr.size());
    return IN.transpose();
}

int main(int argc, char** argv){
		
    Eigen::setNbThreads(4);
	VectorXd label;
	
	int number_of_runs = 50;
	
    for(int k = 0; k < number_of_runs ; k++){
	sleep(1);
    srand(std::time(0));
    int inputNum = 4435;
    int L1 = 50;
    int L2 = 50;
    int L4 = 800;
    vector<vector<double>> arr;
    vector<double> labelArr;

	


    
    int dataSize = 36; //for example width*height for images
    
    
    
	//prepare training data
	MatrixXd IN;
	Sat s("sat/sat.trn",inputNum);
	IN = s.loadSat();
	label = s.getLabels();
	MatrixXd labelM = prepareOutputData(label,7);
	
	
	//test data
	MatrixXd INt;
    MatrixXd OUTl;
	Sat sTest("sat/sat.tst",2000);
	INt = sTest.loadSat();
	VectorXd OUTV = sTest.getLabels();
	OUTl = prepareOutputData(OUTV,7);
	
	
	
	
	
	/*
	LoadMNIST m;
	
	m.readMNIST(inputNum,imageSize,arr,"MNIST/train-images.idx3-ubyte");
	m.readLabel(inputNum,labelArr,"MNIST/train-labels.idx1-ubyte");
	label = VectorXd(labelArr.size());
	
	for(int i = 0; i< labelArr.size(); i++){
		label(i) = labelArr[i];
	}
	
	MatrixXd labelM = prepareOutputData(label,10);
	IN = prepareData(arr);	
	//prepare testing data
    vector<vector<double>> arrTest;
    vector<double> labelArrTest;
    MatrixXd INt;
    MatrixXd OUTl;
    LoadMNIST mt;
    mt.readMNIST(10000,784,arrTest,"MNIST/t10k-images.idx3-ubyte");
    INt = prepareData(arrTest);		
	m.readLabel(10000,labelArrTest,"MNIST/t10k-labels.idx1-ubyte");
	VectorXd OUTlV = VectorXd(labelArrTest.size());
	for(int i = 0; i< labelArrTest.size(); i++){
		OUTlV(i) = labelArrTest[i];
	}
	OUTl = prepareOutputData(OUTlV,10);
	*/
	
	//Constructor of class launches training and testing of the network
	
	HELM h(IN,labelM,INt,OUTl,L1,L2,L4,inputNum,dataSize);
	
}
    return 0;

}


