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

MatrixXd prepareOutputData(VectorXd label,int numOfClasses){
    MatrixXd labelM = MatrixXd::Zero(label.size(),numOfClasses) - MatrixXd::Ones(label.size(),numOfClasses);

    for(int i = 0 ; i<label.size(); i++){
        labelM(i,label(i)) = 1;
    }
    

    return labelM;
}

void removeColumn2(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}



void saveImg(VectorXd oneImg, bool color,string imgName, int row){

    ofstream myfile;
    string path = "/home/ajeje/HELM_MNIST/" + imgName;
    myfile.open(path.c_str());
    if (color){
        myfile << "P3" << "\n" << 32 << " " << 32 << "\n255" << endl;
        for (int i = 0; i < 3072; i++){
            if(i%96==0){
                myfile << endl;
            }
            /* unsigned char r,g,b;
            r = oneImg(i);
            g = oneImg(1024+i);
            b = oneImg(2048+i);
            myfile << (int) r << " " << (int) g << " " << (int) b << " ";*/
            myfile << abs((int) oneImg(i)) << " ";
        }
    }
    else{
        myfile << "P2" << "\n" << row << " " << row << "\n255" << endl;
        for (int i = 0; i < oneImg.size(); i++){
            if(i%row==0){
                myfile << endl;
            }
            unsigned char color;
            color = oneImg(i);
            myfile << (int) color << " ";

        }
    }
    myfile.close();

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
    for(int k = 0; k< 50 ; k++){
	sleep(1);
    srand(std::time(0));
    int inputNum = 25000;
    int L1 = 700;
    int L2 = 700;
    int L4 = 5000;
    vector<vector<double>> arr;
    vector<double> labelArr;

    
    int imageSize = 784; //width*height
	//prepare training data
	MatrixXd IN;
	LoadMNIST m;
	//Sat s("/home/ajeje/HELM_MNIST/sat/sat.trn",4435);
	m.readMNIST(inputNum,imageSize,arr,"/home/ajeje/HELM_MNIST/MNIST/train-images.idx3-ubyte");
	m.readLabel(inputNum,labelArr,"/home/ajeje/HELM_MNIST/MNIST/train-labels.idx1-ubyte");
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
    mt.readMNIST(10000,784,arrTest,"/home/ajeje/HELM_MNIST/MNIST/t10k-images.idx3-ubyte");
    INt = prepareData(arrTest);		
	m.readLabel(10000,labelArrTest,"/home/ajeje/HELM_MNIST/MNIST/t10k-labels.idx1-ubyte");
	VectorXd OUTlV = VectorXd(labelArrTest.size());
	for(int i = 0; i< labelArrTest.size(); i++){
		OUTlV(i) = labelArrTest[i];
	}
	OUTl = prepareOutputData(OUTlV,10);
	
	
	//Constructir of class launches training and testing of the network
	HELM h(IN,labelM,INt,OUTl,L1,L2,L4,inputNum,imageSize);
	
}
    return 0;

}


