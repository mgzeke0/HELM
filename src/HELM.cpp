#include <HELM.h>
#include <iostream>
#include <Stat.h>
#include <fstream>
#include <HELMAutoencoder.h>
using namespace std;
Stat stat;
double l3 = 1;

MatrixXd activateSigmoid(MatrixXd result){
    MatrixXd activation(result.rows(),result.cols());
    for(int i = 0; i< activation.rows(); i++){
        for(int j = 0; j<activation.cols(); j++){
            double x = result(i,j);
            //fast sigmoid
            activation(i,j) = x/(1.0d+(double)abs(x));
            //normal one
            //activation(j) = (1 / (1+(exp(-result(j)))));
        }
    }

    return activation;
}

MatrixXd trainElm(MatrixXd in, MatrixXd labelM,MatrixXd randomInputsW, int L3,ofstream &myfile){
 
    MatrixXd A = in*randomInputsW;
        
    l3 = A.maxCoeff();
    l3 = 1/l3;

    MatrixXd H = activateSigmoid(A*l3);
    
    MatrixXd a = H.transpose()*H;
    cout << "H " << H.rows() << " " << H.cols() << " label " << labelM.rows() << " " << labelM.cols() << endl;
    double c = pow(2,-30);
    MatrixXd AA = (a+MatrixXd::Ones(H.cols(),H.cols())*c);
    MatrixXd B;
    MatrixXd b = H.transpose()*labelM;
    B = AA.llt().solve(b);

    cout << "ok" <<endl;

    MatrixXd result = H*B;
    double accuracy = 0;
    MatrixXf::Index labIndex, resIndex;
    for(int i = 0; i< result.rows(); i++){
        int n;
        labelM.row(i).maxCoeff(&labIndex);
        result.row(i).maxCoeff(&resIndex);

        if(resIndex==labIndex)
            accuracy++;
    }

    float train = accuracy/labelM.rows()*100;
    cout << "Training accuracy " << train << "%" << endl;
    myfile << train << ",";
    return B;
}



HELM::HELM(MatrixXd IN, MatrixXd labelM, MatrixXd INt, MatrixXd OUTl, int L1, int L2, int L4, int inputNum, int imageSize){
	ofstream myfile;
	myfile.open ("results.txt");
    myfile << "accuracy, 1layer, 2layer, 3layer, c = 2^-30"<< endl;
    int iterations = 50; //iterations for autoencoder
	HELMAutoencoder* Henc = new HELMAutoencoder();
    MatrixXd b = MatrixXd::Random(imageSize+1,L1);		
		
	//1 layer
	IN = stat.zscores(IN);
	IN.conservativeResize(NoChange, IN.cols()+1);
	//add bias	
	IN.col(IN.cols()-1) = VectorXd::Ones(IN.rows());	
		
    MatrixXd Ba1 = Henc->extractFeatures(iterations,(IN*b),IN);
    MatrixXd I = activateSigmoid(IN*Ba1.transpose());
    cout << "First Hidden layer size " << I.rows() << " " << I.cols() << endl;	
    
    
    //2 layer
    b = MatrixXd::Random(L1+1,L2);
    I.conservativeResize(NoChange, I.cols()+1);	
    //add bias	
	I.col(I.cols()-1) = VectorXd::Ones(I.rows());
	
    MatrixXd Ba2 = Henc->extractFeatures(iterations,(I*b),I);
    MatrixXd III = activateSigmoid(I*Ba2.transpose());
    
           
    //ELM
    cout << "start ELM" << endl;
    MatrixXd HB = MatrixXd::Random(III.cols()+1,L4);
	III.conservativeResize(NoChange, III.cols()+1);
	//add bias	
	III.col(III.cols()-1) = VectorXd::Ones(III.rows());
	

	MatrixXd B = trainElm(III,labelM,HB, L4, myfile);
    //  testNetwork
    cout << "start testing.." << endl;

    
	//layer 1
	INt = stat.zscores(INt);
    INt.conservativeResize(NoChange,INt.cols()+1);
    //add bias	
	INt.col(INt.cols()-1) = VectorXd::Ones(INt.rows());
    MatrixXd H1 = activateSigmoid(INt*Ba1.transpose());    
    
    
    //layer 2
    H1.conservativeResize(NoChange, H1.cols()+1);	
    //add bias	
	H1.col(H1.cols()-1) = VectorXd::Ones(H1.rows());
    MatrixXd H3 = activateSigmoid(H1*Ba2.transpose());
	
    //ELM
    H3.conservativeResize(NoChange, H3.cols()+1);
    //add bias		
	H3.col(H3.cols()-1) = VectorXd::Ones(H3.rows());
    MatrixXd H4 = activateSigmoid(H3*HB*l3);
    MatrixXd resultTest = H4*B;    
	
    //test accuracy
     double accuracy = 0;
    MatrixXf::Index labIndex, resIndex;
    for(int i = 0; i< resultTest.rows(); i++){
        int n;
        OUTl.row(i).maxCoeff(&labIndex);
        resultTest.row(i).maxCoeff(&resIndex);

        if(resIndex==labIndex)
            accuracy++;
    }

    float train = accuracy/OUTl.rows()*100;
    cout << "Testing accuracy " << train << "%" << endl;
    myfile << train << endl;
	
}
