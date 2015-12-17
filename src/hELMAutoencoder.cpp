#include "HELMAutoencoder.h"
VectorXd in;
VectorXd out;
//MatrixXd b;
MatrixXd H;
VectorXd bias;
int inputNodes = 3;
int hiddenNodes = 30;
int outputNodes = 3;

HELMAutoencoder::HELMAutoencoder(){}

double returnMaxEigenvalue (MatrixXd e){




    return e.eigenvalues().real().maxCoeff();
}



double sign(double a){
    if((0.0d+a)>=0.0d)
    {
        return 1.0d;

    }
    else return -1.0d;
}

MatrixXd soft(MatrixXd x, double T){
    MatrixXd result(x.rows(),x.cols());
    
 
        for(int i = 0; i< x.rows(); i++){
            for(int j = 0; j< x.cols(); j++){
                result(i,j) = max(abs(x(i,j)) - T, 0.0);
                result(i,j) = sign(x(i,j))*result(i,j);
            }
        }
        
    return result;
}

void saveImg2(VectorXd oneImg, bool color,string imgName){

    std::ofstream myfile;
    string path = "/home/ajeje/H-ELM/" + imgName;
    myfile.open(path.c_str());
    if (color){

        myfile << "P3" << "\n" << 32 << " " << 32 << "\n255" << endl;

        for (int i = 0; i < 3072; i++){
            if(i%96==0){
                myfile << endl;
            }
            unsigned char r,g,b;
            /*r = oneImg(i);
            g = oneImg(1024+i);
            b = oneImg(2048+i);
            myfile << (int) r << " " << (int) g << " " << (int) b << " ";*/
            myfile << abs((int) oneImg(i)) << " ";

        }
    }
    else{
        myfile << "P2" << "\n" << 28 << " " << 28 << "\n255" << endl;

        for (int i = 0; i < oneImg.size(); i++){
            if(i%28==0){
                myfile << endl;
            }
            unsigned char color;
            color = oneImg(i);
            myfile << (int) color << " ";
        }
    }
    myfile.close();
}


MatrixXd HELMAutoencoder::extractFeatures(int j, MatrixXd A, MatrixXd b){

    double lambda = 1e-3;

    MatrixXd AA = (A.transpose())*A;
    cout << A.rows() << " A " << A.cols() << endl;
    cout << AA.rows() << " AA " << AA.cols() << endl;
    //FISTA implementation
    //1 - Calculate Lipschitz constant http://www.eecs.berkeley.edu/~yang/paper/YangA_ICIP2010.pdf  section 2.4
    EigenSolver<MatrixXd> eigvals;
    eigvals.compute(AA,false);
    double Lf = returnMaxEigenvalue(AA);
    double Li = 1/Lf;
    double alp = lambda * Li;
    long m = A.cols();
    long n = b.cols();
    MatrixXd x = MatrixXd::Zero(m,n);
    MatrixXd yk = x;
    int tk = 1;

    MatrixXd L1 = 2 * Li * AA;

    MatrixXd L2 = 2 * Li * A.transpose() * b;
    cout <<"qui" << endl;
    //cout << A.rows() << " " << A.cols() << " " << b.transpose().rows() << " " << b.transpose().cols() << endl;
    //2 - argmin
    for (int i = 0; i< 50; i++){
        //cout << L1.rows()<< " L1 " << L1.cols() << " " << yk.rows()<< " yk " << yk.cols() << endl;
        MatrixXd ck = yk - L1*yk + L2;
        MatrixXd x1 = soft(ck,alp);
        double tk1 = 0.5 + 0.5*sqrt(1+(4*tk*tk));
        double tt = (tk-1)/tk1;
        yk = x1 + tt*(x-x1);
        tk = tk1;
        x = x1;
    }
    
    return x;
}


