#include <Mushroom.h>

using namespace std;
using namespace Eigen;

MatrixXd data;
VectorXd labels;

Mushroom::Mushroom(){
	data.resize(8124,23);
    string path = "/home/ajeje/HELM_MNIST/mushroom/agaricus-lepiota.data";
    ifstream myfile(path.c_str());
	int i=0;
	string line;
	while (std::getline(myfile, line)){
		istringstream iss(line);
		char c;
			int j = 0;
			while (iss.get(c)){
				if(c!=','){
					data(i,j) = (double)c;
					//cout << (char)data(i,j);
					j++;
				}			
		}
		//cout << endl;
		i++;
	}
	
	labels = data.col(0);
}


VectorXd Mushroom::getLabels(int from, int to){

	VectorXd l(to-from);
	
	int j = from;
	for(int i = 0; i<l.size(); i++){
		if(j >= to)
			break;
		if((char)labels(j) == 'e')
			l(i) = 0;
		else if((char)labels(j) == 'p')
			l(i) = 1;
			else cout << "impossibile leggere label" << endl;
		j++;
	}
	return l;
}

MatrixXd Mushroom::loadMushroom(int from, int to){
	MatrixXd IN(to-from,23);
	int j = 0;
	for(int i = from ; i <to;i++){
		IN.row(j) = data.row(i);
		j++;
	}

	MatrixXd IN2(IN.rows(),IN.cols()-1);
	for(int i = 0; i< IN2.cols();i++){
		IN2.col(i) = IN.col(i+1);
	}

	return IN2;
}
