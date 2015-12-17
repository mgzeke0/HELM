#include <EigenIncl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

using namespace std;
using namespace Eigen;
class Sat{
	
	public:
	Sat(string path,int lines);
	MatrixXd loadSat();
	VectorXd getLabels();
};
