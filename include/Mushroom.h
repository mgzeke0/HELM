#include <EigenIncl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
using namespace Eigen;
class Mushroom{
	public:
	
	Mushroom();
	
	MatrixXd loadMushroom(int from, int to);
	VectorXd getLabels(int from, int to);

};
