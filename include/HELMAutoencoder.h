#ifndef HELMAUTOENCODER_H
#define HELMAUTOENCODER_H

#include <fstream>
#include <EigenIncl.h>

#include <boost/math/special_functions/sign.hpp>



using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class HELMAutoencoder{
public:
    HELMAutoencoder();
    MatrixXd extractFeatures(int j, MatrixXd input, MatrixXd b);
};
#endif

