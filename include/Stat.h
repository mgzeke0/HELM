
#include <EigenIncl.h>
using namespace Eigen;
class Stat{
public:
Stat();
VectorXd zscoresA(VectorXd data, int n);
MatrixXd zscores(MatrixXd IN);
VectorXd normalize(VectorXd V);
MatrixXd normalizeM(MatrixXd M);
};
