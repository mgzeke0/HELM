#include <Stat.h>
using namespace std;


Stat::Stat()
{

}

VectorXd Stat::zscoresA(VectorXd data, int n){
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i){
        mean=mean+data(i);
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data(i)-mean)*(data(i)-mean);
    double deviation = sqrt(sum_deviation/n);
    
    for(i=0;i<n;++i){
		
	data(i) = (data(i)-mean)/deviation;
	
	}
    
    return data;           
}

MatrixXd Stat::zscores(MatrixXd IN){
		for(int j = 0; j< IN.rows(); j++){
			IN.row(j) = zscoresA(IN.row(j),IN.row(j).size());	
		}	
	return IN;
}

VectorXd Stat::normalize(VectorXd V){

    double max = V.maxCoeff();
    double min = V.minCoeff();

    for(int i = 0; i<V.size(); i++){

        V(i) = (V(i)-min)/(max-min)*2 -1;

    }

    return V;
}

MatrixXd Stat::normalizeM(MatrixXd M){
    for(int i = 0; i<M.rows(); i++){
        M.row(i) = normalize(M.row(i));       		
	}
    return M;
}


