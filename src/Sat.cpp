	
#include <Sat.h>	
	
MatrixXd dataS(4435,36);
VectorXd labelsS;


void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}


void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

Sat::Sat(std::string path,int lines){
	dataS.resize(lines,37);
    //string path = "/home/ajeje/HELM_MNIST/sat/sat.trn";
  
 
	std::fstream myfile(path.c_str(), std::ios_base::in);
	if(myfile.is_open()){
    float a;
    int i =0;
    string line;
   while (std::getline(myfile, line)){
		istringstream iss(line);
		char c;
			int j = 0;
    
    while (iss >> a){
       
       dataS(i,j) = a;
   
       j++;
	}	

	i++;
}

	labelsS = dataS.col(dataS.cols()-1);
} else cerr << "file " << path << " not opened" << endl;
}

MatrixXd Sat::loadSat(){
	
	removeColumn(dataS,dataS.cols()-1);
	return dataS; 
}

VectorXd Sat::getLabels(){ 
	return labelsS;
}

