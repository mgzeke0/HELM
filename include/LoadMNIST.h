#ifndef LOADMNIST_H
#define LOADMNIST_H



#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class LoadMNIST{


public:
    LoadMNIST();
void readLabel(long numberoflabels, std::vector<double> &l, std::string path);
void readMNIST(long NumberOfImages, int DataOfAnImage, std::vector<std::vector<double> > &arr, std::string path);


    int reverseInt(int i);

};
#endif
