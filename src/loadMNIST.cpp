#include "LoadMNIST.h"

using namespace std;

LoadMNIST::LoadMNIST()
{

}
int LoadMNIST::reverseInt(int i){
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void LoadMNIST::readLabel(long numberoflabels, vector<double> &l, string path){
    ifstream file(path.c_str(),ios::binary);
    if (file.is_open()){

        LoadMNIST* m = new LoadMNIST();
        int magic_number=0;
        int number_of_items=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number = m->reverseInt(magic_number);
        file.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items= m->reverseInt(number_of_items);
        l.resize(numberoflabels);
        for(long i=0; i<l.size(); ++i){

            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            l[i] = (double)temp;
        }

    }
}



void LoadMNIST::readMNIST(long NumberOfImages, int DataOfAnImage, vector<vector<double> > &arr, string path){
    //vector<vector<double> > arr;
  arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file (path.c_str(),ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        
        for(int i=0;i<NumberOfImages;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
           
        }
    }

    // return arr;

}
