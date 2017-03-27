#include "stdNlBayes.h"

using namespace std;
using namespace Eigen;

StdNlBayes::StdNlBayes()
{

}

/* 
Load an image into the class
*/
int StdNlBayes::loadImage(const string filePath)
{
  //Load an image
  std::vector<unsigned char> imageLine;
  unsigned width, height;
  unsigned error = lodepng::decode(imageLine,width,height,filePath.c_str());
  if(error != 0)
  {
    cerr << " ERROR : Image located at : " << filePath << " could not be loaded." << endl;
    return EXIT_FAILURE;
  }
  
  //Allocate memory
  for(int channel = 0;channel<3;channel++)
    image[channel] = ImageChannel::Zero(width,height);


  //Store it
  for(int i=0;i<imageLine.size()/4;i++)
  {
    int line = i % width;
    int column = i / width;
    image[0](line,column) = imageLine[4*i];     //R Channel
    image[1](line,column) = imageLine[4*i+1];   //G Channel
    image[2](line,column) = imageLine[4*i+2];   //B Channel
  }

}
