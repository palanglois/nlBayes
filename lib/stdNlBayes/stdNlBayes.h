#ifndef STDNLBAYES_H
#define STDNLBAYES_H

#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <Eigen/Core>
#include "lodepng.h"

typedef Eigen::MatrixXd ImageChannel;
typedef std::array<ImageChannel,3> ImageType;

class StdNlBayes
{
public:
  StdNlBayes();
  int loadImage(const std::string filePath); 
private:
  ImageType image;
};

#endif
