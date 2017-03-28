#ifndef STDNLBAYES_H
#define STDNLBAYES_H

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "lodepng.h"

typedef Eigen::MatrixXd ImageChannel;
typedef std::array<ImageChannel,3> ImageType;

class StdNlBayes
{
public:
  StdNlBayes(double _sigma);
  //Load an image
  int loadImage(const std::string filePath); 
  //Run the algorithm
  void run();
  //Convert an image from RGB to YUV color system
  ImageType rgb2yuv(ImageType src, bool inverse = false) const;
  //Create a 0 image
  ImageType zeroImage(int rows, int cols) const;
private:
  ImageType image;
  ImageType image_yuv;
  Eigen::Matrix3d rgb2yuv_mat;
  Eigen::Matrix3d yuv2rgb_mat;

  //Parameters
  double sigma; //Estimate of the noise std deviation
  int k1;       //Patch size for step 1
  int k2;       //Patch size for step 2
  int n1;       //Neighbourhood size for step 1
  int n2;       //Neighbourhood size for step 2
  int N_1;      //Min number of closest neighbors for step 1
  int N_2;      //Min number of closest neighbors for step 2
  double gamma; //Homogenous factor
  double beta1; //Filtering parameter for step 1
  double beta2; //Filtering parameter for step 2
  double tau0;  //Threshold between similar patches for step 2
};

#endif
