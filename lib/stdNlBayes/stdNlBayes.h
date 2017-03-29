#ifndef STDNLBAYES_H
#define STDNLBAYES_H

#include <string>
#include <vector>
#include <array>
#include <set>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "lodepng.h"

typedef Eigen::MatrixXd ImageChannel;                      //Type for an image channel
typedef std::array<ImageChannel,3> ImageType;              //Type for a RGB or YUV image
typedef std::vector<Eigen::MatrixXd> PatchStack;           //Type for the patch stacks
typedef std::pair<Eigen::MatrixXd,double> Patch;           //Type for pair and its distance
typedef std::vector<std::vector<PatchStack>> ChannelStack; //Stack container for a channel
typedef std::array<ChannelStack,3> ImageStack;             //Stack container for the image
typedef std::array<ImageChannel,3> ImageBuffer;            //Buffer for the aggregation
class PatchCompair //Patch comparator by distance
{
  public:
    bool operator()(const Patch& a, const Patch& b)
    {
      return a.second < b.second;
    }
};
typedef std::set<Patch,PatchCompair> OrderedList; //Ordered list for patch and distance 
typedef OrderedList::iterator OrderedIt;          //Iterator for the ordered lists

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
  //Save an image to a file
  void saveImage(ImageType imToSave, std::string filePath) const;
  //Save the result image
  void saveResult(std::string filePath) const;
  //Compute distance between patches
  double getDistance(ImageChannel chan, int i0, int j0, int i1, int j1, int w, int h) const;
  //Compute distance between patches for second step
  double getDistance2(ImageType im, int i0, int j0, int i1, int j1, int w, int h) const;
  //Estimate standard deviation of a stack of similar patches
  double estimateSigmaP(const PatchStack stack, double M_1) const;
private:
  ImageType image;
  ImageType image_yuv;
  ImageType image_u; //Estimate at the end of the first step
  ImageType image_final;
  ImageType image_final_rgb;
  ImageType image_u_rgb;
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
