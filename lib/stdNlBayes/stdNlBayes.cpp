#include "stdNlBayes.h"

using namespace std;
using namespace Eigen;

StdNlBayes::StdNlBayes(double _sigma) : sigma(_sigma)
{
  //Initiating the color convertion matrices
  rgb2yuv_mat = Matrix3d::Zero(3,3);
  rgb2yuv_mat(0,0) = 1./3.; rgb2yuv_mat(0,1) = 1./3.;  rgb2yuv_mat(0,2) = 1./3.;
  rgb2yuv_mat(1,0) = 1./2.; rgb2yuv_mat(1,1) = 0.;     rgb2yuv_mat(1,2) = -1./2.;
  rgb2yuv_mat(2,0) = 1./4.; rgb2yuv_mat(2,1) = -1./2.; rgb2yuv_mat(2,2) = 1./4.;

  yuv2rgb_mat = Matrix3d::Zero(3,3);
  yuv2rgb_mat(0,0) = sqrt(1./3.); yuv2rgb_mat(0,1) = sqrt(1./2.); yuv2rgb_mat(0,2) = sqrt(1./6.);
  yuv2rgb_mat(1,0) = sqrt(1./3.); yuv2rgb_mat(1,1) = 0.; yuv2rgb_mat(1,2) = -sqrt(2./3.);
  yuv2rgb_mat(2,0) = sqrt(1./3.); yuv2rgb_mat(2,1) = -sqrt(1./2.); yuv2rgb_mat(2,2) = sqrt(1./6.);
  
  //Estimating the parameters

  // k1 and k2
  if(0. <= sigma && sigma < 20)
  {
    k1 = 3;
    k2 = 3;
  }
  else if(20. <= sigma && sigma < 50)
  {
    k1 = 5;
    k2 = 3;
  }
  else if(50 <= sigma && sigma < 70)
  {
    k1 = 7;
    k2 = 5;
  }
  else if(70 <= sigma)
  {
    k1 = 7;
    k2 = 7;
  }
  else
    cerr << "ERROR : sigma must be a positive value !" << endl;
  
  //gamma
  gamma = 1.05;

  //beta1
  beta1 = 1.;

  //beta2
  if(sigma <=50)
    beta2 = 1.;
  else
    beta2 = 1.2;

  //n1 and n2
  n1 = 7*k1;
  n2 = 7*k2;
  //// FOR DEBUG
  //n1 = 2*k1;
  //n1 = 2*k2;

  //N_1 and N_2
  switch(k1)
  {
    case 3:
      N_1 = 30;
      break;
    case 5:
      N_1 = 60;
      break;
    case 7:
      N_1 = 90;
      break;
    default:
      cerr << "ERROR : invalid value for k1." << endl;
  }
  switch(k2)
  {
    case 3:
      N_2 = 30;
      break;
    case 5:
      N_2 = 60;
      break;
    case 7:
      N_2 = 90;
      break;
    default:
      cerr << "ERROR : invalid value for k2." << endl;
  }

  //tau0
  tau0 = 4;

}

/*
Run the algorithm
*/
void StdNlBayes::run()
{
  ////// FIRST STEP //////  

  int w1 = (n1-1)/2; //Neighbourhood size
  int p1 = (k1-1)/2; //patch size

  //Convert the image to yuv
  image_yuv = rgb2yuv(image);
  
  //Computing all the stacks
  // All the patches for the y channel
  ImageStack yuvStack;
  yuvStack.fill(ChannelStack(image[0].rows(),vector<PatchStack>(image[0].cols(),PatchStack(0,MatrixXd::Zero(k1,k1)))));
  for(int i=0;i<image[0].rows();i++)
  {
    for(int j=0;j<image[0].cols();j++)
    {
      OrderedList yList;
      OrderedList uList;
      OrderedList vList;
      //Computing patches
      for(int iPatch = -w1; iPatch <= w1; iPatch++)
      {
        for(int jPatch = -w1;jPatch <= w1; jPatch++)
        {
          if(i-p1 < 0 || image[0].rows() <= i+p1)
            continue;
          if(j-p1 < 0 || image[0].cols() <= j+p1)
            continue;
          if(i+iPatch-p1 < 0 || image[0].rows() <= i+iPatch+p1)
            continue;
          if(j+jPatch-p1 < 0 || image[0].cols() <= j+jPatch+p1)
            continue;
          double curDist = getDistance(image_yuv[0],i+iPatch,j+jPatch,i,j,p1,p1) / pow(k1,2);
          yList.insert(Patch(image_yuv[0].block(i+iPatch-p1,j+jPatch-p1,k1,k1),curDist));
          uList.insert(Patch(image_yuv[1].block(i+iPatch-p1,j+jPatch-p1,k1,k1),curDist));
          vList.insert(Patch(image_yuv[2].block(i+iPatch-p1,j+jPatch-p1,k1,k1),curDist));
        }
      }
      //Storing the N_1 best patches
      int count = 0;
      OrderedIt itU = uList.begin();
      OrderedIt itV = vList.begin();
      for(OrderedIt it = yList.begin(); it != yList.end() && count < N_1;it++)
      {
        yuvStack[0][i][j].push_back(it->first);
        yuvStack[1][i][j].push_back(itU->first);
        yuvStack[2][i][j].push_back(itV->first);
        count++;
        itU++;
        itV++;
      }
    }
    cout << "line : " << i << " out of " << image[0].rows() << endl;
  }
  cout << "Stack computed for the first step." << endl;
  //Collaborative filtering
  double M_1 = N_1 * pow(k1,2);
  ImageStack colStack;
  colStack.fill(ChannelStack(image[0].rows(),vector<PatchStack>(image[0].cols(),PatchStack(0,MatrixXd::Zero(k1,k1)))));
  for(int channel = 0; channel < 3; channel++)
  {
    for(int i=0;i<image[0].rows();i++)
    {
      for(int j=0;j<image[0].cols();j++)
      {
        //Estimating sigma_p
        PatchStack& currentStack = yuvStack[channel][i][j];
        //By definition p is the first patch of the stack
        MatrixXd p;
        if(currentStack.size() > 0) p = currentStack[0]; else continue;
        double sigmaP = estimateSigmaP(currentStack,M_1);
        MatrixXd qBasic = MatrixXd::Zero(k1,k1);
        //Building an estimate for each patch of each stack
        if(sigmaP <= gamma * sigma)
        {
          for(int k=0;k<currentStack.size();k++)
            qBasic += currentStack[k];
          qBasic /= currentStack.size();
          for(int k=0;k<currentStack.size();k++)
            colStack[channel][i][j].push_back(qBasic);
        }
        else
        {
          MatrixXd cp = MatrixXd::Zero(k1,k1);
          MatrixXd pAverage = MatrixXd::Zero(k1,k1);
          if(currentStack.size() < 2) continue;
          for(int k=0;k<currentStack.size();k++)
          {
            MatrixXd diff = currentStack[k] - p;
            cp += diff*diff.transpose();
            pAverage += currentStack[k];
          }
          pAverage /= currentStack.size();
          cp /= (currentStack.size()-1);
          MatrixXd id(k1,k1);
          id.setIdentity();
          for(int k=0;k<currentStack.size();k++)
          {
            qBasic = pAverage + 
                     (cp - beta1 * pow(sigma,2) * id)*cp.inverse()*(currentStack[k]-pAverage);
            colStack[channel][i][j].push_back(qBasic);
          }
        }
      }
    }
    cout << "Performed collaboration for channel " << channel << endl;
  }
  cout << "Collaborative filtering done" << endl;
  //Aggregation
  ImageBuffer bufferIm = zeroImage(image[0].rows(),image[0].cols());
  ImageBuffer bufferW = zeroImage(image[0].rows(),image[0].cols());
  for(int channel = 0; channel < 3; channel++)
  {
    for(int i=0;i<image[0].rows();i++)
    {
      for(int j=0;j<image[0].cols();j++)
      {
        PatchStack& currentStack = colStack[channel][i][j];
        if(currentStack.size() < N_1)
          continue;
        for(int k=0;k<currentStack.size();k++)
        {
          int i0 = i - p1;
          int j0 = j - p1;
          bufferIm[channel].block(i0,j0,k1,k1) += currentStack[k];
          bufferW[channel].block(i0,j0,k1,k1) += MatrixXd::Ones(k1,k1);
        }
      }
    }
  }
  for(int channel = 0; channel<3;channel++)
  {
    image_u[channel] = bufferIm[channel].cwiseQuotient(bufferW[channel]);
  }
  image_u = rgb2yuv(image_u,true);
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

/*
Convert an image from RGB to YUV color system
*/
ImageType StdNlBayes::rgb2yuv(ImageType src, bool inverse) const
{
  int row = src[0].rows();
  int col = src[0].cols();
  ImageType dst = zeroImage(row,col);
  for(int i=0;i<src[0].rows();i++)
    for(int j=0;j<src[0].cols();j++)
    {
      Matrix<double,3,1> colorVec(src[0](i,j),src[1](i,j),src[2](i,j));
      Matrix<double,3,1> newColor; 
      if(inverse)
        newColor = yuv2rgb_mat * colorVec;
      else
        newColor = rgb2yuv_mat * colorVec;
      dst[0](i,j) = newColor(0,0);
      dst[1](i,j) = newColor(1,0);
      dst[2](i,j) = newColor(2,0);
    }
  return dst;
}

/*
Create a 0 image with an appropriate number of rows and columns
*/
ImageType StdNlBayes::zeroImage(int rows, int cols) const
{
  ImageType image;
  for(int i=0;i<3;i++)
    image[i] = ImageChannel::Zero(rows,cols);
  return image;
}

/*
Save an image to a file as png
*/
void StdNlBayes::saveImage(ImageType imToSave, string filePath) const
{
  //Align image
  std::vector<unsigned char> imageLine;
  for(int j=0;j<imToSave[0].cols();j++)
    for(int i=0;i<imToSave[0].rows();i++) 
    {
      imageLine.push_back(imToSave[0](i,j));  //R
      imageLine.push_back(imToSave[1](i,j));  //G
      imageLine.push_back(imToSave[2](i,j));  //B
      imageLine.push_back(255);               //alpha
    }
  //Encode it to png
  std::vector<unsigned char> png;
  lodepng::State state;
  unsigned error = lodepng::encode(png, imageLine, 
                                   imToSave[0].rows(), imToSave[0].cols(), state);
  if(error)
    cerr << "ERROR : while saving file " << filePath << endl;

  //Write image to disk
  lodepng::save_file(png, filePath.c_str());
}

/*
Saving the resulting png image to filePath
*/
void StdNlBayes::saveResult(string filePath) const
{
  saveImage(image_u,filePath);
}

/*
Compute distance between two patches
*/
double StdNlBayes::getDistance(ImageChannel chan, int i0, int j0, int i1, int j1, int w, int h) const
{
  double dist = 0.;
  for(int i=-w;i<=w;i++)
    for(int j=-h;j<=h;j++)
      dist += pow(chan(i0+i,j0+j)-chan(i1+i,j1+j),2);
  return dist;
}

/*
Estimate standard deviation of a stack of similar patches
*/
double StdNlBayes::estimateSigmaP(const PatchStack stack, double M_1) const
{
  double sigmaP = 0;
  for(int i=0;i<stack.size();i++)
    sigmaP += (1/M_1)*stack[i].cwiseProduct(stack[i]).sum() - pow((1/M_1)*stack[i].sum(),2);
  return M_1 / (M_1 - 1.) * sigmaP;
}
