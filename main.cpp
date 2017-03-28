#include <iostream>
#include "stdNlBayes.h"

using namespace std;

int main()
{
  //Parameters
  double sigma = 20;

  //Finding the media directory
  string mediaDir = string(ICPSPARSE_MEDIA_DIR);
  mediaDir = mediaDir.substr(1,mediaDir.length()-2);

  //Load an image
  string lenaSmallFile = mediaDir + "lena_small.png";
  StdNlBayes my_nlbayes(sigma);
  my_nlbayes.loadImage(lenaSmallFile);
  my_nlbayes.run();


  return 0;
}
