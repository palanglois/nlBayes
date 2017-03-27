#include <iostream>
#include "stdNlBayes.h"

using namespace std;

int main()
{
  //Finding the media directory
  string mediaDir = string(ICPSPARSE_MEDIA_DIR);
  mediaDir = mediaDir.substr(1,mediaDir.length()-2);

  //Load an image
  string lenaSmallFile = mediaDir + "lena_small.png";
  StdNlBayes my_nlbayes;
  my_nlbayes.loadImage(lenaSmallFile);


  return 0;
}
