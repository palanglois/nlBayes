#include <iostream>

using namespace std;

int main()
{
  //Finding the media directory
  string mediaDir = string(ICPSPARSE_MEDIA_DIR);
  mediaDir = mediaDir.substr(1,mediaDir.length()-2);

  cout << mediaDir << endl;

  return 0;
}
