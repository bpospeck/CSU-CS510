#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

/* This is the Load, Modify, and Save an Image tutorial
 * available from
 *
 * http://docs.opencv.org/doc/tutorials/introduction/load_save_image/load_save_image.html#load-save-image
 */

int main( int argc, char** argv )
{
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 imwrite( "Gray_Image.jpg", gray_image );

 namedWindow( imageName, CV_WINDOW_AUTOSIZE );
 namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );

 imshow( imageName, image );
 imshow( "Gray image", gray_image );

 waitKey(0);

 return 0;
}