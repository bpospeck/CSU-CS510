#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	if(argc !=2){
		cout << "Missing arg: need a video file\n";
		return -1;
	}

	VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
       cout << "Cannot open video file\n";
       return -1;
    }

    double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    namedWindow("Video with Bounds", CV_WINDOW_AUTOSIZE);

	Mat image;
	image;
	// Look at 1 frame at a time to find the object and bound it
	for(int frame =0; frame < frameCount; frame++){
		bool success = cap.read(image); // Get next frame of video
		if (!success) {
	    	cout << "Cannot read frame\n";
            break;
		}
		// convert to grayscale and threshold image
		Mat gray_image;
 		cvtColor( image, gray_image, CV_BGR2GRAY );
 		threshold(gray_image,gray_image,30,255,THRESH_BINARY);
 		// Look for the first black pixel from each edge
 		int top,left,bottom,right;
 		top = 100000;
 		bottom = 0;
 		left = 100000;
 		right = 0;
 		for(int i=0; i<image.rows; i++){
 			for(int j=0; j<image.cols; j++){
 				Scalar pixel = gray_image.at<uchar>(i,j);
 				if(pixel.val[0] == 0){
 					if(j < left) 	left = j;
 					if(j > right) 	right = j;
 					if(i < top) 	top = i;
 					if(i > bottom) 	bottom = i;
 				}
 			}
 		}
 		/*Below does the same thing as the 1 for loop above, but may be faster in some cases
 		  It looks from each edge of the edge and breaks once it finds a boundary.
 		  Won't look through the entire set of pixels like the above loop*/
 	// 	bool boundFound = false;
		// int top = 0;
		// for( int i=0; i<image.rows; i++){
		// 	for(int j=0; j<image.cols; j++){
		// 		Scalar pixel = gray_image.at<uchar>(i,j);
		// 		if(pixel.val[0] == 0){
		// 			top = i;
		// 			boundFound = true;
		// 			break;
		// 		}
		// 	}
		// 	if(boundFound) break;
		// }
		// boundFound = false;
		// int bottom = 0;
		// for( int i=image.rows-1; i>0; i--){
		// 	for(int j=image.cols-1; j>0; j--){
		// 		Scalar pixel = gray_image.at<uchar>(i,j);
		// 		if(pixel.val[0] == 0){
		// 			bottom = i;
		// 			boundFound = true;
		// 			break;
		// 		}
		// 	}
		// 	if(boundFound) break;
		// }
		// boundFound = false;
		// int left = 0;
		// for( int j=0; j<image.cols; j++){
		// 	for(int i=0; i<image.rows; i++){
		// 		Scalar pixel = gray_image.at<uchar>(i,j);
		// 		if(pixel.val[0] == 0){
		// 			left = j;
		// 			boundFound = true;
		// 			break;
		// 		}
		// 	}
		// 	if(boundFound) break;
		// }
		// boundFound = false;
		// int right = 0;
		// for( int j=image.cols-1; j>0;j--){
		// 	for(int i=image.rows-1; i>0; i--){
		// 		Scalar pixel = gray_image.at<uchar>(i,j);
		// 		if(pixel.val[0] == 0){
		// 			right = j;
		// 			boundFound = true;
		// 			break;
		// 		}
		// 	}
		// 	if(boundFound) break;
		// }
		rectangle(image,Point(left-1,top-2), Point(right+3,bottom+2), Scalar(255,0,255),4); //Draw rectangle on original image
		imshow("Video with Bounds", image);
		if(waitKey(15) == 27) continue;
	}
	return 0;
}
