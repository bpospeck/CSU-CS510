#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

/* Just to show we can, turn a part of the image red
 * This example is hard wired in terms of image size to go
 * with the image file IconFaceLv2.png
 * */

int main(int argc, char** argv) {

	Mat image;
	image = imread("IconFaceLv2.png", 1);

	/* Make a small window into this larger image */
	Mat eye(image, Rect(182, 218, 8, 8));
	cout << "Before color values." << endl;
	cout << "eye = " << endl << " " << eye << endl << endl;

	MatIterator_<Vec3b> it, end;
	for (it = eye.begin<Vec3b>(), end = eye.end<Vec3b>(); it != end; ++it) {
		(*it)[0] = 0;
		(*it)[1] = 0;
		(*it)[2] = 255;
	}

	cout << "After color values." << endl;
	cout << "eye = " << endl << " " << eye << endl << endl;

	namedWindow( "IconFaceLv2.png", CV_WINDOW_AUTOSIZE);
	imshow( "IconFaceLv2.png", image);
	waitKey(0);

	return 0;
}
