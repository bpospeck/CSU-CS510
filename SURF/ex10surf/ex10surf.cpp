#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat img_1, img_keypoints_1;
std::vector<KeyPoint> keypoints_1;
int sliderHessian;
int minHessian = 400;
int maxHessian = 8000;

String window_name = "Open CV Surf Features";

void readme();

void SurfThreshold(int, void*) {
	int hessian = minHessian + sliderHessian;
	// SurfFeatureDetector detector(hessian);
	Ptr<SURF> detector = SURF::create( hessian );
	detector->detect(img_1, keypoints_1);
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
			DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow(window_name, img_keypoints_1);
	cout << "Threshold is: " << hessian << endl;
}

/** @function main */
int main(int argc, char** argv) {
	if (argc != 2) {
		readme();
		return -1;
	}

	img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	if (!img_1.data) {
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	createTrackbar("Response:", window_name, &sliderHessian,
			maxHessian - minHessian, SurfThreshold);
	SurfThreshold(0,0);
	waitKey(0);
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_detector <img1>" << std::endl;
}
