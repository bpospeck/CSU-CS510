#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Mat src, src_gray, dst, cdst, pcdst;

int tauSlider, lowTau;
int minTau = 10;
int const maxTau = 100;
int cannyRatio = 3;
int kernel_size = 3;

int const minVotes = 50;
int const maxVotes = 300;
int votesSlider;
int tauVotes;

string window_name = "Open CV Hough Lines";
string settings;

void help() {
	cout
			<< "\nThis program demonstrates line finding with the Hough transform.\n"
					"Usage:\n"
					"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

void HoughThreshold(int, void*) {
	vector<Vec2f> lines;
	char buff[256];
	lowTau = tauSlider + minTau;
	tauVotes = votesSlider + minVotes;

	/// Reduce noise with a kernel 3x3, then apply Canny, then Hough
	blur(src_gray, dst, Size(3, 3));
	Canny(dst, dst, lowTau, lowTau * cannyRatio, kernel_size);
	HoughLines(dst, lines, 1, CV_PI / 180, tauVotes, 0, 0);
	/// Draw lines on a color version of the Canny Edge image
	cvtColor(dst, cdst, CV_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(30, 157, 255), 3, CV_AA);
	}
    //  Add text indicating threshold settings to a border on top of the edge image.
	copyMakeBorder(cdst, pcdst, 48, 0, 0, 0, BORDER_CONSTANT, cvScalar(128, 128, 128) );
	sprintf(buff, "Min Edge Threshold: %d, Min Votes: %d", lowTau, tauVotes);
	std::string sbuff = buff;
	putText(pcdst, sbuff, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8,
			cvScalar(255, 245, 205), 1, CV_AA, 0);
	imshow(window_name, pcdst);
}

int main(int argc, char** argv) {
	const char* filename = argc >= 2 ? argv[1] : "pic1.jpg";

	src = imread(filename);
	if (src.empty()) {
		help();
		cout << "can not open " << filename << endl;
		return -1;
	}
	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	/// Create a Trackbar for the user to enter lower threshold for the Canny algorithm
	createTrackbar("Edge Threshold:", window_name, &tauSlider,
		maxTau - minTau, HoughThreshold);
	/// Create a Trackbar for user to enter minimum number of Hough votes
	createTrackbar("Vote Threshold:", window_name, &votesSlider,
			maxVotes - minVotes, HoughThreshold);
	HoughThreshold(0, 0);
	waitKey();
	return 0;
}
