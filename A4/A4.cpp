#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void setupWindows() // Sets layout of windows so they don't start on top of each other
{
	namedWindow("Camera", CV_WINDOW_AUTOSIZE);
    moveWindow("Camera",0,0);
    namedWindow("Goal", CV_WINDOW_AUTOSIZE);
    moveWindow("Goal",645,0);
    namedWindow("Exact Filter", CV_WINDOW_AUTOSIZE);
    moveWindow("Exact Filter", 0, 485);
    namedWindow("ASEF", CV_WINDOW_AUTOSIZE);
    moveWindow("ASEF", 645, 485);
}
 
int main(int argc, char** argv) {
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
 	// Camera frames are 480 x 640
 	int top = 480/2 - 60;
	int left = 640/2 - 60;
	int bottom = 480/2 +60;
	int right = 640/2 + 60;
	int initFrames = 30; // num of frames to initialize filter on
	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	} 
	setupWindows();
	Mat frame, nextFrame, gray, nextGray, diff;
	vector<vector<Point>> cnts;
	for(int i=0; i<200; i++){
		stream1.read(frame);
		rectangle(frame,Point(left,top), Point(right,bottom), Scalar(255,0,255),2);
		imshow("Camera", frame);
		waitKey(30);
	}
	for(int i=0; i<initFrames; i++){
		stream1.read(frame);
	}
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(21,21), 0);
	// Loop keeps going until user breaks out of camera
	while (true) {
		stream1.read(nextFrame);
		cvtColor(nextFrame,nextGray, COLOR_BGR2GRAY);
		GaussianBlur(nextGray, nextGray, Size(21,21), 0);

		absdiff(gray,nextGray, diff);
		threshold(diff, diff, 128, 255, THRESH_BINARY);

		//dilate(diff, diff, Mat(), Point(-1,-1), 2);

		imshow("Camera", nextFrame);
		imshow("Goal", gray);
		imshow("ASEF", nextGray);
		imshow("Exact Filter", diff);
		if (waitKey(30) >= 0)
			break;
	}
	return 0;
	}