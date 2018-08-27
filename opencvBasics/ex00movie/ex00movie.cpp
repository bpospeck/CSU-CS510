#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void frameDisplay(Mat frame, int fri);


/** @function main */
int main(int argc, const char** argv) {

    if( argc != 2)
    {
     cout <<" Usage: ex00movie video_file" << endl;
     return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
       cout << "Cannot open the video file" << endl;
       return -1;
    }

    double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
    cap.set(CV_CAP_PROP_POS_FRAMES, 0); //Set index to last frame
    namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);
    Mat frame;
    for (int fri = 0; fri < count; fri++) {
	
	bool success = cap.read(frame);
	if (!success) {
	    cout << "Cannot read  frame " << endl;
            break;
	}
    // Temporary code to test stuff
    cvtColor(frame, frame, CV_BGR2GRAY);
    threshold(frame, frame, 30,255,THRESH_BINARY);
    // End of temporary code to test stuff
        frameDisplay(frame, fri);
	if (waitKey(100) == 27)
           break;
	}
    return 0;
}

/** @function detectAndDisplay */
void frameDisplay(Mat frame, int fri) {
	char s1 [] = "Play back video";
	char s2 [128];
	sprintf(s2, "%s %d", s1, fri);
	string window_name(s2);

	//-- Show single frame
	imshow("MyVideo", frame);
}
