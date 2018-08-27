#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/highgui.h"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv/cv.h"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void complexDivide(Mat& G, Mat& F, Mat& H);
void dftQuadSwap(Mat& img);
void sobelAVG(Mat& Img, Mat& sobel);
void toFreq(Mat& Img, Mat& RI);
void show(Mat& Img, const int& pos, const string& Title);

int main()
{
	// Initialize Planes
	Mat locationRI, filterRI, asefImg;
	Mat sum     = { Mat::zeros(256, 256, CV_64F) };
	Mat average = { Mat::zeros(256, 256, CV_64F) };

	// Creates a Wihite Backdrop to hopefully illuminate the object better
	Mat white;
	white = { Mat::zeros(1080, 1350, CV_32F) };
	white = white + 1 * 255;
	imshow("White", white);
	moveWindow("White", 0, 0);

	// Creates Kernal a Gauss Point the size of the 512 X 512
	// Video and places it at the center 
	// Then Converts to Frequency Domain
	Mat kernelX = getGaussianKernel(11, 11, CV_32FC1);
	Mat kernelY = getGaussianKernel(11, 11, CV_32FC1);
	Mat Gauss = kernelX * kernelY.t();
	normalize(Gauss, Gauss, 0, 1, CV_MINMAX);
	copyMakeBorder(Gauss, Gauss, 121, 124, 121, 124, BORDER_CONSTANT, Scalar::all(0));
	toFreq(Gauss, locationRI);

	// Time to Position Object and Initialize Filter
	int fri = 0;
	int start = 30;
	int middle1 = 31;
	int middle2 = 61;

	// Initializing Other Variables
	int loc = 0;
	double maxX = 256 / 2;
	double maxY = 256 / 2;
	double eta = 0.125;
	Mat GaussLocationLive = Gauss;
	Mat exactfilters[] = { Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F),
						   Mat::zeros(256, 256, CV_64F), Mat::zeros(256, 256, CV_64F) };

	// Retreive Video from Camera
	VideoCapture cap;
	cap.open(1);
	
	while(1)
	{
		// Saves Video frame to Image Files
		Mat image1, image2, image3;
		cap.read(image1);
		cap.read(image2);
		cap.read(image3);

		// Finds Dimensions of the Video
		double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);

		// Counts the number of Frames
		fri++;

		// Loop to Position the Object in the Frame
		if (fri <= start)
		{
			// Creates Window with Point at center of the Frame to Position Object
			Mat PointCropped = Mat(image1, Rect(width / 2 - 128, height / 2 - 128, 256, 256));
			circle(PointCropped, Point(256 / 2, 256 / 2), 3, Scalar(0, 0, 255), -1);
			rectangle(PointCropped, Point(78, 78), Point(178, 178), Scalar(0, 0, 255), 3);
			flip(PointCropped, PointCropped, 1);
			namedWindow("Position the Object...", CV_WINDOW_AUTOSIZE);
			imshow("Position the Object...", PointCropped);
			//moveWindow("Position the Object...", 540, 120);
		}
	
		// Loop to Initialize the ASEF
		else if (fri >= middle1 && fri <= middle2)
		{
			// Creates Window with Point at center of the Frame to Position Object
			Mat PointCropped = Mat(image1, Rect(width / 2 - 128, height / 2 - 128, 256, 256));
			circle(PointCropped, Point(256 / 2, 256 / 2), 3, Scalar(0, 0, 255), -1);
			rectangle(PointCropped, Point(78, 78), Point(178, 178), Scalar(0, 0, 255), 3);
			flip(PointCropped, PointCropped, 1);
			imshow("Initializing...", PointCropped);
			//moveWindow("Initializing...", 540, 120);

			// Create Image of Object at Center
			Mat Object, CroppedObject1;
			Mat CroppedObject = Mat(image3, Rect(width / 2 - 128+78, height / 2 - 128+78, 100, 100));
			CroppedObject.copyTo(CroppedObject1);
			copyMakeBorder(CroppedObject1, Object, 78, 78, 78, 78, 1);
			flip(Object, Object, 1);

			// Creates live video without the Point at the Center 
			Mat Cropped = Mat(image2, Rect(width / 2 - 128, height / 2 - 128, 256, 256));
			flip(Cropped, Cropped, 1);

			// Begining to Track object

			// Perform the Sobel Edge Detection
			Mat grayImgobject, grayImglive, sobelobject, sobellive;
			cvtColor(Object, grayImgobject, CV_BGR2GRAY);
			sobelAVG(grayImgobject, sobelobject);
			cvtColor(Cropped, grayImglive, CV_BGR2GRAY);
			sobelAVG(grayImglive, sobellive);
			imshow("Sobel Edge Detection", sobellive);
			//moveWindow("Sobel Edge Detection", 230, 120);

			// Convert sobel to Frequency Domain
			Mat sobelRI;
			toFreq(sobelobject, sobelRI);

			// Complex Divide both in Frequency Domain
			complexDivide(locationRI, sobelRI, filterRI);

			// Transform Exact Filter to Spacial Domain
			Mat filter;
			dft(filterRI, filter, DFT_INVERSE | DFT_REAL_OUTPUT);
			dftQuadSwap(filter);

			// Display Exact Filter
			int pos = (fri-1) % 30;
			normalize(filter, filter, 0, 1, CV_MINMAX);
			filter.convertTo(filter, CV_64F);
			filter.copyTo(exactfilters[pos]);
			imshow("Exact Filter", exactfilters[pos]);
			//moveWindow("Exact Filter", 850, 120);
			
			// Take the average of the Exact Filters
			asefImg.convertTo(asefImg, CV_64F);
			for (int i = 0; i < 30; i++)
			{
				sum = sum + exactfilters[pos];
			}
			asefImg = sum / (fri-start);
			asefImg.convertTo(asefImg, CV_32F);
			normalize(asefImg, asefImg, 0, 1, CV_MINMAX);
			imshow("ASEF", asefImg);
			//moveWindow("ASEF", 850, 426);
		}
		
		else 
		{
			/*
			Logic to Create Filter

				Sobel * ASEF->LocationLive
				LocationLive / Sobel->Filter
				Filter->ASEF
			*/

			// Creates live video without the Point at the Center
			Mat Cropped = Mat(image2, Rect(width / 2 - 128, height / 2 - 128, 256, 256));
			flip(Cropped, Cropped, 1);

			// Begining to Track object

			// Perform the Sobel Edge Detection
			Mat grayImgLive, sobelLive, sobelLive1, sobelLiveImg;
			cvtColor(Cropped, grayImgLive, CV_BGR2GRAY);
			sobelAVG(grayImgLive, sobelLive);
			sobelLive.copyTo(sobelLiveImg);
			sobelLive = Mat(sobelLive, Rect(maxX - 50, maxY - 50, 100, 100));
			sobelLive.copyTo(sobelLive1);
			copyMakeBorder(sobelLive1, sobelLive, maxY-50, 256 - maxY - 50, maxX-50, 256 - maxX - 50, BORDER_CONSTANT, Scalar::all(0));
			imshow("Sobel Edge Detection", sobelLiveImg);
			//moveWindow("Sobel Edge Detection", 230, 120);

			// Convert sobel to Frequency Domain
			Mat sobelLiveRI;
			toFreq(sobelLive, sobelLiveRI);

			// Multiply ASEF by Sobel to find Location
			Mat asefRI, location;
			toFreq(asefImg, asefRI);
			mulSpectrums(sobelLiveRI, asefRI, locationRI, DFT_COMPLEX_OUTPUT);
			dft(locationRI, location, DFT_INVERSE | DFT_REAL_OUTPUT);
			normalize(location, location, 0, 1, CV_MINMAX);
			dftQuadSwap(location);
			GaussianBlur(location, location, Size(19, 19), 0, 0, BORDER_DEFAULT);

			// Finds Max Position
			Point maxLoc;
			minMaxLoc(location, NULL, NULL, NULL, &maxLoc);
			//circle(location, maxLoc, 10, Scalar::all(255), -1);
			maxX = maxLoc.x;
			maxY = maxLoc.y;
			imshow("Goal", location);
			//moveWindow("Location", 1174, 612);


			// Places Box Around Object
			rectangle(Cropped, Point(maxX-50, maxY-50), Point(maxX+50, maxY+50), Scalar(0, 0, 255), 3);
			imshow("Boxed Location", Cropped);
			//moveWindow("Boxed Location", 230, 426);

			
			// Creates a Gauss Point at the max location 
			// Then Converts to Frequency Domain
			Mat kernelXmax = getGaussianKernel(11, 11, CV_32F);
			Mat kernelYmax = getGaussianKernel(11, 11, CV_32F);
			Mat GaussLocation = kernelXmax * kernelYmax.t();
			normalize(GaussLocation, GaussLocation, 0, 1, CV_MINMAX);
			copyMakeBorder(GaussLocation, GaussLocationLive, maxY-5, 256-maxY-6, maxX-6, 256-maxX-5, BORDER_CONSTANT, Scalar::all(0));
			imshow("Goal Made to Gauss", GaussLocationLive);
			//moveWindow("Location Made to Gauss", 1174, 50);		

			
			// Convert LocationLive to Frequency Domain
			Mat LocationLiveRI;
			toFreq(GaussLocationLive, LocationLiveRI);

			// Complex Divide both in Frequency Domain
			complexDivide(LocationLiveRI, sobelLiveRI, filterRI);

			// Transform Exact Filter to Spacial Domain
			Mat filter;
			dft(filterRI, filter, DFT_INVERSE | DFT_REAL_OUTPUT);
			dftQuadSwap(filter);

			// Display Exact Filter
			normalize(filter, filter, 0, 1, CV_MINMAX);
			imshow("Exact Filter", filter);
			//moveWindow("Exact Filter", 50, 612);
			/*
			// Display Exact Filter
			int pos = (fri - 1) % 30;
			normalize(filter, filter, 0, 1, CV_MINMAX);
			filter.convertTo(filter, CV_64F);
			filter.copyTo(exactfilters[pos]);
			imshow("Exact Filter", exactfilters[pos]);
			moveWindow("Exact Filter", 50, 612);
			
			// Take the average of the Exact Filters
			asefImg.convertTo(asefImg, CV_64F);
			for (int i = 0; i < 30; i++)
			{
				sum = sum + exactfilters[pos];
			}

			asefImg = sum / (fri - start);
			asefImg.convertTo(asefImg, CV_32F);
			normalize(asefImg, asefImg, 0, 1, CV_MINMAX);
			*/

			asefImg = (eta * filter) + ((1 - eta)*asefImg);
			imshow("ASEF", asefImg);
			//moveWindow("ASEF", 612, 612);
		
		}

		waitKey(1);

	}

	return 0;
}

void complexDivide(Mat& G, Mat& F, Mat& H)
{
	Mat top, bot;
	mulSpectrums(G, F, top, DFT_COMPLEX_OUTPUT, true); // Top is G*(F conjugate)
	mulSpectrums(F, F, bot, DFT_COMPLEX_OUTPUT, true); // Bot is F*(F conjugate)

													   // Bottom is strictly real and we should divide real and complex parts by it
	Mat botRe[] = { Mat_<float>(bot),  Mat::zeros(bot.size(),  CV_32F) };
	split(bot, botRe);
	botRe[0].copyTo(botRe[1]);
	merge(botRe, 2, bot);

	// Do the actual division
	divide(top, bot, H);

}

void dftQuadSwap(Mat& img)
{
	int cx = img.cols / 2;
	int cy = img.rows / 2;

	Mat q0(img, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(img, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(img, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(img, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

}

void sobelAVG(Mat& Img, Mat& sobel)
{
	Mat sobelx, abs_grad_x, sobely, abs_grad_y;
	GaussianBlur(Img, Img, Size(7, 7), 0, 0, BORDER_DEFAULT);

	Sobel(Img, sobelx, CV_32F, 1, 0);
	convertScaleAbs(sobelx, abs_grad_x);

	Sobel(Img, sobely, CV_32F, 0, 1);
	convertScaleAbs(sobely, abs_grad_y);

	// Take the Average of the two directions
	addWeighted(abs_grad_x, 0.7, abs_grad_y, 0.3, 0, sobel);
}

void toFreq(Mat& Img, Mat& RI)
{
	Mat Planes[] = { Mat_<float>(Img), Mat::zeros(Img.size(), CV_32F) };
	merge(Planes, 2, RI);
	dft(RI, RI, DFT_COMPLEX_OUTPUT);
}

void show(Mat& Img, const int& pos, const string& Title)
{
	namedWindow(Title, CV_WINDOW_AUTOSIZE);
	moveWindow(Title, pos*(50 + 256), 1);
	imshow(Title, Img);
}
