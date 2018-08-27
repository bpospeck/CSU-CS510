#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

void setupWindows() // Sets layout of windows so they don't start on top of each other
{
	namedWindow("Video", CV_WINDOW_AUTOSIZE);
    moveWindow("Video",0,0);
    namedWindow("Goal", CV_WINDOW_AUTOSIZE);
    moveWindow("Goal",525,0);
    namedWindow("Exact Filter", CV_WINDOW_AUTOSIZE);
    moveWindow("Exact Filter", 0, 580);
    namedWindow("ASEF", CV_WINDOW_AUTOSIZE);
    moveWindow("ASEF", 525, 580);
    namedWindow("Gaussian before goal placement", CV_WINDOW_AUTOSIZE);
    moveWindow("Gaussian before goal placement", 1060, 100);
}

void findEdgeMag(Mat& image) // Find edge magnitude of img
{
	Mat xGrad, yGrad;
 	Sobel(image, xGrad, CV_32F, 1, 0, 3);
 	Sobel(image, yGrad, CV_32F, 0, 1, 3);
 	magnitude(xGrad, yGrad, image);
 	normalize(image, image, 0, 1, CV_MINMAX);
}

void setGaussian(Mat& goal, int dim, float sigma)
{
 	Mat kernelX   = getGaussianKernel(dim, sigma, CV_32FC1);
    Mat kernelY   = getGaussianKernel(dim, sigma, CV_32FC1);
    goal  = kernelX * kernelY.t();
    normalize(goal, goal, 0, 1, CV_MINMAX);
	imshow("Gaussian before goal placement", goal);
}

void findGoal(Mat& goal, Mat& gray) // Find and place gaussian onto goal based on where object is in gray (after thresholding)
{
	threshold(gray,gray,30,255,THRESH_BINARY);
	int top,left,bottom,right;
 	top = 100000;
 	bottom = 0;
 	left = 100000;
 	right = 0;
 	for(int i=0; i<gray.rows; i++){
 		for(int j=0; j<gray.cols; j++){
 			Scalar pixel = gray.at<uchar>(i,j);
 			if(pixel.val[0] == 0){
 				if(j < left) 	left = j;
 				if(j > right) 	right = j;
 				if(i < top) 	top = i;
 				if(i > bottom) 	bottom = i;
 			}
 		}
 	}
 	int xGauss = static_cast<int>((left+right)/2) - 384;
 	int yGauss = static_cast<int>((top+bottom)/2) - 104;
 	int dim = 64;
 	float sigma = 5.0;
 	setGaussian(goal, dim, sigma);
 	left = xGauss - (dim/2);
 	right = 512 - (xGauss + (dim/2));
 	top = yGauss - (dim/2);
 	bottom = 512 - (yGauss + (dim/2));
 	int borderType = BORDER_CONSTANT;
 	Scalar borderVal = Scalar(0,0,0);
 	copyMakeBorder(goal, goal, top, bottom, left, right, borderType, borderVal);
 	// cout << "(top,left,bot,right): (" << top << ","<<left<<","<<bottom<<","<<right<<") and goal size: ";
 	// cout << goal.rows << " x " <<goal.cols << endl;
}

void complexDivide(Mat& G, Mat& F, Mat& H)// Function from a student's piazza post who figured this step out 1st
{
	Mat top, bot;
	mulSpectrums(G, F, top, DFT_COMPLEX_OUTPUT, true); // Top is G*(F conjugate)
	mulSpectrums(F, F, bot, DFT_COMPLEX_OUTPUT, true); // Bot is F*(F conjugate)

	// Bottom is strictly real and we should divide real and complex parts by it
	Mat botRe[]  = {Mat_<float>(bot),  Mat::zeros(bot.size(),  CV_32F)};
	split(bot, botRe);
	botRe[0].copyTo(botRe[1]);
	merge(botRe, 2, bot);

	// Do the actual division
	divide(top, bot, H);
}

void getDFT(Mat& img)
{
	// Set up 2nd channel of zeros on images to store imaginary part after taking dft
	Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
 	merge(planes, 2, img);
 	// Take the dft
 	dft(img, img, DFT_COMPLEX_OUTPUT);
}

void dftQuadSwap (Mat& img)  
{
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = img.cols/2;
    int cy = img.rows/2;

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

void rotateImg(Mat& img, double degrees) // Rotate an image about its center by a specified number of degrees
{
	Point center = Point(img.rows/2,img.cols/2);
	Mat rotation = getRotationMatrix2D(center, degrees, 1.0);
	warpAffine(img, img, rotation, img.size());
}

int main(int argc, char** argv)
{
	if(argc !=2){
		cout << "Usage: A3 <video file name>\n";
		return -1;
	}
	VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
       cout << "Cannot open video file\n";
       return -1;
    }

    double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    setupWindows();

	Mat image,asefImg;
	Mat avg = Mat::zeros(Size(512,512), CV_64FC1);
	// Look at 1 frame at a time to find the exact filter for that frame's goal
	for(int frame =0; frame < frameCount; frame++){
		bool success = cap.read(image); // Get next frame of video
		if (!success) {
	    	cout << "Cannot read frame\n";
            break;
		}
		// convert to grayscale and threshold image
		Mat grayImg; // 1280 x 720 image
 		cvtColor( image, grayImg, CV_BGR2GRAY ); // Convert to grayscale before making edge magnitude image
 		Mat truncatedImg (grayImg, Rect(384, 104, 512, 512)); // Truncate image to 512 x 512
 		findEdgeMag(truncatedImg);
 		Mat goalImg;
 		findGoal(goalImg,grayImg);
 		Mat G, F, filterImg;
 		G = goalImg.clone();		// Let's us avoid having to take the inverse dft of the goal
 		F = truncatedImg.clone();	// Let's us avoid having to take the inverse dft of the truncated image
 		getDFT(F);
 		getDFT(G);
 		complexDivide(G, F, filterImg); // Find the exact filter for this frame
 		dft(filterImg, filterImg, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
 		normalize(filterImg, filterImg, 0, 1, CV_MINMAX);
 		dftQuadSwap(filterImg);
 		rotateImg(filterImg, 180);
 		// Average filters for ASEF
 		Mat temp;
 		filterImg.convertTo(temp, CV_64FC3);
 		avg += temp;
 		avg.convertTo(asefImg, CV_32F, 1.0 / (frame+1));
 		normalize(asefImg,asefImg, 0, 1, CV_MINMAX);

		imshow("Video", truncatedImg);
		imshow("Goal",goalImg);
		imshow("Exact Filter",filterImg);
		imshow("ASEF",asefImg);
		waitKey(30);
	}
	return 0;
}
