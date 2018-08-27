#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Rect g_rectangle;
bool g_bDrawingBox = false;
bool goalSet = false;
Mat goal;

void setupWindows() // Sets layout of windows so they don't start on top of each other
{
	namedWindow("Camera", CV_WINDOW_AUTOSIZE);
    moveWindow("Camera",0,0);
    namedWindow("filter", CV_WINDOW_AUTOSIZE);
	moveWindow("filter",645,0);
    namedWindow("goal", CV_WINDOW_AUTOSIZE);
    moveWindow("goal",645,545);
    namedWindow("edge", CV_WINDOW_AUTOSIZE);
    moveWindow("edge",0,545);
    namedWindow("asef",CV_WINDOW_AUTOSIZE);
    moveWindow("asef", 1290,0);
    namedWindow("correlation",CV_WINDOW_AUTOSIZE);
    moveWindow("correlation",1350,545);
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
	//imshow("Gaussian before goal placement", goal);
}

void setGoal(Mat& img, int xGauss, int yGauss)
{
 	int dim = 32;
 	float sigma = 2.0;
 	setGaussian(goal, dim, sigma);
 	// Find edges of gaussian to determine proper padding length
 	int left = xGauss - (dim/2);
 	int right = img.cols - (xGauss + (dim/2));
 	int top = yGauss - (dim/2);
 	int bottom = img.rows - (yGauss + (dim/2));
 	int borderType = BORDER_CONSTANT;
 	copyMakeBorder(goal, goal, top, bottom, left, right, borderType, Scalar::all(0));
}

void switchLogScale(Mat& img) {
    img += Scalar::all(1);                    // switch to logarithmic scale
    log(img, img);
}

// Found here and tweaked slightly: 
// https://github.com/taochenshh/Rectangles-with-Mouse-in-OpenCV/blob/master/DrawRectangles.cpp
void on_MouseHandle(int event, int x, int y, int flags, void* param) 
{
	Mat& image = *(cv::Mat*) param;
	switch (event) {
	case EVENT_MOUSEMOVE: {    // When mouse moves, get the current rectangle's width and height
		if (g_bDrawingBox) {
			g_rectangle.width = x - g_rectangle.x;
			g_rectangle.height = y - g_rectangle.y;
		}
	}
		break;
	case EVENT_LBUTTONDOWN: {  // when the left mouse button is pressed down,
		                       //get the starting corner's coordinates of the rectangle
		g_bDrawingBox = true;
		g_rectangle = Rect(x, y, 0, 0);
	}
		break;
	case EVENT_LBUTTONUP: {   //when the left mouse button is released,
		                      //set the filter
		g_bDrawingBox = false;
		if (g_rectangle.width < 0) {
			g_rectangle.x += g_rectangle.width;
			g_rectangle.width *= -1;
		}

		if (g_rectangle.height < 0) {
			g_rectangle.y += g_rectangle.height;
			g_rectangle.height *= -1;
		}
		// Setup left, right, top, bottom vars for rectangle
		int top, bottom, left, right;
		top = g_rectangle.tl().y;
		left = g_rectangle.tl().x;
		bottom = g_rectangle.br().y;
		right = g_rectangle.br().x;
		// Find center of square where gaussian will be placed
		int xGauss = static_cast<int>((left+right)/2);
 		int yGauss = static_cast<int>((top+bottom)/2);
		setGoal(image, xGauss, yGauss);
		goalSet = true;
	}
		break;
	}
}

void DrawRectangle(Mat& img, Rect box)
{
	//Draw a rectangle with random color
	rectangle(img, box.tl(), box.br(), Scalar(128,0,128),2);
}

void complexDivide(Mat& G, Mat& F, Mat& H)// Function from a student's piazza post who figured this step out 1st
{
	Mat top, bot;
	mulSpectrums(G, F, top, DFT_COMPLEX_OUTPUT, true); // Top is G*(F conjugate)
	mulSpectrums(F, F, bot, DFT_COMPLEX_OUTPUT, true); // Bot is F*(F conjugate)
	Mat regularize(bot.rows,bot.cols, CV_32FC2, Scalar(.1));
	bot += regularize;

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

void dftQuadSwap(Mat& img)  
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
	Point center = Point(img.cols/2,img.rows/2);
	Mat rotation = getRotationMatrix2D(center, degrees, 1.0);
	warpAffine(img, img, rotation, img.size());
}

void preProcess(Mat& F,Mat& hann)
{
	switchLogScale(F);
	normalize(F, F, 0, 1, CV_MINMAX);
	multiply(F,hann,F);
	getDFT(F);
}

void postProcess(Mat& filter, Mat& hann)
{
	dft(filter, filter, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	dftQuadSwap(filter);
	rotateImg(filter, 180);
	multiply(filter,hann,filter);
	normalize(filter, filter, 0, 1, CV_MINMAX);
}
 
int main(int argc, char** argv) {
	Mat frame, gray, filter, G, F, H, asef, hann, correlate, prev;
	Mat filterTotal = Mat::zeros(Size(640,480), CV_64FC1);
	createHanningWindow(hann, Size(640,480), CV_32F);
	VideoCapture stream(0);   //0 is the id of video device.0 if you have only one camera.
	stream.read(frame);
	if (!stream.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	} 
	setupWindows();
	setMouseCallback("Camera", on_MouseHandle, (void*) &frame);
	int trainCount=0;
	int count = 0;
	float eta = 0.125;// learning rate for running avg of filters
	int index;
	// Loop keeps going until user breaks out of camera
	while (true) {
		stream.read(frame);
		if(!goalSet){
			if(g_bDrawingBox){
				DrawRectangle(frame, g_rectangle);
			}
		}else{
			/*time to do filter averaging stuff for k-frames
			  and figure out at what point to start tracking
			  will it be possible to track with just the initial frame?*/
			if(trainCount < 30){
				DrawRectangle(frame, g_rectangle);
				cvtColor(frame,gray,CV_BGR2GRAY);
				findEdgeMag(gray);
				// multiply(gray,hann,gray);
				gray.copyTo(F);
				preProcess(F,hann);
				goal.copyTo(G);
				getDFT(G);
				G += 0.1;
				complexDivide(G, F, filter); // Find exact filter for this frame
				postProcess(filter,hann);

				// Average filters for ASEF
				Mat temp;
				filter.convertTo(temp, CV_64FC1);
				filterTotal += temp;
				filterTotal.convertTo(H, CV_32F, 1.0/(trainCount + 1));
				normalize(H, H, 0, 1, CV_MINMAX);
				H.copyTo(asef);
				trainCount += 1;
				if(trainCount == 30){
					rotateImg(H, 180);
					dftQuadSwap(H);
					//getDFT(H);
				}
			}else{
				index = count % 30;
				cvtColor(frame,gray,CV_BGR2GRAY);
				findEdgeMag(gray);
				gray.copyTo(F);
  				// goal.copyTo(G);
  				// getDFT(G);
  				preProcess(F, hann);
  				// complexDivide(G, F, filter);
  				// postProcess(filter, hann);
  				// Add this new filter into asef and find new asef
  				// H = filter*eta + (1-eta)*H;
				// rotateImg(H,180);
				// dftQuadSwap(H);
				// Find new goal with new H
				// rotateImg(H, 180);
				// dftQuadSwap(H);
				getDFT(H);
				mulSpectrums(F, H, goal, DFT_COMPLEX_OUTPUT);
				dft(H, H, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
				dftQuadSwap(H);
				rotateImg(H,180);
				dft(goal, goal, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
				// GaussianBlur(goal, goal, Size(7,7),0,0,BORDER_DEFAULT);
				normalize(goal, goal, 0, 1, CV_MINMAX);
				imshow("correlation",goal);
				// threshold(goal,goal,.87,1,THRESH_BINARY);
				float max = 0.0;
				int iMax = 0;
				int jMax = 0;
				for(int i=0; i<goal.rows; i++){
					for(int j=0; j<goal.cols; j++){
						float pixel = goal.at<float>(i,j);
						if(pixel > max){
							max = pixel;
							iMax = i;
							jMax = j;
						}
					}
				}
				int top,left,bottom,right;
				top = iMax - g_rectangle.height/2;
				left = jMax - g_rectangle.width/2;
				bottom = top + g_rectangle.height;
				right = left + g_rectangle.width;				
				rectangle(frame, Point(left,top), Point(right,bottom), Scalar(128,0,128),2);
				setGoal(frame, jMax, iMax);
				goal.copyTo(G);
				getDFT(G);
				G += 0.1;
				complexDivide(G, F, filter);
				postProcess(filter,hann);
				H = filter*eta + (1-eta)*H;
				normalize(H, H, 0,1, CV_MINMAX);
				H.copyTo(asef);
				rotateImg(H,180);
				dftQuadSwap(H);
				dft(G, G, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
				imshow("goal",goal);
				count++;
				// waitKey();
			}
 			// imshow("goal",goal);
 			imshow("edge", gray);
			imshow("filter",filter);
			imshow("asef",asef);
		}
		imshow("Camera", frame);
		if (waitKey(30) == 27)
			break;
	}
	return 0;
}