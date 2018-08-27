#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/*
 * In this example we do smoothing with a Gaussian, but in the
 * Frequency domain.
 *
 * Ross Beveridge

  tutorial website:
  http://www.swarthmore.edu/NatSci/mzucker1/opencv-2.4.10-docs/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
 *
 */

void dftQuadSwap (Mat& img)  {
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

void switchLogScale(Mat& img) {
    img += Scalar::all(1);                    // switch to logarithmic scale
    log(img, img);
}

void complexMultiply(Mat& s1, Mat& s2, Mat& res){
	Mat s1p[]  = {Mat_<float>(s1),  Mat::zeros(s1.size(),  CV_32F)};
    Mat s2p[]  = {Mat_<float>(s2),  Mat::zeros(s2.size(),  CV_32F)};
    Mat resp[] = {Mat_<float>(res), Mat::zeros(res.size(), CV_32F)};
	split(s1, s1p);
	split(s2, s2p);
    split(res, resp);
    // real then the imaginary part of the result
    resp[0] = (s1p[0] * s2p[0]) - (s1p[1] * s2p[1]);
    resp[1] = (s1p[0] * s2p[1]) + (s1p[1] * s2p[0]);
    merge(resp, 2, res);
}

int main(int argc, char ** argv)
{

    //  Start by loading the image to be smoothed
    const char* filename = argc >=2 ? argv[1] : "colostate_quad_bw_512.png";

    Mat inImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if( inImg.empty())
        return -1;

    Mat img;                            //expand input image to optimal size
    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols ); // on the border add zero values
    copyMakeBorder(inImg, img, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    imshow("Padded Source Image", img);

	// Now construct a Gaussian kernel
    // float sigma = 2.0;

    // Mat kernelX   = getGaussianKernel(img.rows, sigma, CV_32FC1);
    // Mat kernelY   = getGaussianKernel(img.cols, sigma, CV_32FC1);
    // Mat kernel  = kernelX * kernelY.t();
    // Mat kernel_d = kernel.clone();
    // normalize(kernel_d, kernel_d, 0, 1, CV_MINMAX);
    Mat kernel;
    kernel = Mat_<float>(3,3) <<
        0,0,0,
        0,0,0,
        0,0,0;
	//imshow("Spatial Domain", kernel_d);
    int rowPad = (img.rows - kernel.rows)/2;
    int colPad = (img.cols - kernel.cols)/2;
    copyMakeBorder(kernel,kernel, rowPad, rowPad, colPad, colPad, BORDER_CONSTANT, Scalar::all(1));
    normalize(kernel, kernel, 0, 1, CV_MINMAX);
    imshow("kernel",kernel);
    // Build complex images for both the source image and the Gaussian kernel
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat kerPlanes[] = {Mat_<float>(kernel), Mat::zeros(kernel.size(), CV_32F)};
    Mat imgRI, kerRI, prdRI;
    merge(imgPlanes, 2, imgRI);
    merge(kerPlanes, 2, kerRI);
    prdRI = imgRI.clone();
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
    //dftQuadSwap(imgRI);
    // dft(kerRI, kerRI, DFT_COMPLEX_OUTPUT);
    complexMultiply(imgRI, kerRI, prdRI);
    mulSpectrums(imgRI, kerRI, prdRI, DFT_COMPLEX_OUTPUT);

    Mat inverseTransform;
    dft(prdRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    //dftQuadSwap(inverseTransform);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    imshow("Reconstructed", inverseTransform);
    waitKey();
    return 0;
}
