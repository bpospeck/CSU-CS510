#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;

/* This code started out as the OpenCV Fourier example
 * http://docs.opencv.org/doc/tutorials/core/
 * discrete_fourier_transform/discrete_fourier_transform.html
 *
 * I am modifying it to play with filtering in the Fourier domain
 * In this example we take the Fourier Transform of a Gaussian Kernel
 *
 * Ross Beveridge
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

int main(int argc, char ** argv)
{
    // Now create an image which is a Gaussian Kernel
	int dim = 512;
    float sigma = 16.0;

    Mat kernelX   = getGaussianKernel(dim, sigma, CV_32FC1);
    Mat kernelY   = getGaussianKernel(dim, sigma, CV_32FC1);
    Mat kernelXY  = kernelX * kernelY.t();
    Mat kernelXYd = kernelXY.clone();
    normalize(kernelXYd, kernelXYd, 0, 1, CV_MINMAX);
	imshow("Spatial Domain", kernelXYd);

    Mat planes[] = {Mat_<float>(kernelXY), Mat::zeros(kernelXY.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI   = planes[0];
    // switchLogScale(magI);
    normalize(magI, magI, 0, 1, CV_MINMAX);

    dftQuadSwap(magI);
    imshow("DFT of Guassian", magI);

    waitKey();

    return 0;
}
