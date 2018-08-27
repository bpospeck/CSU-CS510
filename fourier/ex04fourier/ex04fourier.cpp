#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/* As a prelude to the next example, this is smoothing with a
 * Gaussian in the Spatial Domain.  In other words, standard
 * convolution.
 *
 * Ross Beveridge
 *
 */

int main(int argc, char ** argv)
{
    const char* filename = argc >=2 ? argv[1] : "colostate_quad_bw_512.png";

    Mat inImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Source Image", inImg);
	// Now construct a Gaussian kernel
    float sigma = 8.0;
    Mat kernelX   = getGaussianKernel(4*sigma, sigma, CV_32FC1);
    Mat kernelY   = getGaussianKernel(4*sigma, sigma, CV_32FC1);
    Mat kernel  = kernelX * kernelY.t();
    Mat kernel_d = kernel.clone();
    cout << kernel_d.rows << "\n";
    cout << kernel_d.cols << "\n";
    normalize(kernel_d, kernel_d, 0, 1, CV_MINMAX);
	imshow("Gaussian Kernel", kernel_d);
    // Use the built in convolution function for OpenCV
	Point anchor = Point(-1,-1);
    Mat resImg = inImg.clone();
	filter2D(inImg, resImg, -1, kernel, anchor, BORDER_DEFAULT);
	normalize(resImg, resImg, 0, 255, NORM_MINMAX);
    imshow("Smoothed", resImg);
    waitKey();
    return 0;
}
