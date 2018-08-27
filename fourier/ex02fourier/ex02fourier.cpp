#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;

/* This code is from the OpenCV Fourier example
 * http://docs.opencv.org/doc/tutorials/core/
 * discrete_fourier_transform/discrete_fourier_transform.html
 *
 * I am modifying it to play with moving common operations outside of main.
 *
 * This version also show explicitly how it is proper to permute quadrants
 * in order to arrive at the visualization most people assume is standard,
 * with the DC component in the center.
 *
 * Here is an excellent online reference by  Paul Bourke
 * http://paulbourke.net/miscellaneous/imagefilter/
 *
 * Ross Beveridge
 *
 */

Mat& dftQuadSwap (Mat& img)  {
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

    return(img);
}

Mat& switchLogScale(Mat& img) {
    img += Scalar::all(1);                    // switch to logarithmic scale
    log(img, img);
	return(img);
}

int main(int argc, char ** argv)
{
    const char* filename = argc >=2 ? argv[1] : "colostate_quad_bw_1024.png";

    Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if( I.empty())
        return -1;
    imshow("Source Image", I);
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded to another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI   = planes[0];
    magI = switchLogScale(magI);
    normalize(magI, magI, 0, 1, CV_MINMAX);

    imshow("DFT Output Raw", magI);
    magI = dftQuadSwap(magI);
    imshow("DFT After Quad Swap", magI);

    waitKey();

    return 0;
}
