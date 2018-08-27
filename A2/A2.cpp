#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/*
 * Large portions of this code are used from ex05fourier.cpp written by Ross Beveridge.
   His code was written based off of a tutorial online about DFT in OpenCV
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
    const char* filename = argc >=2 ? argv[1] : "cityscape.jpg";

    Mat inImg = imread(filename, CV_LOAD_IMAGE_COLOR);
    if( inImg.empty())
        return -1;

    Mat img;                            //expand input image to optimal size
    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols ); // on the border add zero values
    copyMakeBorder(inImg, img, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    imshow("Padded Source Image", img);
    cout << img.channels() << endl;

    Mat bgr[3];
    split(img,bgr);

    Mat bPlane[] = {Mat_<float>(bgr[0]), Mat::zeros(bgr[0].size(), CV_32F)};
    Mat gPlane[] = {Mat_<float>(bgr[1]), Mat::zeros(bgr[1].size(), CV_32F)};
    Mat rPlane[] = {Mat_<float>(bgr[2]), Mat::zeros(bgr[2].size(), CV_32F)};

    Mat bRI, gRI, rRI;
    merge(bPlane, 2, bRI);
    merge(gPlane, 2, gRI);
    merge(rPlane, 2, rRI);

    dft(bRI, bRI, DFT_COMPLEX_OUTPUT);
    dft(gRI, gRI, DFT_COMPLEX_OUTPUT);
    dft(rRI, rRI, DFT_COMPLEX_OUTPUT);

    Mat inverseTransform;
    Mat channels[3];
    dft(bRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    //channels[0] = inverseTransform;
    channels[0] = Mat::zeros(bRI.size(), CV_32F);
    imshow("Reconstructed B", channels[0]);
    dft(gRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    //channels[1] = inverseTransform;
    channels[1] = Mat::zeros(gRI.size(), CV_32F);
    imshow("Reconstructed G", channels[1]);
    dft(rRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    channels[2] = inverseTransform;
    //channels[2] = Mat::zeros(rRI.size(), CV_32F);
    imshow("Reconstructed R", channels[2]);
    Mat output;
    merge(channels,3,output);
    cout << "Blue:  " << channels[0].size().width << " x " << channels[0].size().height << " x " << channels[0].channels() << endl;
    cout << "Green: " << channels[1].size().width << " x " << channels[1].size().height << " x " << channels[1].channels() << endl;
    cout << "Red:   " << channels[2].size().width << " x " << channels[2].size().height << " x " << channels[2].channels() << endl;
    cout << "out:   " << output.size().width << " x " << output.size().height << " x " << output.channels() << endl;
    imshow("Reconstructed", output);
    imwrite("cityOut.jpg",output);

    // normalize(bRI, bRI, 0, 1, CV_MINMAX);
    // normalize(gRI, gRI, 0, 1, CV_MINMAX);
    // normalize(rRI, rRI, 0, 1, CV_MINMAX);

    // imshow("DFT of blue", bRI);
    // imshow("DFT of green", gRI);
    // imshow("DFT of red", rRI);

    // split(bRI,bPlane);
    // split(gRI,gPlane);
    // split(rRI,rPlane);

    // magnitude(bPlane[0], bPlane[1], bPlane[0]);
    // magnitude(gPlane[0], gPlane[1], gPlane[0]);
    // magnitude(rPlane[0], rPlane[1], rPlane[0]);

    // Mat magB, magG, magR;
    // magB = bPlane[0];
    // magG = gPlane[0];
    // magR = rPlane[0];

    // switchLogScale(magB);
    // switchLogScale(magG);
    // switchLogScale(magR);

    // normalize(magB, magB, 0,1,CV_MINMAX);
    // normalize(magG, magG, 0,1,CV_MINMAX);
    // normalize(magR, magR, 0,1,CV_MINMAX);

    // dftQuadSwap(magB);
    // dftQuadSwap(magG);
    // dftQuadSwap(magR);

    // imshow("DFT of blue", magB);
    // imshow("DFT of green", magG);
    // imshow("DFT of red", magR);

    // Mat inverseTransform;
    // dft(magB, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    // dftQuadSwap(inverseTransform);
    // normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    // bgr[0] = inverseTransform;

    // dft(magG, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    // dftQuadSwap(inverseTransform);
    // normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    // bgr[1] = inverseTransform;

    // dft(magR, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    // dftQuadSwap(inverseTransform);
    // normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    // bgr[2] = inverseTransform;

    // merge(bgr,3,img);
    // imshow("Reconstructed", img);



    // Now construct a Gaussian kernel
    // float sigma = 2.0;

    // Mat kernelX   = getGaussianKernel(img.rows, sigma, CV_32FC1);
    // Mat kernelY   = getGaussianKernel(img.cols, sigma, CV_32FC1);
    // Mat kernel  = kernelX * kernelY.t();
    // Mat kernel_d = kernel.clone();
    // normalize(kernel_d, kernel_d, 0, 1, CV_MINMAX);
    // imshow("Spatial Domain", kernel_d);

    // Build complex images for both the source image and the Gaussian kernel
    //Mat imgPlanes[] = {Mat_<float>(img(0)),    Mat::zeros(img(0).size(),    CV_32F)};
    // //Mat kerPlanes[] = {Mat_<float>(kernel), Mat::zeros(kernel.size(), CV_32F)};
    // Mat imgRI, prdRI;
    // //Mat kerRI
    // merge(imgPlanes, 2, imgRI);
    // //merge(kerPlanes, 2, kerRI);
    // prdRI = imgRI.clone();
    // dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
    // //dft(kerRI, kerRI, DFT_COMPLEX_OUTPUT);
    // //complexMultiply(imgRI, kerRI, prdRI);
    // //mulSpectrums(imgRI, kerRI, prdRI, DFT_COMPLEX_OUTPUT);

    // Mat inverseTransform;
    // dft(imgRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    // dftQuadSwap(inverseTransform);
    // normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    // imshow("Reconstructed", inverseTransform);
    waitKey();
    return 0;
}
