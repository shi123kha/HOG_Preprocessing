
#include <iostream>
#include <math.h>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <fstream>
#include <ctype.h>


using namespace std;
using namespace cv;
HOGDescriptor hog;
Mat Hogfeat;   /* cantain the HOG features*/
vector<float> features;
Mat Cropping(Mat large)
{
    Mat rgb;   /*color image*/
    Mat rgb2;
    // downsample and use it for processing
    pyrDown(large, rgb);
    Mat small;
    /*convert into gray scale*/
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    adaptiveThreshold( grad, bw, 255, ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,3,0 ); //Threshold the gray


    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    try {
        // filter contours
        for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            Rect rect = boundingRect(contours[idx]);
            Mat maskROI(mask, rect);
            maskROI = Scalar(0, 0, 0);
            // fill the contour
            drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
            // ratio of non-zero pixels in the filled region
            double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

            if (r > .45 /* assume at least 45% of the area is filled if it contains text */
                    &&
                    (rect.height > 8 && rect.width > 8) /* constraints on region size */
                    /* these two conditions alone are not very robust. better to use something
                    like the number of significant peaks in a horizontal projection as a third condition */
               )
            {
                //rectangle(rgb, rect, Scalar(0, 255, 0), 2);
                rgb2=rgb(rect);
            }
        }
    }
    catch( cv::Exception& e )
    {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }
    return rgb2;
}


int Features(Mat inputImage)
{
    Mat outputImage;
    try {

        outputImage = Cropping(inputImage);

        if (outputImage.rows == 0) {
            outputImage = inputImage;
        }
    }

    catch (const std::exception& e)

    {
        std::cout << e.what() << std::endl;
    }

    //resize the image
    resize(outputImage, outputImage, Size(64, 128));
    Mat grayImg;
    //convert the image in grayscale
    cvtColor(outputImage, grayImg, CV_RGB2GRAY);

    // get the points
    vector<Point> locations;
    hog.compute(grayImg, features, Size(10, 10), Size(0, 0), locations);

    Hogfeat.create(features.size(), 1, CV_32FC1);

    return 0;
}

int Test(char *file_names)
{
if(file_names!=NULL)
{
  Mat I;
    char name_csv[512], detail_csv[512];
	if(file_names!=NULL)
	{
		I = imread(file_names, 1);


    try {
        Features(I);
    }
    catch(cv::Exception& e)
    {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }
    //size of features of image
    for (int i = 0; i < features.size(); i++) {
        Hogfeat.at<float>(i, 0) = features.at(i);
        int temp = Hogfeat.at<float>(i, 0);
		cout<<  features.at(i)<<",";
    }
		
	}
}else
{
cout<<"Image not found";	
}
}
int main(int argc, char* argv[])
{
Test(argv[1]);
return 0;
}



