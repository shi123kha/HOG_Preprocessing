
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

//-------------------

Mat negateImage(  Mat src1)
 {

//IplImage *src = cvLoadImage("/home/artivatic/Desktop/18.jpg",1);
IplImage* src = new IplImage(src1);
IplImage *dest = cvCloneImage(src);
 
cvNamedWindow("Original:",1);
cvShowImage("Original:",src);
	 waitKey(0);
 
cvNot(src,dest);//Create a negative image from source image


	 int nNumberOfPixelInImage = dest->width * dest->height;
	// cout<<nNumberOfPixelInImage;
	
	Mat mat= cvarrToMat(dest);
	 
//	cout<<mat.size();


return mat; 
 }
Mat Gaussian(Mat src)
{
	
     Mat dst,src1,diff,img_bw;
	 dilate(src, src1, Mat(), Point(-1, -1), 2, 1, 1);
	imshow("dilate",src1);
	//imwrite( "/home/artivatic/Desktop/dilate.jpg", src1 );

     //Apply median filter
     GaussianBlur( src1, dst, Size( 5, 5 ), 0, 0 );
	imshow("gaussian",dst);
	
	//imwrite( "/home/artivatic/Desktop/gaussian.jpg", dst );
	//255 - absdiff(src, dst);
	absdiff(src,dst,diff);
	//imwrite( "/home/artivatic/Desktop/gaussian.jpg", diff );
	imshow("diff",diff);
	
	Mat diff1;
	
    // imshow("source", src);
	cvtColor(diff, diff1, CV_RGB2GRAY);

	//cvtColor(diff, diff, CV_RGBA2BGRA);
	diff1.convertTo(diff1,CV_8UC1, 255.0);
	cv::threshold(diff1, img_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("threhold",img_bw);
	
		imwrite( "/home/artivatic/Desktop/threhold.jpg", img_bw );
//waitKey(0);
	
	

	//Apha  mask
     return diff;	

}
Mat sobelImage(Mat img)
{
	//  in order to get the edge detection
    img.convertTo(img,CV_32FC1,1.0/255.0);
    Mat h,v,g;
    cv::Sobel(img,h,-1,1,0,3,1.0/8.0);
    cv::Sobel(img,v,-1,0,1,3,1.0/8.0);
    cv::magnitude(h,v,g);

    // Check extremums
    double m,M;
    cv::minMaxLoc(g,&m,&M);
   // cout << m << ":" << M << endl;
    cv::minMaxLoc(h,&m,&M);
    //cout << m << ":" << M << endl;
    cv::minMaxLoc(v,&m,&M);
   // cout << m << ":" << M << endl;


	return g;

}


Mat get_hogdescriptor_visual_image(Mat& origImg,
                                   vector< float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor)
{   
    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
	imshow("visual_image1",visual_image);
 cvtColor(visual_image, visual_image, CV_GRAY2BGR);
	imshow("visual_image",visual_image);
	waitKey(0);

 
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
 int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y< cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x< cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin< gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx< blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky< blocks_in_y_dir; blocky++)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr< 4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin< gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly< cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx< cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin< gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
  //  cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
    // draw cells
    for (int celly=0; celly< cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx< cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(visual_image,
                      Point(drawX*scaleFactor,drawY*scaleFactor),
                      Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin< gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(visual_image,
                     Point(x1*scaleFactor,y1*scaleFactor),
                     Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y< cells_in_y_dir; y++)
    {
      for (int x=0; x< cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visual_image;
 
}





//--------------
int Features(Mat inputImage)
{
   // Mat outputImage;
	
		Mat img3=negateImage(inputImage);
		Mat img2=sobelImage(img3);
	
	Mat outputImage1,outMat;
		Mat outputImage=Gaussian(img2);
		cvtColor(outputImage, outputImage, CV_RGB2GRAY);
	//convert into single 8 bit image
     outputImage.convertTo(outputImage,CV_8UC1, 255.0);

        if(outputImage.rows == 0)
        {
            outputImage=inputImage;

        }
    
       
     Mat r_img1_gray;
 resize(outputImage, r_img1_gray, Size(32, 16));
	 HOGDescriptor d1( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
 // Size(32,16), //winSize
 // Size(8,8), //blocksize
 // Size(4,4), //blockStride,
 // Size(4,4), //cellSize,
 // 9, //nbins,


 //feature compare
 vector< float> descriptorsValues1;
 vector< Point> locations1;
 d1.compute( r_img1_gray, descriptorsValues1, Size(0,0), Size(0,0), locations1);


 //hog visualization
 Mat r1 = get_hogdescriptor_visual_image(r_img1_gray,
                                   descriptorsValues1,
                                   Size(32,16),
                                   Size(4,4),                                   
                                   10,
                                   5);

 imshow("features visualization", r1);
 waitKey(0);
	
	
	
	
	
	/*
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

*/
    //resize the image
    resize(outputImage, outputImage, Size(64, 128));
   // Mat grayImg;
    //convert the image in grayscale
   // cvtColor(outputImage, grayImg, CV_RGB2GRAY);

    // get the points
    vector<Point> locations;
	
	
    hog.compute(outputImage, features, Size(10, 10), Size(0, 0), locations);

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
		//cout<<features.size();
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



