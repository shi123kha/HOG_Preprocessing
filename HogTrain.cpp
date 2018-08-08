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
vector<String> filenames;   /*directory name*/
int label=1;

Mat negateImage(Mat src1)
 {
		// In order to get the negate of image 
		 IplImage* src = new IplImage(src1);
		IplImage *dest = cvCloneImage(src);
		cvNot(src,dest);//Create a negative image from source image
		int nNumberOfPixelInImage = dest->width * dest->height;
		Mat mat= cvarrToMat(dest);
        return mat; 
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
  //  cout << m << ":" << M << endl;
    cv::minMaxLoc(h,&m,&M);
   // cout << m << ":" << M << endl;
    cv::minMaxLoc(v,&m,&M);
 
	return g;

}
Mat Gaussian(Mat src)
{	
     Mat dst,src1,diff,img_bw;
	 dilate(src, src, Mat(), Point(-1, -1), 2, 1, 1);
    //Apply median filter
     GaussianBlur( src, dst, Size( 5, 5 ), 0, 0 );
	absdiff(src,dst,diff);
     return diff;	

}


/* cropping the image*/

Mat Cropping(Mat large)
{
    Mat rgb;   /*color image*/
    Mat rgb2;
    // downsample and use it for processing
    pyrDown(large, rgb);
    Mat character;
	
    /*convert into gray scale*/
    cvtColor(rgb, character, CV_BGR2GRAY);
	character.convertTo(character, CV_8U);
threshold(character, character, 0, 255, 1);
	
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(character, grad, MORPH_GRADIENT, morphKernel);
	

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
   /* findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
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
                  /*  &&
                    (rect.height > 8 && rect.width > 8) /* constraints on region size */
                    /* these two conditions alone are not very robust. better to use something
                    like the number of significant peaks in a horizontal projection as a third condition */
        /*       )
				   
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
	*/
	return connected;
}

/*Getting the features of images*/


int Features(Mat inputImage)
{
    Mat outputImage,img1,img2;
	
    try
    {
		img1=negateImage(inputImage);
		 img2=sobelImage(img1);
	
	
		outputImage=Gaussian(img2);
		cvtColor(outputImage, outputImage, CV_RGB2GRAY);
	//convert into single 8 bit image
     outputImage.convertTo(outputImage,CV_8UC1, 255.0);

        if(outputImage.rows == 0)
        {
            outputImage=inputImage;

        }
    }

    catch(const std::exception& e)

    {
        std::cout << e.what() << std::endl;
    }


    //resize the image
    resize(outputImage, outputImage, Size(64,128) );


    // get the points 
    vector<Point> locations;
    hog.compute(outputImage,features,Size(10,10), Size(0,0),locations);
    Hogfeat.create(features.size(),1,CV_32FC1);

    return 0;
	
}


/*
For train  all the image of the folder
*/

int Training(char *image_dir,char *csv_path)
{

    char dir_path[512];
    Mat image;
    char name_csv[512],detail_csv[512];
    DIR *dir;
    struct dirent *ent;
    String dirName;

	if(image_dir!=NULL && csv_path!=NULL)
	{
	//cout<<"image not found";	
	
	
    //contain the name of dir folder  in the csv

    sprintf(name_csv,"%s/dynamic_name.csv",csv_path);

    // contain the detail of each image in the csv file

    sprintf(detail_csv,"%s/dynamic_detail.csv",csv_path);
    std::ofstream ofs_detail,ofs_name;

    ofs_name.open(name_csv,std::ofstream::out);
    ofs_detail.open (detail_csv, std::ofstream::out);
    try {

        if ((dir = opendir (image_dir)) != NULL) {
            // print all the files and directories within directory 
            // start of while loop 


            while ((ent = readdir (dir)) != NULL) {
                //Get the directory name
                dirName=ent->d_name;
                std::string str(image_dir);

                if(dirName.compare(".")!= 0 && dirName.compare("..")!= 0)
                {
                    String dir_path=str+(ent->d_name)+"/";
                    int count1=0;
				//	cout<<dirName<<endl;
                    //add the directory name in csv file
                    ofs_name<<dirName<<',';
                    //get the image of each directory
                    try {
                        glob(dir_path, filenames);
				//		cout<<filenames.size()<<endl;
                        for (size_t i=0; i<filenames.size(); i++)
                        {
						//	cout<<filenames[i]<<endl;
							// get the image of each directory
                            image = imread(filenames[i]);
							
                            //for Hog features of image
                            Features(image);
                            //total features of image
							//cout<<"size"<<features.size();
							//cout<<"size"<<features.size();
                            for(int i=0; i<=features.size(); i++)

                            {
                                if(i==features.size())
                                {

                                    Hogfeat.at<float>(i,0)=label;
                                    //at the end add the label that defile the class in the csv file
                                    ofs_detail <<label <<',';
									//cout<<label;
                                }
                                else {
                                    Hogfeat.at<float>(i,0)=features.at(i);
                                    //add the features in csv file
                                    ofs_detail <<features.at(i) <<',';

                                }
                            }
                            // for new line
                            ofs_detail << '\n';

                        }
                    }

                    catch( cv::Exception& e )
                    {
                        const char* err_msg = e.what();
                        std::cout << "exception caught: " << err_msg << std::endl;
                    }
                    label++;
                    ofs_name<<'\n';

                }

            }// End of while loop



            cout<<"Training is done";
            closedir (dir);

        }
        else {
            // could not open directory 
            perror ("");
            return EXIT_FAILURE;
        }
    }
    catch( const runtime_error& erro )
    {
        cout << erro.what() << "\n";
    }
    ofs_name.close();   //end of name csv file
    ofs_detail.close();  
	//end of detail csv file
}
	else
	{
		cout<<"Path not found";
	}
    return 0;
}

int main(int argc, char *argv[])
{
    Training(argv[1],argv[2]);   /*for Train the image*/
    return 0;
}