
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
ofstream myfile;

ifstream myfile1("/home/artivatic/Desktop/MachineLearning/Desktop/new.csv");
string filename="/home/artivatic/Desktop/MachineLearning/Desktop/new.csv";

void saveMatToCsv(Mat m){
     
	myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;     
}
main()
{
	std::vector<double> om2;
	// std::ofstream ofs_detail;
  myfile.open(filename.c_str());	
Mat img1=imread("/home/artivatic/Desktop/MachineLearning/Desktop/52.jpg",1);
resize(img1, img1, Size(10,10) );

	
Mat img2=imread("/home/artivatic/Desktop/MachineLearning/Desktop/threhold.jpg",1);
	resize(img2, img2, Size(10,10) );
	
	int count=0;
	cout<<img1.size()<<endl;
	Mat vec1 = img1.reshape(0,1);
	Mat vec2=img2.reshape(0,1);
	cout<<vec1.channels();
	//cout<<vec1.size();
	Mat outputMat=Mat(1,vec2.cols,CV_32FC3, cv::Scalar(0, 0, 255));
	
	//cout<<vec1<<endl;
	//cout<<vec2<<endl;
	
	//Mat img3;

//	cout<<vec2.cols<<"endl";
	std::vector<uchar> last;
	 std::vector<std::vector<uchar> > output; 
	cout<<outputMat.size();
	 
	for(int i=0;i<vec2.cols;i++)
	{
		//cout<<i<<endl;
	 	int data2 = vec2.at<uchar>(0,i);
		int data1 = vec1.at<uchar>(0,i);
		//cout<<data1<<"value"<<count++<<endl;
		if(data2==0)
		{
			
			  outputMat.at<uchar>(0,i) = data2;
	
			//cout<<data2;
		}
		else
		{
			
			outputMat.at<uchar>(0,i) = data1;
			//cout<<data1;
			
			//last.push_back(data1);
			
		}
		
		//outputMat.at<uchar>(0,i)=((data2==0)?0:data1);
		
		//output.push_back(last);
		
		//std::cout << getValueAs<uchar>(i) << " " << (int)getValueAs<uchar>(i);*/
		
		//cout<<(data1 & data2)<<" ";
		
	}
	Mat temp2=outputMat.reshape(10,10);
	cout<<"\n"<<temp2;
imshow("temp2",temp2);
	//waitKey();
//cout<<vec3.size();
//Mat input_feature_vector(output.size(),output.at(0).size(),CV_32F);

   /* for(int i=0;i<output.size();i++)
    {
       //for(int j=0;j<output.at(0).size();j++)
        {
          // In order to get the value offeatures from csv file
			//input_feature_vector.at<float>(j,i)=output.at(j).at(i);
			//cout<<output.at(i).at(0)<<"endl";
			
        }
    }*/

//	cout<<input_feature_vector;
	//cout<<input_feature_vector.size();
	//cout<<output.at(0).size()<<endl;
	//saveMatToCsv(vec1);

//	 myfile << '\n';
	//saveMatToCsv(vec2);
	 myfile.close();
	string line;

/*
	for (int i=0;i<vec1.cols;i++)
{
    if(vec2.at<float>(0,i)==0)
    {
		cout<<vec2.at<float>(0,i);
     // std::cout<<vec2[i]<<"\t"<<vec1[i]<<": Match: \n";
    }
    else {
      cout<<vec2.at<float>(0,i)<<endl;
     // std::cout<<vec[i]<<"\t"<<vec1[i]<<":Doesnt Match: \n";
    }

}

*/     if (myfile1.is_open()) { // detail csv is open
            int count = 0;
            float sum = 0;
		  
            // get the line of csv file
            while (getline(myfile1, line, ',')) {
				//cout<<"count";
				if(count==300)
				{	
				//cout<<"end"<<endl;	
				count=0;
				}
				else
				{
				
				count++;	
				}
				
                // totall number of features is 3780
             
			}
		   myfile1.close();
	   }
	
}
