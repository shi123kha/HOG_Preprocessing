


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
Mat Hogfeat;
vector<float> features;
vector<String> filenames;

int Path(char* csv_path ,char * path,char *  index)
{
if(csv_path!=NULL&& path!=NULL&&index!=NULL )
{
	    string line1, csvItem;
	int category=atoi(index);
	
            char output_file_name[512], file[512];

            sprintf(file, "%s/dynamic_name.csv", csv_path);

ifstream myfile1(file);
            int lineNumber = 0;
            char dir_name[100];
            char match_url[500];

            if (myfile1.is_open()) { //name csv is open
				
                while (getline(myfile1, line1)) {
                    lineNumber++;
                    /* check the lineNumber with label we get
                    and increase the value of lineNumber untill
                    equal to label
                    */
					//	cout<<"came";
					//cout<<"index"<<index;
                    if (lineNumber == category) {
                        istringstream myline(line1);
                        // get the line in the csv file
                        while (getline(myline, csvItem, ',')) {
                            //copy the string 
                            strcpy(dir_name, csvItem.c_str());
                            // add the path
                            sprintf(match_url, "%s/%s", path, dir_name);
                            // output path of the match image
                            sprintf(output_file_name, "%s/image_0013.jpg", match_url);
                            // print the path of match image
                            cout << output_file_name;
                        }
                    }
                }

                myfile1.close(); //close the read of name csv file
            }
}
	else
	{
	cout<<"Not correct Path";	
	}
}
int main(int argc ,char* argv[])
{
	
	Path(argv[1],argv[2], (argv[3]));
	return 0;
	
}