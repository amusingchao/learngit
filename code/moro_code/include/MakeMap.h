#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>  
#include"opencv2/highgui/highgui.hpp"    
#include"opencv2/imgproc/imgproc.hpp"
#include <stdio.h> 
using namespace std;
using namespace cv;
class BuildMap{
	
	public://声明在类的外部可以访问的开放的成员函数 
	  BuildMap();
	  cv::Mat FloorMap(cv::Mat& inputRgb, cv::Mat& inputDepth,float wheel_x,float wheel_y,float wheel_yaw,float head_yaw,float head_pitch,int ID);
	  cv::Mat Last_Map;
	  cv::Mat Global_Map;
	  cv::Mat Img_Save;//(480,640,CV_8UC3);
	  cv::Mat Global_Map_Count;
	  //int
	  cv::Mat Point_Track;
	  int Point_Track_2[1000][2];
	  
	  //FILE *f1 = fopen("f4.txt","w");
	  float pose_save[1000][5];
	  
	  //cv::Mat Track;
	  //int Last_x;
	  //int Current_y;
	  //int Current_x;
	  //cv::Mat Last_no_obtacle_Map;
	  //cv::Mat Local_Map;
	  int Input_Num;
	  
	  //cv::Mat Last_Map_No_Obtacle;
	  //cv::Mat Last_Map_Obtacle;
	  void print_value();
	  typedef struct _ROBOT_INFOR
    {
	    float camera_yaw;
	    float camera_pitch;
	    float camera_roll;
	    float wheel_xposition;
	    float wheel_yposition;
	    float wheel_yaw;
    }ROBOT_INFOR;
    typedef struct _Quadruple_INFOR
    {
	    float w;
	    float x;
	    float y;
	    float z;
    }Quadruple_INFOR;
    //cv::Mat Rotate_Matrix;
};
