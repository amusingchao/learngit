#include<iostream>
#include <stdio.h>
#include"MakeMap.h"
#include <fstream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>  
#include"opencv2/highgui/highgui.hpp"    
#include"opencv2/imgproc/imgproc.hpp"
#include<sys/time.h>
using namespace std;
using namespace cv;
#define num_image 570
void getData(float data[570][5])
{
    FILE *fp;
    int i = 0, j = 0;
    fp = fopen("posdata.txt", "r");
    if(fp == NULL)
    {
        printf("cannot open file!\n");
        return;
    }
    while(!feof(fp))
    {
        if(j != 4)
            fscanf(fp, "%f  ", &data[i][j++]);//注意此处有个空格
        else
        {
            fscanf(fp, "%f", &data[i][j]);
            j = 0;
            ++i;
        }
        //printf("aa is %d ", i);
    }
    fclose(fp);
}
int main(){

  float data[570][5];
  //int i, j;
  getData(data);
  
  cv::Mat imD;
  cv::Mat im_rgb;
  BuildMap build_map;
  cv::Mat outputMap_now;
  float wheel_x,wheel_y,wheel_yaw,head_yaw,head_pitch;
  int jj = 0;
  //char filename[100] = "/home/xinchao/dd/explore/ORB_SLAM2/prj/ZedData/";
  char cmd[50]= ".png"; 
  string png = ".png";
  string mm;
  string nn;
  
  cv::Mat rgb_img = cv::imread("rgb_img1.png");
  cv::Mat rgb_img1 = rgb_img(Range(0,720),Range(0,1280)) ;
  cv::imwrite("uuu.png",rgb_img1);
  cout<< rgb_img1.size()<<endl;
  cv::Mat img_3c(480,640,CV_8UC3);
  //img_3c.create(480,640,3,CV_32FC3);  
  
  //string ee;
  for(int i=0;i<570;i++)
  {
    
    char dd[15]; 
    //sprintf(dd,"%d",i+1); 
    //ee = strcat(dd,cmd);
    mm = std::to_string(i+1);
    std::cout<<mm+png<<std::endl;
    //imD = cv::imread(strcat(dd,cmd),CV_LOAD_IMAGE_UNCHANGED);
    imD = cv::imread(mm+png,CV_LOAD_IMAGE_UNCHANGED);
    //sprintf(dd,"%d",i+201); 
    //cout<<strcat(dd,cmd)<<endl;
    //mm = std::to_string(i+201);
    //std::cout<<png<<std::endl;
    nn = std::to_string(i + 1801);
    im_rgb = cv::imread(nn + png,0);
    
    //im_rgb = cv::imread("201.png",0);
    //cv::imwrite("mm.png", im_rgb);
    wheel_yaw = data[i][2];//0.001687*600;//0.001687;
    head_yaw = data[i][4];//*600;// -0.017959;
    head_pitch = data[i][3]; //3.1415926/4;     //0.002587*600;// 0.002587*600;//0;//0.002587*600;//0.002587;
    wheel_x = data[i][0];  
    wheel_y = data[i][1]; 
    cout<<"pic num:"<<i+1<<endl;
    cout<<"wheel_x:"<<wheel_x<<endl;
    cout<<"wheel_y"<<wheel_y<<endl;
    cout<<"wheel_yaw:"<<wheel_yaw<<endl;
    cout<<"head_yaw"<<head_yaw<<endl;
    cout<<"head_pitch"<<head_pitch<<endl;
    std::cout<<"aaaaaaaaaaaaaa"<<std::endl;
    outputMap_now = build_map.FloorMap(im_rgb,imD,wheel_x,wheel_y,wheel_yaw,head_yaw,head_pitch,1);
     
     
     
    cout<<"save:"<<dd<<endl;
    sprintf(dd,"%d",i+901); 
    //cv::imwrite(strcat(c,cmd), tmpImg);
    cv::imwrite(strcat(dd,cmd), outputMap_now);
  }
  
  
  
  
  
  
  
  //RobX 0.677832  RobY 0.000444  RobYaw 0.002822 HeadPitch 0.002587  HeadYaw -0.017959
  
  
	return 0;
}
