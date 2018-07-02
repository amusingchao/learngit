#include<iostream>
#include"MakeMap.h"
#include"opencv2/core/core.hpp"    
#include"opencv2/highgui/highgui.hpp"    
#include"opencv2/imgproc/imgproc.hpp"
#define INPUT_DISP 0
static float Dxlow = -20.f;
static float Dxup = 20.f;
static float Dylow = -20.f;
static float Dyup = 20.f;
static int Dwidth = 1000;
static int Dheight = 1000;
static int Degree = 30;
static int Pi = 3.1415926;
//static float fx = 331.4490965;
//static float fy = 331.4490965;
//static float cx = 318.571991;
//static float cy = 178.753937;
static float a = 0.0232774;
static float b = -0.558345;
static float c = -0.395061;
static float fx = 468.9011;
static float fy = 468.9011;
static float cx = 324.0461;
static float cy =  236.3762;
//const float DepthMapFactor= 0.3/65536;
static float DepthMapFactor= 5000;
using namespace std;
static int fun(float x0, float y0,int *index_x,int *index_y)
{
    int index_x_tmp,index_y_tmp;
    //cout<<"x0 "<<x0<<endl;
    //cout<<"y0 "<<y0<<endl;
    if (x0 <= Dxlow || x0 >= Dxup || y0 <= Dylow || y0 >= Dyup)
    {
        cout<< "-1"<<endl;
        return -1;
    }
    else
    {
        index_x_tmp = (x0 - Dxlow)/(Dxup - Dxlow)*Dwidth;
        index_y_tmp = (y0 - Dylow)/(Dyup - Dylow)*Dheight;
        *index_x=index_x_tmp;
        *index_y=index_y_tmp;
        //cout<<index_x_tmp<<endl;
        //cout<<index_x_tmp<<endl;
        
        return 1;
    }
}
void BuildMap::print_value()
{
  cout<<"make map :"<<endl;
}
void RotatePlane(const cv::Mat& normVec, cv::Mat& rotateAxis)
{
    float x = normVec.at<float>(0);
    float y = normVec.at<float>(1);
    float z = normVec.at<float>(2);
    
    float norm_xy = x*x + y*y;
    float norm_xyz = sqrt(norm_xy + z*z);
    norm_xy = sqrt(norm_xy);

    float cos_alpha = z/ norm_xyz;
    float sin_alpha = norm_xy/norm_xyz;

    float nx = y/norm_xy;
    float ny = -x/norm_xy;
    float nz = 0;

    float base_data[] = {0, -nz, ny, nz, 0, -nx, -ny, nx, 0};
    float I_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    cv::Mat base(3, 3, CV_32F, base_data);
    cv::Mat I(3, 3, CV_32F, I_data);

    rotateAxis = I + sin_alpha*base + (1-cos_alpha)*base*base;
    //rotateAxisInverse = rotateAxis.inv();
}
void BuildMap::FloorMap(cv::Mat& inputGgb,cv::Mat& inputDepth,float *wheel_x,float *wheel_y,float *wheel_z,float *yaw,cv::Mat outputMap)
{
  cv::Mat coord;
  int W = inputGgb.cols;
  int H = inputGgb.rows;
  
  float c1=-1.0/c;
  float a1=a*c1;
  float b1=b*c1;
  float inorm_abc=1.0 / sqrt(a *a+ b*b+ c*c);
  
  coord.create(H*W, 3, CV_32F);
  float* ptr = coord.ptr<float>(0);
  unsigned short int* pdepth = inputDepth.ptr<unsigned short int>(0);
  
  double degree_yaw;
  double tmp_degree;
  float degree_y;
  float dist_yuandian;
  int map_index_x=0;
  int map_index_y=0;
  cv::Mat Twc_2d(3, 3, CV_32F);   //2dpose
  Twc_2d.setTo(0.f);
  Twc_2d.at<float>(0,0) = cos(degree_yaw);
  Twc_2d.at<float>(0,1) = -1*sin(degree_yaw);
  Twc_2d.at<float>(1,1) = cos(degree_yaw);
  Twc_2d.at<float>(1,0) = sin(degree_yaw);
  Twc_2d.at<float>(0,2) = *wheel_x;
  Twc_2d.at<float>(1,2) = *wheel_z;
  Twc_2d.at<float>(2,2) = 1.0;
  cv::Mat local_point_2d(3, 1, CV_32F);
  local_point_2d.setTo(0.f);
  cv::Mat global_point_2d(3, 1, CV_32F);
  global_point_2d.setTo(0.f);
  
  cv::Mat outputMap1(1000, 1000, CV_8UC1);
  outputMap1.setTo(0);
  
  //Mat img(500, 500, CV_8U, Scalar(0));  
  
  //Point root_points[1][3];  
  //root_points[0][0] = Point(130,500);  
  //root_points[0][1] = Point(230,600);  
  //root_points[0][2] = Point(230,400);  
  //root_points[0][3] = Point(235,465);  

  //const Point* ppt[1] = {root_points[0]};  
  //int npt[] = {3};  
  //polylines(outputMap1, ppt, npt, 1, 1, Scalar(255),1,8,0);   
  //fillPoly(outputMap1, ppt, npt, 1, Scalar(255));  
  //imshow("Test", outputMap1);  
  //waitKey(); 
  
  //int r = 50;//半径  
  //Point center = Point(150, 50);  
  //circle(picture, center, r, Scalar(123, 21, 32), -1);
  unsigned char* poutmap = outputMap1.ptr<unsigned char>(0);
  int jj=11;
  for (int mm=400;mm<500;mm++)
  {
  outputMap1.at<unsigned char>(500,mm)=255;
  
  }
  for (int i = 0; i < H; ++i)
  {
    for (int j = 0; j < W; ++j, ptr+=3, pdepth+=1)
    {
      
      //ptr[0] = (39.713978/(640*(*pdepth)) * (j - cx) /fx;       
      //ptr[1] = (39.713978/(640*(*pdepth)*DepthMapFactor))  * (i - cy) /fy;       
      //ptr[2] = (39.713978/(640*(*pdepth)*DepthMapFactor)) ;
    
      ptr[0] = (*pdepth)/5000.0* (j - cx) /fx;       
      ptr[1] = (*pdepth)/5000.0 * (i - cy) /fy;       
      ptr[2] = (*pdepth)/5000.0;
      
      if(0)
      {
        std::cout<<"*pdepth "<<(*pdepth)/5000<<std::endl;
        std::cout<<"cx"<<cx<<std::endl;
        std::cout<<"cy"<<cy<<std::endl;
        std::cout<<"invfx"<<fx<<std::endl;
        std::cout<<"invfy"<<fy<<std::endl;
        std::cout<<"ptr[0]"<<(39.713978/(640*(*pdepth)*DepthMapFactor)) * (j - cx)/fx<<std::endl;
        std::cout<<"ptr[1]"<<(39.713978/(640*(*pdepth)*DepthMapFactor))  * (i - cy)/fy<<std::endl;
        std::cout<<"ptr[2]"<<(39.713978/(640*(*pdepth)*DepthMapFactor))<<std::endl;
        std::cout<<"xxx"<<ptr[0]<<std::endl;
      }
      std::cout<<"ptr[2]"<<(*pdepth)/5000.0<<std::endl;
      //coord.at<float>(0,0)
      //std::cout<<"*pdepth"<<(*pdepth)<<std::endl;
      float dist = (a*ptr[0] + b*ptr[1] + c*ptr[2] + 1) * inorm_abc;
      float zz = (a1*ptr[0]+b1*ptr[1]+c1+a1*a1*ptr[2]+b1*b1*ptr[2])/(a1*a1+b1*b1+1);
      float xx = ptr[0]+a1*(ptr[2]-zz);
      float yy = ptr[1]+b1*(ptr[2]-zz);   //假设地面方程不变化，即全局地面方程也不变，xx，zz就是二维平面投影坐标投影点
      //std::cout<<"dist"<<dist<<std::endl;
      //std::cout<<"xx"<<xx<<std::endl;
      //std::cout<<"yy"<<yy<<std::endl;
      //std::cout<<"zz"<<zz<<std::endl;
      
      
      if(abs(dist) < 0.09)
      {   
          degree_y=(atan(abs(xx/zz)))/Pi*180;
          dist_yuandian=sqrt(xx*xx+zz*zz);
          //cout<<"degree_y:"<<degree_y<<endl;
          if((degree_y<30) && (dist_yuandian<10))
          {
            
          //local_point_2d.at<float>(0)=xx;
          //local_point_2d.at<float>(1)=zz;
          //local_point_2d.at<float>(2)=1.0;
          //global_point_2d = Twc_2d*local_point_2d;
          int map_flag = fun(xx, zz,&map_index_x,&map_index_y);
          //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_x);
          poutmap[map_index_y] = 255;
          }
          
          //cout<<"outmap: "<<outputMap1.at<unsigned char>(map_index_x,map_index_y)<<endl;
          //cout<<"outmap: "<<"print ok"<<endl;
          //cout<<"tmpImg: "<<tmpImg.at<unsigned char>(1,1)<<endl;
          //cout<<"index: "<<map_index_y<<endl;
          //outputMap.at<unsigned char>(map_index_x,map_index_y)=255;
         // unsigned char* pomap = outputMap.ptr<unsigned char>(0);
         // pomap[index]=255;
          
      }
      if((0.1<abs(dist))&& (abs(dist)<1.8))
      {   
          degree_y=(atan(abs(xx/zz)))/Pi*180;
          dist_yuandian=sqrt(xx*xx+zz*zz);
          //cout<<"degree_y:"<<degree_y<<endl;
          if((degree_y<30) && (dist_yuandian<10))
          {
            
          //local_point_2d.at<float>(0)=xx;
          //local_point_2d.at<float>(1)=zz;
          //local_point_2d.at<float>(2)=1.0;
          //global_point_2d = Twc_2d*local_point_2d;
          int map_flag = fun(xx, zz,&map_index_x,&map_index_y);
          //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_x);
          poutmap[map_index_y] = 125;
          }
          
          //cout<<"outmap: "<<outputMap1.at<unsigned char>(map_index_x,map_index_y)<<endl;
          //cout<<"outmap: "<<"print ok"<<endl;
          //cout<<"tmpImg: "<<tmpImg.at<unsigned char>(1,1)<<endl;
          //cout<<"index: "<<map_index_y<<endl;
          //outputMap.at<unsigned char>(map_index_x,map_index_y)=255;
         // unsigned char* pomap = outputMap.ptr<unsigned char>(0);
         // pomap[index]=255;
          
      }
    }
  }
  char cmd[50]= ".png";  
  char cc[8];  
  sprintf(cc,"%d",jj);  
  cout<<"save:"<<cc<<endl;
  //cv::imwrite(strcat(c,cmd), tmpImg);
  cv::imwrite(strcat(cc,cmd), outputMap1);
  jj++;
 
  outputMap1.copyTo(outputMap);
  //return outputMap;
}
