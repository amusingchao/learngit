#include<iostream>
#include <math.h>
#include"MakeMap.h"
#include"opencv2/core/core.hpp"    
#include"opencv2/highgui/highgui.hpp"    
#include"opencv2/imgproc/imgproc.hpp"
 
#include <stdlib.h>  
 
#include <vector>  
#include<sys/time.h>
#include <omp.h>
#define _GLIBCXX_USE_CXX11_ABI 0
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
static float a =-0.069682054;//0.045408707;//-0.057114515;//0.0330547082016;//0.084217004;// 0.0330547082016;
static float b =-0.98465014;//-0.82245082;//-0.83588785;//-0.895213494974;//-0.89223164;// -0.895213494974;
static float c =-0.15059012;//0.098771609;//0.06275171;//0.111914819345; //0.028351974;//0.111914819345;
static float fx = 662.898193;
static float fy = 662.898193;
static float cx = 637.143982;
static float cy = 268.1309055;
static float fb = 79.427956;

static int X0 = 670;
static int X1 = 869;
static int Y0 = 477;
static int Y1 = 619;

//const float DepthMapFactor= 0.3/65536;
static float DepthMapFactor = 512.0;
using namespace std; 


#define CLAMP(x , min , max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : x))

 


BuildMap::Quadruple_INFOR EulerAngleToQuadruple(BuildMap::ROBOT_INFOR  *robot_infor)
{
	BuildMap::Quadruple_INFOR qd;
	float fCosHRoll = cos(robot_infor->camera_roll * .5f);
	float fSinHRoll = sin(robot_infor->camera_roll * .5f);
	float fCosHPitch = cos(robot_infor->camera_pitch * .5f);
	float fSinHPitch = sin(robot_infor->camera_pitch * .5f);
	float fCosHYaw = cos(robot_infor->camera_yaw * .5f);
	float fSinHYaw = sin(robot_infor->camera_yaw * .5f);

	/// Cartesian coordinate System
	//w = fCosHRoll * fCosHPitch * fCosHYaw + fSinHRoll * fSinHPitch * fSinHYaw;
	//x = fSinHRoll * fCosHPitch * fCosHYaw - fCosHRoll * fSinHPitch * fSinHYaw;
	//y = fCosHRoll * fSinHPitch * fCosHYaw + fSinHRoll * fCosHPitch * fSinHYaw;
	//z = fCosHRoll * fCosHPitch * fSinHYaw - fSinHRoll * fSinHPitch * fCosHYaw;

	qd.w = fCosHRoll * fCosHPitch * fCosHYaw + fSinHRoll * fSinHPitch * fSinHYaw;
	qd.x = fCosHRoll * fSinHPitch * fCosHYaw + fSinHRoll * fCosHPitch * fSinHYaw;
	qd.y = fCosHRoll * fCosHPitch * fSinHYaw - fSinHRoll * fSinHPitch * fCosHYaw;
	qd.z = fSinHRoll * fCosHPitch * fCosHYaw - fCosHRoll * fSinHPitch * fSinHYaw;
	return qd;
}

BuildMap::ROBOT_INFOR QuadrupleToEulerAngle(BuildMap::Quadruple_INFOR  *quadruple_infor)
{
	BuildMap::ROBOT_INFOR  ea;

	/// Cartesian coordinate System 
	//ea.m_fRoll  = atan2(2 * (w * x + y * z) , 1 - 2 * (x * x + y * y));
	//ea.m_fPitch = asin(2 * (w * y - z * x));
	//ea.m_fYaw   = atan2(2 * (w * z + x * y) , 1 - 2 * (y * y + z * z));
	float w = quadruple_infor->w;
	float x = quadruple_infor->x;
	float y = quadruple_infor->y;
	float z = quadruple_infor->z;

	ea.camera_roll  = atan2(2 * (w * z + x * y) , 1 - 2 * (z * z + x * x));
	ea.camera_pitch = asin(CLAMP(2 * (w * x - y * z) , -1.0f , 1.0f));
	ea.camera_yaw   = atan2(2 * (w * y + z * x) , 1 - 2 * (x * x + y * y));

	return ea;
}
cv::Mat QuadrupleToRotateMatrix(BuildMap::Quadruple_INFOR  *quadruple_infor)
{
  
  
  cv::Mat RotateMatrix(4, 4, CV_32F);
  RotateMatrix.setTo(0);
  float w = quadruple_infor->w;
	float x = quadruple_infor->x;
	float y = quadruple_infor->y;
	float z = quadruple_infor->z;
  float x2 = x * x;
  float y2 = y * y;
  float z2 = z * z;
  float xy = x * y;
  float xz = x * z;
  float yz = y * z;
  float wx = w * x;
  float wy = w * y;
  float wz = w * z;

  RotateMatrix.at<float>(0,0) = 1.0f - 2.0f * (y2 + z2);
  RotateMatrix.at<float>(0,1) = 2.0f * (xy - wz);
  RotateMatrix.at<float>(0,2) = 2.0f * (xz + wy);
  RotateMatrix.at<float>(0,3) = 0;
  RotateMatrix.at<float>(1,0) = 2.0f * (xy + wz);
  RotateMatrix.at<float>(1,1) = 1.0f - 2.0f * (x2 + z2);
  RotateMatrix.at<float>(1,2) = 2.0f * (yz - wx);
  RotateMatrix.at<float>(1,3) = 0;
  RotateMatrix.at<float>(2,0) = 2.0f * (xz - wy);
  RotateMatrix.at<float>(2,1) = 2.0f * (yz + wx);
  RotateMatrix.at<float>(2,2) = 1.0f - 2.0f * (x2 + y2);
  RotateMatrix.at<float>(2,3) = 0;
  RotateMatrix.at<float>(3,0) = 0;
  RotateMatrix.at<float>(3,1) = 0;
  RotateMatrix.at<float>(3,2) = 0; 
  RotateMatrix.at<float>(3,3) = 1;
  
  return RotateMatrix;
}

static int fun_2d(float x0, float y0,int *index_x,int *index_y)
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
        //cout<<Dxlow<<endl;
        //cout<<Dxup<<endl;
        //cout<<Dylow<<endl;
        //cout<<Dyup<<endl;
        //cout<<Dwidth<<endl;
        //cout<<Dheight<<endl;
        //cout<<x0<<endl;
        //cout<<y0<<endl;
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

    rotateAxis =  I + sin_alpha*base + (1-cos_alpha)*base*base;
    //rotateAxisInverse = rotateAxis.inv();
}

BuildMap::BuildMap()
{
 
  Last_Map.create(1000, 1000, CV_8UC1);
    
}
cv::Mat BuildMap::FloorMap(cv::Mat& inputDepth,float wheel_x,float wheel_y,float wheel_yaw,float head_yaw,float head_pitch,int ID)
{
  
  float time_use=0;
  struct timeval start;
  struct timeval end;
  gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
  printf("start.tv_sec:%d\n",start.tv_sec);
  printf("start.tv_usec:%d\n",start.tv_usec);
  cv::Mat inputDepth1;
  cv::resize(inputDepth, inputDepth1, cv::Size(1280, 720), (0, 0), (0, 0), cv::INTER_LINEAR);
  int W = inputDepth1.cols;
  int H = inputDepth1.rows;
  cout<<W<<endl<<H<<endl;
  
  cv::Mat outputMap;
  float inorm_abc=1.0 / sqrt(a *a+ b*b+ c*c);
  
  cv::Mat coord;
  coord.create(H*W, 3, CV_32F);
  float* ptr = coord.ptr<float>(0);
  cv::Mat ptx(4, 1, CV_32F);
  float *pptx = ptx.ptr<float>(0); pptx[3] = 1.f;
  
  cv::Mat point(4, 1, CV_32F);
  float *ppt  = point.ptr<float>(0);
  
  cv::Mat fpoint(3, 1, CV_32F);
  float *fpt  = fpoint.ptr<float>(0);
  
  cv::Mat plane(3, 1, CV_32F);
  float aa = a * inorm_abc;
  float bb = b * inorm_abc;
  float cc = c * inorm_abc;
  plane.at<float>(0,0)=a;
  plane.at<float>(1,0)=b;
  plane.at<float>(2,0)=c;
  //cout<<"aa"<<aa<<endl;
  //cout<<"bb"<<bb<<endl;
  //cout<<"cc"<<cc<<endl;
  //cout<<"a"<<a<<endl;
  //cout<<"b"<<b<<endl;
  //cout<<"c"<<c<<endl;
  cv::Mat rotateAxis, rotateAxisInverse;
  RotatePlane(plane, rotateAxis);
  rotateAxisInverse = rotateAxis.inv();
  unsigned short int* pdepth = inputDepth1.ptr<unsigned short int>(0);
  
  
  
  int map_index_x=0;
  int map_index_y=0;
  
  
  
  cv::Mat outputMap1(1000, 1000, CV_8UC1);
  outputMap1.setTo(0);
  
  
  float degree_y;
  float dist_yuandian;
  int flag1=0;
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
  
  BuildMap::Quadruple_INFOR qd_orb;
  BuildMap::ROBOT_INFOR     ea_orb;
  BuildMap::Quadruple_INFOR qd_out;
  if(0)
  {
    qd_orb.x =-0.0012185;
    qd_orb.y = 0.0205426; 
    qd_orb.z = 0.0028898; 
    qd_orb.w = 0.9997841;
    ea_orb = QuadrupleToEulerAngle(&qd_orb);
    std::cout<< "ea_orb.camera_roll:"<<ea_orb.camera_roll<<std::endl;
    std::cout<< "ea_orb.camera_pitch"<<ea_orb.camera_pitch<<std::endl;
    std::cout<< "ea_orb.camera_yaw"<<ea_orb.camera_yaw<<std::endl;
    
    qd_out=EulerAngleToQuadruple(&ea_orb);
    std::cout<< "qd_out.w:"<<qd_out.w<<std::endl;
    std::cout<< "qd_out.x"<<qd_out.x<<std::endl;
    std::cout<< "qd_out.y"<<qd_out.y<<std::endl;
    std::cout<< "qd_out.z"<<qd_out.z<<std::endl;
  }
  //float RobYaw =0.001687*600;//0.001687;
  //float HeadYaw =-0.017959*600;// -0.017959;
  //float HeadPitch =0.002587; //3.1415926/4;     //0.002587*600;// 0.002587*600;//0;//0.002587*600;//0.002587;
  //float RobX = 0.486209;  
  //float RobY = 0.000072; 
  ea_orb.camera_yaw = -1*(wheel_yaw*600 + head_yaw) ;//wheel_yaw + head_yaw ;
  ea_orb.camera_pitch = -1*head_pitch;//head_pitch; 
  ea_orb.camera_roll= 0.0;
  //std::cout<< "ea_orb.camera_roll:"<<ea_orb.camera_roll<<std::endl;
  //std::cout<< "ea_orb.camera_pitch"<<ea_orb.camera_pitch<<std::endl;
  //std::cout<< "ea_orb.camera_yaw"<<ea_orb.camera_yaw<<std::endl;
  qd_out=EulerAngleToQuadruple(&ea_orb);
  cv::Mat Twc;
  Twc = QuadrupleToRotateMatrix(&qd_out);
  //Eigen::Matrix<double, 3, 3> matrix_33;
  //Eigen::Matrix3d<float, 3, 3> uuu;
  //Eigen::Matrix3d uuu;
  //cv::Mat vvvv;
  //matrix_33=euler2RotationMatrix(ea_orb.camera_roll, ea_orb.camera_pitch, ea_orb.camera_yaw);
  //eigen2cv(uuu,vvvv);
  //cout<<"eigen"<<endl;
  //cout<<matrix_33.matrix()<<endl;
  //cout<<Twc.at<float>(0,0)<<endl;
  //cout<<Twc.at<float>(0,1)<<endl;
  Twc.at<float>(0,3) = wheel_x;//;
  Twc.at<float>(1,3) = 0;
  Twc.at<float>(2,3) = wheel_y;//wheel_y;
  float tmp_depth;
  
  //cv::Mat tmp;
  //float *ptr_tmp = tmp.ptr<float>(0);
  //cout<<"Twc"<<endl;
  /*
  for (int iii = 0; iii < Twc.rows; iii++)
  {
      //pTwc = Twc.ptr<float>(iii);
      for (int jjj = 0; jjj < Twc.cols; jjj++)
      {
        //cout<<(*pTwc)<<endl ;
        cout<< Twc.at<float>(iii,jjj)<<endl;
        //p.g = Twc.data[ vv*Twc.step+uu*Twc.channels()+1 ];
        //p.r = Twc.data[ vv*Twc.step+uu*Twc.channels()+2 ];
     }
  }*/
  
  
  //std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
  //Eigen::Vector3d mmmm1;
  //mmmm1=Quaterniond2Euler(qd_orb.x,qd_orb.y,qd_orb.z,qd_orb.w); 
  
  
  //cv::Rect mroi(cv::Point(Y0,Y1), cv::Point(X0,X1));
  //cout<<X0<<endl<<X1<<endl<<Y0<<endl<<Y1<<endl;
  
  //cout<<mroi.area()<<endl;
  //cout<<"aaaa"<<endl;
  cv::Mat xydata((Y1-Y0)*(X1-X0), 3, CV_32F);
  float *pdata = xydata.ptr<float>(0);
  
  //for (int mm=400;mm<500;mm++)
  //{
 // outputMap1.at<unsigned char>(500,mm)=255;
  
  //}
  
  //#pragma omp parallel for
  for (int i = 0; i < H; ++i)
  {
    //#pragma omp parallel for
    for (int j = 0; j < W; ++j, ptr+=3, pdepth+=1)
    {
      
      //ptr[0] = (39.713978/(640*(*pdepth)) * (j - cx) /fx;       
      //ptr[1] = (39.713978/(640*(*pdepth)*DepthMapFactor))  * (i - cy) /fy;       
      //ptr[2] = (39.713978/(640*(*pdepth)*DepthMapFactor)) ;
      tmp_depth = fb/((*pdepth)/DepthMapFactor);
      //ptr[0] = tmp_depth * (j - cx) /fx;       
      //ptr[1] = tmp_depth * (i - cy) /fy;       
      //ptr[2] = tmp_depth;
      //cout<<"*pdepth"<<(*pdepth)/DepthMapFactor<<endl;
      //pptx[0] = ptr[0];
      //pptx[2] = ptr[1];
      //pptx[1] = ptr[2];
      //pptx[3] = 1.0;
      ptx.at<float>(0)=tmp_depth * (j - cx) /fx;//ptr[0];
      ptx.at<float>(1)=tmp_depth * (i - cy) /fy;//* (i - cy) /fy;//ptr[2];
      ptx.at<float>(2)=tmp_depth ; //ptr[1];
      ptx.at<float>(3)=1;
      //cout<<ptx<<endl;
      point = Twc * ptx;   //point = Twc * ptx;
      //ppt[0]=ppt[0];
      //ppt[1]=ppt[2];
      //ppt[2]=ppt[1];
      //cout<<"after"<<endl;
      /*
      if(0)
      {
      float tmp_point;
      point.at<float>(0)=point.at<float>(0);
      tmp_point=point.at<float>(1);
      point.at<float>(1)=point.at<float>(2);
      point.at<float>(2)=tmp_point;
      tmp_point=ptx.at<float>(1);
      ptx.at<float>(1)=ptx.at<float>(2);
      ptx.at<float>(2)=tmp_point;
      }*/
      
      /*
      if((flag1==1))
      {
        std::cout<< "ptx:"<<ptx.at<float>(0)<<std::endl;
        std::cout<< "ptx:"<<ptx.at<float>(1)<<std::endl;
        std::cout<< "ptx:"<<ptx.at<float>(2)<<std::endl;
        std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
        std::cout<< "point:"<<point.at<float>(0)<<std::endl;
        std::cout<< "point:"<<point.at<float>(1)<<std::endl;
        std::cout<< "point:"<<point.at<float>(2)<<std::endl;
        std::cout<< "point:"<<point.at<float>(3)<<std::endl;
        flag1++;
      }*/
      
      fpoint = rotateAxis *point.rowRange(0,3);   // 
      if(0)
      {
      std::cout<< "ptx:"<<ptx.at<float>(0)<<std::endl;
      std::cout<< "ptx:"<<ptx.at<float>(1)<<std::endl;
      std::cout<< "ptx:"<<ptx.at<float>(2)<<std::endl;
      std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
      std::cout<< "point:"<<point.at<float>(0)<<std::endl;
      std::cout<< "point:"<<point.at<float>(1)<<std::endl;
      std::cout<< "point:"<<point.at<float>(2)<<std::endl;
      std::cout<< "point:"<<point.at<float>(3)<<std::endl;
      cout<<"fpoint.at<float>[0]"<<fpoint.at<float>(0)<<endl;
      cout<<"fpoint.at<float>[1]"<<fpoint.at<float>(1)<<endl;
      cout<<"fpoint.at<float>[2]"<<fpoint.at<float>(2)<<endl;
      cout<<"fpoint.at<float>[2]"<<inorm_abc<<endl;
      }
      //cv::Mat nnnn= rotateAxis*plane;//.rowRange(0,3); 
      //cout<<"nnnn"<<nnnn.at<float>(0)<<endl;
      //cout<<"nnnn"<<nnnn.at<float>(1)<<endl;
      //cout<<"nnnn"<<nnnn.at<float>(2)<<endl;
      //cout<<"inorm_abc"<<inorm_abc<<endl;
      //std::cout<<"dist"<<dist<<std::endl;
      //std::cout<<"xx"<<xx<<std::endl;
      //std::cout<<"yy"<<yy<<std::endl;
      //std::cout<<"zz"<<zz<<std::endl;
      
      bool uxy=((i>=Y0) &&(i<Y1)) && ((j>=X0) && (j<X1)) ;//j lie
      if (uxy)
      {
        if(0)
        {
          cout<<i<<" "<<j<<endl;
          float dist_1 = (a*pptx[0] + b*pptx[1] + c*pptx[2] + 1) * inorm_abc;
          std::cout<< "dist_1:"<<dist_1<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(0)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(1)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(2)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
          std::cout<< "point:"<<point.at<float>(0)<<std::endl;
          std::cout<< "point:"<<point.at<float>(1)<<std::endl;
          std::cout<< "point:"<<point.at<float>(2)<<std::endl;
          std::cout<< "point:"<<point.at<float>(3)<<std::endl;
          cout<<"fpoint.at<float>[0]"<<fpoint.at<float>(0)<<endl;
          cout<<"fpoint.at<float>[1]"<<fpoint.at<float>(1)<<endl;
          cout<<"fpoint.at<float>[2]"<<fpoint.at<float>(2)<<endl;
          cout<<"fpoint.at<float>[2]"<<inorm_abc<<endl;
        }
        //cout<<"pdata[2]"<<pdata<<endl;
        //cout<<xydata.ptr<float>(0)<<endl;
        //cout<<xydata.rows<<endl;
        pdata[0] = ptx.at<float>(0);//ppt[0];
        pdata[1] = ptx.at<float>(1);//ppt[1];
        pdata[2] = ptx.at<float>(2);//ppt[2];
        //cout<<"pdata[0]"<<pdata[0]<<endl;
        //cout<<"pdata[1]"<<pdata[1]<<endl;
        //cout<<"pdata[2]"<<pdata[2]<<endl;
        
        //cout<<"bug"<<endl;
        //pdata[3] = k;
        //int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
         //Last_Map.at<unsigned char>(map_index_x,map_index_y)=255;
        pdata += 3;
      }
      
      if(abs(fpt[2] + inorm_abc) <0.15)
      {
          float dist_1 = (a*pptx[0] + b*pptx[1] + c*pptx[2] + 1) * inorm_abc;
          
          if(ID==0)
          {
          std::cout<<tmp_depth<<endl;
          std::cout<< "dist_1:"<<dist_1<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(0)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(1)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(2)<<std::endl;
          std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
          std::cout<< "point:"<<point.at<float>(0)<<std::endl;
          std::cout<< "point:"<<point.at<float>(1)<<std::endl;
          std::cout<< "point:"<<point.at<float>(2)<<std::endl;
          std::cout<< "point:"<<point.at<float>(3)<<std::endl;
          cout<<"fpoint.at<float>[0]"<<fpoint.at<float>(0)<<endl;
          cout<<"fpoint.at<float>[1]"<<fpoint.at<float>(1)<<endl;
          cout<<"fpoint.at<float>[2]"<<fpoint.at<float>(2)<<endl;
          cout<<"fpoint.at<float>[2]"<<inorm_abc<<endl;
          }
          
            //std::cout<< "fpt[2]:"<<fpt[2]<<std::endl;
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          
          if((degree_y<40) && (dist_yuandian<4))
          {
          int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
          //cout<<fpt[0]<<endl;
          //cout<<fpt[1]<<endl;
          //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_y);     //x for heng
          poutmap[map_index_x] = 255;
          Last_Map.at<unsigned char>(map_index_x,map_index_y)=255;
          //Last_Map
            
          }
          
      }
      if((fpt[2] + inorm_abc) < 1.8 && (fpt[2] + inorm_abc) >0.15)
      {
          //std::cout<< "fpt[2]:"<<fpt[2]<<std::endl;
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          //pdraw[0] = 255;

          
          if((degree_y<40) && (dist_yuandian<4))
          {
          int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
         //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_y);
          poutmap[map_index_x] = 128;
          Last_Map.at<unsigned char>(map_index_x,map_index_y)=128;
            
          }
          
      }
      if((fpt[2] + inorm_abc) < 0.1)
      {
          
         
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          if((degree_y<40) && (dist_yuandian<4))
          {
            int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
          //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
            poutmap = outputMap1.ptr<unsigned char>(map_index_x);
            poutmap[map_index_y] = 255;
            Last_Map.at<unsigned char>(map_index_x,map_index_y)=255;
            
          }
      }
    }
  }
  
  gettimeofday(&end,NULL);
  printf("end.tv_sec:%d\n",end.tv_sec);
  printf("end.tv_usec:%d\n",end.tv_usec);

  time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
  printf("time_use is %.10f\n",time_use);
  
  cv::Mat b((Y1-Y0)*(X1-X0), 1, CV_32F);
  b.setTo(-1.f);
  cv::Mat planeInfo(3, 1, CV_32F);
  if (!cv::solve(xydata, b, planeInfo, cv::DECOMP_SVD))
  {
      assert(0 > 1);
  }

  std::cout<<planeInfo.t()<<std::endl;
  float aaa= planeInfo.at<float>(0);
  float bbb= planeInfo.at<float>(1);
  float ccc= planeInfo.at<float>(2);
  float sqrt_abc = sqrt(aaa*aaa+bbb*bbb+ccc*ccc);
  float sqrt_height = 1.0 / sqrt_abc;
  float degree_b = (acos(abs(bbb/sqrt_abc)))/Pi*180;
  float robot_pitch = 0.655387/Pi*180;
  std::cout<<"robot_pitch"<<robot_pitch<<std::endl;
  std::cout<<"degree_b"<<degree_b<<std::endl;
  std::cout<<"sqrt_height"<<sqrt_height<<std::endl;
  char cmd[50]= ".png";  
  //char cmd1[50]= "mmm.png"; 
  char dd[8];  
  sprintf(dd,"%d",jj+600);  
  cout<<"save:"<<cc<<endl;
  cv::imwrite(strcat(c,cmd), tmpImg);
  //cv::imwrite(strcat(dd,cmd), Last_Map);
  //if(ID==2){cv::imwrite("aa.png", Last_Map);}
  //jj++;
  //if(ID==2)
    //cout<<Last_Map<<endl;
  //outputMap1.at<unsigned char>(0,0)=1;
  //Last_Map.at<unsigned char>(0,0)=1;
  //outputMap1.copyTo(Last_Map);
  
  //outputMap1.copyTo(outputMap);
  //return outputMap;
  return Last_Map;
}
