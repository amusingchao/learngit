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
/*
static float a =0.013787434;//-0.0095618721;//0.0330547082016;//0.084217004;// 0.0330547082016;
static float b =-0.82089496;//-0.88887048;//-0.895213494974;//-0.89223164;// -0.895213494974;
static float c =-0.10762557;//0.111914819345; //0.028351974;//0.111914819345;*/
static float a =-0.069682054;//0.045408707;//-0.057114515;//0.0330547082016;//0.084217004;// 0.0330547082016;
static float b =-0.98465014;//-0.82245082;//-0.83588785;//-0.895213494974;//-0.89223164;// -0.895213494974;
static float c =-0.15059012;//0.098771609;//0.06275171;//0.111914819345; //0.028351974;//0.111914819345;
static int resize_divide = 2;
static float fx = 662.898193/float(resize_divide);
static float fy = 662.898193/float(resize_divide);
static float cx = 637.143982/float(resize_divide);
static float cy = 268.1309055/float(resize_divide);
static float fb = 79.427956;      //not divide

static float Observe_Degree = 30;
static float Observe_Distance = 4;      

static float Ground_Height = 0.12;
static float Tianhuaban_Height = 1.8;

//const float DepthMapFactor= 0.3/65536;
static float DepthMapFactor = 512.0;
using namespace std; 
using namespace cv;
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
 
  //Last_Map.create(1000, 1000, CV_8UC1);
  Last_Map = cv::Mat::zeros(1000, 1000, CV_8UC1);
  
  //circle(Last_Map,cvPoint(500,500),1,CV_RGB(0,255,255),2,8,0);  
  Last_Map_No_Obtacle = cv::Mat::zeros(1000, 1000, CV_8UC1);
  Last_Map_Obtacle = cv::Mat::zeros(1000, 1000, CV_8UC1);
  Input_Num = 0;
    
}

float TwoPoints_Degree(float dst_x0, float dst_y0, float src_x0, float src_y0)
{
  float tmp_degree = abs((dst_x0-src_x0)/(dst_y0-src_y0));
  float degree = (atan(tmp_degree))/Pi*180;
  return degree; 
}

float TwoPoints_Diostance(float dst_x0, float dst_y0, float src_x0, float src_y0)
{
  float tmp_distance = sqrt((dst_x0-src_x0)*(dst_x0-src_x0)+(dst_y0-src_y0)*(dst_y0-src_y0));
  
  return tmp_distance; 
}
/*
void Gound_Modify(float *a_tmp, float *b_tmp, float *c_tmp,cv::Mat& tmp_inputDepth)
{
  int tmp_h = tmp_inputDepth.rows;
  int tmp_w = tmp_inputDepth.cols;
  unsigned short int* tmp_pdepth = tmp_inputDepth.ptr<unsigned short int>(0);
  
  cv::Mat ptx_in(4, 1, CV_32F);
  for (int i = 0; i < H; ++i)
  {
    
    for (int j = 0; j < W; ++j, ptr+=3, pdepth_in+=1)
    {
      tmp_depth_in = fb/((*pdepth)/DepthMapFactor);
      ptx_in.at<float>(0)=tmp_depth_in * (j - cx) /fx;//ptr[0];
      ptx_in.at<float>(1)=tmp_depth_in * (i - cy) /fy;//* (i - cy) /fy;//ptr[2];
      ptx_in.at<float>(2)=tmp_depth_in ; //ptr[1];
      ptx_in.at<float>(3)=1;
      point_in = Twc_in * ptx_in; 
    }
    
   }
}*/

cv::Mat BuildMap::FloorMap(cv::Mat& inputDepth,float wheel_x,float wheel_y,float wheel_yaw,float head_yaw,float head_pitch,int ID)
{
  
  cv::Mat Local_Map;
  Local_Map = cv::Mat::zeros(1000, 1000, CV_8UC1);
  cout<<"start make map:"<<endl;
  int W = inputDepth.cols;
  int H = inputDepth.rows;
  Input_Num +=1;
  cout<<"Input_Num:"<<Input_Num<<endl;
  if(Input_Num >= 50000)
     Input_Num = 0;
     
  if((W==0 && H==0) || (Input_Num <=10) || (Input_Num == 13))
      return Local_Map;//return Local_Map;//return Last_Map;
  if(Input_Num == 11)
    Gound_Modify(&a, &b, &c, inputDepth);
  cout<<"hello:"<<endl;
  
  float time_use=0;
 
  struct timeval start;
  struct timeval end;
  gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
  printf("start.tv_sec:%d\n",start.tv_sec);
  printf("start.tv_usec:%d\n",start.tv_usec);
  
  cv::Mat inputDepth1;
  //resize(inputDepth,inputDepth1,Size(inputDepth.cols/2,inputDepth.rows/2),0,0,INTER_LINEAR);
  cv::resize(inputDepth, inputDepth1, cv::Size(inputDepth.cols/resize_divide, inputDepth.rows/resize_divide), (0, 0), (0, 0), cv::INTER_LINEAR);
  //inputDepth1 = inputDepth;
  W = inputDepth1.cols;
  H = inputDepth1.rows;
  
  cout<<W<<endl;
  cout<<H<<endl;
  cv:Mat ptx = cv::Mat::zeros(H*W,4,CV_32F);
  //cout<<ptx.cols<<endl;
  float* pptx = ptx.ptr<float>(0);
  cv::Mat ptx_t;   
  
  float tmp_depth;//fb/((*pdepth)/DepthMapFactor);
  unsigned short int* pdepth = inputDepth1.ptr<unsigned short int>(0);
  //int ii,jj;
  
  BuildMap::Quadruple_INFOR qd_orb;
  BuildMap::ROBOT_INFOR     ea_orb;
  BuildMap::Quadruple_INFOR qd_out;
  ea_orb.camera_yaw = -1*((wheel_yaw-0.004247)*600 + head_yaw) ;//wheel_yaw + head_yaw ;
  ea_orb.camera_pitch = -1*head_pitch;//head_pitch; 
  ea_orb.camera_roll= 0.0;
  ea_orb.wheel_xposition = wheel_x+0.057204;
  ea_orb.wheel_yposition = wheel_y+0.000370;
  
  qd_out=EulerAngleToQuadruple(&ea_orb);
  cv::Mat Twc = cv::Mat::zeros(4,4,CV_32F);;
  Twc = QuadrupleToRotateMatrix(&qd_out);
  Twc.at<float>(0,3) = wheel_x+0.057204;//;
  Twc.at<float>(1,3) = 0;
  Twc.at<float>(2,3) = wheel_y+0.000370;//wheel_y;
  
  //cv::Mat world_ptx(H*W,4,CV_32F);
  
  cv::Mat plane = cv::Mat::zeros(3, 1, CV_32F);
  float inorm_abc=1.0 / sqrt(a *a+ b*b+ c*c);
  float aa = a * inorm_abc;
  float bb = b * inorm_abc;
  float cc = c * inorm_abc;
  plane.at<float>(0,0)=a;
  plane.at<float>(1,0)=b;
  plane.at<float>(2,0)=c;
  
  cv::Mat rotateAxis;
  RotatePlane(plane, rotateAxis);
  
  cv::Mat projection_point = cv::Mat::zeros(3,H*W,CV_32F);
  
  float degree_y;
  float dist_yuandian;
  
  int map_index_x=0;
  int map_index_y=0;
  //#pragma omp parallel for
  int num_index=0;
  int uuu = 0;
  pptx = ptx.ptr<float>(0);
  int ii,jj;
  int H_W = H*W;
//#pragma omp parallel for
  for(int i=0;i<H_W;i+=1)
  {
    //
    ii = i/W;
    jj = i%W;
    pptx = ptx.ptr<float>(i);
    tmp_depth = (fb/((*pdepth)/DepthMapFactor));
    pptx[0] = tmp_depth * (jj - cx) /fx;  //x zuobiao
    pptx[1] = tmp_depth * (ii - cy) /fy;
    pptx[2] = tmp_depth;
    pptx[3] = 1;//tmp_depth;
    pdepth++;
     //pptx+=4;
   }
  
  //float pprj = ptx.ptr<float>(i);
  //cout<<"num_index"<<num_index<<endl;
  //if(ID==1)
    //cout<<ptx<<endl;
  //ptx_t = ptx.t();
  cv::Mat world_ptx(4,H*W,CV_32F);
  
  //cout<<ptx.cols<<endl;
  //cout<<ptx.rows<<endl;
  //cv::Mat ptx_t;
  ptx_t = ptx.t();
  world_ptx = Twc * ptx_t;
  
  cout<<"as"<<endl;
  cv::Mat world_ptx_3 = cv::Mat::zeros(3,H*W,CV_32F);
  
  world_ptx_3 = world_ptx.rowRange(0,3).clone();
  
  projection_point = rotateAxis * world_ptx_3;
  
  /*if(0)
  {
  cout<<"ptx.at<float>(0,0)"<<endl;
  cout<<ptx.at<float>(554*1280+153,0)<<endl;
  cout<<ptx.at<float>(554*1280+153,1)<<endl;
  cout<<ptx.at<float>(554*1280+153,2)<<endl;
  
  cout<<"world_ptx.at<float>(0,0)"<<endl;
  cout<<world_ptx.at<float>(0,554*1280+153)<<endl;
  cout<<world_ptx.at<float>(1,554*1280+153)<<endl;
  cout<<world_ptx.at<float>(2,554*1280+153)<<endl;
  cout<<"projection_point.at<float>(0,0)"<<endl;
  cout<<projection_point.at<float>(0,554*1280+153)<<endl;
  cout<<projection_point.at<float>(1,554*1280+153)<<endl;
  cout<<projection_point.at<float>(2,554*1280+153)<<endl;
  }*/
  //cout<<projection_point<<endl;
  for(int i=0; i<H*W; i++)
  {
    //cout<<"1222222"<<endl;
    //cout<<projection_point.at<float>(i,2)<<endl;
    //int jj=i/W;
    //int ii=i%W;
    //bool uxy=((ii>=554) &&(ii<624)) && ((jj>=153) &&(jj<298)) ;
    /*if(0)
    {
    cout<<"ptx.at<float>(0,0)"<<endl;
    cout<<ptx.at<float>(i,0)<<endl;
    cout<<ptx.at<float>(i,1)<<endl;
    cout<<ptx.at<float>(i,2)<<endl;
    
    cout<<"world_ptx.at<float>(0,0)"<<endl;
    cout<<world_ptx.at<float>(0,i)<<endl;
    cout<<world_ptx.at<float>(1,i)<<endl;
    cout<<world_ptx.at<float>(2,i)<<endl;
    cout<<"projection_point.at<float>(0,0)"<<endl;
    cout<<projection_point.at<float>(0,i)<<endl;
    cout<<projection_point.at<float>(1,i)<<endl;
    cout<<projection_point.at<float>(2,i)<<endl;
    }*/
    if(abs(projection_point.at<float>(2,i)+ inorm_abc) <Ground_Height)
    {
      if(0)
      {
      cout<<"ptx.at<float>(0,0)"<<endl;
      cout<<ptx.at<float>(i,0)<<endl;
      cout<<ptx.at<float>(i,1)<<endl;
      cout<<ptx.at<float>(i,2)<<endl;
      
      cout<<"world_ptx.at<float>(0,0)"<<endl;
      cout<<world_ptx.at<float>(0,i)<<endl;
      cout<<world_ptx.at<float>(1,i)<<endl;
      cout<<world_ptx.at<float>(2,i)<<endl;
      cout<<"projection_point.at<float>(0,0)"<<endl;
      cout<<projection_point.at<float>(0,i)<<endl;
      cout<<projection_point.at<float>(1,i)<<endl;
      cout<<projection_point.at<float>(2,i)<<endl;
      cout<<inorm_abc<<endl;
      }
      degree_y = TwoPoints_Degree(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //degree_y = (atan(abs(ptx.at<float>(i,0)/ptx.at<float>(i,2))))/Pi*180;    //degree not right
      dist_yuandian = TwoPoints_Diostance(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //dist_yuandian = sqrt(ptx.at<float>(i,0)*ptx.at<float>(i,0)+ptx.at<float>(i,2)*ptx.at<float>(i,2));
    
      if((degree_y<Observe_Degree) && (dist_yuandian<Observe_Distance))
      {
        int map_flag = fun_2d(projection_point.at<float>(0,i), projection_point.at<float>(1,i),&map_index_x,&map_index_y);
        Last_Map.at<unsigned char>(map_index_x,map_index_y) = 255;
        Local_Map.at<unsigned char>(map_index_x,map_index_y) = 255;
      //Last_Map
      }
     }
    if((projection_point.at<float>(2,i)+ inorm_abc) > Ground_Height && (projection_point.at<float>(2,i)+ inorm_abc) <Tianhuaban_Height)
    {
      degree_y = TwoPoints_Degree(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //degree_y = (atan(abs(ptx.at<float>(i,0)/ptx.at<float>(i,2))))/Pi*180;    //degree not right
      dist_yuandian = TwoPoints_Diostance(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //dist_yuandian = sqrt(ptx.at<float>(i,0)*ptx.at<float>(i,0)+ptx.at<float>(i,2)*ptx.at<float>(i,2));
    
      if((degree_y<Observe_Degree) && (dist_yuandian<Observe_Distance))
      {
        int map_flag = fun_2d(projection_point.at<float>(0,i), projection_point.at<float>(1,i),&map_index_x,&map_index_y);
        Last_Map.at<unsigned char>(map_index_x,map_index_y) = 128;
        Local_Map.at<unsigned char>(map_index_x,map_index_y) = 128;
      //Last_Map
      }
     }
     if((projection_point.at<float>(2,i)+ inorm_abc) < Ground_Height)
    {

      
      degree_y = TwoPoints_Degree(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //degree_y = (atan(abs(ptx.at<float>(i,0)/ptx.at<float>(i,2))))/Pi*180;    //degree not right
      dist_yuandian = TwoPoints_Diostance(ptx.at<float>(i,0), ptx.at<float>(i,2), ea_orb.wheel_xposition, ea_orb.wheel_yposition);
      //dist_yuandian = sqrt(ptx.at<float>(i,0)*ptx.at<float>(i,0)+ptx.at<float>(i,2)*ptx.at<float>(i,2));
    
      if((degree_y<Observe_Degree) && (dist_yuandian<Observe_Distance))
      {
        int map_flag = fun_2d(projection_point.at<float>(0,i), projection_point.at<float>(1,i),&map_index_x,&map_index_y);
        Last_Map.at<unsigned char>(map_index_x,map_index_y) = 255;
        Local_Map.at<unsigned char>(map_index_x,map_index_y) = 255;
      //Last_Map
      }
     }
  }
  
  
  gettimeofday(&end,NULL);
  printf("end.tv_sec:%d\n",end.tv_sec);
  printf("end.tv_usec:%d\n",end.tv_usec);

  time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
  printf("time_use is %.10f\n",time_use);
  
  
  //cout<<ptx.at<float>(0,0)<<endl;
  
  
 
  //return Local_Map;
     
  return Local_Map;
}
