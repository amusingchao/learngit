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
static int Degree = 20;
static int Pi = 3.1415926;
//static float fx = 331.4490965;
//static float fy = 331.4490965;
//static float cx = 318.571991;
//static float cy = 178.753937;
/*
static float a =0.013787434;//-0.0095618721;//0.0330547082016;//0.084217004;// 0.0330547082016;
static float b =-0.82089496;//-0.88887048;//-0.895213494974;//-0.89223164;// -0.895213494974;
static float c =-0.10762557;//0.111914819345; //0.028351974;//0.111914819345;*/
static float a = -0.000952103346921;//0.045408707;//-0.057114515;//0.0330547082016;//0.084217004;// 0.0330547082016;
static float b =-0.668814503351;//-0.82245082;//-0.83588785;//-0.895213494974;//-0.89223164;// -0.895213494974;
static float c =-0.585212607871;//0.098771609;//0.06275171;//0.111914819345; //0.028351974;//0.111914819345;
static int resize_divide = 1;
//static float fx = 662.898193/float(resize_divide);
//static float fy = 662.898193/float(resize_divide);
//static float cx = 637.143982/float(resize_divide);
//static float cy = 268.1309055/float(resize_divide);
//static float fb = 79.427956;      //not divide
static float fx = 331.4490965;
static float fy = 441.932128667;
static float cx = 318.571991;
static float cy = 238.338582667;
static float fb = 39.713978;      //not divide

static float Observe_Degree = 40;
static float Observe_Distance = 4;      
static float Observe_Middle_Distance = 2; 
static float Ground_Height = 0.1;
static float Ground_Farway_Height = 0.2;
static float Tianhuaban_Height = 1.8;

static float Camera_To_Robot = 0.16;

//const float DepthMapFactor= 0.3/65536;
static float DepthMapFactor = 512.0;

static float GroundFactor = 20.0;  //3m for 1+3/60
#define Plane_Test 0
#define Obtacle_Flag 1
#define Ground_Flag 0

#define Value1 5
#define Value2 5

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

static int fun_2d(float x0, float y0,int *index_x,int *index_y,int nnn)
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
        index_y_tmp = 1000-((y0 - Dylow)/(Dyup - Dylow)*Dheight);
        *index_x = index_x_tmp;
        *index_y = index_y_tmp;
        if(0)//nnn==170)
        {
          cout<<"fun:"<<endl;
          cout<<y0<<endl;
          cout<<index_y_tmp<<endl;
        }
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
  Global_Map = cv::Mat::ones(1000, 1000, CV_8UC1) * 128;
  
  Img_Save = cv::Mat::zeros(1000,1000,CV_8UC3);
  //circle(Last_Map,cvPoint(500,500),1,CV_RGB(0,255,255),2,8,0);  
  //Last_Map_No_Obtacle = cv::Mat::zeros(1000, 1000, CV_8UC1);
  //Last_Map_Obtacle = cv::Mat::zeros(1000, 1000, CV_8UC1);
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
void camera_to_world_2d(float xr, float yr, float *xw, float *yw, float yaw_degree, float tx, float ty)
{
  *xw = xr*cos(yaw_degree)-yr*sin(yaw_degree)+tx;
  
  *yw = xr*sin(yaw_degree)+yr*cos(yaw_degree)+ty;
}

void rotate_pitch(float y_p,float z_p,float pitch_degree,float *y_w,float *z_w)
{    
  *y_w = y_p * cos(pitch_degree) + z_p * sin(pitch_degree);
  *z_w = z_p * cos(pitch_degree) - y_p * sin(pitch_degree);

    //return y_w,z_w
}

void rotate_transfer_yuanlai(float xx_p,float zz_p,float total_yaw,float transfer_x,float transfer_y,float *xx_w,float *zz_w,int mmm)
{    
  //*x_w = x_p*cos(total_yaw) + z_p*sin(total_yaw)-transfer_y; //left + up +
  
  //*z_w = z_p * cos(total_yaw) - x_p * sin(total_yaw)+transfer_x;
  *xx_w = xx_p * cos(total_yaw) - zz_p * sin(total_yaw)- transfer_y; //left + up +
  *zz_w = zz_p * cos(total_yaw) + xx_p * sin(total_yaw)+ transfer_x + Camera_To_Robot;// + Camera_To_Robot;
  if(0)//(mmm==17)
  {
    cout<<"indef:"<<endl;
    cout<<xx_p<<endl;
    cout<<zz_p<<endl;
    cout<<*zz_w<<endl;
    cout<<transfer_x<<endl;
    cout<<cos(total_yaw)<<endl;
    cout<<sin(total_yaw)<<endl;
    cout<<"end"<<endl;
  }
    //return y_w,z_w
}

void rotate_transfer(float xx_p,float zz_p,float total_camera_yaw,float total_wheel_yaw, float transfer_x,float transfer_y,float *xx_w,float *zz_w,int mmm)
{    
  //*x_w = x_p*cos(total_yaw) + z_p*sin(total_yaw)-transfer_y; //left + up +
  
  //*z_w = z_p * cos(total_yaw) - x_p * sin(total_yaw)+transfer_x;
  float tmp_x,tmp_y;
  tmp_x = xx_p * cos(total_camera_yaw) - zz_p * sin(total_camera_yaw);
  tmp_y = xx_p * sin(total_camera_yaw) + zz_p * cos(total_camera_yaw) + Camera_To_Robot;
  
  *xx_w = tmp_x * cos(total_wheel_yaw) - tmp_y * sin(total_wheel_yaw) - transfer_y ; //left + up +
  *zz_w = tmp_y * cos(total_wheel_yaw) + tmp_x * sin(total_wheel_yaw) + transfer_x ; 
  //*xx_w = xx_p * cos(total_yaw) - zz_p * sin(total_yaw)- transfer_y; //left + up +
  //*zz_w = zz_p * cos(total_yaw) + xx_p * sin(total_yaw)+ transfer_x + Camera_To_Robot;// + Camera_To_Robot;
  
    //return y_w,z_w
}
/*
void update_global(int tmp_y, int tmp_x, bool tmp_flag)
{
  
  cv::
  if((tmp_flag == Obtacle_Flag) && (Global_Map.at<unsigned char>(map_index_y,map_index_x)-Value1) > 0)
     Global_Map.at<unsigned char>(map_index_y,map_index_x) -= Value1;
  
  else if((tmp_flag == Ground_Flag) && (Global_Map.at<unsigned char>(map_index_y,map_index_x) + Value2) <= 255)
     Global_Map.at<unsigned char>(map_index_y,map_index_x) += Value2;
}*/
cv::Mat BuildMap::FloorMap(cv::Mat& inputRgb, cv::Mat& inputDepth,float wheel_x,float wheel_y,float wheel_yaw,float head_yaw,float head_pitch,int ID)
{
  
  cv::Mat Local_Map;
  //Local_Map = cv::Mat::zeros(1000, 1000, CV_8UC1);
  Local_Map = cv::Mat::ones(1000, 1000, CV_8UC1) * 128;
  //cv::Mat Local_Map;
  //Local_Map = cv::Mat::zeros(1000, 1000, CV_8UC1);
  
  cv::Mat Local_no_obtacle_Map;
  Local_no_obtacle_Map = cv::Mat::zeros(1000, 1000, CV_8UC1);
  cout<<"start make map:"<<endl;
  int W = inputDepth.cols;
  int H = inputDepth.rows;
  Input_Num +=1;
  cout<<"Input_Num:"<<Input_Num<<endl;
  if(Input_Num >= 50000)
     Input_Num = 0;
     
  //if((W==0 && H==0) || (Input_Num <=10) || (Input_Num == 13))
    //  return Last_Map;//return Local_Map;//return Last_Map;
  //if(Input_Num == 11)
    //Gound_Modify(&a, &b, &c, inputDepth);
  cout<<"hello:"<<endl;
  
  float time_use=0;
  
  struct timeval start;
  struct timeval end;
  gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
  printf("start.tv_sec:%d\n",start.tv_sec);
  printf("start.tv_usec:%d\n",start.tv_usec);
  
  cv::Mat inputDepth1;
  inputDepth1 = cv::Mat::zeros(480, 640, CV_8UC1);
  
  //cv::Mat inputDepth2;
  //inputDepth2 = inputDepth; // 640.0;
  //resize(inputDepth,inputDepth1,Size(inputDepth.cols/2,inputDepth.rows/2),0,0,INTER_LINEAR);
  
  cv::resize(inputDepth, inputDepth1, cv::Size(640,480), 0, 0, cv::INTER_LINEAR);
  cout<<"aaaaa"<<endl;
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
  ea_orb.camera_yaw = head_yaw ;//wheel_yaw + head_yaw ;
  ea_orb.wheel_yaw = wheel_yaw ;
  ea_orb.camera_pitch = head_pitch;//head_pitch; 
  ea_orb.camera_roll= 0.0;
  ea_orb.wheel_xposition = wheel_x;
  ea_orb.wheel_yposition = wheel_y;
  
  qd_out=EulerAngleToQuadruple(&ea_orb);
  cv::Mat Twc = cv::Mat::zeros(4,4,CV_32F);;
  Twc = QuadrupleToRotateMatrix(&qd_out);
  Twc.at<float>(0,3) = wheel_x;//;
  Twc.at<float>(1,3) = 0;
  Twc.at<float>(2,3) = wheel_y;//wheel_y;
  
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
  world_ptx = ptx_t;
  
 
  cv::Mat world_ptx_3 = cv::Mat::zeros(3,H*W,CV_32F);
  
  world_ptx_3 = world_ptx.rowRange(0,3).clone();
  
  projection_point = rotateAxis * world_ptx_3;
  cout<<H<<endl;
  cout<<W<<endl;
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
  float Y_W;
  float Z_W;
  float X_W;
  float Y_tmp_W;
  float Z_tmp_W;
  float X_tmp_W;
  int map_flag;
  //bool tmp_flag;
  for(int i=0; i<H*W; i++)
  {
    //cout<<"1222222"<<endl;
    //cout<<projection_point.at<float>(i,2)<<endl;
    float dist_1 = (a * world_ptx.at<float>(0,i) + b * world_ptx.at<float>(1,i) + c * world_ptx.at<float>(2,i) + 1) * inorm_abc;
    
    int jj=i/W;
    int ii=i%W;   //ii for cols
    projection_point.at<float>(1,i) = -1*projection_point.at<float>(1,i);
    //bool uxy=((ii>=670) &&(ii<869)) && ((jj>=477) &&(jj<619)) ;
    dist_yuandian = TwoPoints_Diostance(world_ptx.at<float>(0,i), world_ptx.at<float>(2,i), 0, 0);
    //cout<<"as"<<endl;
    if(dist_yuandian<Observe_Middle_Distance)
    {
      if(abs(dist_1) <0.3)//(abs(dist_1) < (Ground_Height + dist_yuandian / GroundFactor)) 
        {
          //tmp_flag 
          rotate_pitch(world_ptx.at<float>(1,i),world_ptx.at<float>(2,i),ea_orb.camera_pitch, &Y_tmp_W,&Z_tmp_W);
          X_tmp_W = world_ptx.at<float>(0,i);
          rotate_transfer(world_ptx.at<float>(0,i),Z_tmp_W,ea_orb.camera_yaw,ea_orb.wheel_yaw,ea_orb.wheel_xposition,ea_orb.wheel_yposition,&X_W,&Z_W,Input_Num);
          degree_y = (atan(abs(world_ptx.at<float>(0,i)/world_ptx.at<float>(2,i))))/Pi*180;    //degree not right
          
          if(degree_y < Observe_Degree)
            {
              map_flag = fun_2d(X_W, Z_W,&map_index_x,&map_index_y,Input_Num);  //y for hang x for lie
              inputRgb.at<unsigned char>(jj,ii) = 0;
              //Img_Save.at<unsigned char>(jj,ii)[0] = 0;
              //Img_Save.at<unsigned char>(jj,ii)[1] = 0;
              //Img_Save.at<unsigned char>(jj,ii)[2] = 0;
              Img_Save.at<Vec3b>(map_index_y,map_index_x)[0]= 255;//B    
              Img_Save.at<Vec3b>(map_index_y,map_index_x)[1]= 255;//G    
              Img_Save.at<Vec3b>(map_index_y,map_index_x)[2]= 255;
              //Img_Save.at<Vec3b>(jj, ii) = Vec3b(255, 255, 255);
              //Global_Map = update_global(map_index_y, map_index_x, Ground_Flag);
              if((Global_Map.at<unsigned char>(map_index_y,map_index_x) + Value2) <= 255)
               {
                Global_Map.at<unsigned char>(map_index_y,map_index_x) += Value2;
                //Local_Map.at<unsigned char>(map_index_y,map_index_x) += Value2;
               }
              Local_Map.at<unsigned char>(map_index_y,map_index_x) = 255; 
              /*if((Local_Map.at<unsigned char>(map_index_y,map_index_x) + Value2) <= 255)
               {
                
                Local_Map.at<unsigned char>(map_index_y,map_index_x) = Value2;
               }*/
            }
        } 
       if(dist_1>0.3 && dist_1<Tianhuaban_Height)   //0.1
        {
          //tmp_flag 
          rotate_pitch(world_ptx.at<float>(1,i),world_ptx.at<float>(2,i),ea_orb.camera_pitch, &Y_tmp_W,&Z_tmp_W);
          X_tmp_W = world_ptx.at<float>(0,i);
          rotate_transfer(world_ptx.at<float>(0,i),Z_tmp_W,ea_orb.camera_yaw,ea_orb.wheel_yaw,ea_orb.wheel_xposition,ea_orb.wheel_yposition,&X_W,&Z_W,Input_Num);
          degree_y = (atan(abs(world_ptx.at<float>(0,i)/world_ptx.at<float>(2,i))))/Pi*180;    //degree not right
          
          if(degree_y < Observe_Degree)
            {
              map_flag = fun_2d(X_W, Z_W,&map_index_x,&map_index_y,Input_Num);  //y for hang x for lie
              //Global_Map = update_global(map_index_y, map_index_x, Ground_Flag);
              if((Global_Map.at<unsigned char>(map_index_y,map_index_x)-Value1) > 0)
              {
                Global_Map.at<unsigned char>(map_index_y,map_index_x) = Global_Map.at<unsigned char>(map_index_y,map_index_x) - Value1;
                //Local_Map.at<unsigned char>(map_index_y,map_index_x) = Local_Map.at<unsigned char>(map_index_y,map_index_x) - Value1;
                Img_Save.at<Vec3b>(map_index_y,map_index_x)[0]= 128;//B    
                Img_Save.at< Vec3b >(map_index_y,map_index_x)[1]= 128;//G    
                Img_Save.at< Vec3b >(map_index_y,map_index_x)[2]= 128;
              }
              Local_Map.at<unsigned char>(map_index_y,map_index_x) = 0; 
             /*if((Local_Map.at<unsigned char>(map_index_y,map_index_x) + Value2) <= 255)
             {
              
              Local_Map.at<unsigned char>(map_index_y,map_index_x) += Value2;
             }*/
            }
        }    
    } //  endif if(dist_yuandian<Observe_Middle_Distance)
    if(0)//((dist_yuandian > Observe_Middle_Distance) && (dist_yuandian < Observe_Distance))
    {
      if(abs(dist_1) < 0.2) 
        {
          //tmp_flag 
          rotate_pitch(world_ptx.at<float>(1,i),world_ptx.at<float>(2,i),ea_orb.camera_pitch, &Y_tmp_W,&Z_tmp_W);
          X_tmp_W = world_ptx.at<float>(0,i);
          rotate_transfer(world_ptx.at<float>(0,i),Z_tmp_W,ea_orb.camera_yaw,ea_orb.wheel_yaw,ea_orb.wheel_xposition,ea_orb.wheel_yposition,&X_W,&Z_W,Input_Num);
          degree_y = (atan(abs(world_ptx.at<float>(0,i)/world_ptx.at<float>(2,i))))/Pi*180;    //degree not right
          
          if(degree_y < Observe_Degree)
            {
              map_flag = fun_2d(X_W, Z_W,&map_index_x,&map_index_y,Input_Num);  //y for hang x for lie
              //inputRgb.at<unsigned char>(jj,ii) = 0;
              //Global_Map = update_global(map_index_y, map_index_x, Ground_Flag);
              if((Global_Map.at<unsigned char>(map_index_y,map_index_x) + Value2) <= 255)
                Global_Map.at<unsigned char>(map_index_y,map_index_x) += Value2;
            }
        } 
         
    } //  endif if(dist_yuandian<Observe_Middle_Distance)
    
  
     
  }//end for
  //for global_modify
  if(1)
  {
    for( int nrow = 0; nrow < Global_Map.rows; nrow++)  
    {  
       for(int ncol = 0; ncol < Global_Map.cols; ncol++)  
       {  
           unsigned char val = Global_Map.at<unsigned char>(nrow,ncol);
           //cout<< image_diff.at<unsigned char>(nrow,ncol)<<endl;
           //cout<< val<<endl;
           if(val<20)
             Last_Map.at<unsigned char>(nrow,ncol) = 128;
           if(val>200)
             Last_Map.at<unsigned char>(nrow,ncol) = 255;
           if(val>20 && val<200)
             Last_Map.at<unsigned char>(nrow,ncol) = 0;
           //cout<<val<<endl;     
       }  
    } 
  }
  float Xx_tmp_W, Yy_tmp_W, Zz_tmp_W;
  float Xx_W, Zz_W;
  int x0_map, y0_map;
  int x1_map, y1_map;
  rotate_pitch(0,0,ea_orb.camera_pitch, &Yy_tmp_W,&Zz_tmp_W);
  Xx_tmp_W = 0;
  rotate_transfer(Xx_tmp_W,Zz_tmp_W,ea_orb.camera_yaw,ea_orb.wheel_yaw,ea_orb.wheel_xposition,ea_orb.wheel_yposition,&Xx_W,&Zz_W,Input_Num);
  map_flag = fun_2d(Xx_W, Zz_W, &x0_map, &y0_map, Input_Num);
  
  rotate_pitch(0,2,ea_orb.camera_pitch, &Yy_tmp_W,&Zz_tmp_W);
  Xx_tmp_W = 0;
  rotate_transfer(Xx_tmp_W, Zz_tmp_W, ea_orb.camera_yaw, ea_orb.wheel_yaw, ea_orb.wheel_xposition, ea_orb.wheel_yposition, &Xx_W,&Zz_W, Input_Num);
  map_flag = fun_2d(Xx_W, Zz_W, &x1_map, &y1_map, Input_Num);
  
  line(Last_Map,Point(x0_map,y0_map),Point(x1_map,y1_map),Scalar(128,128,128),1,CV_AA);  
  //cv::circle(image, pointInterest, 2, cv::Scalar(0, 0, 255));//在图像中画出特征点，2是圆的半径 
  cv::circle(Local_Map, Point(x0_map,y0_map), 50, cv::Scalar(0, 0, 0));
  gettimeofday(&end,NULL);
  printf("end.tv_sec:%d\n",end.tv_sec);
  printf("end.tv_usec:%d\n",end.tv_usec);

  time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
  
  printf("time_use is %.10f\n",time_use);
  char cmd11[50]= ".png";  
  //char cmd1[50]= "mmm.png"; 
  char dd[10];  
  sprintf(dd,"%d",Input_Num+2900);  
  //cout<<"save:"<<cc<<endl;
  cv::imwrite(strcat(dd,cmd11), inputRgb);
  //return Last_Map;
  return Local_Map;
}
