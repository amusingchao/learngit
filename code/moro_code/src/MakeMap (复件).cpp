#include<iostream>
#include <math.h>
#include"MakeMap.h"
#include"opencv2/core/core.hpp"    
#include"opencv2/highgui/highgui.hpp"    
#include"opencv2/imgproc/imgproc.hpp"
#include <Eigen/Eigen>  
#include <stdlib.h>  
#include <Eigen/Geometry>  
#include <Eigen/Core>  
#include <vector>  
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
static float a = 0.011115;
static float b = -1.04616;
static float c = -0.142031;
static float fx = 468.9011;
static float fy = 468.9011;
static float cx = 324.0461;
static float cy =  236.3762;

//const float DepthMapFactor= 0.3/65536;
static float DepthMapFactor = 512;
using namespace std;
using namespace Eigen; 


#define CLAMP(x , min , max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : x))

 
  
Eigen::Quaterniond euler2Quaternion(const double roll, const double pitch, const double yaw)  
{  
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());  
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());  
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());  
  
    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;  
    cout << "Euler2Quaternion result is:" <<endl;  
    cout << "x = " << q.x() <<endl;  
    cout << "y = " << q.y() <<endl;  
    cout << "z = " << q.z() <<endl;  
    cout << "w = " << q.w() <<endl<<endl;  
    return q;  
}  
  
Eigen::Vector3d Quaterniond2Euler(const double x,const double y,const double z,const double w)  
{  
    Eigen::Quaterniond q;  
    q.x() = x;  
    q.y() = y;  
    q.z() = z;  
    q.w() = w;  
  
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);  
    cout << "Quaterniond2Euler result is:" <<endl;  
    cout << "x = "<< euler[2] << endl ;  
    cout << "y = "<< euler[1] << endl ;  
    cout << "z = "<< euler[0] << endl << endl;  
    return euler;
}  
  
Eigen::Matrix3d Quaternion2RotationMatrix(const double x,const double y,const double z,const double w)  
{  
    Eigen::Quaterniond q;  
    q.x() = x;  
    q.y() = y;  
    q.z() = z;  
    q.w() = w;  
  
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();  
    cout << "Quaternion2RotationMatrix result is:" <<endl;  
    cout << "R = " << endl << R << endl<< endl;  
    return R;  
}  
  
  
Eigen::Quaterniond rotationMatrix2Quaterniond(Eigen::Matrix3d R)  
{  
    Eigen::Quaterniond q = Eigen::Quaterniond(R);  
    q.normalize();  
    cout << "RotationMatrix2Quaterniond result is:" <<endl;  
    cout << "x = " << q.x() <<endl;  
    cout << "y = " << q.y() <<endl;  
    cout << "z = " << q.z() <<endl;  
    cout << "w = " << q.w() <<endl<<endl;  
    return q;  
}  
  
Eigen::Matrix3d euler2RotationMatrix(const double roll, const double pitch, const double yaw)  
{  
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());  
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());  
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());  
  
    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;  
    Eigen::Matrix3d R = q.matrix();  
    cout << "Euler2RotationMatrix result is:" <<endl;  
    cout << "R = " << endl << R << endl<<endl;  
    return R;  
}  
  
Eigen::Vector3d RotationMatrix2euler(Eigen::Matrix3d R)  
{  
    Eigen::Matrix3d m;  
    m = R;  
    Eigen::Vector3d euler = m.eulerAngles(0, 1, 2);  
    cout << "RotationMatrix2euler result is:" << endl;  
    cout << "x = "<< euler[2] << endl ;  
    cout << "y = "<< euler[1] << endl ;  
    cout << "z = "<< euler[0] << endl << endl;  
    return euler;  
}  

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
        cout<<Dxlow<<endl;
        cout<<Dxup<<endl;
        cout<<Dylow<<endl;
        cout<<Dyup<<endl;
        cout<<Dwidth<<endl;
        cout<<Dheight<<endl;
        cout<<x0<<endl;
        cout<<y0<<endl;
        cout<<index_x_tmp<<endl;
        cout<<index_x_tmp<<endl;
        
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
void BuildMap::FloorMap(cv::Mat& inputDepth,float *wheel_x,float *wheel_y,float *wheel_z,float *yaw,cv::Mat outputMap)
{
  
  int W = inputDepth.cols;
  int H = inputDepth.rows;
  

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
  plane.at<float>(0,0)=a;
  plane.at<float>(1,0)=b;
  plane.at<float>(2,0)=c;
  cv::Mat rotateAxis, rotateAxisInverse;
  RotatePlane(plane, rotateAxis);
  rotateAxisInverse = rotateAxis.inv();
  unsigned short int* pdepth = inputDepth.ptr<unsigned short int>(0);
  
  
  
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
  
  
  //std::cout<< "ptx:"<<ptx.at<float>(3)<<std::endl;
  //Eigen::Vector3d mmmm1;
  //mmmm1=Quaterniond2Euler(qd_orb.x,qd_orb.y,qd_orb.z,qd_orb.w); 
  for (int i = 0; i < H; ++i)
  {
    for (int j = 0; j < W; ++j, ptr+=3, pdepth+=1)
    {
      
      //ptr[0] = (39.713978/(640*(*pdepth)) * (j - cx) /fx;       
      //ptr[1] = (39.713978/(640*(*pdepth)*DepthMapFactor))  * (i - cy) /fy;       
      //ptr[2] = (39.713978/(640*(*pdepth)*DepthMapFactor)) ;
    
      ptr[0] = (*pdepth)/DepthMapFactor* (j - cx) /fx;       
      ptr[1] = (*pdepth)/DepthMapFactor * (i - cy) /fy;       
      ptr[2] = (*pdepth)/DepthMapFactor;
      //cout<<"*pdepth"<<(*pdepth)/DepthMapFactor<<endl;
      pptx[0] = ptr[0];
      pptx[1] = ptr[1];
      pptx[2] = ptr[2];
      point = ptx;   //point = Twc * ptx;
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
      }
      fpoint = rotateAxis * point.rowRange(0,3);
      
      

      //std::cout<<"dist"<<dist<<std::endl;
      //std::cout<<"xx"<<xx<<std::endl;
      //std::cout<<"yy"<<yy<<std::endl;
      //std::cout<<"zz"<<zz<<std::endl;
      if(abs(fpt[2] + inorm_abc) < 0.1)
      {
          //std::cout<< "fpt[2]:"<<fpt[2]<<std::endl;
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          
          if((degree_y<30) && (dist_yuandian<5))
          {
          int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
          cout<<fpt[0]<<endl;
          cout<<fpt[1]<<endl;
          cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_x);
          poutmap[map_index_y] = 255;
            
          }
          
      }
      if((fpt[2] + inorm_abc) < 1.8 && (fpt[2] + inorm_abc) >0.1)
      {
          //std::cout<< "fpt[2]:"<<fpt[2]<<std::endl;
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          //pdraw[0] = 255;

          
          if((degree_y<30) && (dist_yuandian<5))
          {
          int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
         //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
          poutmap = outputMap1.ptr<unsigned char>(map_index_x);
          poutmap[map_index_y] = 128;
            
          }
          
      }
      if((fpt[2] + inorm_abc) < 0.1)
      {
          
         
          degree_y=(atan(abs(ptx.at<float>(0)/ptx.at<float>(2))))/Pi*180;
          dist_yuandian=sqrt(ptx.at<float>(0)*ptx.at<float>(0)+ptx.at<float>(2)*ptx.at<float>(2));
          if((degree_y<30) && (dist_yuandian<5))
          {
            int map_flag = fun_2d(fpt[0], fpt[1],&map_index_x,&map_index_y);
          //cout<<"map_index_y: "<<map_index_x<<map_index_y<<endl;
            poutmap = outputMap1.ptr<unsigned char>(map_index_x);
            poutmap[map_index_y] = 255;
            
          }
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
