#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <unistd.h>

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
using namespace std;

#include "mi_sys_datatype.h"
#include "mi_ipu.h"
#include "mi_sys.h"

#include "math.h"
#include <Eigen/Dense>
#include <sys/time.h>


#define yolov8_CLASSES (6)
// #define yolov8_Num_box (5040)
// #define yolov8_Num_box (2835)
// #define yolov8_Num_box (2184)
#define yolov8_Num_box (1260)
#define yolov8_CONF_THRESHOLD (0.50)
#define yolov8_NMS_THRESHOLD (0.45)

// 推理模型的宽高

// #define MODEL_Y8_W (640.0)
// #define MODEL_Y8_H (384.0)
#define MODEL_Y8_W (320.0)
#define MODEL_Y8_H (192.0)
// #define MODEL_Y8_W (416.0)
// #define MODEL_Y8_H (256.0)
// #define MODEL_Y8_W (320.0)
// #define MODEL_Y8_H (192.0)



#define Input_IMG_W (1280.0)
#define Input_IMG_H (720.0)

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
static inline float logistic(float x){return (1.f / (1.f + exp(-x)));}


struct yolov8_DetectionBBoxInfo {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float score;
  int   classID;
};
struct yolov8_BboxInfo {
  float x;
  float y;
  float width;
  float height;
  float score;
  int   classID;
};

struct yolov8_hand_info 
{
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float area;
  float dis;
  float way; 
  float angle;
  int   classID;
};

std::vector<yolov8_DetectionBBoxInfo> yolov8_infer_postprocess(cv::Mat imgSrc, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_SubNet_InputOutputDesc_t &desc);
// load model                            
void yolov8_load_detection_model(char *pModelImgPath, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_OfflineModelStaticInfo_t &OfflineModelInfo, 
                          MI_IPU_SubNet_InputOutputDesc_t &desc);
//preprocess img
cv::Mat yolov8_preprocess_img(cv::Mat &img, int input_w, int input_h, std::vector<int> &padsize);

//model infer
void yolov8_infer(cv::Mat imgDst, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_SubNet_InputOutputDesc_t &desc);

// postprocess
std::vector<yolov8_DetectionBBoxInfo> yolov8_postprocess(cv::Mat imgSrc,std::vector<int> padsize,
                              MI_IPU_TensorVector_t OutputTensorVector);

//yolov8 NMS
std::vector<yolov8_BboxInfo> yolov8_NMS(std::vector<yolov8_BboxInfo>& Bboxes);

//get Iou
float yolov8_calcIoU(yolov8_BboxInfo bbox1, yolov8_BboxInfo bbox2);

//get DetectionBBoxInfo
std::vector<yolov8_DetectionBBoxInfo>  yolov8_GetDetections(std::vector<yolov8_BboxInfo> output);
//WriteVisualizeBBox
vector<cv::Scalar> yolov8_GetColors(const int n);
cv::Scalar yolov8_HSV2RGB(const int i);
void yolov8_WriteVisualizeBBox(string strImageName,
                   const vector<yolov8_DetectionBBoxInfo > detections,
                   const vector<cv::Scalar>& colors);
                  
int yolov8_mkpath(std::string sDir, mode_t mode=0777);


