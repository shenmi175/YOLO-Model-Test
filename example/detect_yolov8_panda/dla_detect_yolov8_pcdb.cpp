/* Copyright (c) 2018-2019 Sigmastar Technology Corp.
 All rights reserved.

  Unless otherwise stipulated in writing, any and all information contained
 herein regardless in any format shall remain the sole proprietary of
 Sigmastar Technology Corp. and be kept in strict confidence
 (��Sigmastar Confidential Information��) by the recipient.
 Any unauthorized act including without limitation unauthorized disclosure,
 copying, use, reproduction, sale, distribution, modification, disassembling,
 reverse engineering and compiling of the contents of Sigmastar Confidential
 Information is unlawful and strictly prohibited. Sigmastar hereby reserves the
 rights to any and all damages, losses, costs and expenses resulting therefrom.
*/

#define Eyes_follow 1

using namespace std;
#include "detect_y8_pcdb.h"


int main(int argc,char *argv[])
{
  //   ./dla_detect_yolov4_test ep100-l1.98_8_fixed.sim_sgsimg.img ValidSet/cat3 pcd.txt
  if (argc < 3)
  {
    std::cout << "USAGE: " << argv[0] <<": <ipu_firmware> <xxsgsimg.img>" \
    << std::endl;
    exit(0);
  }

  char * pFirmwarePath = NULL;
  char * pModelImgPath = argv[1]; //model path
  char * pImagePath    = argv[2]; //image path

  MI_U32 u32ChannelID  = 0;
  MI_IPU_SubNet_InputOutputDesc_t desc;
  MI_IPU_TensorVector_t InputTensorVector;
  MI_IPU_TensorVector_t OutputTensorVector;
  MI_IPU_OfflineModelStaticInfo_t OfflineModelInfo;

  //////load model
  yolov8_load_detection_model(pModelImgPath,
                        u32ChannelID,
                        InputTensorVector,
                        OutputTensorVector,
                        OfflineModelInfo,
                        desc);
  cv::Mat imgSrc;
  std::vector<std::string> imgNames;
  //读取图片路径
  cv::glob(pImagePath, imgNames);
  char imgNameCount[128];
  struct  timeval    all_tv_start;
  struct  timeval    all_tv_end;
  struct  timeval    all_tv_start1;
  struct  timeval    all_tv_end1;
  int all_elasped_time;
  // while(1){
  for(std::vector<std::string> :: iterator imgName = imgNames.begin(); imgName != imgNames.end(); imgName ++)
  {

    sprintf(imgNameCount, "%s", imgName->c_str());
    // 读取原图
    imgSrc = cv::imread(imgNameCount);

    gettimeofday(&all_tv_start,NULL);

    std::cout<<"Opencv Version:" << CV_VERSION << endl;
    std::vector<yolov8_DetectionBBoxInfo> detect_info;

    detect_info = yolov8_infer_postprocess(imgSrc,
                u32ChannelID,
                InputTensorVector,
                OutputTensorVector,
                desc);
    // usleep(77000);

    gettimeofday(&all_tv_end,NULL); 
    all_elasped_time = (all_tv_end.tv_sec-all_tv_start.tv_sec)*1000+(all_tv_end.tv_usec-all_tv_start.tv_usec)/1000;
    cout<<"----------------------------> all time is:"<<all_elasped_time<<", "<< (float(all_elasped_time)) / 1000.0<<std::endl;
    
    //可视化结果 
    std::cout << "detect_info.size()---:" << detect_info.size() << std::endl;
    for (unsigned int j = 0 ; j < detect_info.size();j++)
    {
        const int label = detect_info[j].classID;
        const float score = detect_info[j].score;
        // std::cout << "label---:" << label <<"   "<< "score---:" << score << std::endl;
        // std::cout << "xmin---:" << detect_info[j].xmin * Input_IMG_W <<"   "<< "ymin---:" << detect_info[j].ymin * Input_IMG_H << endl;
        // std::cout << "xmax---:" << detect_info[j].xmax * Input_IMG_W <<"   "<< "ymax---:" << detect_info[j].ymax * Input_IMG_H << endl;
        // const float len_x = (detect_info[j].xmax * Input_IMG_W) - (detect_info[j].xmin * Input_IMG_W);
        // const float len_y = (detect_info[j].ymax * Input_IMG_H) - (detect_info[j].ymin * Input_IMG_H);
        // std::cout << "xmax - xmin = " << len_x << "   " <<"ymax - ymin = " << len_y << endl;
        // std::cout << "len_x / len_y = " << len_x / len_y << endl;
    }
    int show_result = 1;
    if(show_result && detect_info.size() > 0)
    {
      vector<cv::Scalar> colors = yolov8_GetColors(yolov8_CLASSES);
      yolov8_WriteVisualizeBBox(imgNameCount, detect_info, colors);
    }  
  }  
  return 0;
}
