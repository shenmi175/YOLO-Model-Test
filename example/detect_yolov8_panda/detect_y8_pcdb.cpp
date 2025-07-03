
using namespace std;
#include "detect_y8_pcdb.h"
#include <sys/stat.h>

const std::vector<std::string> yolov8_class_names = {
    "person", "cat", "dog", "catface", "dogface", "hand"};

int yolov8_mkpath(std::string sDir, mode_t mode)
{
  int mdret;
  
  if((mdret = ::mkdir(sDir.c_str(), mode)) && errno!=EEXIST){
    return mdret;
  }
  
  return mdret;
}

void yolov8_WriteVisualizeBBox(string strImageName,
                   const vector<yolov8_DetectionBBoxInfo > detections,
                   const vector<cv::Scalar>& colors)
{
    cv::Mat image = cv::imread(strImageName, -1);
    map<int, vector<yolov8_DetectionBBoxInfo> > detectionsInImage;
    char buffer[50];

    std::string name = strImageName;
    // cout << "name---" << name <<endl;
    unsigned int pos = strImageName.rfind("/");
    if (pos > 0 && pos < strImageName.size()) {
        name = name.substr(pos + 1);
    }
    int pos1 = 0;
    int i = 0;
    int dirNamePosStart = 0;
    int dirNamePosEnd = -1;
    while((pos1 = strImageName.find("/", pos1)) != string::npos)
    {
      dirNamePosStart = dirNamePosEnd;
      dirNamePosEnd = pos1;
      pos1 ++;
      i ++;
    }
    std::string savedir(strImageName.substr(dirNamePosStart + 1, dirNamePosEnd - dirNamePosStart - 1) + "Rst");
    int ret = yolov8_mkpath(savedir);
    if (ret == 0){

    }else if (ret == -1){

    }
    std::string strOutImageName = name;
    
    strOutImageName = strOutImageName.replace(strOutImageName.size()-4, 4, ".png");
    cout << "strOutImageName---" << strOutImageName << endl;
    char saveName[128];
    sprintf(saveName, "%s/%s", savedir.c_str(), strOutImageName.c_str());
    for (unsigned int j = 0; j < detections.size(); j++) {
        yolov8_DetectionBBoxInfo bbox;
        const int label = detections[j].classID;
        const float score = detections[j].score;
        bbox.xmin =  detections[j].xmin * image.cols;
        bbox.xmin = bbox.xmin < 0 ? 0 : bbox.xmin ;
        bbox.ymin =  detections[j].ymin * image.rows;
        bbox.ymin = bbox.ymin < 0 ? 0 : bbox.ymin ;
        bbox.xmax =  detections[j].xmax * image.cols;
        bbox.xmax = bbox.xmax > image.cols ? image.cols : bbox.xmax;
        bbox.ymax =  detections[j].ymax * image.rows;
        bbox.ymax = bbox.ymax > image.rows ? image.rows : bbox.ymax ;
        cv::Point top_left_pt(int(bbox.xmin), int(bbox.ymin));
        cv::Point bottom_right_pt(int(bbox.xmax), int(bbox.ymax));
        const cv::Scalar& color = colors[label];

        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 2);
        

        std ::string classname = yolov8_class_names[label];
        snprintf(buffer, sizeof(buffer), "%s: %.2f", classname.c_str(), score);
        cv::putText(image, buffer, cv::Point(int(bbox.xmin), int(bbox.ymax)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(128, 0, 128), 2, 8);

        
    }
    cv::imwrite(saveName, image);

}

cv::Scalar yolov8_HSV2RGB(const int i) {
  float r, g, b;
  // std :: cout << "--f--" << f <<"--p--"<< p <<"--q--"<< q << "--t--"<<t<< std :: endl;
  // std :: cout << "-----h_i----" << h_i << std :: endl;
  switch (i) {
    case 0:
      r = 0; g = 0; b = 1;
      break;
    case 1:
      r = 0; g = 1; b = 0;
      break;
    case 2:
      r = 0; g = 1; b = 1;
      break;
    case 3:
      r = 0.88627; g = 0.16862; b = 0.54117;
      break;
    case 4:
      // r = t; g = p; b = v;
      r = 0; g = 0.27058; b = 1;
      break;
    case 5:
      r = 0.40; g = 0.69803; b = 1;
      // r = v; g = 1; b = q;
      break;
    default:
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}
vector<cv::Scalar> yolov8_GetColors(const int n)
{
  vector<cv::Scalar> colors;
  // cv::RNG rng(12345);
  // int rand = rng;
  // // std :: cout << "-------rand----" << rand << std :: endl;
  // const float golden_ratio_conjugate = 0.618033988749895;
  // const float s = 0.3;
  // const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    // const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate, 1.f);
    // std :: cout << "-------h----" << h << std :: endl;
    colors.push_back(yolov8_HSV2RGB(i));
  }
  return colors;
}


#define INNER_MOST_ALIGNMENT (4)
#define ALIGN_UP(val, alignment) ((( (val)+(alignment)-1)/(alignment))*(alignment))

std::vector<yolov8_DetectionBBoxInfo>  yolov8_GetDetections(std::vector<yolov8_BboxInfo> output){
    std::vector<yolov8_DetectionBBoxInfo > detections;
    for(int i=0;i<output.size();i++)
    {
      yolov8_DetectionBBoxInfo  detection;
      memset(&detection,0,sizeof(yolov8_DetectionBBoxInfo));
      detection.xmin =  output[i].x;
      detection.ymin =  output[i].y;
      detection.xmax =  output[i].x + output[i].width;
      detection.ymax =  output[i].y + output[i].height;
      //归一化
      detection.xmin = detection.xmin / Input_IMG_W ;
      detection.ymin = detection.ymin / Input_IMG_H ;
      detection.xmax = detection.xmax / Input_IMG_W ;
      detection.ymax = detection.ymax / Input_IMG_H ;

      detection.xmin = detection.xmin < 0 ? 0 : detection.xmin ;
      detection.ymin = detection.ymin < 0 ? 0 : detection.ymin ;
      detection.xmax = detection.xmax > 1 ? 1 : detection.xmax ;
      detection.ymax = detection.ymax > 1 ? 1 : detection.ymax ;

      detection.score = output[i].score;
      
      detection.classID = output[i].classID;

      const float len_x = (detection.xmax * Input_IMG_W) - (detection.xmin * Input_IMG_W);
      const float len_y = (detection.ymax * Input_IMG_H) - (detection.ymin * Input_IMG_H);
      const float rio_p = len_x / len_y ;
      
      detections.push_back(detection);
    }
    return detections;
}

MI_S32 IPUCreateDevice(char *pFirmwarePath,MI_U32 u32VarBufSize)
{
  MI_S32 s32Ret = MI_SUCCESS;
  MI_IPU_DevAttr_t stDevAttr;
  stDevAttr.u32MaxVariableBufSize = u32VarBufSize;
  stDevAttr.u32YUV420_W_Pitch_Alignment = 16;
  stDevAttr.u32YUV420_H_Pitch_Alignment = 2;
  stDevAttr.u32XRGB_W_Pitch_Alignment = 16;
  s32Ret = MI_IPU_CreateDevice(&stDevAttr, NULL, pFirmwarePath, 0);
  return s32Ret;
}

MI_S32 IPUCreateChannel(MI_U32 *s32Channel, char *pModelImage)
{
  MI_S32 s32Ret ;
  MI_SYS_GlobalPrivPoolConfig_t stGlobalPrivPoolConf;
  MI_IPUChnAttr_t stChnAttr;

  //create channel
  memset(&stChnAttr, 0, sizeof(stChnAttr));
  stChnAttr.u32InputBufDepth = 1;
  stChnAttr.u32OutputBufDepth = 1;
  return MI_IPU_CreateCHN(s32Channel, &stChnAttr, NULL, pModelImage);
}

MI_S32 IPUDestroyChannel(MI_U32 s32Channel)
{
  return MI_IPU_DestroyCHN(s32Channel);
}


void yolov8_load_detection_model(char *pModelImgPath, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_OfflineModelStaticInfo_t &OfflineModelInfo, 
                          MI_IPU_SubNet_InputOutputDesc_t &desc)
{
  
  MI_SYS_Init(0);
    
    // 1.create device
  // 获取离线模型运行需要的 variable buffer size 和离线模型文件大小
  if(MI_SUCCESS != MI_IPU_GetOfflineModeStaticInfo(NULL, pModelImgPath, &OfflineModelInfo))
  {
    cout<<"get model variable buffer size failed!" << std::endl;
    return;
  }
  if(MI_SUCCESS != IPUCreateDevice(NULL, OfflineModelInfo.u32VariableBufferSize))
  {
    cout<< "create ipu device failed!" << std::endl;
    return;
  }

  //2.create channel
  if(MI_SUCCESS!=IPUCreateChannel(&u32ChannelID, pModelImgPath))
  {
    cout<<"create ipu channel failed!"<<std::endl;
    MI_IPU_DestroyDevice();
    return;
  }

  //3.get input/output tensor
  MI_IPU_GetInOutTensorDesc(u32ChannelID, &desc);
  // cout << "Num of output:" << desc.u32OutputTensorCount << endl;
}
std::vector<yolov8_DetectionBBoxInfo>  yolov8_infer_postprocess(cv::Mat imgSrc, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_SubNet_InputOutputDesc_t &desc)
{
  struct  timeval    tv_start;
  struct  timeval    tv_end;
  int elasped_time;
  cv::Mat imgSrc1;

  //preprocess_img
  gettimeofday(&tv_start, NULL);
  std::vector<int> padsize;
  // cv::cvtColor(imgSrc, imgSrc1, cv::COLOR_BGR2RGB);
  cv::Mat imgDst = yolov8_preprocess_img(imgSrc, MODEL_Y8_W, MODEL_Y8_H,padsize);
  // cout << "imgDst.size()----" << imgDst.size() << endl;
  gettimeofday(&tv_end,NULL);
  elasped_time = (tv_end.tv_sec-tv_start.tv_sec)*1000+(tv_end.tv_usec-tv_start.tv_usec)/1000;
  cout<<"----------------------------> preprocess img time is:"<<elasped_time<<", "<< (float(elasped_time)) / 1000.0<<std::endl;
  gettimeofday(&tv_start, NULL);
  //model infer
  yolov8_infer(imgDst,
                u32ChannelID,
                InputTensorVector,
                OutputTensorVector,
                desc);
  gettimeofday(&tv_end,NULL);
  elasped_time = (tv_end.tv_sec-tv_start.tv_sec)*1000+(tv_end.tv_usec-tv_start.tv_usec)/1000;
  cout<<"----------------------------> infer time is:"<<elasped_time<<", "<< (float(elasped_time)) / 1000.0<<std::endl;
  //Postprocess
  gettimeofday(&tv_start, NULL);
  std::vector<yolov8_DetectionBBoxInfo> detect_info;
  detect_info = yolov8_postprocess(imgSrc, padsize,OutputTensorVector);
  gettimeofday(&tv_end,NULL);
  elasped_time = (tv_end.tv_sec-tv_start.tv_sec)*1000+(tv_end.tv_usec-tv_start.tv_usec)/1000;
  cout<<"----------------------------> Postprocess time is:"<<elasped_time<<", "<< (float(elasped_time)) / 1000.0<<std::endl;
  
  return detect_info;
}
void yolov8_infer(cv::Mat imgDst, 
                          MI_U32 &u32ChannelID,
                          MI_IPU_TensorVector_t &InputTensorVector,
                          MI_IPU_TensorVector_t &OutputTensorVector,
                          MI_IPU_SubNet_InputOutputDesc_t &desc)
{
    int iResizeH = desc.astMI_InputTensorDescs[0].u32TensorShape[1];
    int iResizeW = desc.astMI_InputTensorDescs[0].u32TensorShape[2];
    int iResizeC = desc.astMI_InputTensorDescs[0].u32TensorShape[3];
    // std::cout << "h:" << iResizeH << ";w:" << iResizeW << ";c:" << iResizeC << std::endl;
    MI_IPU_GetInputTensors(u32ChannelID, &InputTensorVector);

    memcpy(InputTensorVector.astArrayTensors[0].ptTensorData[0],imgDst.data,iResizeH*iResizeW*iResizeC);
    MI_SYS_FlushInvCache(InputTensorVector.astArrayTensors[0].ptTensorData[0], iResizeH*iResizeW*iResizeC);
    
    MI_S32 s32Ret = MI_IPU_GetOutputTensors(u32ChannelID, &OutputTensorVector);
    if (s32Ret != MI_SUCCESS) {
      printf("fail to get buffer, please try again\n");
      // return -1;
    }
    //4.invoke
    if(MI_SUCCESS!=MI_IPU_Invoke(u32ChannelID, &InputTensorVector, &OutputTensorVector))
    {
      cout<<"IPU invoke failed!!"<<endl;
      // delete pu8ImageData;
      IPUDestroyChannel(u32ChannelID);
      MI_IPU_DestroyDevice();
    }

    //释放指定通道的输入/输出Tensor Buffer
    MI_IPU_PutInputTensors(u32ChannelID,&InputTensorVector);
    MI_IPU_PutOutputTensors(u32ChannelID,&OutputTensorVector);
    
}

std::vector<yolov8_DetectionBBoxInfo> yolov8_postprocess(cv::Mat imgSrc,std::vector<int> padsize,
                              MI_IPU_TensorVector_t OutputTensorVector)
{
  struct  timeval    tv_start;
  struct  timeval    tv_end;
  int elasped_time;
  int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
  // std :: cout << "newh---neww---padh---padw:"<< newh << " " << neww << " " << padh << " " << padw << std :: endl;
  float ratio_h = (float) imgSrc.rows / newh;
  float ratio_w = (float) imgSrc.cols / neww;
  // std :: cout << "---ratio_h---ratio_w---:"<< ratio_h << " " << ratio_w << std::endl;
  const int OUTPUT_SIZE = yolov8_Num_box * (yolov8_CLASSES + 4);//output0
  // cout << "OUTPUT_SIZE--" << OUTPUT_SIZE << endl;
  float prob0[OUTPUT_SIZE];
  float *data0 = (float *)OutputTensorVector.astArrayTensors[0].ptTensorData[0];

  memcpy(prob0, (void *)(data0), OUTPUT_SIZE * sizeof(float));
  
  yolov8_BboxInfo Bbox;
  std::vector<yolov8_BboxInfo> Bboxes;
  int net_length = yolov8_CLASSES + 4;
  cv::Mat out1 = cv::Mat(net_length, yolov8_Num_box, CV_32F, prob0);
  // cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob1);
  // std :: cout << "out1-------" << out1.size()<<std :: endl;
  // std :: cout << "Num_box-------" << Num_box<<std :: endl;
  
  for (int i = 0; i < yolov8_Num_box; i++) {
      //输出是1*net_length*Num_box;所以每个box的属性是每隔Num_box取一个值，共net_length个值
      cv::Mat scores = out1(cv::Rect(i, 4, 1, yolov8_CLASSES)).clone();
      cv ::Point classIdPoint;
      double max_class_socre;
      minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
      // std :: cout << "max_class_socre-------" << max_class_socre<<std :: endl;
      max_class_socre = (float) max_class_socre;
      if (max_class_socre >= yolov8_CONF_THRESHOLD) {
        //   std :: cout << "max_class_socre-------" << max_class_socre<<std :: endl;
          float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
          float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
          float w = out1.at<float>(2, i) * ratio_w;  //w
          float h = out1.at<float>(3, i) * ratio_h;  //h
        //   std :: cout << "x----y----w---h----: " << x << " " << y << " " << w << " "<< h<<std :: endl;
          int left = MAX((x - 0.5 * w), 0);
          int top = MAX((y - 0.5 * h), 0);
          int width = (int) w;
          int height = (int) h;
          // std :: cout << "i----left----top----width--height----: " << i << "  " << left << " " << top << " " << width << " "<< height<<std :: endl;
          if (width <= 0 || height <= 0) { continue; }
          // cout << "classIdPoint---" << classIdPoint << endl;
          Bbox.x = left;
          Bbox.y = top;
          Bbox.width = width;
          Bbox.height = height;
          Bbox.score = max_class_socre;
          Bbox.classID = classIdPoint.y;
          Bboxes.push_back(Bbox);
        }

    }
    // std :: cout << "Bboxes.size()-------" << Bboxes.size()<<std :: endl;
    std::vector<yolov8_BboxInfo> picked_boxes;
    picked_boxes = yolov8_NMS(Bboxes);
    // std :: cout << "picked_boxes.size()-------" << picked_boxes.size()<<std :: endl;
    std::vector<yolov8_DetectionBBoxInfo> detections = yolov8_GetDetections(picked_boxes);
    
    return detections;

}

cv::Mat yolov8_preprocess_img(cv::Mat &img, int input_w, int input_h, std::vector<int> &padsize) {
    
    int w, h;
    int dw, dh;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        dw = 0;
        dh = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        dw = (input_w - w) / 2;
        dh = 0;
    }
    // std :: cout << "w---h---dw---dh:"<< w << " " << h << " " << dw << " " << dh << std :: endl;
    // std :: cout << "---img.size()---:"<< img.size() << std::endl;
    cv::Mat re(h, w, CV_8UC3);
    // std :: cout << "---re.size()---:"<< re.size() << std::endl;
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(int(dw), int(dh), re.cols, re.rows)));
    padsize.push_back(h);
    padsize.push_back(w);
    padsize.push_back(dh);
    padsize.push_back(dw);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
    
    return out;
}

std::vector<yolov8_BboxInfo> yolov8_NMS(std::vector<yolov8_BboxInfo>& Bboxes) {

    std::vector<yolov8_BboxInfo> picked_boxes;

    // 首先按照分数对框进行排序
    std::sort(Bboxes.begin(), Bboxes.end(), [](yolov8_BboxInfo& a, yolov8_BboxInfo& b) {
        return a.score > b.score;
    });

    while (Bboxes.size() > 0) {
        // 选出分数最高的框
        yolov8_BboxInfo best_box = Bboxes[0];
        picked_boxes.push_back(best_box);

        int index = 1;

        // 对剩余的框进行遍历，删除与选出的框有较大iou值的框
        while (index < Bboxes.size()) {
          float iou = yolov8_calcIoU(Bboxes[0],Bboxes[index]);
          // cout << "iou---"<<iou << endl;
          
          if (iou > yolov8_NMS_THRESHOLD){
            Bboxes.erase(Bboxes.begin() + index);
          }
          else{
            index ++;
          }
        }
        Bboxes.erase(Bboxes.begin());
    }
    return picked_boxes;
}


float yolov8_calcIoU(yolov8_BboxInfo bbox1, yolov8_BboxInfo bbox2) {
    float ret=0.0;
    const float intersection_xmin = max(bbox1.x, bbox2.x);
    const float intersection_ymin = max(bbox1.y, bbox2.y);
    const float intersection_xmax = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    const float intersection_ymax = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);
    const float intersection_area =
      max(intersection_ymax - intersection_ymin, 0.0) *
      max(intersection_xmax - intersection_xmin, 0.0);
    const float union_area = (bbox1.width * bbox1.height) + (bbox2.width * bbox2.height) - intersection_area;
    
    ret = intersection_area / union_area;
    // cout << "ret--"<<ret<<endl;
    return ret;
}
