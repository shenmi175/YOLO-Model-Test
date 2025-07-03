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

#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

static std::vector<std::string> yolov8_labels = {
    "person", "cat", "dog", "catface", "dogface", "hand", "background"};

std::vector<std::string> yolov8_collect_images(const std::string& root) {
    std::vector<std::string> files;
    for (auto const& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        auto ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

bool yolov8_parse_xml(const std::string& xml_path,
                      std::vector<yolov8_DetectionBBoxInfo>& boxes,
                      const std::map<std::string, int>& label_map) {
    std::ifstream ifs(xml_path);
    if (!ifs.is_open())
        return false;
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());

    std::regex size_re("<size>.*?<width>([^<]+)</width>.*?<height>([^<]+)</height>",
                       std::regex::icase | std::regex::dotall);
    std::smatch msize;
    float w = Input_IMG_W;
    float h = Input_IMG_H;
    if (std::regex_search(content, msize, size_re)) {
        try {
            w = std::stof(msize[1]);
            h = std::stof(msize[2]);
        } catch (...) {
        }
    }

    std::regex obj_re(
        "<object>.*?<name>([^<]+)</name>.*?<xmin>([^<]+)</xmin>.*?<ymin>([^<]+)"
        "</ymin>.*?<xmax>([^<]+)</xmax>.*?<ymax>([^<]+)</ymax>",
        std::regex::icase | std::regex::dotall);
    auto it = std::sregex_iterator(content.begin(), content.end(), obj_re);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        std::smatch m = *it;
        auto name = m[1].str();
        if (label_map.count(name) == 0)
            continue;
        yolov8_DetectionBBoxInfo b{};
        b.classID = label_map.at(name);
        try {
            b.xmin = std::stof(m[2]) * Input_IMG_W / w;
            b.ymin = std::stof(m[3]) * Input_IMG_H / h;
            b.xmax = std::stof(m[4]) * Input_IMG_W / w;
            b.ymax = std::stof(m[5]) * Input_IMG_H / h;
        } catch (...) {
            continue;
        }
        b.score = 1.0f;
        boxes.push_back(b);
    }
    return !boxes.empty();
}

static float yolov8_iou(const yolov8_DetectionBBoxInfo& a,
                        const yolov8_DetectionBBoxInfo& b) {
    float x1 = std::max(a.xmin, b.xmin);
    float y1 = std::max(a.ymin, b.ymin);
    float x2 = std::min(a.xmax, b.xmax);
    float y2 = std::min(a.ymax, b.ymax);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    if (inter <= 0)
        return 0.f;
    float area1 = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float area2 = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    return inter / (area1 + area2 - inter);
}

void yolov8_update_confusion(std::vector<std::vector<int>>& matrix,
                             const std::vector<yolov8_DetectionBBoxInfo>& preds,
                             const std::vector<yolov8_DetectionBBoxInfo>& gts) {
    std::vector<int> gt_used(gts.size(), 0);
    std::vector<int> pred_used(preds.size(), 0);
    int bg = yolov8_CLASSES; // background index

    for (size_t i = 0; i < preds.size(); ++i) {
        float best_iou = 0.f;
        int best_j = -1;
        for (size_t j = 0; j < gts.size(); ++j) {
            if (gt_used[j])
                continue;
            float iv = yolov8_iou(preds[i], gts[j]);
            if (iv >= 0.5f && iv > best_iou) {
                best_iou = iv;
                best_j = j;
            }
        }
        if (best_j >= 0) {
            gt_used[best_j] = 1;
            pred_used[i] = 1;
            int gt_idx = gts[best_j].classID;
            int pd_idx = preds[i].classID;
            matrix[gt_idx][pd_idx] += 1;
        }
    }

    for (size_t j = 0; j < gts.size(); ++j) {
        if (!gt_used[j]) {
            int gt_idx = gts[j].classID;
            matrix[gt_idx][bg] += 1;
        }
    }

    for (size_t i = 0; i < preds.size(); ++i) {
        if (!pred_used[i]) {
            int pd_idx = preds[i].classID;
            matrix[bg][pd_idx] += 1;
        }
    }
}

void yolov8_draw_confusion(const std::vector<std::vector<int>>& matrix,
                           const std::vector<std::string>& labels,
                           const std::string& save_path) {
    int n = labels.size();
    int cell = 60;
    cv::Mat img((n + 1) * cell, (n + 1) * cell, CV_8UC3, cv::Scalar(255, 255, 255));

    int max_val = 0;
    for (auto const& row : matrix) {
        for (auto v : row)
            if (v > max_val)
                max_val = v;
    }
    max_val = std::max(1, max_val);

    for (int i = 0; i < n; ++i) {
        cv::putText(img, labels[i], cv::Point((i + 1) * cell + 5, cell - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::putText(img, labels[i], cv::Point(5, (i + 1) * cell + cell / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            int val = matrix[r][c];
            int intensity = 255 - static_cast<int>(255.0 * val / max_val);
            cv::rectangle(img, cv::Rect((c + 1) * cell, (r + 1) * cell, cell, cell),
                          cv::Scalar(intensity, intensity, 255), cv::FILLED);
            char buf[16];
            sprintf(buf, "%d", val);
            cv::putText(img, buf,
                        cv::Point((c + 1) * cell + cell / 4, (r + 1) * cell + cell / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
    }

    cv::imwrite(save_path, img);
}

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
  std::vector<std::string> imgNames = yolov8_collect_images(pImagePath);
  std::map<std::string, int> label_map;
  for (size_t i = 0; i < yolov8_labels.size() - 1; ++i) {
      label_map[yolov8_labels[i]] = i;
  }
  std::map<std::string, std::vector<std::vector<int>>> confusion;
  auto init_matrix = [&](){
      return std::vector<std::vector<int>>(yolov8_CLASSES + 1,
                                           std::vector<int>(yolov8_CLASSES + 1, 0));
  };
  char imgNameCount[512];

  struct  timeval    all_tv_start;
  struct  timeval    all_tv_end;

  int all_elasped_time;
  for(const auto& imgPath : imgNames)

  {
    sprintf(imgNameCount, "%s", imgPath.c_str());

    imgSrc = cv::imread(imgNameCount);

    gettimeofday(&all_tv_start,NULL);

    std::cout<<"Opencv Version:" << CV_VERSION << endl;
    std::vector<yolov8_DetectionBBoxInfo> detect_info;

    detect_info = yolov8_infer_postprocess(imgSrc,
                u32ChannelID,
                InputTensorVector,
                OutputTensorVector,
                desc);
    fs::path xml_path = fs::path(imgPath).replace_extension(".xml");
    if (fs::exists(xml_path)) {
      std::vector<yolov8_DetectionBBoxInfo> gt_boxes;
      if (yolov8_parse_xml(xml_path.string(), gt_boxes, label_map)) {
        std::string folder = fs::relative(xml_path.parent_path(), pImagePath).string();
        if(folder.empty()) folder = "root";
        if(!confusion.count(folder)) confusion[folder] = init_matrix();
        if(!confusion.count("overall")) confusion["overall"] = init_matrix();
        yolov8_update_confusion(confusion[folder], detect_info, gt_boxes);
        yolov8_update_confusion(confusion["overall"], detect_info, gt_boxes);
      }
    }
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

  for (auto& kv : confusion) {
    std::string folder = kv.first;
    std::vector<std::vector<int>>& mat = kv.second;
    fs::path out_dir = fs::path(pImagePath) / folder / "eval";
    fs::create_directories(out_dir);
    fs::path out_path = out_dir / "confusion_matrix.png";
    yolov8_draw_confusion(mat, yolov8_labels, out_path.string());
  }

  return 0;
}
