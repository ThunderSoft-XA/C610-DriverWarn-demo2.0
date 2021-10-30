#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>

#include "opencv2/opencv.hpp"
#include "ai_inference.h"
#include "run_model/include/tflite_blazeface.h"
#include "run_model/include/face_detection.h"

using namespace ai2nference;
using namespace cv;

#define DEFAULT_AI_CONFIG "../res/ai_config.ini"
#define LANDMARK_PARSE_FILE "../models/landmark_contours.txt"

int main(int argc, char ** argv)
{
    IniConf ini_conf;
    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)DEFAULT_AI_CONFIG,&ini_conf);

    std::vector <AiInference *> ai_inference_vec;
    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "AI runtime config info " << i << ",as follow:" <<std::endl;
        ai_inference_vec.push_back(new AiInference(DEFAULT_AI_CONFIG,i));
    }

    for(auto ai_inference : ai_inference_vec) {
        std::cout << "loading tflite model ......" <<std::endl;
        ai_inference->loadTfliteModel();
    }

    cv::Mat landmark_mat = cv::imread("../models/img/gst_zero_37.png",cv::COLOR_BGR2RGB);
    vector<uchar> landmark_vec = convertMat2Vector<uchar>(landmark_mat);
    cv::Mat sign_detection_mat = cv::imread("../models/img/00670.png",cv::COLOR_BGR2RGB);
    vector<uchar> sign_detection_vec = convertMat2Vector<uchar>(sign_detection_mat);

    std::cout << "landmark info = " << landmark_mat.rows << "," << landmark_mat.cols  << "," << landmark_mat.channels();

    std::thread ai_thread_1([=]{
        ai_inference_vec[0]->loadTfliteData(landmark_mat.rows,landmark_mat.cols,landmark_mat.channels(),landmark_vec);
        std::vector<std::vector<float>> inference_result;
        ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
        std::vector<FaceInfo> face_info_vec;
#if 0
        std::list<face_t> face_bounds_list;
        blazeface_result_t *face_result;
        float score_thresh = 0.5;
        decode_bounds(inference_result,face_bounds_list,score_thresh,128,128);
        #if 0 /* USE NMS */
            float iou_thresh = 0.3;
            std::list<face_t> face_nms_list;

            non_max_suppression (face_bounds_list, face_nms_list, iou_thresh);
            pack_face_result (face_result, face_nms_list);
        #else
            pack_face_result (face_result, face_bounds_list);
        #endif

        if(face_result != nullptr) {
            std::cout << "now, show all face result ......" << std::endl;
            for(int index = 0; index < face_result->num; index++ ) {
                std::cout << "face top left point info:" << std::endl;
                std::cout << face_result->faces->topleft.x << "," << face_result->faces->topleft.y << std::endl;
            }
        }
#endif
        decode_face_result(inference_result,landmark_mat,face_info_vec,0.5,0.3);
        for (auto face : face_info_vec){
            std::cout << " face info :" << std::endl;
            std::cout << "face score = " << face.score << "point = " << face.x1 << "," << face.y1 << std::endl;
        }

        std::vector<cv::Point3_<float>> parse_results;
        for(auto result : inference_result) {
            std::cout << "result size " << result.size() << ",inferece result as follow:" << std::endl;
            if(result.size() < 3) {
                for(auto value : result){
                    std::cout << value << std::endl;
                } 
            } else {
                for (int index = 0; index <  result.size()/3; ++index) {
                    parse_results.push_back(cv::Point3_<float>(result[3*index],result[3*index+1],result[3*index+2]));
                    std::cout << parse_results[index] << std::endl;
                }
            }
        }
        // following code ,read param parse file include point info.
        std::vector<std::vector<int>> landmark_point_vec;
        std::vector<cv::Point3_<float>> left_eye_point_vec;
        std::vector<cv::Point3_<float>> right_eye_point_vec;
        {
            ifstream landmark_parse_file;
            string landmark_line;
            std::vector<string> landmark_str;
            std::vector<int> landmark_point;
            landmark_parse_file.open(LANDMARK_PARSE_FILE);
            getline(landmark_parse_file,landmark_line);
            while(landmark_parse_file && !landmark_line.empty()) {
                std::cout << "landmark line :" << landmark_line << std::endl;
                landmark_str = selfSplit(landmark_line," ");
                landmark_str.erase(std::begin(landmark_str));
                for(auto pos_value : landmark_str){
                    landmark_point.push_back(std::stoi(pos_value));
                }
                sort(landmark_point.begin(),landmark_point.end());
                landmark_point.erase(unique(landmark_point.begin(), landmark_point.end()), landmark_point.end());
                landmark_point_vec.push_back(landmark_point);
                getline(landmark_parse_file,landmark_line);
            }
            landmark_parse_file.close();
        }
        // get Site feature point
        for(auto index : landmark_point_vec[1]) {
            left_eye_point_vec.push_back(parse_results[index]);
        }
        for(auto index : landmark_point_vec[2]) {
            right_eye_point_vec.push_back(parse_results[index]);
        }

        // get eye center point
        cv::Point3f left_eye_center;
        int left_point_num = left_eye_point_vec.size();
        for(auto pos : left_eye_point_vec){
            left_eye_center.x += pos.x;
            left_eye_center.y += pos.y;
            left_eye_center.z += pos.z;
        }
        left_eye_center.x = left_eye_center.x / left_point_num;
        left_eye_center.y = left_eye_center.y / left_point_num;
        left_eye_center.z = left_eye_center.z / left_point_num;

        std::cout << "left eye center point :" << left_eye_center << std::endl;
        cv::Mat source_mat = cv::imread("../models/img/face.jpeg",cv::COLOR_BGR2RGB);
        cv::Mat tmp_mat;
        cv::resize(source_mat,tmp_mat,cv::Size(192,192));
        cv::Mat left_eye_mat = tmp_mat(cv::Rect2f(left_eye_center.x,left_eye_center.y,64,64));
        cv::Mat con_left_eye_mat= left_eye_mat.clone();
        vector<uchar> left_eye_vec = convertMat2Vector<uchar>(con_left_eye_mat);

        ai_inference_vec[1]->loadTfliteData(left_eye_mat.rows,left_eye_mat.cols,left_eye_mat.channels(),left_eye_vec);
        std::vector<std::vector<float>> left_eye_inference_result;
        ai_inference_vec[1]->runAndGetResult<float>(&left_eye_inference_result);
        for(auto result : left_eye_inference_result) {
            std::cout << "inferece result as follow:" << std::endl;
            for(auto value : result) {
                std::cout << value;
            }
            std::cout << std::endl;
        }

    });
    ai_thread_1.join();

    std::thread ai_thread_2([=]{
        ai_inference_vec[2]->loadTfliteData(sign_detection_mat.rows,sign_detection_mat.cols,sign_detection_mat.channels(),sign_detection_vec);
        std::vector<std::vector<float>> inference_result;
        ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
        for(auto result : inference_result) {
            std::cout << "inferece result as follow:" << std::endl;
            for(auto value : result) {
                std::cout << value;
            }
            std::cout << std::endl;
        }
    });
    ai_thread_2.join();

    while(1);

}
