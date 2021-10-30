#ifndef __DATA_POOL_H__
#define __DATA_POOL_H__

#include <iostream>
#include <string>
#include <ctime>
#include <thread>
#include <vector>
#include <deque>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include "fastcv.h"
#include "../utils/timeutil.h"
#include "libyuv.h"
#include "bufmanager.hpp"

using namespace cv;

namespace datapool {

#ifdef __cplusplus
extern "C" {
#endif

typedef struct  _TimeStamp
{
    std::time_t create_time;
    std::time_t convert_time;
    std::time_t inference_time;
} TimeStamp;

typedef struct _DataInfo {
    int width;
    int height;
    int channel;
    bool is_keyframe;
    bool is_calibrated;
    bool is_hwdec;
} DataInfo;

typedef enum _DataState {
    Inited,
    Converted,
    Inferenced,
    Rendered
} DataState;

#ifdef __cplusplus
}
#endif

/**
 * In order to maintain the independence of the module, 
 * it is necessary to ensure that the packets are available directly and avoid direct correlation with upstream
 * So I removed the member methods that parse GstSample here
 * */

template <typename T>
class DataPacket {

public:
    DataPacket();
    DataPacket(T* _data,long _index);
    DataPacket(T* _data,int data_size, long _index, string source_name,DataInfo data_info);
    ~DataPacket(){
        delete this->data;
        this->data = NULL;
    }

    TimeStamp time_info;
    DataInfo data_info;
    int data_index;
    int holder_id;
    string source_name;   //Specify which camera the data packet comes from 
    std::vector<T>* data;
    cv::Mat data_mat;

    bool is_keyframe;
    bool is_calibrated;
    bool is_hwdec;

    DataState data_state;

    void getData();
    void imageCalibration();

    int hires_num;   //remove packet when hires num is 0 
    std::time_t timeout;

    void judgeKeyFrame();

};

/**
 * To be clear, the primary purpose of a data center is to support data correction and distribution
 * 
 * 
 * */

template <typename T>
class DataPool {

public:
    ~DataPool();
    DataPool(int _max_num,bool _en_policy) : max_num(_max_num),en_policy(_en_policy){

    }
    int getDataNum();
    static long packet_index;
    void addPacket(T* _data,int data_size,long _index,string source_name,DataInfo data_info);
    DataPacket<T>* getPacket(int _data_index);
    void runDataPool();
    void rmPackage(DataPacket<T>* _packet);
    std::deque<DataPacket<T>*> data_packet_vec;

        // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;

private:
    int max_num;
    bool en_policy;

    bool stop;

    // std::vector<DataPacket<T>*> data_packet_vec;

    void rmInvalidPackage();

};

template<typename T> long DataPool<T>::packet_index = 0;

template class DataPacket<unsigned char>;

template <typename T>
DataPacket<T>::DataPacket(T * _data,long _index)
{
    this->data = new std::vector<T>(*_data);
    this->data_state = DataState::Inited;
    this->time_info.create_time = getCurrentTime();
    this->data_index = _index;

}

template <typename T>
DataPacket<T>::DataPacket(T* _data,int data_size, long _index, string source_name,DataInfo data_info)
{
    // this->data = new std::vector<T>();
    // this->data->insert(this->data->end(), _data, _data + data_size);
    // Mat matImg = Mat(ysize, xsize, CV_8UC1, ucImg, 0);
    this->data_mat = Mat(data_info.width,data_info.height,CV_8UC4,_data,0);
    this->data_mat = this->data_mat.reshape(4,data_info.height).clone();
    this->data_state = DataState::Inited;
    this->time_info.create_time = getCurrentTime();
    this->data_index = _index;
    this->source_name = source_name;
    this->data_info.width = data_info.width;
    this->data_info.height = data_info.height;
    this->data_info.channel = data_info.channel;
    this->data_info.is_keyframe = data_info.is_keyframe;
    this->data_info.is_hwdec = data_info.is_hwdec;
    this->data_info.is_calibrated = data_info.is_calibrated;

    if(this->data_info.is_calibrated) {
        this->imageCalibration();
    } else {
        this->data_state = DataState::Converted;
    }

}

const cv::Mat intrinsic_matrix = (cv::Mat_<double>(3,3) << 945.511,0,1545.17,0,945.511,1528.21,0,0,1);
const cv::Mat D = ( cv::Mat_<double> ( 8,1 ) << 0.19948,0.0422889,0,0,0.00202519,0.522254,0.0630432,0.011681);
const cv::Mat fish_D = ( cv::Mat_<double> ( 4,1 ) << 0.19948,0.0422889,0,0);

static int count_packet = 0;
template <typename T>
void DataPacket<T>::imageCalibration()
{
    cv::Mat map1, map2;
    // cv::Mat tmp_mat = cv::Mat(*this->data,CV_8UC4);
    cv::Mat UndistortImage;
    cv::Size imageSize(this->data_info.width, this->data_info.height);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::Matx33d NewCameraMatrix;
    // cv::Mat calibration_mat = tmp_mat.reshape(4,this->data_info.height).clone();
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix,fish_D,imageSize,R,NewCameraMatrix,0.0);
    // NewCameraMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, D, imageSize, alpha, imageSize, 0);
    cv::fisheye::initUndistortRectifyMap(intrinsic_matrix, fish_D,R, NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    cv::remap(this->data_mat, UndistortImage, map1, map2, cv::INTER_LINEAR,cv::BORDER_CONSTANT);

    this->data_mat = UndistortImage.clone();

    // stringstream str_name;
    // str_name << "./packet_" << count_packet << ".jpg";
    // cv::imwrite(str_name.str(),UndistortImage);
    // count_packet++;
    // delete this->data;
    // this->data = new std::vector<T>((std::vector<T>)UndistortImage.reshape(1,1));
    // this->data = &convertMat2Vector<T>(UndistortImage);
    this->data_state = DataState::Converted;

}

template <typename T>
void DataPacket<T>::judgeKeyFrame()
{


}

// Gets the number of current packages
template <typename T>
int DataPool<T>::getDataNum()
{
    return this->data_packet_vec.size();
}


template <typename T>
DataPacket<T>* DataPool<T>::getPacket(int _data_index)
{
    return this->data_packet_vec[_data_index];
}

static int count = 0;

template <typename T>
void DataPool<T>::addPacket(T* _data,int data_size, long _index, string source_name,DataInfo data_info)
{
    // DataPacket<T>*tmp_packet = new DataPacket<T>(_data,data_size, _index, source_name,data_info);
    std::unique_lock<std::mutex> locker(this->queue_mutex);
    DataPacket<T>* tmp_packet  = new DataPacket<T>(_data,data_size, _index, source_name,data_info);
    this->data_packet_vec.emplace_back(tmp_packet);
    std::cout << "data packet size " << this->data_packet_vec.size() << std::endl;
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    locker.unlock();
    this->condition.notify_one();
    // std::this_thread::sleep_for( std::chrono::milliseconds(300) );
}

template <typename T>
void DataPool<T>::rmInvalidPackage()
{
    for(auto packet : this->data_packet_vec) {
        std::time_t timeout = getCurrentTime() - packet.time_info.create_time;
        if((packet.is_keyframe == false)
        && (packet.DataState == DataState::Inferenced || timeout > packet.timeout)){
            this->data_packet_vec.erase(packet);
        }
    }

}

template <typename T>
void DataPool<T>::rmPackage(DataPacket<T>* _packet)
{
    _packet->~DataPacket();
}

template <typename T>
void DataPool<T>::runDataPool()
{
#if 0
    for(;;) {
             
        if(this->data_packet_vec.size() >= this->max_num) {
            this->rmInvalidPackage();
        }
        for(auto packet : this->data_packet_vec) {
            if(packet.is_calibrated != TRUE) {
                std::thread gst_camera_th(&DataPool::getDataFromSample,this);
                gst_camera_th.detach();
            }
        }
    }
#endif
}

template <typename T>
DataPool<T>::~DataPool()
{


}


}

#endif