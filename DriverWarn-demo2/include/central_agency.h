#ifndef CENTRAL_AGENCY_H
#define CENTRAL_AGENCY_H


#include <iostream>
#include <string>
#include <time.h>
#include <pthread.h>
#include <thread>
#include <vector>
#include <omp.h>
#include <mutex>
#include <condition_variable>

#include "fastcv.h"
#include "gstcamera.h"
#include "ai_inference.h"
#include "thread_pool.h"
#include "data_pool.h"

using namespace ai2nference;
using namespace threadpool;
using namespace datapool;

namespace agency
{
/**
 * this class built for deal with video data from gst_camera,
 * Then,create a data pool to provide img data to AI model thread,
 * And, the AI model thread was produced by the agency.
 * the agency will provide data when the thread  Trigger signal or event or other.
 **/

class CentralAgency
{
public:
    CentralAgency(/* args */);
    CentralAgency(string _gst_config_file,string _ai_config_file);
    ~CentralAgency();
    void Init();
    void RunCenter();

private:
    /* data */
    string gst_config_file;
    string ai_config_file;

    std::vector<GstCamera *> gst_camera_vec;
    datapool::DataPool <unsigned char> *data_pool; 
    ThreadPool* thread_pool;
    std::vector <AiInference *> ai_inference_vec;

    std::vector< std::shared_future<int> > thread_results; 

        // synchronization
    std::mutex source_mutex;
    std::condition_variable source_condition;

    void bind_data_and_thread_pool(ThreadPool *_thread_pool);

    void initComponent();
    void saveToDataPool();
    void convertInDataPool();
    void assignToThread();
    void renderImgData();

};


} // namespace agency


#endif