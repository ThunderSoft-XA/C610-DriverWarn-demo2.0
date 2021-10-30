#ifndef GST_CAMERA_H
#define GST_CAMERA_H

#include <iostream>
#include <sstream>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <gst/app/app.h>
#include <cairo/cairo.h>
#include "utils/timeutil.h"
#include "publicattr.h"
#include "bufmanager.hpp"
#include "config/include/param_ops.h"
#include "data_pool.h"

using namespace datapool;

// extern std::shared_ptr<BufManager<cv::Mat>> rgb_object_frame;

namespace gstcamera{

class MulitGstCamera
{
private:
    static GstElement *g_pPipeline;

public:
    static void GstEnvInit() {
        gst_init (NULL, NULL);
    }

    static GstElement *GetPipeline() {
        return g_pPipeline;
    }
    static void GstEnvDeinit() {
        if(g_pPipeline != NULL)
        {
            gst_element_set_state (g_pPipeline, GST_STATE_NULL);
            gst_object_unref (g_pPipeline);
            g_pPipeline = NULL;
        }
    }
};

class GstCamera : public VideoAttr{
private:
    string pipeline_str;
    string pipe_name;
    string sink_name;
    GstType gst_type;
    bool hw_dec;
    bool need_calibration;

    GstElement *pipeline,*appsink;
    GstBus *gstbus;
    GError *error;

    struct timeval gst_time_start;

    std::shared_ptr<BufManager<GstSample> > frame_cache;

    void buildLocalPipeLine(bool isHwDec);
    void buildRtspPipeLine(bool isHwDec);
    void buildCameraPipeLine(bool isHwDec, int camera_num = -1);
    bool launchPipeLine(string _pipeline_str);
    void handleAppsink();
    
    static void onEOS(GstAppSink *appsink, void *user_data);
    static GstFlowReturn onPreroll(GstAppSink *appsink, void *user_data);
    static GstFlowReturn onBuffer(GstAppSink *appsink, void *user_data);

    static gboolean MY_BUS_CALLBACK (GstBus *bus, GstMessage *message, gpointer data);

    /**
     * this function only get buffer data from sample,then construct a data packet.
     * param: _camera ----- gst camera class of create original sample;
     *                  _data_pool ---- call addpacket() function for add data packet(original data without any treatment ) to data pool.
     * */
    template <typename T>
    void getDataFromSample(GstCamera* _camera, datapool::DataPool<T>* _data_pool) {
        std::shared_ptr<GstSample> sample;
        GstCamera* pCam = _camera;
        static int count = 0;

        DEBUG_FUNC();
        for ( ; ;) {
            pCam->CaptureNoWait(sample);
            if(NULL == sample || NULL == sample.get()) {
#ifdef DEBUG
                std::cout << "the sample is null or invaild" << std::endl;
#endif
                continue;
            }

            DEBUG_FUNC();
            GstCaps *sample_caps = gst_sample_get_caps(sample.get());
            // std::shared_ptr<cv::Mat> object_frame;
            gint sample_width,sample_height;

            GstStructure *structure = gst_caps_get_structure(sample_caps,0);
            gst_structure_get_int(structure,"width",&sample_width);
            gst_structure_get_int(structure,"height",&sample_height);
            DEBUG_FUNC();

            GstBuffer *gst_buffer = gst_sample_get_buffer(sample.get());
            if (NULL == gst_buffer || !sample_width || !sample_height) {
                continue;
            }
            cout <<"==sample width="<< sample_width <<",sample height = " << sample_height <<  endl;
            GstMapInfo sample_map;
            gst_buffer_map(gst_buffer,&sample_map,GST_MAP_READ);
            std::cout << "smaple map size: " << sample_map.size << std::endl;

            unsigned char *data_buffer = (unsigned char*)malloc(sizeof(unsigned char)*sample_map.size);
            if(data_buffer != nullptr) {
                DEBUG_FUNC();
                memset(data_buffer,0x00, sample_map.size);
                memcpy(data_buffer, (guchar *)sample_map.data, sample_map.size);
                if (_data_pool != nullptr) {
                    DEBUG_FUNC();
                    DataInfo data_info = {sample_width,sample_height,3,false,_camera->is_hwdec(),_camera->get_need_calibration()};
                    cv::Mat tmp_mat = cv::Mat(sample_height,sample_width,CV_8UC4,data_buffer,0);
                    std::shared_ptr<cv::Mat> object_frame =  std::make_shared<cv::Mat>(tmp_mat);
                    // rgb_object_frame->feed(object_frame);
                    // stringstream str_name;
                    // str_name << "record_" << count << ".png";
                    // count++;
                    // cv::imwrite(str_name.str(),tmp_mat);
                    _data_pool->addPacket(data_buffer,sample_map.size,datapool::DataPool<T>::packet_index++,pCam->get_pipe_name(),data_info);
                    std::cout << __FILE__ << "================" << __LINE__ << std::endl;
                }
            }
            free(data_buffer);
            //Avoiding memory leaks
            gst_buffer_unmap(gst_buffer, &sample_map);
            DEBUG_FUNC();
        }
    }

public:
    GstCamera(const char *_conf_path,int _conf_item);
    GstCamera(int argc, char **argv);
    ~GstCamera();

    bool stream_end;
    float trackPS;

    void Init();
    template <typename T>
    void RunGst(datapool::DataPool<T>* _data_pool = nullptr){
        this->launchPipeLine(this->pipeline_str);

        this->handleAppsink();

        /* Run the pipeline for Start playing */
        DEBUG_FUNC();
        if (GST_STATE_CHANGE_FAILURE == gst_element_set_state (this->pipeline, GST_STATE_PLAYING)) {
            g_printerr ("Unable to set the pipeline to the playing state.\n");
        }
        //clear pipeline buffer

        /* Putting a Message handler */
        DEBUG_FUNC();
        this->gstbus = gst_pipeline_get_bus (GST_PIPELINE (this->pipeline));
        gst_bus_add_watch (this->gstbus, MY_BUS_CALLBACK, reinterpret_cast<void *>(this));
        gst_object_unref (this->gstbus);

        // this->getDataFromSample(this,_data_pool);

    }
    GstStateChangeReturn setPipeState(GstState _state);
    std::shared_ptr<GstSample> get_gst_sample();
    void Deinit();
    void CaptureNoWait(std::shared_ptr<GstSample>& dst);

    string get_pipeline_str() {
        return this->pipeline_str;
    }

    void set_pipe_name(string _name) {
        this->pipe_name = _name;
    }
    string get_pipe_name(){
        return this->pipe_name;
    }

    void set_sink_name(string _name) {
        this->sink_name = _name;
    }
    string get_sink_name(){
        return this->sink_name;
    }

    void set_gst_type(GstType _type) {
        this->gst_type = _type;
    }
    GstType get_gst_type() {
        return this->gst_type;
    }

    void set_hw_dec(bool _hw) {
        this->hw_dec = _hw;
    }
    bool is_hwdec() {
        return this->hw_dec;
    }

    void set_need_calibration(bool _need){
        this->need_calibration = _need;
    }
    bool get_need_calibration(){
        return this->need_calibration;
    }

};

}

#endif /*__GST_CAMERA_H__*/
