#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include "gstcamera.h"
#include "libyuv.h"

using namespace std;
using namespace libyuv;

namespace gstcamera{

GstElement *MulitGstCamera::g_pPipeline = NULL;

GstCamera::GstCamera(const char *_conf_file,int _conf_item)
{
    StreamConf gstCameraConf;
    memset(&gstCameraConf, 0, sizeof(StreamConf));
    sprintf(gstCameraConf.gst_dic, "gst_%d",_conf_item);
    gst_param_load((char *)_conf_file,&gstCameraConf);

    this->pipe_name = gstCameraConf.gst_name;
    this->sink_name = gstCameraConf.gst_sink;
    this->gst_type = gstCameraConf.gst_type;
    this->set_width(gstCameraConf.width);
    this->set_height(gstCameraConf.height);
    this->set_decode_type(gstCameraConf.decode);
    this->set_format(gstCameraConf.format);
    this->set_framerate(gstCameraConf.framerate);
    this->set_path(gstCameraConf.path);
    this->set_hw_dec((gstCameraConf.hw_dec == 1 ? true : false));
    this->set_need_calibration((gstCameraConf.calibration == 1 ? true : false));

}

void GstCamera::Init()
{
    this->set_index(0);
    this->pipeline = gst_pipeline_new (this->get_pipe_name().c_str());

    gst_time_start.tv_sec = 0;
    gst_time_start.tv_usec = 0;

    this->stream_end = false;

    frame_cache = std::make_shared<BufManager<GstSample> > ();
    error = NULL;
    if(this->pipe_name.empty()) {
        set_pipe_name("qmmf");
    }
    if(this->sink_name.empty()) {
        set_sink_name("app_sink");
    }
    //DEBUG_FUNC();
    if(GstType::LOCALFILE == this->gst_type) {
        buildLocalPipeLine(this->hw_dec);
    } else if (GstType::CAMERA == this->gst_type) {
        buildCameraPipeLine(this->hw_dec,atoi(this->get_path().c_str()));
    } else if (GstType::RTSP == this->gst_type) {
        buildRtspPipeLine(this->hw_dec);
    }
}

#if 0
void GstCamera::RunGst(unsigned char *_buffer_data)
{
    this->launchPipeLine(this->pipeline_str);

    this->handleAppsink();

       /* Run the pipeline for Start playing */
    //DEBUG_FUNC();
    if (GST_STATE_CHANGE_FAILURE == gst_element_set_state (this->pipeline, GST_STATE_PLAYING)) {
        g_printerr ("Unable to set the pipeline to the playing state.\n");
    }
    //clear pipeline buffer

    /* Putting a Message handler */
    //DEBUG_FUNC();
    this->gstbus = gst_pipeline_get_bus (GST_PIPELINE (this->pipeline));
    gst_bus_add_watch (this->gstbus, MY_BUS_CALLBACK, reinterpret_cast<void *>(this));
    gst_object_unref (this->gstbus);

    for( ; ;) {
        gst_camera->getDataFromSample(gst_camera,data_buffer);
    }

}
#endif

void GstCamera::handleAppsink()
{
    /* get sink */
    this->appsink = gst_bin_get_by_name (GST_BIN (this->pipeline), this->get_sink_name().c_str());
    std::cout << "appsink name = " << gst_object_get_name(GST_OBJECT(this->appsink)) << std::endl;

    /*set sink prop*/
    gst_app_sink_set_emit_signals((GstAppSink*)this->appsink, true);
    gst_app_sink_set_drop((GstAppSink*)this->appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)this->appsink, 1);
    gst_base_sink_set_sync(GST_BASE_SINK(this->appsink),false);
    gst_base_sink_set_last_sample_enabled(GST_BASE_SINK(this->appsink), true);
    //gst_base_sink_set_drop_out_of_segment(GST_BASE_SINK(this->appsink), true);
    gst_base_sink_set_max_lateness(GST_BASE_SINK(this->appsink), 0);

    {//avoid goto check
        GstAppSinkCallbacks callbacks = { onEOS, onPreroll, onBuffer };
        gst_app_sink_set_callbacks (GST_APP_SINK(this->appsink), &callbacks, reinterpret_cast<void *>(this), NULL);
    }
}

bool GstCamera::launchPipeLine(string _pipeline_str)
{
    this->pipeline = gst_parse_launch(_pipeline_str.c_str(),&this->error);
    if (this->error != NULL) {
        printf ("could not construct pipeline: %s\n", error->message);
        g_clear_error (&error);
        goto exit;
    }
    return true;
exit:
    if(this->pipeline != NULL) {
        gst_element_set_state (this->pipeline, GST_STATE_NULL);
        gst_object_unref (this->pipeline);
        this->pipeline = NULL;
    }
    return false;
}

GstStateChangeReturn GstCamera::setPipeState(GstState _state)
{
    GstStateChangeReturn ret =  gst_element_set_state(this->pipeline, _state);
    if (GST_STATE_CHANGE_FAILURE == ret) {
        return ret;
    }

    return GstStateChangeReturn::GST_STATE_CHANGE_SUCCESS;
}


void GstCamera::buildLocalPipeLine(bool isHwDec)
{
    std::ostringstream cameraPath;
    cameraPath << "filesrc location=" << get_path() << " ! " << "qtdemux name=demux demux. ! queue ! h264parse ! ";
    if(isHwDec) {
        //GstPipeline:pipeline0/GstOMXH264Dec-omxh264dec:omxh264dec-omxh264dec0: Could not initialize supporting library in 610
        cameraPath << "omx" << get_decode_type() << "dec " << " ! ";
    } else {
        cameraPath << "avdec_" << get_decode_type() << " ! ";
    }
    cameraPath << "videoscale ! video/x-raw,width=" << get_width() << ",height=" << get_height() << " ! appsink name=" << get_sink_name() << " sync=false  max-lateness=0 max-buffers=1 drop=true";
    this->pipeline_str = cameraPath.str();
    std::cout << "local file Pipeline: " << this->pipeline_str << std::endl;
}

void GstCamera::buildRtspPipeLine(bool isHwDec)
{
    std::ostringstream cameraPath;
    cameraPath << "rtspsrc location=" << get_path() << " latency=0 tcp-timeout=500 drop-on-latency=true ntp-sync=true" << " ! ";
    cameraPath << "queue ! rtp" << get_decode_type() << "depay ! "<< get_decode_type() << "parse ! queue ! ";
    if(isHwDec) {
        cameraPath << "omx" << get_decode_type() << "dec " << " ! ";
    } else {
        cameraPath << "avdec_" << get_decode_type() << " ! ";
    }
    cameraPath << "videoscale ! video/x-raw,width=" << get_width() << ",height=" << get_height() <<  " ! appsink name=" << get_sink_name() << " sync=false  max-lateness=0 max-buffers=1 drop=true";
    this->pipeline_str = cameraPath.str();
    std::cout << "rtsp Pipeline: " << this->pipeline_str << std::endl;
}

void GstCamera::buildCameraPipeLine(bool isHwDec, int camera_num)
{
    std::ostringstream cameraPath;
    cameraPath << "qtiqmmfsrc af-mode=auto name="<< get_pipe_name();
    if(-1 == camera_num) {
        cameraPath << " ! ";
    } else {
        cameraPath << " camera=" << camera_num << " " << get_pipe_name()<<".video_0 ! ";
    }
    cameraPath << "video/x-"<< get_decode_type() << ",format=" << get_format() << ",width="<< get_width() << ",height="<< get_height() <<",framerate="<< get_framerate() <<"/1" << " ! ";
    if(isHwDec) {
        cameraPath << get_decode_type() << "parse ! queue ! qtivdec ! qtivtransform ! video/x-raw,format=BGRA ! "; //qtivtransform rotate=90CW
    } else {
        cameraPath << "queue ! avdec_" << get_decode_type() <<" ! videoscale ! ";
    }
    cameraPath << "appsink name=" << get_sink_name() << " sync=false  max-lateness=0 max-buffers=1 drop=true";
    this->pipeline_str = cameraPath.str();
    std::cout << "GST Pipeline: " << this->pipeline_str << std::endl;
}

// onEOS
void GstCamera::onEOS(GstAppSink *appsink, void *user_data)
{
    GstCamera *dec = reinterpret_cast<GstCamera *>(user_data);
    dec->stream_end = true;
    printf("gstreamer decoder onEOS\n");
}

// onPreroll
GstFlowReturn GstCamera::onPreroll(GstAppSink *appsink, void *user_data)
{
    // GstCamera *dec = reinterpret_cast<GstCamera *>(user_data);
    printf("gstreamer decoder onPreroll\n");
    return GST_FLOW_OK;
}

static void deleterGstSample(GstSample* x) {
    //std::cout << "DELETER FUNCTION CALLED\n";
    if(x != NULL) {
        gst_sample_unref (x);
    }
}

// onBuffer
GstFlowReturn GstCamera::onBuffer(GstAppSink *appsink, void *user_data)
{
    //DEBUG_FUNC();
    GstCamera *dec = NULL;
    GstSample *sample = NULL;
    double elapsed_time = 0.0;
    struct timeval one_buffer_time_end;
    dec = reinterpret_cast<GstCamera *>(user_data);
    if(dec == NULL || appsink == NULL) {
        printf ("decode or appsink is null\n");
        return GST_FLOW_ERROR;
    }

    // dec->gst_pull_block();

    if(!dec->get_index()) gettimeofday(&dec->gst_time_start,nullptr);

    // sample = gst_app_sink_pull_sample(appsink);
	sample = gst_base_sink_get_last_sample(GST_BASE_SINK(appsink));
    if(sample == NULL) {
        printf ("pull sample is null\n");
    } else {
        dec->frame_cache->feed(std::shared_ptr<GstSample>(sample, deleterGstSample));
        dec->set_index(dec->get_index()+1);
    }

    GstCaps *smaple_caps = gst_sample_get_caps(sample);
    std::shared_ptr<cv::Mat> object_frame;
    gint smaple_width,smaple_height;

    GstStructure *structure = gst_caps_get_structure(smaple_caps,0);
    gst_structure_get_int(structure,"width",&smaple_width);
    gst_structure_get_int(structure,"height",&smaple_height);

    //DEBUG_FUNC();
    gettimeofday(&one_buffer_time_end,nullptr);
    elapsed_time = (one_buffer_time_end.tv_sec - dec->gst_time_start.tv_sec) * 1000 +
                                (one_buffer_time_end.tv_usec - dec->gst_time_start.tv_usec) / 1000;
    if(elapsed_time > 1000) {
		if(dec->get_index()*1000/elapsed_time < dec->get_framerate())
            printf("%s : framerate=%f\n", GetLocalTimeWithMs().c_str(),dec->get_index()*1000/elapsed_time);
        dec->set_index(0);
    }
    //DEBUG_FUNC();
    return GST_FLOW_OK;
}

gboolean GstCamera::MY_BUS_CALLBACK (GstBus *bus, GstMessage *message, gpointer data) 
{
    GstCamera *_gst_camera = reinterpret_cast<GstCamera *>(data);
    switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR: {
        GError *err;
        gchar *debug;

        gst_message_parse_error (message, &err, &debug);
        g_print ("Error: %s\n", err->message);
        g_error_free (err);
        g_free (debug);

        gst_element_set_state(_gst_camera->pipeline,GST_STATE_READY);
        break;
    }
    case GST_MESSAGE_EOS:
        /* end-of-stream */
        gst_element_set_state(_gst_camera->pipeline,GST_STATE_NULL);
        break;
    default:
      /* unhandled message */
        break;
    }
    /* we want to be notified again the next time there is a message
    * on the bus, so returning TRUE (FALSE means we want to stop watching
    * for messages on the bus and our callback should not be called again)
    */
    return TRUE;
}

void GstCamera::CaptureNoWait(std::shared_ptr<GstSample>& dst)
{
    dst = frame_cache->fetch();
}

#if 0
cv::Mat GstCamera::getDataFromSample(GstCamera* _camera, unsigned char *_data)
{
    std::shared_ptr<GstSample> sample;
    GstCamera* pCam = _camera;

    //DEBUG_FUNC();
    pCam->CaptureNoWait(sample);
    if(NULL == sample || NULL == sample.get()) {
        exit(0);
    }

    //DEBUG_FUNC();
    GstCaps *smaple_caps = gst_sample_get_caps(sample.get());
    std::shared_ptr<cv::Mat> object_frame;
    gint smaple_width,smaple_height;

    GstStructure *structure = gst_caps_get_structure(smaple_caps,0);
    gst_structure_get_int(structure,"width",&smaple_width);
    gst_structure_get_int(structure,"height",&smaple_height);
    //DEBUG_FUNC();

    GstBuffer *buffer = gst_sample_get_buffer(sample.get());
    // if (NULL == buffer || !smaple_width || !smaple_height) {
    //     continue;
    // }
    cout <<"==sample width="<< smaple_width <<",sample height = " << smaple_height <<  endl;
    GstMapInfo smaple_map;
    gst_buffer_map(buffer,&smaple_map,GST_MAP_READ);

    // Convert gstreamer data to OpenCV Mat
    // still exixst some risk , maybe need corvert style ex.NV12 to RGBA,or other
    cv::Mat tmp_mat;
    unsigned char rgb24[smaple_width * smaple_height*3];
    if(pCam->is_hwdec()) {
        cout << "enter hardware decoder" << endl;
        unsigned char *ybase = (unsigned char *)smaple_map.data;
        unsigned char *vubase = &smaple_map.data[smaple_width * smaple_height];
        //NV12 convert RGB24
        libyuv::NV12ToRGB24(ybase, smaple_width, vubase, smaple_width, rgb24, smaple_width*3, smaple_width, smaple_height);
        std::cout << __FILE__ << "==finished nv12 convert to rgb24==" << __LINE__ << std::endl;
        // get video frame 
        tmp_mat = cv::Mat(smaple_height, smaple_width, CV_8UC3, (unsigned char *)rgb24, cv::Mat::AUTO_STEP);
    } else {
        cout << "enter software decoder" << endl;
        unsigned char rgb24[smaple_width * smaple_height*3];
        unsigned char *ybase = (unsigned char *)smaple_map.data;
        unsigned char *ubase = &smaple_map.data[smaple_width * smaple_height];
        unsigned char *vbase = &smaple_map.data[smaple_width * smaple_height * 5 / 4];
        //YUV420P convert RGB24
        libyuv::I420ToRGB24(ybase, smaple_width, ubase, smaple_width / 2, vbase, smaple_width / 2,rgb24,smaple_width * 3, smaple_width, smaple_height);
        tmp_mat = cv::Mat(smaple_height, smaple_width, CV_8UC3, (unsigned char *)rgb24, cv::Mat::AUTO_STEP);
    }
    if(_data != NULL) {
        memcpy(_data, rgb24, sizeof(rgb24));
    }
    //Avoiding memory leaks
    gst_buffer_unmap(buffer, &smaple_map);
    // Make sure the frame rate is close to the true value
    return tmp_mat;
}

#endif

void GstCamera::Deinit()
{
    if(appsink!=NULL)
    {
    }
    if(pipeline!=NULL) {
        gst_element_set_state (pipeline, GST_STATE_NULL);
        gst_object_unref (pipeline);
        pipeline = NULL;
    }
}

std::shared_ptr<GstSample> GstCamera::get_gst_sample()
{
    return frame_cache->fetch();
}

GstCamera::~GstCamera()
{
}

}