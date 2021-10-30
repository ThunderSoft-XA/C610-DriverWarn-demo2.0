#include "central_agency.h"
#include "thread_task.h"

extern void run_gst_task(GstCamera *_camera);
extern void run_convert_task(std::vector<datapool::DataPacket<unsigned char>> _data);

#define DEFAULT_GST_CONFIG "../res/gst_config.ini"
#define DEFAULT_AI_CONFIG "../res/ai_config.ini"
#define LANDMARK_PARSE_FILE "../models/landmark_contours.txt"

namespace agency
{
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));
}

template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
	return dest;
}

//字符串分割函数
std::vector<std::string> selfSplit(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}


CentralAgency::CentralAgency(/* args */)
{
    this->gst_config_file = DEFAULT_GST_CONFIG;
    this->ai_config_file = DEFAULT_AI_CONFIG;
    std::cout << "Using default config file !!!" <<std::endl;
}

CentralAgency::CentralAgency(string _gst_config_file,string _ai_config_file)
{
    if( !_gst_config_file.empty()){
        this->gst_config_file = _gst_config_file;
    } else {
        std::cout << "invaild GST configure file,Using default config file !!!" <<std::endl;
        this->gst_config_file = DEFAULT_GST_CONFIG;
    }
    if ( !_ai_config_file.empty()) {
        this->ai_config_file = _ai_config_file;
    } else {
        std::cout << "invaild AI configure file,Using default config file !!!" <<std::endl;
        this->ai_config_file = DEFAULT_AI_CONFIG;
    }
}

void CentralAgency::Init()
{
    //Strictly control all initialization processes and only include data initialization 
    //initialize the gst data source have priority,only config param,The actual initialization is done in the alone thread 
    MulitGstCamera::GstEnvInit();
    IniConf ini_conf;
    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)this->gst_config_file.c_str(),&ini_conf);

    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "gstreamer config info " << i << ",as follow:" <<std::endl;
        this->gst_camera_vec.push_back(new GstCamera(this->gst_config_file.c_str(),i));
    }

    this->data_pool = new DataPool<unsigned char>(1024,FALSE);

    for(auto gst_camera : this->gst_camera_vec) {
        gst_camera->Init();
        // std::thread gst_thread(&GstCamera::RunGst,gst_camera);
        std::thread gst_thread([=]{
            DEBUG_FUNC();
            gst_camera->RunGst(this->data_pool);
        });
        gst_thread.detach();
    }
    DEBUG_FUNC();
    //initialize AI resource,load model file ,prepare Interpreter 
    get_ini_info((char *)this->ai_config_file.c_str(),&ini_conf);
    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "AI runtime config info " << i << ",as follow:" <<std::endl;
        this->ai_inference_vec.push_back(new AiInference(this->ai_config_file,i));
    }
#if 1
    DEBUG_FUNC();
    for(auto ai_inference : this->ai_inference_vec) {
        ai_inference->loadTfliteModel();
    }
#endif

    PoolConfig thread_pool_conf;
    thread_pool_conf.core_threads = 5;
    thread_pool_conf.max_task_size = 16;
    thread_pool_conf.max_threads = 8;
    thread_pool_conf.thread_type = ThreadType::SOURCE_THREAD;
    thread_pool_conf.time_out = std::chrono::seconds(5);

    //initialize three thread pool,don`t insert task
    this->thread_pool = new ThreadPool(thread_pool_conf);

    DEBUG_FUNC();
}

void CentralAgency::RunCenter()
{
    DEBUG_FUNC();
    for(;;) {
        // std::cout << "now data packet number is " << this->data_pool->getDataNum() << std::endl;
        // for(int packet_index = 0; packet_index < this->data_pool->getDataNum(); packet_index++) {
        //     DataPacket<unsigned char>* packet = this->data_pool->getPacket(packet_index);
        //     DEBUG_FUNC();
        //     if(packet->data_info.is_calibrated == true && packet->data_state == DataState::Inited) {
        //         DEBUG_FUNC();
        //         this->thread_pool->AddTask([packet]{
        //             DEBUG_FUNC();
        //             packet->imageCalibration();
        //         });
        //     }
        //     DEBUG_FUNC();
        //     for(auto ai_inference : this->ai_inference_vec) {
        //         DEBUG_FUNC();
        //         if(packet->source_name == ai_inference->get_ai_data_source()) {
        //             // DEBUG_FUNC();
        //             this->thread_pool->AddTask([ai_inference,packet]{
        //                 DEBUG_FUNC();
        //                 ai_inference->loadTfliteData(packet->data_info.width,packet->data_info.height,packet->data_info.channel,*packet->data);
        //                 std::vector<std::vector<float>> inference_result;
        //                 ai_inference->runAndGetResult(&inference_result);
        //                 packet->data_state = DataState::Inferenced;
        //                 DEBUG_FUNC();
        //             });
        //         }
        //     }
        //     this->data_pool->rmPackage(packet);
        //     packet_index--;
        //     DEBUG_FUNC();
        // }
        int packet_total = this->data_pool->getDataNum();
        if( packet_total > 0) {
        } else {
            continue;
        }
        for(int packet_index = 0; packet_index < packet_total; packet_index++) {
            DataPacket<unsigned char>* packet = this->data_pool->getPacket(packet_index);
            DEBUG_FUNC();
            if(packet == nullptr ) {
                continue;
            }
            DEBUG_FUNC();
            if(packet->source_name == "gst_zero"){
                std::thread ai_thread_1([=]{
                    DEBUG_FUNC();
                    ai_inference_vec[0]->loadTfliteData(packet->data_info.width,packet->data_info.height,packet->data_info.channel,*packet->data);
                    std::vector<std::vector<float>> inference_result;
                    DEBUG_FUNC();
                    ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
                    DEBUG_FUNC();
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
                    DEBUG_FUNC();
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
                            DEBUG_FUNC();
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
                    // this->data_pool->rmPackage(packet);
                    DEBUG_FUNC();
                });
                ai_thread_1.join();
            }else if(packet->source_name == "gst_one") {
                DEBUG_FUNC();
                std::thread ai_thread_2([=]{
                    DEBUG_FUNC();
                    ai_inference_vec[2]->loadTfliteData(packet->data_info.width,packet->data_info.height,packet->data_info.channel,*packet->data);
                    std::vector<std::vector<float>> inference_result;
                    DEBUG_FUNC();
                    ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
                    for(auto result : inference_result) {
                        std::cout << "inferece result as follow:" << std::endl;
                        for(auto value : result) {
                            std::cout << value;
                        }
                        std::cout << std::endl;
                    }
                    // this->data_pool->rmPackage(packet);
                    DEBUG_FUNC();
                });
                DEBUG_FUNC();
                ai_thread_2.join();
            }
        }
    }

}


CentralAgency::~CentralAgency()
{
}



    
} // namespace agency

