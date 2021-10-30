#include "gstcamera.h"
#include "data_pool.h"
#include <thread>
#include "ai_inference.h"

using namespace ai2nference;
using namespace cv;

#define DEFAULT_AI_CONFIG "../res/ai_config.ini"
#define LANDMARK_PARSE_FILE "../models/landmark_contours.txt"
#define TRAFFIC_SIGN_FILE "../models/traffi_sign_label.txt"

#define DEFAULT_GST_CONFIG "../res/gst_config.ini"

using namespace datapool;

CascadeClassifier faceDetector;

// std::shared_ptr<BufManager<cv::Mat>> rgb_object_frame;
std::shared_ptr<BufManager<LandmarkMat>> rgb_object_frame;
std::shared_ptr<BufManager<TrafficSignMat>> traffic_sign_frame;
std::vector<GstCamera *> gst_camera_vec;
std::vector <AiInference *> ai_inference_vec;
std::vector<std::vector<int>> landmark_point_vec;    // record face landmark point from landmark contours.txt
std::vector<string> traffic_sign_vec;

std::shared_ptr<BufManager<LandmarkResult>> landmark_inference_result;
std::shared_ptr<BufManager<std::vector<cv::Point3_<float>>>> iris_inference_result;
std::shared_ptr<BufManager<TrafficSignResult>> traffic_sign_inference_result;


cv::Mat map1, map2;
const cv::Mat intrinsic_matrix_K = (cv::Mat_<double>(3,3) << 945.511,0,1545.17,0,945.511,1528.21,0,0,1);
const cv::Mat intrinsic_matrix_D = ( cv::Mat_<double> ( 8,1 ) << 0.19948,0.0422889,0,0,0.00202519,0.522254,0.0630432,0.011681);
const cv::Mat fish_matrix_D = ( cv::Mat_<double> ( 4,1 ) << 0.19948,0.0422889,0,0);


cv::Mat imageCalibration(cv::Mat _src_mat)
{
    // cv::Mat tmp_mat = cv::Mat(*this->data,CV_8UC4);
    cv::Mat UndistortImage,okay_mat;
    cv::resize(_src_mat,UndistortImage,cv::Size(1920,1080));
    cv::Size imageSize(_src_mat.cols, _src_mat.rows);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::Matx33d NewCameraMatrix;
    // cv::Mat calibration_mat = tmp_mat.reshape(4,this->data_info.height).clone();
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix_K,fish_matrix_D,imageSize,R,NewCameraMatrix,0.0);
    // NewCameraMatrix = getOptimalNewCameraMatrix(intrinsic_matrix, D, imageSize, alpha, imageSize, 0);
    cv::fisheye::initUndistortRectifyMap(intrinsic_matrix_K, fish_matrix_D,R, NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    cv::remap(_src_mat, UndistortImage, map1, map2, cv::INTER_LINEAR,cv::BORDER_CONSTANT);
    cv::resize(UndistortImage,okay_mat,cv::Size(640,480));

    return okay_mat.clone();

}

void NV12_T_RGB(unsigned int width , unsigned int height , unsigned char *yuyv , unsigned char *rgb)
{
const int nv_start = width * height ;
    unsigned int  i, j, index = 0, rgb_index = 0;
    unsigned char y, u, v;
    int r, g, b, nv_index = 0;
	
 
    for(i = 0; i <  height ; i++)
    {
		for(j = 0; j < width; j ++){
			//nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
			nv_index = i / 2  * width + j - j % 2;
 
			y = yuyv[rgb_index];
			v = yuyv[nv_start + nv_index ];
			u = yuyv[nv_start + nv_index + 1];
			
		
			r = y + (140 * (v-128))/100;  //r
			g = y - (34 * (u-128))/100 - (71 * (v-128))/100; //g
			b = y + (177 * (u-128))/100; //b
				
			if(r > 255)   r = 255;
			if(g > 255)   g = 255;
			if(b > 255)   b = 255;
       		if(r < 0)     r = 0;
			if(g < 0)     g = 0;
			if(b < 0)     b = 0;
			
			index = rgb_index % width + (height - i - 1) * width;
			rgb[index * 3+0] = b;
			rgb[index * 3+1] = g;
			rgb[index * 3+2] = r;
			rgb_index++;
		}
    }
}

void NV12_T_BGR(unsigned int width, unsigned int height, unsigned char *yuyv,
         unsigned char *bgr) {
    const int nv_start = width * height;
    int i, j, index = 0, rgb_index = 0;
    unsigned char y, u, v;
    int r, g, b, nv_index = 0;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            //nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
            nv_index = i / 2 * width + j - j % 2;

            y = yuyv[rgb_index];
            v = yuyv[nv_start + nv_index];
            u = yuyv[nv_start + nv_index + 1];
//            u = yuyv[nv_start + nv_index ];
//            v = yuyv[nv_start + nv_index + 1];

            r = y + (140 * (v - 128)) / 100;  //r
            g = y - (34 * (u - 128)) / 100 - (71 * (v - 128)) / 100; //g
            b = y + (177 * (u - 128)) / 100; //b

            if (r > 255)
                r = 255;
            if (g > 255)
                g = 255;
            if (b > 255)
                b = 255;
            if (r < 0)
                r = 0;
            if (g < 0)
                g = 0;
            if (b < 0)
                b = 0;

            index = rgb_index % width + (height - i - 1) * width;
            bgr[index * 3 + 2] = r;
            bgr[index * 3 + 1] = g;
            bgr[index * 3 + 0] = b;
            rgb_index++;
        }
    }
}

#define clamp_g(x, minValue, maxValue) ((x) < (minValue) ? (minValue) : ((x) > (maxValue) ? (maxValue) : (x)))
int NV21ToBGR(unsigned int width,unsigned int height,unsigned char * srcYVU, unsigned char * destBGR)
{

    unsigned char * srcVU = srcYVU + width * height;

    unsigned char Y, U, V;
    int B, G, R;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            Y = srcYVU[i * width + j];
            V = srcVU[(i / 2 * width / 2 + j / 2) * 2 + 0];
            U = srcVU[(i / 2 * width / 2 + j / 2) * 2 + 1];


            R = 1.164*(Y - 16) + 1.596*(V - 128);
            G = 1.164*(Y - 16) - 0.813*(V - 128) - 0.392*(U - 128);
            B = 1.164*(Y - 16) + 2.017*(U - 128);

            destBGR[(i * width + j) * 3 + 0] = clamp_g(B, 0, 255);
            destBGR[(i * width + j) * 3 + 1] = clamp_g(G, 0, 255);
            destBGR[(i * width + j) * 3 + 2] = clamp_g(R, 0, 255);


        }
    }
    return 0;
}

//hikvision yuv420p for yvu not yuv
void YUV420P_to_BGR24(int width, int height,unsigned char *data, unsigned char *bgr)
{
    int index = 0;
    unsigned char *ybase = data;
    unsigned char *vbase = &data[width * height ];
    unsigned char *ubase = &data[width * height * 5 / 4];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //YYYYYYYYUUVV
            u_char Y = ybase[x + y * width];
            u_char U = ubase[y / 2 * width / 2 + (x / 2)];
            u_char V = vbase[y / 2 * width / 2 + (x / 2)];
            bgr[index++] = Y + 1.402 * (V - 128); //R
            bgr[index++] = Y - 0.34413 * (U - 128) - 0.71414 * (V - 128); //G
            bgr[index++] = Y + 1.772 * (U - 128); //B
        }
    }
}

void Yuv420p2Bgr32(int width, int height,unsigned char *yuvBuffer_in, unsigned char *rgbBuffer_out)
{
    uchar *yuvBuffer = (uchar *)yuvBuffer_in;
    uchar *rgb32Buffer = (uchar *)rgbBuffer_out;

    int channels = 3;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            int indexY = y * width + x;
            int indexU = width * height + y / 2 * width / 2 + x / 2;
            int indexV = width * height + width * height / 4 + y / 2 * width / 2 + x / 2;

            uchar Y = yuvBuffer[indexY];
            uchar U = yuvBuffer[indexU];
            uchar V = yuvBuffer[indexV];
            
            int R = Y + 1.402 * (V - 128);
            int G = Y - 0.34413 * (U - 128) - 0.71414*(V - 128);
            int B = Y + 1.772*(U - 128);
            R = (R < 0) ? 0 : R;
            G = (G < 0) ? 0 : G;
            B = (B < 0) ? 0 : B;
            R = (R > 255) ? 255 : R;
            G = (G > 255) ? 255 : G;
            B = (B > 255) ? 255 : B;

            rgb32Buffer[(y*width + x)*channels + 2] = uchar(R);
            rgb32Buffer[(y*width + x)*channels + 1] = uchar(G);
            rgb32Buffer[(y*width + x)*channels + 0] = uchar(B);
        }
    }
}


void Pic2Gray(cv::Mat camerFrame,cv::Mat &gray)
{
	//common PC 3 channel BGR,mobile phone 4 channel
	if (camerFrame.channels() == 3)
	{
		cv::cvtColor(camerFrame, gray, CV_BGR2GRAY);
	}
	else if (camerFrame.channels() == 4)
	{
		cv::cvtColor(camerFrame, gray, CV_BGRA2GRAY);
	}
	else
		gray = camerFrame;
}

std::vector<uint8_t> decode_mat(cv::Mat _src,int row_size) 
{
    int height = _src.rows;
    int width = _src.cols;
    int channels = _src.channels();
    uchar * input = _src.data;
    std::vector<uint8_t> output(height * width * channels);
    for (int i = 0; i < height; i++) {
        int src_pos;
        int dst_pos;

        for (int j = 0; j < width; j++) {
 
        src_pos = i * row_size + j * channels;

        dst_pos = (i * width + j) * channels;

        switch (channels) {
            case 1:
            output[dst_pos] = input[src_pos];
            break;
            case 3:
            // BGR -> RGB
            output[dst_pos] = input[src_pos + 2];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos];
            break;
            case 4:
            // BGRA -> RGBA
            output[dst_pos] = input[src_pos + 2];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos];
            output[dst_pos + 3] = input[src_pos + 3];
            break;
            default:
            LOG(FATAL) << "Unexpected number of channels: " << channels;
            break;
        }
        }
    }
    return output;
}

void getDataFromSample() {

    std::shared_ptr<GstSample> sample;
    for(;;) {
        if(gst_camera_vec.empty()) {
            //DEBUG_FUNC();
            continue;
        }
        for(auto _camera : gst_camera_vec) {
            std::shared_ptr<GstSample> sample;
            GstCamera* pCam = _camera;
            static int count = 0;

            pCam->CaptureNoWait(sample);
            if(NULL == sample || NULL == sample.get()) {
                #ifdef DEBUG
                            std::cout << "the sample is null or invaild" << std::endl;
                #endif
                continue;
            }

            //DEBUG_FUNC();
            GstCaps *sample_caps = gst_sample_get_caps(sample.get());
            if( sample_caps == NULL) {
                //DEBUG_FUNC();
                continue;
            }
            // std::shared_ptr<cv::Mat> object_frame;
            gint sample_width,sample_height;

            GstStructure *structure = gst_caps_get_structure(sample_caps,0);
            gst_structure_get_int(structure,"width",&sample_width);
            gst_structure_get_int(structure,"height",&sample_height);
            //DEBUG_FUNC();

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
                //DEBUG_FUNC();
                memset(data_buffer,0x00, sample_map.size);
                memcpy(data_buffer, (guchar *)sample_map.data, sample_map.size);
                //DEBUG_FUNC();
                cv::Mat tmp_mat;
                if(pCam->get_gst_type() == GstType::CAMERA) {
                    // DataInfo data_info = {sample_width,sample_height,3,false,_camera->is_hwdec(),_camera->get_need_calibration()};
                    tmp_mat = cv::Mat(sample_height,sample_width,CV_8UC4,data_buffer,0);
                } else if(pCam->get_gst_type() == GstType::RTSP) {
                    unsigned char *bgr_data = (unsigned char*)malloc(sizeof(unsigned char) * sample_width * sample_height * 3);
                    Yuv420p2Bgr32(sample_width,sample_height,data_buffer,bgr_data);
                    tmp_mat = cv::Mat(sample_height,sample_width,CV_8UC3,bgr_data,0);
                    // cv::cvtColor(tmp_mat,tmp_mat,CV_YUV420p2BGR);
                    free(bgr_data);
                }
                // std::shared_ptr<cv::Mat> object_frame =  std::make_shared<cv::Mat>(tmp_mat);
                // rgb_object_frame->feed(object_frame);
                // if(count < 5) {
                //     continue;
                // }
                cv::Mat calibration_mat;
                if(pCam->get_need_calibration()) {
                    calibration_mat = imageCalibration(tmp_mat);
                    tmp_mat = calibration_mat;
                }
                if( pCam->get_pipe_name() == "gst_zero"){
                    std::shared_ptr<LandmarkMat> object_frame =  std::make_shared<LandmarkMat>(tmp_mat,pCam->get_pipe_name());
                    rgb_object_frame->feed(object_frame);
                    // stringstream str_name;
                    // str_name << "./gst_pipe/"<< pCam->get_pipe_name() << "_"<< count++ << ".png";
                    // cv::imwrite(str_name.str(),object_frame->data_mat);

                } else if(pCam->get_pipe_name() == "gst_one") {
                    std::shared_ptr<TrafficSignMat> object_frame =  std::make_shared<TrafficSignMat>(tmp_mat,pCam->get_pipe_name());
                    traffic_sign_frame->feed(object_frame);
                    // stringstream str_name;
                    // str_name << "./gst_pipe/"<< pCam->get_pipe_name() << "_"<< count++ << ".png";
                    // cv::imwrite(str_name.str(),object_frame->data_mat);
                }
                stringstream str_name;
                str_name << "./gst_img/"<< pCam->get_pipe_name() << "_"<< count++ << ".png";
                cv::imwrite(str_name.str(),tmp_mat);
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(50));
            free(data_buffer);
            //Avoiding memory leaks
            gst_buffer_unmap(gst_buffer, &sample_map);
            //DEBUG_FUNC();
        }
    }
}

int cmpMapWishchin(const void *p1, const void *p2)
{
    int v = 1;
    std::pair<float, int >  *pp1, *pp2;
    pp1 = (std::pair<float, int > *) p1;
    pp2 = (std::pair<float, int > *) p2;

    //max value sort
    if (pp1->first - pp2->first < 0){
        v = 1;
    }
    else{
        v = -1;
    }

return (v);
}


void ai2nfrencelandmark(){
    std::shared_ptr<LandmarkMat> tmp_landmark;
    static int count = 0;
    for(;;) {
        cv::Mat input_mat;
        string data_source_name;

        if(gst_camera_vec.empty() || ai_inference_vec.empty()) {
            //DEBUG_FUNC();
            continue;
        }
        if(ai_inference_vec.size() != 3) {
            //DEBUG_FUNC();
            continue;
        }
        if(!rgb_object_frame || rgb_object_frame->fetch() == tmp_landmark) {
            //DEBUG_FUNC();
            continue;
        }
        cv::Mat inference_mat;
        if(rgb_object_frame->fetch()->data_mat.empty()) {
            //DEBUG_FUNC();
            continue;
        } else {
            //DEBUG_FUNC();
            tmp_landmark =  rgb_object_frame->fetch();
            // input_mat = rgb_object_frame->fetch()->data_mat;
            // data_source_name = rgb_object_frame->fetch()->data_source;
            input_mat = tmp_landmark->data_mat;
            data_source_name = tmp_landmark->data_source;
            // cv::resize(input_mat,inference_mat,cv::Size(192,192));

            // stringstream str_name;
            // str_name << "./gst_data/" << data_source_name << "_"<< count++ << ".png";
            // cv::imwrite(str_name.str(),inference_mat);
        }

        //DEBUG_FUNC();
        // std::vector<uchar> temp_packt = convertMat2Vector<uchar>(input_mat);
        std::cout << "gst data source name is " << data_source_name  << std::endl;
        if(data_source_name == "gst_zero") {
            cv::Mat face_gray_mat;
            Pic2Gray(input_mat,face_gray_mat);
            //DEBUG_FUNC();
            cv::Mat face_equalized_mat;
            cv::equalizeHist(face_gray_mat,face_equalized_mat);
            int face_flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH;
            // int face_flags = CASCADE_SCALE_IMAGE;
            cv::Size minFeatureSize(50, 50);
            float searchScaleFactor = 1.1f;
            int minNeighbors = 3;
            std::vector<int> rejlevel;
            std::vector<double> levelw;
            std::vector<Rect> faces;
            //DEBUG_FUNC();
            // faceDetector.detectMultiScale(face_equalized_mat, faces, searchScaleFactor, minNeighbors, face_flags, minFeatureSize);
            faceDetector.detectMultiScale(face_equalized_mat, faces, rejlevel,levelw, searchScaleFactor, minNeighbors, face_flags, minFeatureSize, cv::Size(), true);
            cv::Mat face_land_input_mat;
            static int face_count = 0;
            //DEBUG_FUNC();
            if(faces.size() == 0) {
                std::shared_ptr<LandmarkResult> inference_result_ptr =  std::make_shared<LandmarkResult>(input_mat,std::vector<cv::Point3_<float>>());
                landmark_inference_result->feed(inference_result_ptr);
                continue;
            }
            std::vector<std::pair<double, int> > faceConfi;
            for (int i = 0; i < (int)(faces.size()); i++){
                faceConfi.push_back(std::make_pair(levelw[i], i));
            }
            //DEBUG_FUNC();
            std::qsort(&faceConfi[0], faceConfi.size(), sizeof(faceConfi[0]), cmpMapWishchin);
            //DEBUG_FUNC();
            if(faceConfi[0].first < 3) {
                continue;
            }
            for (size_t i = 0; i < (int)faces.size() && i < 1; i++)
            {
                std::cout << "the face" << i << "confidence value is " << faceConfi[i].first << std::endl;
                if (faces[i].height > 0 && faces[i].width > 0)
                {
                    //DEBUG_FUNC();
                    cv::Rect zone = faces[faceConfi[i].second];
                    int a= faces[i].width;
                    int b= faces[i].height;
                    int offSetLeft=a/4;//x offeset
                    int offSetTop=b*0.5;
                    if( (faces[i].x-offSetLeft) > 0) {
                        zone.x=faces[i].x-offSetLeft;
                    }
                    if( (faces[i].y-offSetTop) > 0) {
                        zone.y=faces[i].y-offSetTop;
                    }
                    if( (zone.x + (a/4 *2+a)) <  input_mat.cols) {
                        zone.width= a/4 *2+a;
                    }
                    if(( zone.y + 2* b) < input_mat.rows) {
                        // zone.height=zone.width*(input_mat.cols / input_mat.rows);
                        zone.height =2* b;
                    }
                    // zone.height = b + offSetTop;

                    //DEBUG_FUNC();
                    face_land_input_mat = input_mat(zone).clone();
                    // stringstream str_name;
                    // str_name << "./gst_face/" << data_source_name << "_"<< face_count++ << ".png";
                    // cv::imwrite(str_name.str(),face_land_input_mat);
                    cv::rectangle(input_mat, zone, cv::Scalar(255, 0, 0), 1, 8, 0);
                }
            }
            if(face_land_input_mat.empty()) {
                continue;
            }
            cv::Mat face_land_mat;
            cv::resize(face_land_input_mat,face_land_mat,cv::Size(192,192));
            // if(face_land_input_mat.type() == CV_8UC4 || face_land_input_mat.channels() == 4) {
            //     cv::cvtColor(face_land_input_mat,face_land_mat,CV_BGRA2RGBA);
            // }
            //DEBUG_FUNC();
            // cv::cvtColor(face_land_input_mat,face_land_input_mat,COLOR_BGRA2RGBA);
            // std::vector<uchar> temp_packt = convertMat2Vector<uchar>(face_land_input_mat);
            std::vector<uchar> temp_packet = decode_mat(face_land_mat,192);
            std::cout <<"single row mat size = " << temp_packet.size() << std::endl;
            ai_inference_vec[0]->loadTfliteData(face_land_mat.rows,face_land_mat.cols, face_land_mat.channels(),temp_packet);
            std::vector<std::vector<float>> inference_result;
            ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
            std::vector<cv::Point3_<float>> parse_results;
            //DEBUG_FUNC();
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
            //DEBUG_FUNC();
            // std::shared_ptr<LandmarkResult> tmp_inference_result;
            // tmp_inference_result->result_vec = parse_results;
            //DEBUG_FUNC();
            std::shared_ptr<LandmarkResult> inference_result_ptr =  std::make_shared<LandmarkResult>(input_mat,parse_results);
            landmark_inference_result->feed(inference_result_ptr);
        }
#if 0
            std::vector<cv::Point3_<float>> left_eye_point_vec;
            std::vector<cv::Point3_<float>> right_eye_point_vec;
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

            cv::Point3f right_eye_center;
            int right_point_num = right_eye_point_vec.size();
            for(auto pos : right_eye_point_vec){
                right_eye_center.x += pos.x;
                right_eye_center.y += pos.y;
                right_eye_center.z += pos.z;
            }
            right_eye_center.x = right_eye_center.x / right_point_num;
            right_eye_center.y = right_eye_center.y / right_point_num;
            right_eye_center.z = right_eye_center.z / right_point_num;


            std::cout << "left eye center point :" << left_eye_center << std::endl;
            cv::Mat iris_left_mat = input_mat.clone();
            cv::Mat tmp_mat;
            cv::resize(iris_left_mat,tmp_mat,cv::Size(192,192));
            cv::Mat left_eye_mat = tmp_mat(cv::Rect2f(left_eye_center.x,left_eye_center.y,64,64));
            cv::Mat con_left_eye_mat= left_eye_mat.clone();
            vector<uchar> left_eye_vec = convertMat2Vector<uchar>(con_left_eye_mat);

            ai_inference_vec[1]->loadTfliteData(left_eye_mat.rows,left_eye_mat.cols,left_eye_mat.channels(),left_eye_vec);
            std::vector<std::vector<float>> left_eye_inference_result;
            ai_inference_vec[1]->runAndGetResult<float>(&left_eye_inference_result);
            std::vector<cv::Point3_<float>> iris_parse_results;
            for(auto result : left_eye_inference_result) {
                std::cout << "inferece result as follow:" << std::endl;
                int i = 0;
                for(auto value : result) {
                    if(result.size() == 15 && i < (result.size()/3)) {
                        cv::Point3_<float> iris_point = cv::Point3_<float>(result[i*3],result[i*3 + 1],result[i*3 + 2]);
                        i++;
                        iris_parse_results.push_back(iris_point);
                    }
                    std::cout << value;
                }
                std::cout << std::endl;
            }

            std::cout << "right eye center point :" << right_eye_center << std::endl;
            cv::Mat iris_right_mat = input_mat.clone();
            cv::resize(iris_right_mat,tmp_mat,cv::Size(192,192));
            cv::Mat right_eye_mat = tmp_mat(cv::Rect2f(right_eye_center.x,right_eye_center.y,64,64));
            cv::Mat con_right_eye_mat= right_eye_mat.clone();
            vector<uchar> right_eye_vec = convertMat2Vector<uchar>(con_right_eye_mat);

            ai_inference_vec[1]->loadTfliteData(right_eye_mat.rows,right_eye_mat.cols,right_eye_mat.channels(),right_eye_vec);
            std::vector<std::vector<float>> right_eye_inference_result;
            ai_inference_vec[1]->runAndGetResult<float>(&right_eye_inference_result);
            for(auto result : right_eye_inference_result) {
                std::cout << "inferece result as follow:" << std::endl;
                for(auto value : result) {
                    int i = 0;
                    if(result.size() == 15 && i < (result.size()/3)) {
                        cv::Point3_<float> iris_point = cv::Point3_<float>(result[i*3],result[i*3 + 1],result[i*3 + 2]);
                        i++;
                        iris_parse_results.push_back(iris_point);
                    }
                    std::cout << value;
                }
                std::cout << std::endl;
            }
            std::shared_ptr<std::vector<cv::Point3_<float>>> iris_parse_results_ptr = std::make_shared<std::vector<cv::Point3_<float>>>(iris_parse_results);
            iris_inference_result.get()->feed(iris_parse_results_ptr);

        } else if(data_source_name == "gst_one") {
            //DEBUG_FUNC();
            ai_inference_vec[2]->loadTfliteData(1080,1920,3,temp_packt);
            std::vector<std::vector<float>> inference_result;
            ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
            //DEBUG_FUNC();
            for(auto result : inference_result) {
                std::cout << "inferece result as follow:" << std::endl;
                for(auto value : result) {
                    std::cout << value;
                }
                std::cout << std::endl;
            }
            //DEBUG_FUNC();
            cv::Mat inference_mat;
            cv::resize(input_mat,inference_mat,cv::Size(320,320));
            //DEBUG_FUNC();
            std::shared_ptr<TrafficSignResult> traffic_sign_result_ptr =  std::make_shared<TrafficSignResult>(inference_mat,inference_result);
            traffic_sign_inference_result->feed(traffic_sign_result_ptr);
            //DEBUG_FUNC();
        }
#endif
    }
    // }
}

void ai2nfrencesign() {
    std::shared_ptr<TrafficSignMat> tmp_traffic_sign;
    static int count = 0;
    for(;;) {
        cv::Mat input_mat;
        string data_source_name;
        cv::Mat inference_mat;

        if(gst_camera_vec.empty() || ai_inference_vec.empty()) {
            //DEBUG_FUNC();
            continue;
        }
        if(ai_inference_vec.size() != 3) {
            //DEBUG_FUNC();
            continue;
        }

        if(!traffic_sign_frame || traffic_sign_frame->fetch() == tmp_traffic_sign) {
            //DEBUG_FUNC();
            continue;
        }

        if(traffic_sign_frame->fetch()->data_mat.empty()) {
            //DEBUG_FUNC();
            continue;
        } else {
            //DEBUG_FUNC();
            tmp_traffic_sign = traffic_sign_frame->fetch();
            // input_mat = traffic_sign_frame->fetch()->data_mat;
            // data_source_name = traffic_sign_frame->fetch()->data_source;

            input_mat = tmp_traffic_sign->data_mat;
            data_source_name = tmp_traffic_sign->data_source;

            //DEBUG_FUNC();
            // cv::resize(input_mat,inference_mat,cv::Size(320,320));
            // stringstream str_name;
            // str_name << "./gst_one_data/" << data_source_name  << "_"<< count++ << ".png";
            // cv::imwrite(str_name.str(),inference_mat);
        }
        // cv::Mat input_RGB_mat;
        // static int rgb_count = 0;
        // if(input_mat.type() == CV_8UC4 || input_mat.channels() == 4) {
        //     cv::cvtColor(input_mat,input_RGB_mat,CV_BGRA2RGBA);
        //     stringstream str_name;
        //     str_name << "./gst_one_rgb/" << data_source_name  << "_"<< rgb_count++ << ".png";
        //     cv::imwrite(str_name.str(),input_RGB_mat);
        // }
        // cv::cvtColor(input_mat,input_mat,COLOR_BGRA2RGBA);
        cv::Mat sign_input_mat;
            cv::resize(input_mat,sign_input_mat,cv::Size(320,320));
        // static int rgb_count = 0;
        // stringstream str_name;
        // str_name << "./" << data_source_name  << "_"<< rgb_count++ << ".png";
        // cv::imwrite(str_name.str(),input_mat);
        std::vector<uchar> temp_packt = convertMat2Vector<uchar>(sign_input_mat);

        //DEBUG_FUNC();
        if(data_source_name == "gst_one") {
            ai_inference_vec[2]->loadTfliteData(sign_input_mat.rows,sign_input_mat.rows,sign_input_mat.channels(),temp_packt);
            std::vector<std::vector<float>> inference_result;
            ai_inference_vec[2]->runAndGetResult<float>(&inference_result);
            //DEBUG_FUNC();
            for(auto result : inference_result) {
                std::cout << "inferece result as follow:" << std::endl;
                for(auto value : result) {
                    std::cout << value;
                }
                std::cout << std::endl;
            }
            //DEBUG_FUNC();
            std::shared_ptr<TrafficSignResult> traffic_sign_result_ptr =  std::make_shared<TrafficSignResult>(sign_input_mat,inference_result);
            traffic_sign_inference_result->feed(traffic_sign_result_ptr);
            //DEBUG_FUNC();
        }
    }
}

void preview(std::shared_ptr<cv::Mat> imgframe)
{
    cv::Mat showframe;
#if 1 
        // cv::cvtColor(*imgframe,showframe,CV_BGR2RGBA);
        // imgframe = std::make_shared<cv::Mat> (showframe);
        cv::resize(*imgframe, showframe, cv::Size(1920,1080), 0, 0, cv::INTER_LINEAR);
        cv::imshow("sink", showframe);
        cv::waitKey(1);
#endif
		return ;
}

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

static void result_show(void)
{
    std::shared_ptr<TrafficSignResult> tmp_traffic_sign_ptr;
    std::shared_ptr<LandmarkResult> tmp_landmark_ptr;
    // Multichannel display block diagram segmentation calculation
    int num_w = sqrt(gst_camera_vec.size());
    num_w = num_w + (pow(num_w, 2) < gst_camera_vec.size() ? 1 : 0);
    int num_h = gst_camera_vec.size()/num_w + (gst_camera_vec.size()%num_w > 0 ? 1 :0);
    cout << "nuw_w = " << num_w << "num_h =" << num_h << endl;

    int width = 640;
    int height = 480;
    int show_left,show_top;
    //DEBUG_FUNC();
    static int count_frame = 0;
    static int show_frame = 0;
    bool land_show;
    bool sign_show;
    for(;;) {
        if(gst_camera_vec.empty() || ai_inference_vec.empty()) {
            continue;
        }
        //DEBUG_FUNC();
        if(landmark_inference_result == NULL || traffic_sign_inference_result == NULL){
            continue;
        }
        land_show = false;
        sign_show = false;
        //DEBUG_FUNC();
        std::shared_ptr<cv::Mat> img_show;
        std::shared_ptr<cv::Mat> sub_imgframe;
        cv::Mat imageShow(cv::Size(width*num_w, height*num_h), CV_8UC4);
        img_show = std::make_shared<cv::Mat>(imageShow);

        //DEBUG_FUNC();
        for(int i = 0; i < gst_camera_vec.size(); i++) {
            GstCamera* pCam = gst_camera_vec.at(i);
            show_left = i % num_w * (width);
            show_top = i / num_w * (height); 
           if(pCam->get_pipe_name() == "gst_one") {
                if(traffic_sign_inference_result->fetch() == NULL) {
                    //DEBUG_FUNC();
                    continue;
                }
                if(traffic_sign_inference_result->fetch() == tmp_traffic_sign_ptr) {
                   continue;
                } else {
                    tmp_traffic_sign_ptr = traffic_sign_inference_result->fetch();
                }
               if(tmp_traffic_sign_ptr->show_frame.empty() || tmp_traffic_sign_ptr->result_vec.empty()) {
                    //DEBUG_FUNC();
                   continue;
               }
                //DEBUG_FUNC();
                std::vector<cv::Rect2f> rect_vec;
               sub_imgframe = std::make_shared<cv::Mat> (tmp_traffic_sign_ptr->show_frame);
                    //DEBUG_FUNC();
                std::vector<std::vector<float>> result_vec = tmp_traffic_sign_ptr->result_vec;
                int sign_num = 0;
                if( result_vec.empty()) {
                    goto drict_show;
                }
                if(result_vec.size() != 4 || result_vec[2][0] < 0.15) {
                    continue;
                }
                //DEBUG_FUNC();
                for(int i = 0; i < result_vec[3][0];i++) {
                    if(result_vec[2][i] > 0.17) {
                        //DEBUG_FUNC();
                        cv::Rect2f tmp_rect = cv::Rect2f(cv::Point2f(result_vec[0][i*4]*320,result_vec[0][i*4+1]*320),cv::Point2f(result_vec[0][i*4+2]*320,result_vec[0][i*4+3]*320));
                        // cv::putText(*sub_imgframe,"gst_one", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50),1);
                        if(tmp_rect.x > 0 && tmp_rect.y > 0 && (tmp_rect.width > 0 && tmp_rect.height > 0)) {
                            cv::rectangle(*sub_imgframe, (cv::Rect)tmp_rect, cv::Scalar(0, 200, 0));
                            cv::putText(*sub_imgframe,to_string(result_vec[2][i]), cv::Point(tmp_rect.x + 20, tmp_rect.y  + 20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                            if(!traffic_sign_vec.empty()) {
                                //DEBUG_FUNC();
                                cv::putText(*sub_imgframe,traffic_sign_vec[result_vec[1][i]], cv::Point(tmp_rect.x + 20, tmp_rect.y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                            }
                        }
                        sign_num++;
                        rect_vec.push_back(tmp_rect);
                        //DEBUG_FUNC();
                    }
                    //DEBUG_FUNC();
                }
            drict_show:
                //DEBUG_FUNC();
                std::shared_ptr<cv::Mat> tmp_img = std::make_shared<cv::Mat>();
                //DEBUG_FUNC();
                cv::resize(*sub_imgframe,*tmp_img,cv::Size(640,480));
                //DEBUG_FUNC();
                // stringstream str_name;
                // str_name << "./gst_sign/sign_" << count_frame++ << ".jpg";
                // cv::imwrite(str_name.str(),*sub_imgframe);
                //DEBUG_FUNC();
                std::cout << "show camera 1 mat channel " << tmp_img->channels() << std::endl;
                if( tmp_img->type() == CV_8UC3) {
                    cv::cvtColor(*tmp_img,*tmp_img,CV_RGB2RGBA);
                }
                tmp_img->copyTo(imageShow(cv::Rect(show_left,show_top,width,height)));
                sign_show = true;
                //DEBUG_FUNC();
            } else if(pCam->get_pipe_name() == "gst_zero") {
                //DEBUG_FUNC();
                if(landmark_inference_result->fetch() == NULL) {
                    //DEBUG_FUNC();
                    continue;
                }
                if(landmark_inference_result->fetch() == tmp_landmark_ptr) {
                   continue;
                } else {
                    tmp_landmark_ptr = landmark_inference_result->fetch();
                }
                if(tmp_landmark_ptr->show_frame.empty()/* || tmp_landmark_ptr->result_vec.empty()*/) {
                    //DEBUG_FUNC();
                    continue;
                }
                //DEBUG_FUNC();
               sub_imgframe = std::make_shared<cv::Mat> (tmp_landmark_ptr->show_frame);
                //DEBUG_FUNC();
                std::vector<cv::Point3f> result_vec = tmp_landmark_ptr->result_vec;
                if(!result_vec.empty()) {
                    // cv::resize(*sub_imgframe,*sub_imgframe,cv::Size(192,192));
                    std::vector<cv::Point3_<float>> lips_point_vec;
                    std::vector<cv::Point3_<float>> left_eye_point_vec;
                    std::vector<cv::Point3_<float>> right_eye_point_vec;

                    //Draw face landmark point
                    // for(auto index : landmark_point_vec[0]) {
                    //     cv::circle(*sub_imgframe,cv::Point2f(result_vec[index].x,result_vec[index].y),1,cv::Scalar(50, 170, 50), -1);
                    //     lips_point_vec.push_back(result_vec[index]);
                    // }
                    // for(auto index : landmark_point_vec[1]) {
                    //     cv::circle(*sub_imgframe,cv::Point2f(result_vec[index].x,result_vec[index].y),1,cv::Scalar(50, 170, 50), -1);
                    //     left_eye_point_vec.push_back(result_vec[index]);
                    // }
                    // for(auto index : landmark_point_vec[2]) {
                    //     cv::circle(*sub_imgframe,cv::Point2f(result_vec[index].x,result_vec[index].y),1,cv::Scalar(50, 170, 50), -1);
                    //     right_eye_point_vec.push_back(result_vec[index]);
                    // }
                    float distance_h = result_vec[133].x - result_vec[33].x;
                    float distance_v_1 = result_vec[145].y - result_vec[159].y;
                    float distance_v_2 = result_vec[153].y - result_vec[158].y;

                    float lips_distance_h = result_vec[409].x - result_vec[185].x;
                    float lips_distance_v_1 = result_vec[84].y - result_vec[37].y;
                    float lips_distance_v_2 = result_vec[17].y - result_vec[0].y;
                    float lips_distance_v_3 = result_vec[314].y - result_vec[267].y;

                    float lips_ear = (lips_distance_v_1 + lips_distance_v_2 + lips_distance_v_3) / (3 * lips_distance_h);
                    float left_ear = (distance_v_1 + distance_v_2) / (2 * distance_h);

                    std::cout << "lips_ear = " << lips_ear << ",left_eye_ear = " << left_ear << std::endl;
                    //DEBUG_FUNC();
                    if( left_ear < 0.16/* || lips_ear > 0.65*/)  {
                        //DEBUG_FUNC();
                        cv::putText(*sub_imgframe,"Fatigue driving", cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                        //DEBUG_FUNC();
                    } else {
                        //DEBUG_FUNC();
                        cv::putText(*sub_imgframe,"Safe driving", cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                        //DEBUG_FUNC();
                    }
                    cv::putText(*sub_imgframe,"eye_ear:" + SSTR(float(left_ear)), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                    cv::putText(*sub_imgframe,"lips_ear:"+ SSTR(float(lips_ear)), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(50, 170, 50), 1);
                }
                //DEBUG_FUNC();
                std::shared_ptr<cv::Mat> tmp_img = std::make_shared<cv::Mat>();
                //DEBUG_FUNC();
                cv::resize(*sub_imgframe,*tmp_img,cv::Size(640,480));
                //DEBUG_FUNC();
                // stringstream str_name;
                // str_name << "./gst_land/land_" << count_frame++ << ".jpg";
                // cv::imwrite(str_name.str(),*sub_imgframe);
                //DEBUG_FUNC();
                std::cout << "show camera 0 mat channel " << tmp_img->channels() << std::endl;
                // cv::cvtColor(*tmp_img,*tmp_img,CV_RGBA2BGRA);
                tmp_img->copyTo(imageShow(cv::Rect(show_left,show_top,width,height)));
                land_show = true;
                //DEBUG_FUNC();
            }
            //DEBUG_FUNC();
        }
        // if( land_show || sign_show) {
        //     //DEBUG_FUNC();
        //     stringstream str_name;
        //     str_name << "./gst_show/show_" << show_frame++ << ".jpg";
        //     cv::imwrite(str_name.str(),imageShow);
        //     //DEBUG_FUNC();
        // }

        if( land_show || sign_show) {
            DEBUG_FUNC();
            stringstream str_name;
            // str_name << "./gst_show/show_" << show_frame++ << ".jpg";
            // cv::imwrite(str_name.str(),imageShow);
            // DEBUG_FUNC();
            preview(std::make_shared<cv::Mat> (imageShow));
        }
    }
}

int main (int argc, char ** argv)
{
    MulitGstCamera::GstEnvInit();
    GMainLoop *main_loop = g_main_loop_new(NULL,false);
    IniConf ini_conf;
    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)DEFAULT_GST_CONFIG,&ini_conf);

    rgb_object_frame = std::make_shared<BufManager<LandmarkMat>> ();
    traffic_sign_frame  = std::make_shared<BufManager<TrafficSignMat>> ();
    std::cout << "rgb_object_frame address " << rgb_object_frame << ",traffic_sign_frame address " << traffic_sign_frame << std::endl;
    landmark_inference_result = std::make_shared<BufManager<LandmarkResult>> ();
    iris_inference_result = std::make_shared<BufManager<std::vector<cv::Point3_<float>>>>();
    traffic_sign_inference_result = std::make_shared<BufManager<TrafficSignResult>> ();
    
    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "gstreamer config info " << i << ",as follow:" <<std::endl;
        gst_camera_vec.push_back(new GstCamera(DEFAULT_GST_CONFIG,i));
    }

    datapool::DataPool <unsigned char> *data_pool = new DataPool<unsigned char>(1024,FALSE);
    for(auto gst_camera : gst_camera_vec) {
        gst_camera->Init();
        // std::thread gst_thread(&GstCamera::RunGst,gst_camera);
        std::thread gst_thread([=]{
            //DEBUG_FUNC();
            gst_camera->RunGst(data_pool);
        });
        gst_thread.join();
    }

    std::string faceCascadeFilename = "haarcascade_frontalface_default.xml";
	//error info tips
	try{
		faceDetector.load(faceCascadeFilename);
	}
	catch (cv::Exception e){}
	if (faceDetector.empty())
	{
		std::cerr << "don`t load face detection (" << faceCascadeFilename << ")!" << std::endl;
		exit(1);
	}

    memset(&ini_conf, 0, sizeof(IniConf));
    sprintf(ini_conf.ini_node, "conf_info");
    get_ini_info((char *)DEFAULT_AI_CONFIG,&ini_conf);

    for(int i = 0; i < ini_conf.conf_count; i++) {
        std::cout << "AI runtime config info " << i << ",as follow:" <<std::endl;
        ai_inference_vec.push_back(new AiInference(DEFAULT_AI_CONFIG,i));
    }

    for(auto ai_inference : ai_inference_vec) {
        std::cout << "loading tflite model ......" <<std::endl;
        ai_inference->loadTfliteModel();
    }

    // following code ,read param parse file include point info.
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
    {
        ifstream traffic_sign_file;
        string traffic_sign_line;
        traffic_sign_file.open(TRAFFIC_SIGN_FILE);
        getline(traffic_sign_file,traffic_sign_line);
        while(traffic_sign_file && !traffic_sign_line.empty()) {
            std::cout << "landmark line :" << traffic_sign_line << std::endl;
            traffic_sign_vec.push_back(traffic_sign_line);
            getline(traffic_sign_file,traffic_sign_line);
        }
        traffic_sign_file.close();
    }

    //DEBUG_FUNC();
    std::thread handleThread(getDataFromSample);
    handleThread.detach();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    //DEBUG_FUNC();
    std::thread inferenceThread_landmark(ai2nfrencelandmark);
    inferenceThread_landmark.detach();

    std::thread inferenceThread_sign(ai2nfrencesign);
    inferenceThread_sign.detach();

    std::this_thread::sleep_for(std::chrono::seconds(2));

    //DEBUG_FUNC();
    std::thread showThread(result_show);
    showThread.join();

    g_main_loop_run(main_loop);
    MulitGstCamera::GstEnvDeinit();
    g_main_loop_unref(main_loop);

#if 0
    // following code ,read param parse file include point info.
    std::vector<std::vector<int>> landmark_point_vec;
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

    static int count = 0;
    while(1){
        if(!data_pool->data_packet_vec.empty()) {

        } else {
            continue;
        }
        std::cout << __FILE__ << "================" << __LINE__ << std::endl;
        std::deque<datapool::DataPacket<unsigned char>*>::iterator packet_iter = data_pool->data_packet_vec.begin();
        std::deque<datapool::DataPacket<unsigned char>*>::iterator packet_iter_end = data_pool->data_packet_vec.end();
        for(;packet_iter != packet_iter_end;packet_iter++) {
            std::cout << "data pool size " <<  data_pool->data_packet_vec.size() << std::endl;
            std::cout << __FILE__ << "================" << __LINE__ << std::endl;

            std::cout << "data packet info:" << (*packet_iter)->data_info.channel << "," \
                << (*packet_iter)->data_info.width << "," << (*packet_iter)->data_info.height << std::endl;
            // cv::Mat tmp_mat = cv::Mat(*(*packet_iter)->data,CV_8UC4);
            if((*packet_iter)->source_name == "gst_zero") {
                std::vector<uchar> temp_packt = convertMat2Vector<uchar>((*packet_iter)->data_mat);
                std::thread ai_thread_1([=]{
                    ai_inference_vec[0]->loadTfliteData((*packet_iter)->data_mat.rows,(*packet_iter)->data_mat.cols,(*packet_iter)->data_mat.channels(),temp_packt);
                    std::vector<std::vector<float>> inference_result;
                    ai_inference_vec[0]->runAndGetResult<float>(&inference_result);
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
                    std::vector<cv::Point3_<float>> left_eye_point_vec;
                    std::vector<cv::Point3_<float>> right_eye_point_vec;
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
                ai_thread_1.detach();
            } else if ((*packet_iter)->source_name == "gst_one") {
                std::vector<uchar> temp_packt = convertMat2Vector<uchar>((*packet_iter)->data_mat);
                std::thread ai_thread_2([=]{
                    ai_inference_vec[2]->loadTfliteData((*packet_iter)->data_mat.rows,(*packet_iter)->data_mat.cols,(*packet_iter)->data_mat.channels(),temp_packt);
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
                ai_thread_2.detach();
            }

            // cv::Mat tmp_mat = (*packet_iter)->data_mat;
            // std::cout << "gst test mat channels,rows,cols:" << tmp_mat.channels() << "," << tmp_mat.rows << "," << tmp_mat.cols << std::endl;;
            // // cv::Mat result_mat = tmp_mat.reshape(4,(*packet_iter)->data_info.height).clone();
            // stringstream str_name;
            // str_name << "./gst_record_" << count++ << ".jpg";
            // // cv::imwrite(str_name.str(),result_mat);
            // cv::imwrite(str_name.str(),tmp_mat);
            // std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            // // (*packet_iter)->~DataPacket();
            // std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            std::unique_lock<std::mutex> locker(data_pool->queue_mutex);
            data_pool->condition.wait(locker);
            // data_pool->rmPackage((*packet_iter));
            (*packet_iter)->data_mat.~Mat();
            data_pool->data_packet_vec.erase(packet_iter);
            // std::deque<datapool::DataPacket<unsigned char>*>(data_pool->data_packet_vec).swap(data_pool->data_packet_vec);
            if(!data_pool->data_packet_vec.empty()) {
                locker.unlock();
                packet_iter = data_pool->data_packet_vec.begin();
                packet_iter_end = data_pool->data_packet_vec.end();
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
            } else {
                locker.unlock();
                std::cout << __FILE__ << "================" << __LINE__ << std::endl;
                break;
            }
        }
    }
#endif

}