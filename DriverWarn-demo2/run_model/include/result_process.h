#ifndef RESULT_PROCESS_H_
#define RESULT_PROCESS_H_

#include "result_process_imp.h"

namespace ai2nference {

template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating);

template <class T>
void landmark_result(T* prediction, int prediction_size, 
                std::vector<cv::Point3_<T>>* results,bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void get_top_n<uint8_t>(uint8_t*, int, size_t, float,
                                 std::vector<std::pair<float, int>>*, bool);
template void get_top_n<float>(float*, int, size_t, float,
                               std::vector<std::pair<float, int>>*, bool);


template
void landmark_result<uint8_t>(uint8_t* prediction, int prediction_size, 
                std::vector<cv::Point3_<uint8_t>>* results,bool input_floating);
template
void landmark_result <float>(float* prediction, int prediction_size, 
                std::vector<cv::Point3_<float>>* results,bool input_floating);


}



#endif