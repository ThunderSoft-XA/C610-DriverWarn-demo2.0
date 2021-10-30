#include <algorithm>
#include <functional>
#include <queue>
#include <opencv2/opencv.hpp>

using namespace cv;

namespace ai2nference {

extern bool input_floating;

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

  const long count = prediction_size;  // NOLINT(runtime/int)
  for (int i = 0; i < count; ++i) {
    float value;
    if (input_floating)
      value = prediction[i];
    else
      value = prediction[i] / 255.0;
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

// for getting face /iris landmark (x,y,z) point value
template <class T>
void landmark_result(T* prediction, int prediction_size, std::vector<cv::Point3_<T>>* results,bool input_floating)
{
	const long count = prediction_size;  // NOLINT(runtime/int)
	cv::Point3_<T> point_value;
	for (int index = 0; index < (count + 1) /3; ++index) {
		if (input_floating) {
			point_value = cv::Point3_<T>(prediction[3*index],prediction[3*index+1],prediction[3*index+2]);
		} else {
			point_value = cv::Point3_<T>(prediction[3*index] /255.0,prediction[3*index+1]/255.0,prediction[3*index+2]/255.0);
		}
		results->push_back(point_value);
	}
}

}