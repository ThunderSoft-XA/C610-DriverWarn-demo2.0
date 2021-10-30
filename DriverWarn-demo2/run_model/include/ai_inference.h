#ifndef __AI_INFERENCE_H__
#define __AI_INFERENCE_H__

#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../config/include/param_ops.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "result_process.h"
#include "../../utils/timeutil.h"

using namespace tflite;

#define LOG(x) std::cerr

namespace ai2nference
{

template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));
}

template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//vector ---> Single row mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：must clone()，or wrong
	return dest;
}

//string sqlit func
std::vector<std::string> selfSplit(std::string str, std::string pattern);

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;


//convert original size image to model wanted size image, from tflite example label_image
//in fact,the function is a whole tflite model inference process
template <class T>
void resize(T* out, uint8_t* _input, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, AIConf* _settings) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;
  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  if(out == nullptr) {
      std::cout << "the out data point is null" << std::endl; 
  }

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;

  ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,nullptr);

  interpreter->AllocateTensors();
  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
  // fill input image
  // input[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = _input[i];
  }

  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
  int fact_input = interpreter->inputs()[0];
  auto output = interpreter->typed_tensor<float>(fact_input);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;
  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    for (int i = 0; i < output_number_of_pixels; i++) {
        if (_settings->input_mean) {
            out[i] = (output[i] - _settings->input_mean) / _settings->std_mean;
        } else {
            out[i] = (uint8_t)output[i];
        }
    }
  std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
}

class AiInference {
public:
    AiInference(){}
    AiInference(string _conf_file,int _conf_item);
    AiInference(AIConf _ai_conf);

    void loadTfliteModel();
    TfLiteDelegatePtrMap getDelegatesFromConf();
    void setGelegatesToRT(TfLiteDelegatePtrMap _delegate);
    template<class T> void loadTfliteData(int  image_width,int image_height,int image_channels,std::vector<T> _input_data ) {
        
        // const std::vector<int> inputs = this->interpreter->inputs();
        // const std::vector<int> outputs = this->interpreter->outputs();
        //record input,output node`s index value
        this->input_index_vec = this->interpreter->inputs();
        this->output_index_vec = this->interpreter->outputs();

        auto delegates_ = getDelegatesFromConf();
        for (const auto& delegate : delegates_) {
            if (this->interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=kTfLiteOk) {
                LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
            } else {
                LOG(INFO) << "Applied " << delegate.first << " delegate.";
            }
        }

        if (this->interpreter->AllocateTensors() != kTfLiteOk) {
            LOG(FATAL) << "Failed to allocate tensors!";
        }

        if (false) {
            PrintInterpreterState(interpreter.get());
        }

        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        for(auto input : this->input_index_vec) {
            LOG(INFO) << "input node index value: " << input << std::endl;
            TfLiteIntArray* dims = this->interpreter->tensor(input)->dims;
            int wanted_height = dims->data[1];
            int wanted_width = dims->data[2];
            int wanted_channels = dims->data[3];

            std::cout << "wanted_height = " << wanted_height << ",wanted_width = " << wanted_width << ",wanted_channels = " << wanted_channels;

            std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
            switch (interpreter->tensor(input)->type) {
                case kTfLiteFloat32:
                    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
                    resize<float>(interpreter->typed_tensor<float>(input), _input_data.data(),
                                    image_height, image_width, image_channels, wanted_height,
                                    wanted_width, wanted_channels, &this->settings);
                break;
                case kTfLiteUInt8:
                    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
                    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), _input_data.data(),
                                    image_height, image_width, image_channels, wanted_height,
                                    wanted_width, wanted_channels, &this->settings);
                break;
                default:
                    LOG(FATAL) << "cannot handle input type "
                                << interpreter->tensor(input)->type << " yet";
                    exit(-1);
            }
        }
        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    }
#if 0
    int recursive_parse(int size) {
        if(size != 0) {
            for(auto output_node : this->output_index_vec) {
                std::cout << "this is output node " << output_node << std::endl;
                TfLiteIntArray* output_dims = interpreter->tensor(output_node)->dims;
                // assume output dims to be something like (1, 1, ... ,size)
                // this output_dims->size ,example ,1*1*10*2 ----->size = 4
                //then,output_dims->data,---->data[0] = 1,data[2] = 10, data[3] = 3
                auto output_size = output_dims->data[output_dims->size - 1];
                std::cout << "output size " << output_size << std::endl;
                std::vector<T> result_vec;
                const long count = output_size;  // NOLINT(runtime/int)
                switch (interpreter->tensor(output_node)->type) {
                    case kTfLiteFloat32:
                        for (int i = 0; i < count; ++i) {
                            float value;
                            //0 == outputs()[index] == actually output node index
                            //example, output1`s index 123,output1`s index 124,output1`s index 125,so, the index in (0~2),124 = outputs()[1] 
                            value = this->interpreter->typed_output_tensor<float>(0)[i];
                            result_vec.push_back(value);
                        }
                        // if (0) {
                        //     get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                        //         5, threshold, &top_results, true);
                        // } else {
                        //     landmark_result<float>(interpreter->typed_output_tensor<float>(0), output_size,&point_results,true);
                        // }
                        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
                        break;
                }
            }
            return size;
        } else {
            return recursive_parse(--size);
        }
    }
#endif

    template<class T> void runAndGetResult(std::vector<std::vector<T>>*_result){
        struct timeval start_time, stop_time;
        gettimeofday(&start_time, nullptr);

        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        if (this->interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
        }

        gettimeofday(&stop_time, nullptr);
        LOG(INFO) << "invoked \n";
        LOG(INFO) << "average time: "
                << (get_us(stop_time) - get_us(start_time)) / (/*s->loop_count*/1 * 1000)
                << " ms \n";

        const float threshold = 0.001f;

        std::vector<std::pair<float, int>> top_results;

        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        //Multiple output
        int output_node_index = 0;
        for(auto output_node : this->output_index_vec) {
            std::cout << "this is output node " << output_node << std::endl;
            TfLiteIntArray* output_dims = interpreter->tensor(output_node)->dims;
            // assume output dims to be something like (1, 1, ... ,size)
            // this output_dims->size ,example ,1*1*10*2 ----->size = 4
            //then,output_dims->data,---->data[0] = 1,data[2] = 10, data[3] = 3
            // auto output_size = output_dims->data[output_dims->size - 1];
            auto output_size = 1;
            std::vector<int> output_dims_vec;
            for(int i = 0; i < output_dims->size; i++ ){
                output_dims_vec.push_back(output_dims->data[i]);
                output_size *= output_dims->data[i];
            }
            std::cout << "output size " << output_size << std::endl;
            std::vector<T> result_vec;
            const long count = output_size;  // NOLINT(runtime/int)
            switch (interpreter->tensor(output_node)->type) {
            case kTfLiteFloat32:
                for (int i = 0; i < count; ++i) {
                    float value;
                    //0 == outputs()[index] == actually output node index
                    //example, output1`s index 123,output1`s index 124,output1`s index 125,so, the index in (0~2),124 = outputs()[1] 
                    value = this->interpreter->typed_output_tensor<float>(output_node_index)[i];
                    result_vec.push_back(value);
                }
                // if (0) {
                //     get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                //         5, threshold, &top_results, true);
                // } else {
                //     landmark_result<float>(interpreter->typed_output_tensor<float>(0), output_size,&point_results,true);
                // }
                std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
                break;
            case kTfLiteUInt8:
                uint8_t value;
                for (int i = 0; i < count; ++i) {
                    value = this->interpreter->typed_output_tensor<float>(0)[i] / 255.0;
                    result_vec.push_back(value);
                }
                get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                                    output_size, 5, threshold,
                                    &top_results, false);
                std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
                break;
            default:
                LOG(FATAL) << "cannot handle output type "
                            << interpreter->tensor(output_node)->type << " yet";
                exit(-1);
            }
            _result->push_back(result_vec);
            if(output_node_index < this->output_index_vec.size()){
                output_node_index++;
            }
        }
        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        std::cout << "result vector size is " << _result->size()<< std::endl;
    }
    std::vector<std::vector<std::pair<float, int>>> all_result_vec;
    TfLiteType getTensorType(int node_index){
        return this->interpreter->tensor(node_index)->type;
    }

    std::vector<int> getOutputNodeIndex(){
        return this->interpreter->outputs();
    }

    string get_ai_data_source() {
        return string(this->settings.data_source, this->settings.data_source + strlen(this->settings.data_source));
    }

private:
    AIConf settings;
    string model_path;
    string data_source;
    std::vector<int> input_index_vec,output_index_vec;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

};
    
} // namespace ai2nference


#endif