#include "../../utils/timeutil.h"
#include "ai_inference.h"
#include "result_process.h"

#define LOG(x) std::cerr

using namespace tflite;

namespace ai2nference {


//string split function
std::vector<std::string> selfSplit(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//extro string for easy opration
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

/**
 * constructor provide two construction type,
 * once, the init param from .ini config file,another,an AIConf object;
 * */
AiInference::AiInference(string _conf_file,int _conf_item)
{
    AIConf aiConf;
    memset(&this->settings,0,sizeof(AIConf));
    sprintf(this->settings.ai_node,"ai_thread_%d",_conf_item);
    ai_param_load((char *)_conf_file.c_str(),&this->settings);

    std::cout << "fact ai inference info: " << this->settings.model_path << std::endl;
}

AiInference::AiInference(AIConf _ai_conf)
{
    this->settings = _ai_conf;
}

void AiInference::loadTfliteModel()
{
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    this->model_path = settings.model_path;
    if (this->model_path.empty()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }
    std::cout << "model path: " << this->model_path << std::endl;
    this->model = tflite::FlatBufferModel::BuildFromFile(this->model_path.c_str());
    if(! this->model) {
        LOG(FATAL) << "\nFailed to mmap model " << settings.ai_name << "\n";
        exit(-1);
    }
    this->model->error_reporter();
    LOG(INFO) << "resolved reporter\n";
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;


    tflite::InterpreterBuilder(*this->model,this->resolver)(&this->interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    // the line code for indicating it that you will use old NNAPI
    this->interpreter->UseNNAPI(false);
    this->interpreter->SetAllowFp16PrecisionForFp32(false);

    // show tflite model all tensor info
    if(false) {
        std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        LOG(INFO) << "tensors size: " << this->interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << this->interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << this->interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << this->interpreter->GetInputName(0) << "\n";

        int t_size = this->interpreter->tensors_size();
        for (int i = 0; i < t_size; i++) {
            std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
            if (this->interpreter->tensor(i)->name)
                LOG(INFO) << i << ": " << this->interpreter->tensor(i)->name << ", "
                        << this->interpreter->tensor(i)->bytes << ", "
                        << this->interpreter->tensor(i)->type << ", "
                        << this->interpreter->tensor(i)->params.scale << ", "
                        << this->interpreter->tensor(i)->params.zero_point << "\n";
                std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
        }
    }
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
}

// the function for setting Delegates of hardware inference.
// Notice,now ,GPU Delegate temporarily unavailable in c610 open kit.
TfLiteDelegatePtrMap AiInference::getDelegatesFromConf()
{
    TfLiteDelegatePtrMap delegates;
    if(this->settings.delegate == DelegateType::GPU) {
        auto delegate = tflite::evaluation::CreateGPUDelegate();
        if (!delegate) {
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";  
        } else {
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (this->settings.delegate == DelegateType::NNAPI) {
        auto delegate = tflite::evaluation::CreateNNAPIDelegate();
        if (!delegate) {
            LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("NNAPI", std::move(delegate));
        }
    }

    if (this->settings.delegate == DelegateType::HEXAGON) {
        const std::string libhexagon_path("/data/local/tmp");
        auto delegate = 
            tflite::evaluation::CreateHexagonDelegate(libhexagon_path, true);
        if (!delegate) {
            LOG(INFO) << "Hexagon acceleration is unsupported on this platform.";
        } else {
            delegates.emplace("Hexagon", std::move(delegate));
        }
    }

    return delegates;
}

void AiInference::setGelegatesToRT(TfLiteDelegatePtrMap _delegate)
{
    for (const auto& delegate : _delegate) {
    if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
      LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
    } else {
      LOG(INFO) << "Applied " << delegate.first << " delegate.";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }
}

#if 0
//in fact, get original output result is greater choices in here.
template<class T> void AiInference::runAndGetResult(std::vector<std::vector<T>>*_result)
{
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
    for(auto output_node : this->output_index_vec) {
        TfLiteIntArray* output_dims = interpreter->tensor(output_node)->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];
        if(output_size == 0) {
            continue;
        }
        std::vector<T> result_vec;
        const long count = output_size;  // NOLINT(runtime/int)
        switch (interpreter->tensor(output_node)->type) {
        case kTfLiteFloat32:
            for (int i = 0; i < count; ++i) {
                float value;
                if (input_floating) {
                    value = this->interpreter->typed_output_tensor<float>(0)[i];
                } 
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
        _result->pop_back(result_vec);
    }
    std::cout << __FILE__ << "=======" << __LINE__ << std::endl;
    std::cout << "result vector size is " << _result.size << std::endl;
    // if(0) {
    //     for (auto result : top_results) {
    //         std::cout << "top output result,as follow:" << result.first << ":" << result.second << std::endl;
    //     }
    // } else {
    //     std::vector<cv::Point3_<float>>::iterator it;

    //     for(it = point_results.begin(); it != point_results.end(); it++){
    //         std::cout << *it << std::endl;
    //     }
    // }

}
#endif

}