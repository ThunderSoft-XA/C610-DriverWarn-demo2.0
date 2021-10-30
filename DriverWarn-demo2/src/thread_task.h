#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "gstcamera.h"
#include "../run_model/include/ai_inference.h"
#include "data_pool.h"

void run_gst_task(GstCamera *_camera);

template <class T>
void run_convert_task(std::vector<datapool::DataPacket<T>> _data);

#endif