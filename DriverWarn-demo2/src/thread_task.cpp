#include "thread_task.h"

void run_gst_task(GstCamera *_camera) 
{
    GMainLoop *loop;
    loop = g_main_loop_new(NULL,false);
    _camera->RunGst();

    g_main_loop_run(loop);
    g_print("g_main_loop_unref/n");
    g_main_loop_unref(loop);
}

template <class T>
void run_convert_task(datapool::DataPacket<T> _data)
{
    for(;;) {
            if (data.is_converted == true) {
                break;
            } else {
        }
    }
}