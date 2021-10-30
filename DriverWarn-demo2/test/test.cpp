#include "central_agency.h"
#include "utils/configenv.hpp"

using namespace agency;

#define DEFAULT_GST_CONFIG "../res/gst_config.ini"
#define DEFAULT_AI_CONFIG "../res/ai_config.ini"

int main(int argc, char **argv)
{
    cout << __FILE__ << __LINE__ << endl;
    if(setCameraEnv()) {
        printf("camera env init failed\n");
        return -1;
    }
    system("source /etc/gstreamer1.0/set_gst_env.sh");
    MulitGstCamera::GstEnvInit();
    GMainLoop *main_loop = g_main_loop_new(NULL,false);

    CentralAgency* center = new CentralAgency(DEFAULT_GST_CONFIG, DEFAULT_AI_CONFIG);
    center->Init();
    center->RunCenter();
    center->~CentralAgency();


    g_main_loop_run(main_loop);
    MulitGstCamera::GstEnvDeinit();
    g_main_loop_unref(main_loop);

    return 0;
}