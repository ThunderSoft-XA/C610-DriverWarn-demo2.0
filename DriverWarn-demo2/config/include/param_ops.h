#ifndef __PARAM_H__
#define __PARAM_H__
#include "../../include/publicattr.h"

using namespace gstcamera;


#ifdef __cplusplus
extern "C"
{
#endif
typedef struct __stream_conf
{
	char path[1024];
	char gst_dic[16];
	char gst_name[16];
	char gst_sink[16];
	char format[8];
	char decode[5];//h264 h265
	char enable[4]; //on off  tempoary don`t use
	int gst_id;
	int width;
	int height;
	int framerate;
	int hw_dec;
	int calibration;
	AIType ai_type;   //0 = None, 1 = objectTracker
	GstType gst_type;
} StreamConf;

typedef enum _DelegateType {
    CPU = 0,
    GPU,
    NNAPI,
	HEXAGON
}DelegateType;

typedef struct __ai_conf {
	char model_path[64];
	char ai_node[16];
	char ai_name[16];
	char data_source[16];
	int ai_id;
	int input_width;
	int input_height;
	int channel;
	DelegateType delegate;
	float input_mean;
	float std_mean;
	char input_layer_type[8];
	int max_profiling_buffer_entries;
	int number_of_warmup_runs;
} AIConf;

typedef struct _ini_conf {
	int conf_count;
	char ini_node[16];
} IniConf;


int param_init(void);
void param_deinit(void);
int param_set_int(const char *section, const char *key, int val);
const char *param_get_string(const char *section, const char *key, const char *notfound);
int param_get_int(const char *section, const char *key, int notfound);
double param_get_double(const char *section, const char *key, double notfound);

int param_set_string(const char *section, const char *key, char *val);

int gst_param_load(char *fileName, StreamConf* pstStreamConf);
int ai_param_load(char *fileName, AIConf* ai_conf);
int get_ini_info(char *fileName,IniConf *ini_conf);
#ifdef __cplusplus
}
#endif
#endif
