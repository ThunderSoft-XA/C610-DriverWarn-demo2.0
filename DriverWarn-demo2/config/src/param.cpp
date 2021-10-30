#include "config/include/iniparser.h"
#include "config/include/file_ops.h"
#include "config/include/param_ops.h"
#include "errno.h"


static char mfileName[1024]={0};
static dictionary *conf_dic;

static FILE *fp_dic = NULL;

int is_file_exist(char *file_path)
{
	if (!file_path)
		return -1;

	if (access(file_path, 0) != -1)
		return 0;

	return -1;
}


/*bref; sync
 * 		employee_time= 
 * 		visitor_time=
 * 		illegal_time= 
 * 		statistic
 * 		recognized_cnt= */
int param_init(void)
{
	int ret = is_file_exist(mfileName);
	if(ret != 0)
	{
		printf("%s: %s not exsit \n",__func__,mfileName);
	return -1;
	}
	conf_dic = iniparser_load(mfileName);
	if (!conf_dic) {
		printf("failed to load app config file:%s", mfileName);
		return -1;
	}


	return 0;
}

void param_deinit(void)
{
	if (conf_dic)
		iniparser_freedict(conf_dic);
	if(fp_dic)
    	fclose(fp_dic);
}


int param_set_int(const char *section, const char *key, int val)
{
	char buf[32];
	int notfound = -1;
	int ret = 0;

	if (!conf_dic)
		return notfound;

	snprintf(buf, sizeof(buf), "%s:%s", section, key);
	char int_buf[32];
	snprintf(int_buf, sizeof(int_buf), "%d",val);

	printf("%s: buf %s val %d ",__func__,buf,val);

	ret = iniparser_set(conf_dic, (const char *)buf, (const char *)int_buf);

//写入文件
     fp_dic = fopen(mfileName, "w");
    if( fp_dic == NULL ) {
        printf("stone:fopen error!\n");
		return -1;
    }
    iniparser_dump_ini(conf_dic,fp_dic);
	fclose(fp_dic);

	return ret;
}

int param_set_string(const char *section, const char *key, char *val)
{
	char buf[32];
	int notfound = -1;
	int ret = 0;

	if (!conf_dic)
		return notfound;

	snprintf(buf, sizeof(buf), "%s:%s", section, key);
	printf("%s: buf %s val %s ",__func__,buf,val);

	ret = iniparser_set(conf_dic, (const char *)buf, (const char *)val);
//写入文件
     fp_dic = fopen(mfileName, "w");
    if( fp_dic == NULL ) {
        printf("stone:fopen error!\n");
		return -1;
    }

    iniparser_dump_ini(conf_dic,fp_dic);
	fclose(fp_dic);

	return ret;
}

const char *param_get_string(const char *section, const char *key, const char *notfound)
{
	char buf[32];
	char *str = NULL;

	if (!conf_dic)
		return notfound;

	snprintf(buf, sizeof(buf), "%s:%s", section, key);

	str =  (char *)iniparser_getstring(conf_dic, buf, notfound);
	return (const char *)str;
}

int param_get_int(const char *section, const char *key, int notfound)
{
	char buf[32];
	int ret = 0;

	if (!conf_dic)
		return notfound;

	snprintf(buf, sizeof(buf), "%s:%s", section, key);

	ret = iniparser_getint(conf_dic, buf, notfound);

	return ret;
}

double param_get_double(const char *section, const char *key, double notfound)
{
	char buf[32];
	double ret = 0;

	if (!conf_dic)
		return notfound;

	snprintf(buf, sizeof(buf), "%s:%s", section, key);

	ret = iniparser_getdouble(conf_dic, buf, notfound);

	return ret;
}

int gst_param_load(char *fileName, StreamConf* pstCameraConf)
{
	sprintf(mfileName,"%s",fileName);
	
	int ret = param_init();
	if(ret)
		return -1;

	sprintf(pstCameraConf->gst_name, "%s",param_get_string(pstCameraConf->gst_dic,"gstname","gst_zero"));
	sprintf(pstCameraConf->gst_sink, "%s",param_get_string(pstCameraConf->gst_dic,"sinkname","gst_sink"));
	sprintf(pstCameraConf->decode, "%s",param_get_string(pstCameraConf->gst_dic,"decode","NULL"));
	sprintf(pstCameraConf->format, "%s",param_get_string(pstCameraConf->gst_dic,"format","NV12"));
	sprintf(pstCameraConf->enable, "%s",param_get_string(pstCameraConf->gst_dic,"enable","off"));
	sprintf(pstCameraConf->path, "%s",param_get_string(pstCameraConf->gst_dic,"path","NULL"));

	pstCameraConf->gst_type = (GstType)param_get_int(pstCameraConf->gst_dic,"gsttype",0);
	pstCameraConf->ai_type = (AIType)param_get_int(pstCameraConf->gst_dic,"AIType",0);
	pstCameraConf->framerate = param_get_int(pstCameraConf->gst_dic,"framerate",30);
	pstCameraConf->height = param_get_int(pstCameraConf->gst_dic,"height",1080);
	pstCameraConf->width = param_get_int(pstCameraConf->gst_dic,"width",1920);
	pstCameraConf->gst_id = param_get_int(pstCameraConf->gst_dic,"gstid",0);
	pstCameraConf->hw_dec = param_get_int(pstCameraConf->gst_dic,"hw_dec",1);
	pstCameraConf->calibration = param_get_int(pstCameraConf->gst_dic,"calibration",1);
	

	param_deinit();
	printf("%s\n",__FUNCTION__);
	printf("gstID: %d\n",pstCameraConf->gst_id);
	printf("gstName: %s\n",pstCameraConf->gst_name);
	printf("sinkName: %s\n",pstCameraConf->gst_sink);
	printf("gsttype: %d\n",pstCameraConf->gst_type);
	printf("AIType: %d\n",pstCameraConf->ai_type);
	printf("enable: %s\n",pstCameraConf->enable);
	printf("path: %s\n",pstCameraConf->path);
	printf("decode: %s\n",pstCameraConf->decode);
	printf("framerate: %d\n",pstCameraConf->framerate);
	printf("height: %d\n",pstCameraConf->height);
	printf("width: %d\n",pstCameraConf->width);
	printf("hardware decode: %s\n",(pstCameraConf->hw_dec == 1 ? "hw_dec" : "sw_dec"));
	printf("need calibration: %s\n",(pstCameraConf->calibration == 1 ? "need" : "not need"));

	if(strncmp(pstCameraConf->decode, "NULL",strlen("NULL")) == 0 || strncmp(pstCameraConf->path,"NULL",strlen("NULL")) == 0)
	{
		printf("%s: error\n",__FUNCTION__);
		return -1;
	}
	return 0;
}


int ai_param_load(char *fileName, AIConf* ai_conf)
{
	sprintf(mfileName,"%s",fileName);
	
	int ret = param_init();
	if(ret)
		return -1;

	sprintf(ai_conf->ai_name, "%s",param_get_string(ai_conf->ai_node,"ai_name",""));
	sprintf(ai_conf->model_path, "%s",param_get_string(ai_conf->ai_node,"model_path",""));
	sprintf(ai_conf->input_layer_type, "%s",param_get_string(ai_conf->ai_node,"input_layer_type","uint_8"));
	sprintf(ai_conf->data_source, "%s",param_get_string(ai_conf->ai_node,"data_source","gst_zero"));

	ai_conf->ai_id = param_get_int(ai_conf->ai_node,"ai_id",rand());
	ai_conf->input_width = param_get_int(ai_conf->ai_node,"input_width",620);
	ai_conf->input_height = param_get_int(ai_conf->ai_node,"input_height",480);
	ai_conf->channel = param_get_int(ai_conf->ai_node,"channel",3);
	ai_conf->input_mean = param_get_double(ai_conf->ai_node,"input_mean",127.5);
	ai_conf->std_mean = param_get_double(ai_conf->ai_node,"std_mean",127.5);
	ai_conf->max_profiling_buffer_entries = param_get_int(ai_conf->ai_node,"max_profiling_buffer_entries",1024);
	ai_conf->number_of_warmup_runs = param_get_int(ai_conf->ai_node,"number_of_warmup_runs",1);
	ai_conf->delegate = (DelegateType)param_get_int(ai_conf->ai_node,"delegate",1);

	param_deinit();
	printf("%s\n",__FUNCTION__);
	printf("AI id: %d\n",ai_conf->ai_id);
	printf("AI Name: %s\n",ai_conf->ai_name);
	printf("Model Path: %s\n",ai_conf->model_path);
	printf("data source: %s\n",ai_conf->data_source);
	printf("input layer type: %s\n",ai_conf->input_layer_type);
	printf("Input Width: %d\n",ai_conf->input_width);
	printf("Input Height: %d\n",ai_conf->input_height);
	printf("channel: %d\n",ai_conf->channel);
	printf("delegate: %d\n",ai_conf->delegate);
	printf("input_mean: %f\n",ai_conf->input_mean);
	printf("std_mean: %f\n",ai_conf->std_mean);
	printf("max_profiling_buffer_entries: %d\n",ai_conf->max_profiling_buffer_entries);
	printf("number_of_warmup_runs: %d\n",ai_conf->number_of_warmup_runs);


	if(strncmp(ai_conf->model_path,"NULL",strlen("NULL")) == 0)
	{
		printf("%s: error\n",__FUNCTION__);
		return -1;
	}
	return 0;
}

int get_ini_info(char *fileName,IniConf *ini_conf)
{
	sprintf(mfileName,"%s",fileName);
	
	int ret = param_init();
	if(ret)
		return -1;

	ini_conf->conf_count = param_get_int(ini_conf->ini_node,"conf_count",-1);

	param_deinit();

	return ini_conf->conf_count;
}
