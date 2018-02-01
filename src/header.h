#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
# define CONSOLE_DISPLAY 1
#define INNER_PI 3.14159265358979323846f
/*--------------------------------------------
プログラム実行時に変更可能なパラメータ
--------------------------------------------*/
#define RECON_THREADx  16
#define RECON_THREADy  16
#define PIC_SET_NUM 16 //1回の逆投影カーネルで処理する投影像の枚数
#define Z_PIXEL  128			//1回の逆投影で処理するZ方向のピクセル数
#define PROJECTION_SIZE 1024
#define STREAM_NUM PIC_SET_NUM				//フィルタリングのストリーム数


/*--------------------------------------------
再構成用のパラメータを格納する構造体
--------------------------------------------*/

typedef struct{

	float sid;							//from X-ray source to projection
	float sod;							//from X-ray source to volume center(0,0,0)
	float midplane;						//Ｘ線光軸高さ[pixel]
	float center;						//回転中心を検出器に投影した[pixel]
	float dpu;							//検出器の水平方向　画素サイズ[mm]
	float dpv;							//検出器の垂直方向　画素サイズ[mm]
	short nu, nv, np;					//2D Projection Pixel Size，枚数NP
	short nx, ny, nz;					//3D Volume Pixel Size
	float centerx, centery, centerz;	//回転原点からの再構成中心オフセット[mm]
	float pixel_pitch, slice_pitch;		//
	float u_keisu, v_keisu;				//
	float c0, c1;						//
}PictureParameter;

/*--------------------------------------------
再構成用の投影像ポインタを格納する構造体
--------------------------------------------*/

typedef struct{

	float **picArray;

}PicturePtr;


/*--------------------------------------------
strcture for filtering and reconstruction
--------------------------------------------*/
typedef struct{
	PictureParameter	*picParam;
	PicturePtr		picture;
	float*			subVolume;
	int				pixel_num, pic_data_size;
	long long		sub_voxel_num, subVol_data_size;
}VolumeParameter;

/*--------------------------------------------
関数のプロトタイプ宣言
--------------------------------------------*/

void get_param(VolumeParameter *, char *);
cudaStream_t * allocate_stream(PictureParameter);				//ストリームを確保する関数
void set_para(PictureParameter, VolumeParameter *, int);		//処理する投影像，ボリュームを設定
void free_host_mem(VolumeParameter *);							//確保した領域を解放する関数
void calc_parameter(PictureParameter, VolumeParameter *);		//各スレッドで共通するパラメータを計算
void OutputToDisk(VolumeParameter *, int Sub_Vol_ID, char *);		//ボリュームデータの書き込み
void initialize_texture(void);									//テクスチャ情報を初期化する関数
void bind_texture(int, cudaArray *);							//テクスチャをバインドする関数
VolumeParameter initialize_gpu(VolumeParameter);				//GPUを初期化する関数
//struct texObjtStrut { cudaTextureObject_t texAry[SPERATE_NUM]; };

__global__ void	convolution(float *PICTURE, float* RESULT, PictureParameter picParam);
__global__ void
reconstruction(float* RESULT, int picture_ID, int Sub_Vol_ID, cudaTextureObject_t texObjt, PictureParameter );
