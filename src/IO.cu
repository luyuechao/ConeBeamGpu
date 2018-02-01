#include "header.h"

using namespace std;

/*--------------------------------------------
パラメータファイルを読み込む関数
--------------------------------------------*/
void get_param(VolumeParameter *VolParam,char *paramFile){		//再構成パラメータを取得する関数
	FILE *fp1;

	if ((fp1 = fopen(paramFile, "rb")) == NULL){			//Reading Binary file! be CAREFULL with 'b'!
		fprintf(stderr, "%s is not exist\n", paramFile);
		exit(1);
	}

	/* size_t fread ( void * ptr, size_t size, size_t count, FILE * stream );
	Reads an array of count elements, each one with a size of size bytes,
	from the stream and stores them in the block of memory specified by ptr.
	*/
	fread(VolParam->picParam, sizeof(PictureParameter), 1, fp1);						//共通のパラメータ

	VolParam->pixel_num = VolParam->picParam->nu * VolParam->picParam->nv;				//投影像の画素数を計算
	VolParam->pic_data_size = sizeof(float) * VolParam->pixel_num;						//投影像のデータサイズを計算
	VolParam->sub_voxel_num = VolParam->picParam->nx * VolParam->picParam->ny * Z_PIXEL;	//サブボリュームのボクセル数を計算
	VolParam->subVol_data_size = sizeof(float) * VolParam->sub_voxel_num;					//サブボリュームのデータサイズを計算
	fclose(fp1);
	/*-------------------------Allocate host memory for picture ----------------------------*/



	/*----------------- Copy CT projection picture from disk to host memeory ----------------------------------*/
	if ((VolParam->picture.picArray = (float **)malloc(sizeof(float *) * VolParam->picParam->np)) == NULL){
		fprintf(stderr, "Can't allocate projection region.\n");
		exit(1);
	}


	/*---------------------ボリューム格納領域の確保----------------------------------*/
/*
	if ((VolParam->volume.volArray = (float **)malloc(sizeof(float *) * VolParam->picParam->nz / Z_PIXEL)) == NULL){
		fprintf(stderr, "Can't allocate volume region.\n");
		exit(1);
	}

	for (int i = 0; i < VolParam->picParam->nz / Z_PIXEL; i++){
		checkCudaErrors(cudaHostAlloc((void**)&VolParam->volume.volArray[i], VolParam->subVol_data_size, cudaHostAllocDefault));
	}
*/
	checkCudaErrors(cudaHostAlloc((void**)&VolParam->subVolume, VolParam->subVol_data_size, cudaHostAllocDefault));


	//Calcuate_Filter(VolParam->picParam);
	//printf("Parameter file name:	%s\n", paramFile);
	printf("SID:			%f\n", VolParam->picParam->sid);
	printf("SOD:			%f\n", VolParam->picParam->sod);
	printf("midplane:		%f\n", VolParam->picParam->midplane);
	printf("center:			%f\n", VolParam->picParam->center);
	printf("dpu:			%f\n", VolParam->picParam->dpu);
	printf("dpv:			%f\n", VolParam->picParam->dpv);
	printf("nu:			%d\n", VolParam->picParam->nu);
	printf("nv:			%d\n", VolParam->picParam->nv);
	printf("np:			%d\n", VolParam->picParam->np);
	printf("nx:			%d\n", VolParam->picParam->nx);
	printf("ny:			%d\n", VolParam->picParam->ny);
	printf("nz:			%d\n", VolParam->picParam->nz);
	printf("centerx:		%f\n", VolParam->picParam->centerx);
	printf("centery:		%f\n", VolParam->picParam->centery);
	printf("centerz:		%f\n", VolParam->picParam->centerz);
	printf("pixel_pitch:		%f\n", VolParam->picParam->pixel_pitch);
	printf("slice_pitch:		%f\n", VolParam->picParam->slice_pitch);
	printf("u_keisu:		%f\n", VolParam->picParam->u_keisu);
	printf("v_keisu:		%f\n", VolParam->picParam->v_keisu);
	printf("c0:			%f\n", VolParam->picParam->c0);
	printf("c1:			%f\n", VolParam->picParam->c1);
	printf("one picture data size:	%d [MB]\n", VolParam->pic_data_size / 1048576);
	printf("SubVolume data size:	%lld [MB]\n", (long long)VolParam->subVol_data_size / 1048576);
}


/*--------------------------------------------
生成したボリュームを書き出す関数
--------------------------------------------*/

void OutputToDisk(VolumeParameter *volPrm, int Sub_Vol_ID,char *Volname){	//ボリュームをファイルに書き出す関数
	FILE *volFile;
	char filename[1024];

	sprintf(filename, "%sOutputVol_%d.dat",Volname,Sub_Vol_ID);
	if ((volFile = fopen(filename, "wb")) == NULL){
		printf("%s is not exist\n", filename);
		exit(1);
	}

	float *volDump;//buffer for one piece of picture
	if ((volDump = (float *)malloc(sizeof(float) *  volPrm->picParam->nx *  volPrm->picParam->ny)) == NULL){
		fprintf(stderr, "memory allocate error!\n");
		exit(1);
	}
	/*data is transfered into short little endian*/
	for (int z = 0; z < Z_PIXEL; z++){
		for (int y = 0; y < volPrm->picParam->ny; y++){
			for (int x = 0; x < volPrm->picParam->nx; x++){
				volDump[x + y * volPrm->picParam->nx] =
						volPrm->subVolume[x + y *volPrm->picParam->nx + z *volPrm->picParam->nx *volPrm->picParam->ny];
			}
		}
		fwrite(volDump, sizeof(float), volPrm->picParam->nx *  volPrm->picParam->ny, volFile);
		if (z % 50 == 0){printf(".");}
	}
	free(volDump);
	fclose(volFile);

}


/*--------------------------------------------
確保したメモリ領域を解放する関数
--------------------------------------------*/

void free_host_mem(VolumeParameter *VolPrm){

	for (int i = 0; i < VolPrm->picParam->np; i++){
		cudaFreeHost(VolPrm->picture.picArray[i]);
	}

	free(VolPrm->picture.picArray);


	cudaFreeHost(VolPrm->subVolume);

}
