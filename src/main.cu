#include "header.h"

using namespace std;
static int TimberON;

static void Host(VolumeParameter *VolPrm, char *picFileName, char *volFileName);

int main(int argc, char* argv[]){

	VolumeParameter PrmMaster;
	PictureParameter PicParam;
	PrmMaster.picParam = &PicParam;
	if (argc < 7){	//実行ファイル名，パラメータファイル，入力ファイル，出力ファイル，最大GPU数を指定する必要がある．
		fprintf(stderr, "USAGE: reconstruction.exe [1]ParameterFile [2]InputFile [3]OutputFile [4]TimerSetting [5]Release [6]ThreadX [7]ThreadY\n");
		exit(1);
	}
	else{
		TimberON = atoi(argv[4]);

	}

	/*-------Display CUDA device properties----------------------*/
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", 0, deviceProp.name, deviceProp.major, deviceProp.minor);
	//unsigned int CPU_THREAD_NUM = omp_get_num_procs();
	//printf("Number of CPU threads = %d\n", CPU_THREAD_NUM);

	/*--------再構成パラメータの取得と必要な領域の確保---------------*/
	get_param(&PrmMaster, argv[1]);									//再構成パラメータを取得


	/*---------------main process ---------------------------------*/
	Host(&PrmMaster, argv[2], argv[3]);

	/*------------ Output to Disk ---------------------------------*/
	//if (atoi(argv[5]) == 1){
	//	OutputToDisk(&PrmMaster, argv[3]);
	//}

	free_host_mem(&PrmMaster);
	cudaDeviceReset();
	if (TimberON){
		printf("finished!\n");
		//printf("The end."); char str1[1];	scanf("%s", &str1);
	}
	return 0;
}

static void Host(VolumeParameter *VolPrm, char *picFileName, char *volFileName){

	/*--------------------------------------------タイマー設定--------------------------------------------*/
	cudaEvent_t	start0, stop0, start1, stop1, start2, stop2, start3, stop3, start4, stop4, start5, stop5,
		start6, stop6, start7, stop7, start8, stop8, start9, stop9, start10, stop10, start11, stop11;
	cudaEventCreate(&start0);	cudaEventCreate(&start1);	cudaEventCreate(&start2);	cudaEventCreate(&start3);
	cudaEventCreate(&start4);	cudaEventCreate(&start5);	cudaEventCreate(&start6);	cudaEventCreate(&start7);
	cudaEventCreate(&start8);	cudaEventCreate(&start9);	cudaEventCreate(&start10);	cudaEventCreate(&start11);
	cudaEventCreate(&stop0);	cudaEventCreate(&stop1);	cudaEventCreate(&stop2);	cudaEventCreate(&stop3);
	cudaEventCreate(&stop4);	cudaEventCreate(&stop5);	cudaEventCreate(&stop6);	cudaEventCreate(&stop7);
	cudaEventCreate(&stop8);	cudaEventCreate(&stop9);	cudaEventCreate(&stop10);	cudaEventCreate(&stop11);

	float timer0 = 0.0F, timer1 = 0.0F, timer2 = 0.0F, timer3 = 0.0F, timer4 = 0.0F, timer5 = 0.0F,
		timer6 = 0.0F, timer7 = 0.0F, timer8 = 0.0F, timer9 = 0.0F, timer10 = 0.0F, timer11 = 0.0F;

	if (TimberON)	{ cudaEventRecord(start0, 0); }
	int EndPicture = VolPrm->picParam->np;		//投影像の枚


	/*---------------- Grid size, thread size setup  --------------------------*/

	dim3 grid_fft(2, VolPrm->picParam->nv);
	dim3 threads_fft(VolPrm->picParam->nu / 2, 1);


	dim3 grid_rcs(VolPrm->picParam->nx / RECON_THREADx, VolPrm->picParam->ny / RECON_THREADy);
	dim3 threads_rcs(RECON_THREADx, RECON_THREADy);
	//printf("Reconstruction thread:\t\t\t(X,Y,Z) = (%d,%d,%d)\n", RECON_THREADx, RECON_THREADy, THREAD_SIZE_Z);
	//printf("Reconstruction block:\t\t\t(X,Y,Z) = (%d,%d,%d)\n", VolPrm->picParam->nx / RECON_THREADx, VolPrm->picParam->ny / RECON_THREADy, BLOCK_SIZE_Z);

	/*----------------ストリーム用の領域を確保--------------------------------*/
	cudaStream_t *streams = NULL;

	//フィルタリングに使用するストリーム数の領域を確保
	if ((streams = ((cudaStream_t *)malloc(STREAM_NUM * sizeof(cudaStream_t)))) == NULL){
		printf("Stream malloc error\n");
		exit(1);
	}
	for (int i = 0; i < STREAM_NUM; i++){
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}
	/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Allocate Host Memeory ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

	/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Allocate Device Memeory ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

	//if (CONSOLE_DISPLAY)	{ printf("Allocate device memory for raw picture.\n"); }
	float *dev_raw_pic_buf[PIC_SET_NUM];
	float *dev_filtered_pic_buf[PIC_SET_NUM];

	for (int i = 0; i < PIC_SET_NUM; i++){
		checkCudaErrors(cudaMalloc((void**)&dev_raw_pic_buf[i], VolPrm->pic_data_size));
		checkCudaErrors(cudaMalloc((void**)&dev_filtered_pic_buf[i], VolPrm->pic_data_size));

		checkCudaErrors(cudaMemset(dev_raw_pic_buf[i], 0, VolPrm->pic_data_size));
		checkCudaErrors(cudaMemset(dev_filtered_pic_buf[i], 0, VolPrm->pic_data_size));
	}
	float * dev_merge_pic;
	checkCudaErrors(cudaMalloc((void**)&dev_merge_pic, VolPrm->pic_data_size * PIC_SET_NUM));



	/*------------------- texture memory ---------------------------------------*/
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaExtent extent;
	extent.width = VolPrm->picParam->nu;
	extent.height = VolPrm->picParam->nv;
	extent.depth = PIC_SET_NUM;

	cudaArray_t tex_buf;
	checkCudaErrors(cudaMalloc3DArray(&tex_buf, &desc, extent, cudaArrayLayered));

	cudaResourceDesc resdesc;
	resdesc.resType = cudaResourceTypeArray;
	resdesc.res.array.array = tex_buf;

	cudaTextureDesc texdesc;
	memset(&texdesc, 0, sizeof(cudaTextureDesc));
	texdesc.normalizedCoords = 0;		//indicates whether texture reads are normalized or not
	texdesc.filterMode = cudaFilterModeLinear;
	texdesc.addressMode[0] = cudaAddressModeClamp;
	texdesc.addressMode[1] = cudaAddressModeClamp;
	texdesc.addressMode[2] = cudaAddressModeClamp;
	texdesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texObjt;
	checkCudaErrors(cudaCreateTextureObject(&texObjt, &resdesc, &texdesc, NULL));

	/*---------------- parameter for copy data to texture memory --------------------- */
	cudaMemcpy3DParms memcpy3dparam = { 0 };
	memcpy3dparam.srcPos = make_cudaPos(0, 0, 0);
	memcpy3dparam.dstPos = make_cudaPos(0, 0, 0);
	memcpy3dparam.srcPtr = make_cudaPitchedPtr(dev_merge_pic, VolPrm->picParam->nu * sizeof(float), VolPrm->picParam->nu, VolPrm->picParam->nv);
	memcpy3dparam.dstArray = tex_buf;
	memcpy3dparam.extent = make_cudaExtent(VolPrm->picParam->nu, VolPrm->picParam->nv, PIC_SET_NUM);
	memcpy3dparam.kind = cudaMemcpyDeviceToDevice;

	/*-------------- device memeory allocation for sub volume --------------*/
	float* dev_vol_buf;
	cudaMalloc((void**)&dev_vol_buf, VolPrm->subVol_data_size);
	cudaMemset(dev_vol_buf, 0, VolPrm->subVol_data_size);

	if (CONSOLE_DISPLAY)	{ printf("memory allocation finished.\n"); }


	/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Allocate Device Memeory ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
	FILE *picFile;
	if ((picFile = fopen(picFileName, "rb")) == NULL){
		printf("%s is not exist\n", picFileName);
		exit(1);
	}
        /*The following is for angle - time test*/
	ofstream angletime;
	angletime.open("angletime.csv");
	float timeArray[1200/PIC_SET_NUM] = {0.0f};// 1200/20=60
	/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ File Open finished ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

	float ftemp;
	if (TimberON)	{ cudaEventRecord(start2, 0); }

	for (int picture_ID = 0; picture_ID < EndPicture; picture_ID += PIC_SET_NUM){

		/*---------------------Copy raw picture from host to device  -----------------------------*/

		for (int i = 0; i < STREAM_NUM; i++){
			checkCudaErrors(cudaHostAlloc((void**)&VolPrm->picture.picArray[picture_ID + i], VolPrm->pic_data_size, cudaHostAllocDefault));
			fread(VolPrm->picture.picArray[picture_ID + i], sizeof(float), VolPrm->pixel_num, picFile);
			checkCudaErrors(cudaMemcpyAsync(dev_raw_pic_buf[i],
				VolPrm->picture.picArray[picture_ID + i], VolPrm->pic_data_size,
				cudaMemcpyHostToDevice, streams[i]));
		}

		/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Filter CUDA kernel ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
		if (TimberON)	{ cudaEventRecord(start3, 0); }
		printf("f");
/*
		for (int i = 0; i < STREAM_NUM; i++){
			convolution <<< grid_fft, threads_fft, 2 * VolPrm->picParam->nu * sizeof(float), streams[i]
			                       >>> (dev_raw_pic_buf[i], dev_filtered_pic_buf[i], *(VolPrm->picParam));
			getLastCudaError("filter kernel failed\n");
		}
*/
		//cudaThreadSynchronize();
		if (TimberON)	{ cudaEventRecord(stop3, 0); cudaEventSynchronize(stop3); }
		cudaEventElapsedTime(&ftemp, start3, stop3);
		timer3 += ftemp;
		/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Filter CUDA kernel ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

		/*----------------------Copy filtered picture from device to host.(OVERWRITE)---------------------*/
		for (int i = 0; i < STREAM_NUM; i++){
			checkCudaErrors(cudaMemcpyAsync(VolPrm->picture.picArray[picture_ID + i],
				dev_raw_pic_buf[i], VolPrm->pic_data_size,
				cudaMemcpyDeviceToHost, streams[i]));
		}

	}
	if (TimberON)	{ cudaEventRecord(stop2, 0); cudaEventSynchronize(stop2); }
	fclose(picFile);

	if (TimberON)	{ cudaEventRecord(start7, 0); }

		/*------------------ Reconstruct the Slice sequentially----------------------------*/
//	for (int Sub_Vol_ID = StartVolume; Sub_Vol_ID < VolPrm->picParam->nz / Z_PIXEL; Sub_Vol_ID++){
	for (int Sub_Vol_ID = 0; Sub_Vol_ID < 1; Sub_Vol_ID++){

	/*---------------------Copy filtered first PIC_SET_NUM pictures from host to device texture memory----------*/
		for (int picture_ID = 0; picture_ID < EndPicture; picture_ID += PIC_SET_NUM){
//		for (int picture_ID = 0; picture_ID < PIC_SET_NUM; picture_ID += PIC_SET_NUM){
			if (picture_ID == 0){

				for (int i = 0; i < STREAM_NUM; i++){
					checkCudaErrors(cudaMemcpyAsync(&dev_merge_pic[i * VolPrm->pixel_num],
							VolPrm->picture.picArray[picture_ID + i],
							VolPrm->pic_data_size, cudaMemcpyHostToDevice, streams[i]));
				}
				checkCudaErrors(cudaMemcpy3D(&memcpy3dparam));//Can't be async here!
			}

		/*---------------------Copy NEXT loop filtered picture from host to device global memory(temporary buffer)---*/

			if ((picture_ID + PIC_SET_NUM) < EndPicture){
				for (int i = 0; i < STREAM_NUM; i++){
					checkCudaErrors(cudaMemcpyAsync(&dev_merge_pic[i * VolPrm->pixel_num],
							VolPrm->picture.picArray[picture_ID + i + PIC_SET_NUM],		//data for NEXT loop
							VolPrm->pic_data_size, cudaMemcpyHostToDevice, streams[i]));
				}
			}

		/*----------------------- reconstruction kernel ---------------------------------------------- */
			if (CONSOLE_DISPLAY)	{ printf("r"); }
			cudaProfilerStart();
			if (TimberON)	{ cudaEventRecord(start6, 0); }
			reconstruction <<< grid_rcs, threads_rcs >>>
					(dev_vol_buf, picture_ID, Sub_Vol_ID, texObjt, *(VolPrm->picParam));
			getLastCudaError("reconstruction CUDA kernel failed\n");
			cudaThreadSynchronize();  /*Block CPU until ALL GPU finish job*/
			if (TimberON)	{ cudaEventRecord(stop6, 0); cudaEventSynchronize(stop6); }
			cudaEventElapsedTime(&ftemp, start6, stop6);
			cudaProfilerStop();
			timer6 += ftemp;
			/*for angle test*/timeArray[picture_ID / PIC_SET_NUM]+= ftemp;
			checkCudaErrors(cudaMemcpy3DAsync(&memcpy3dparam));
		}

		/*-----------------------copy volume result from device to host-----------------------------------------*/
		checkCudaErrors(cudaMemcpy(VolPrm->subVolume,
				dev_vol_buf, VolPrm->subVol_data_size, cudaMemcpyDeviceToHost));

	/*--------------clear the volume buffer from next sub volume ----------------------------*/
		checkCudaErrors(cudaMemsetAsync(dev_vol_buf, 0, VolPrm->subVol_data_size));

		{OutputToDisk(VolPrm, Sub_Vol_ID, volFileName);}
	} //for (int Sub_Vol_ID = StartVolume; Sub_Vol_ID < VolPrm->picParam->nz / Z_PIXEL; Sub_Vol_ID++)

	if (TimberON)	{ cudaEventRecord(stop7, 0); cudaEventSynchronize(stop7); }
	printf("\nReconstruction finished!\n");

	/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Finish restruction ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
/*
	FILE *filtered;
	if ((filtered = fopen("filteredPic", "wb")) == NULL){
		printf("filteredPic can not open\n");
		exit(1);
	}
	printf("Copying filtered picture to disk");
	for (int i = 0; i < VolPrm->picParam->np; i++){
		if (i % 50 == 0){
			printf(".");
		}
		fwrite(VolPrm->picture.picArray[i], sizeof(float), VolPrm->pixel_num, filtered);
	}
	printf("done\n");
*/

	/*The following is for test angle*/
	for(int i=0; i<(1200 / PIC_SET_NUM); i++){
		angletime << timeArray[i] <<endl;
	}
	angletime << timer6;
	angletime.close();
	
	if (TimberON)	{ cudaEventRecord(stop0, 0); cudaEventSynchronize(stop0); }
	/*-------------- Free Memeory and Destory object --------------------------------------*/

	checkCudaErrors(cudaFree(dev_vol_buf));
	checkCudaErrors(cudaFree(dev_merge_pic));
	checkCudaErrors(cudaFreeArray(tex_buf));
	checkCudaErrors(cudaDestroyTextureObject(texObjt));

	for (int i = 0; i < PIC_SET_NUM; i++){
		checkCudaErrors(cudaFree(dev_raw_pic_buf[i]));
		checkCudaErrors(cudaFree(dev_filtered_pic_buf[i]));
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}

	/*-----------------------実行時間の表示------------------------------------------*/
	if (TimberON){
		cudaEventElapsedTime(&timer1, start1, stop1);
		cudaEventElapsedTime(&timer2, start2, stop2);
		cudaEventElapsedTime(&timer4, start4, stop4);
		cudaEventElapsedTime(&timer5, start5, stop5);

		cudaEventElapsedTime(&timer7, start7, stop7);
		cudaEventElapsedTime(&timer8, start8, stop8);
		cudaEventElapsedTime(&timer9, start9, stop9);
		cudaEventElapsedTime(&timer10, start10, stop10);
		cudaEventElapsedTime(&timer11, start11, stop11);

		//printf("Parameter translation time          : %.2f\n", timer1);
		//printf("VRAM allocation time                : %.2f\n", timer9);
		//printf("GPU synchronization time            : %.2f\n", timer5);
		//printf("Volume initialization time          : %.2f\n", timer8);
		//printf("Projection download stream time     : %.4f\n", timer2);
		//printf("Projection download non stream time : %.4f\n", timer10);
		//printf("Projection copy time                : %.2f\n", timer11);
		//printf("Projection writeback stream time    : %.2f\n", timer4);
		//printf("Volume readback time                : %.2f\n", timer7);
		printf("Copy and filter time               : %.0f[ms]\n", timer2);
		printf("filter only time		     : %.0f[ms]\n", timer3);

		printf("Reconstruction time    : %.0f[ms]\n", timer6);
	}

	printf("Total time of reconstruction         : %.0f[ms]\n", timer7);
	cudaEventElapsedTime(&timer0, start0, stop0);
	printf("Total time                          : %.0f[ms]\n", timer0);

	cudaEventDestroy(start0);	cudaEventDestroy(stop0);
	cudaEventDestroy(start1);	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);	cudaEventDestroy(stop2);
	cudaEventDestroy(start3);	cudaEventDestroy(stop3);
	cudaEventDestroy(start4);	cudaEventDestroy(stop4);
	cudaEventDestroy(start5);	cudaEventDestroy(stop5);
	cudaEventDestroy(start6);	cudaEventDestroy(stop6);
	cudaEventDestroy(start7);	cudaEventDestroy(stop7);
	cudaEventDestroy(start8);	cudaEventDestroy(stop8);
	cudaEventDestroy(start9);	cudaEventDestroy(stop9);
	cudaEventDestroy(start10);	cudaEventDestroy(stop10);
	cudaEventDestroy(start11);	cudaEventDestroy(stop11);

}
