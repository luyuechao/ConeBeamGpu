#include "header.h"

__global__  void
reconstruction(float* RESULT, int picture_ID, int Sub_Vol_ID, cudaTextureObject_t texObjt,  PictureParameter picParam){

	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;

	/*------------------------座標変換------------------------------------*/
	float x = -X + picParam.nx * 0.5f;
	float y = -Y + picParam.ny * 0.5f;
	float z = - Sub_Vol_ID * Z_PIXEL + picParam.nz * 0.5f;

	float V[PIC_SET_NUM], W[PIC_SET_NUM], U[PIC_SET_NUM], KEISU[PIC_SET_NUM];
	float sinData, cosData, ftemp = 0.0f;
        float theta = 2.0f  * INNER_PI / picParam.np;	//angle change per picture
        float dn = picParam.sod / picParam.pixel_pitch;

	/*------------各投影画像について逆投影する座標を計算-------------------*/
#pragma unroll
	for (int i = 0; i < PIC_SET_NUM; i++){

		ftemp = (picture_ID + i) * theta;
		sinData = sin(ftemp);
		cosData = cos(ftemp);

		ftemp = dn / (dn + x * cosData + y * sinData);
		W[i] = picParam.c1 * ftemp * ftemp;

		U[i] = ftemp * picParam.u_keisu * (-x * sinData + y * cosData) +  float(picParam.nu /2);
		KEISU[i] = ftemp * picParam.v_keisu;
		V[i] = KEISU[i] * z + picParam.nv/2;
	}

	int output_voxel_index = X + Y * picParam.nx;
#pragma unroll
	for (int i = 0; i < Z_PIXEL; i++){
		ftemp = 0.0f;

		for (int j = 0; j < PIC_SET_NUM; j++){
			ftemp += W[j] * tex2DLayered<float>(texObjt, U[j], V[j], j);
			V[j] -= KEISU[j];// next z layer v 
		}
		atomicAdd(&RESULT[output_voxel_index], ftemp);
		//RESULT[output_voxel_index] += ftemp;

		/* --------------next Z layer ------------------------*/
		output_voxel_index += picParam.nx * picParam.ny;
	}

}
