#include "header.h"

#define FILTER_PARAM (2.0f / (INNER_PI * INNER_PI ))

__global__ void	convolution(float* PICTURE, float* RESULT, PictureParameter picParam){

	extern __shared__ float shared_line[];
	int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;
	int Row = by * blockDim.y + ty,			Col = bx * blockDim.x + tx;
	int share_mem_index;
	int pic_height = gridDim.y * blockDim.y;

	double weight;

	float V2, SID2V2, PRE_U;
    int FILTER_RADIUS = picParam.nu /2;
	/* zero padding for left and right */
	shared_line[tx] = 0;
	shared_line[tx + FILTER_RADIUS * 2] = 0;
	__syncthreads();
	V2 = (Row - pic_height / 2) * picParam.dpv;
	V2 = V2 * V2;
	SID2V2 = picParam.sid * picParam.sid + V2;

	int pixel_index = Row * picParam.nu + Col;

	if (bx == 0){//left
		PRE_U = (float)(blockDim.x - tx) * picParam.dpu;
		weight = (double)(picParam.sid / sqrt(SID2V2 + PRE_U * PRE_U));
		shared_line[blockDim.x + tx] = PICTURE[pixel_index] * weight;

		PRE_U = (float) tx * picParam.dpu;
		weight = (double)(picParam.sid / sqrt(SID2V2 + PRE_U * PRE_U));
		shared_line[blockDim.x * 2 + tx] = PICTURE[pixel_index + blockDim.x] * weight;

	}else if (bx == 1){//right
		PRE_U = (float)tx * picParam.dpu;
		weight = (double)(picParam.sid / sqrt(SID2V2 + PRE_U * PRE_U));
		shared_line[blockDim.x + tx] = PICTURE[pixel_index - blockDim.x] * weight;

		PRE_U = (float)(blockDim.x - tx) * picParam.dpu;
		weight = (double)(picParam.sid / sqrt(SID2V2 + PRE_U * PRE_U));
		shared_line[blockDim.x * 2 + tx] = PICTURE[pixel_index] * weight;

	}
	__syncthreads();

	share_mem_index = blockDim.x * (bx + 1) + tx;
	float temp = 0.0f, filter = 0.0f;

	/*becuase every bank will be accessed by the adjecent thread, there is 2-way bank conflict
	if you write:
		temp += (shared_line[share_mem_index - j] + shared_line[share_mem_index + j]* filter);
	*/

#pragma unroll
	for (int j = 1; j < (PROJECTION_SIZE / 2); j++){

		filter = FILTER_PARAM / (1.0f - 4.0f * j * j);
		temp += shared_line[share_mem_index - j] * filter;
		temp += shared_line[share_mem_index + j] * filter;

	}

	temp += shared_line[share_mem_index] * FILTER_PARAM;	//own point
	RESULT[pixel_index] = temp;
}
