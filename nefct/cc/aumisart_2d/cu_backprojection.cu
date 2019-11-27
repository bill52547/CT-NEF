#include "cu_backprojection.h"

__host__ void host2_backprojection(float *d_img, float *d_proj, float *float_para, int *int_para)
{

}

__host__ void host_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, int nx, int ny)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent = make_cudaExtent(na, 1, 1);
    cudaArray *array_proj;
    cudaMalloc3DArray(&array_proj, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    cudaPitchedPtr dp_proj = make_cudaPitchedPtr((void*) d_proj, na * sizeof(float), na, 1);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_proj;
    copyParams.dstArray = array_proj;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    resDesc.res.array.array = array_proj;
	cudaTextureObject_t tex_proj = 0;
    // cudaTextureObject_t tex_proj = host_create_texture_object(d_proj, nb, na, 1);
    cudaCreateTextureObject(&tex_proj, &resDesc, &texDesc, NULL);

    const dim3 gridSize_img((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
	kernel_backprojection<<<gridSize_img, blockSize>>>(d_img, tex_proj, angle, SO, SD, da, na, ai, nx, ny);
    cudaDeviceSynchronize();

    cudaFreeArray(array_proj);
    cudaDestroyTextureObject(tex_proj);
}


__global__ void kernel_backprojection(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;

    int id = ix + iy * nx;

    img[id] = 0.0f;
	float sphi = __sinf(angle);
	float cphi = __cosf(angle);
	// float dd_voxel[3];
	float xc, yc;
	xc = (float)ix - nx / 2 + 0.5f;
	yc = (float)iy - ny / 2 + 0.5f;

	// voxel boundary coordinates
	float xll, yll, xlr, ylr, xrl, yrl, xrr, yrr;

	xll = +xc * cphi + yc * sphi - 0.5f;
    yll = -xc * sphi + yc * cphi - 0.5f;
    xrr = +xc * cphi + yc * sphi + 0.5f;
    yrr = -xc * sphi + yc * cphi + 0.5f;
    xrl = +xc * cphi + yc * sphi + 0.5f;
    yrl = -xc * sphi + yc * cphi - 0.5f;
    xlr = +xc * cphi + yc * sphi - 0.5f;
    ylr = -xc * sphi + yc * cphi + 0.5f;
    
	// the coordinates of source and detector plane here are after rotation
	float ratio, all, alr, arl, arr, a_max, a_min;
	// calculate a value for each boundary coordinates
	

	// the a and b here are all absolute positions from isocenter, which are on detector planes
	ratio = SD / (xll + SO);
	all = ratio * yll;
	ratio = SD / (xrr + SO);
	arr = ratio * yrr;
	ratio = SD / (xlr + SO);
	alr = ratio * ylr;
	ratio = SD / (xrl + SO);
	arl = ratio * yrl;

	a_max = MAX4(all ,arr, alr, arl);
	a_min = MIN4(all ,arr, alr, arl);

	// the related positions on detector plane from start points
	a_max = a_max / da - ai + 0.5f; //  now they are the detector coordinates
	a_min = a_min / da - ai + 0.5f;

	int a_ind_max = (int)floorf(a_max); 	
	int a_ind_min = (int)floorf(a_min); 

	float bin_bound_1, bin_bound_2, wa;
	for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia ++){
		bin_bound_1 = ia + 0.0f;
		bin_bound_2 = ia + 1.0f;
		
		wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);// wa /= a_max - a_min;


		img[id] += wa * tex3D<float>(tex_proj, (ia + 0.5f), 0.5f, 0.5f);
	}
}
