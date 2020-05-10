#include "mex.h"
__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_deform_invert(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_invert(float *mx2, float *my2, float *mx, float *my, int nx, int ny);
__global__ void kernel_invert(float *mx2, float *my2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, int nx, int ny);
__global__ void kernel_forwardDVF(float *mx, float *my, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y, float volume, float flow, int nx, int ny);
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny);

__host__ void host_backprojection(float *d_img, float *d_proj, float angle,float SO, float SD, float da, int na, float ai, int nx, int ny);
__global__ void kernel_backprojection(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny);
__host__ void host_initial(float *img, int nx, int ny, float value);
__global__ void kernel_initial(float *img, int nx, int ny, float value);
__host__ void host_add(float *img1, float *img, int nx, int ny, float weight);

__global__ void kernel_add(float *img1, float *img, int nx, int ny, float weight);
#define MAX(a,b) (((a) > (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

#define MIN(a,b) (((a) < (b)) ? a : b)
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#define ABS(x) ((x) > 0 ? x : -(x))
#define PI 3.141592653589793f

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_PROJ prhs[0]
#define GEO_PARA prhs[1]
#define ITER_PARA prhs[2]
#define OUT_IMG plhs[0]


int nx, ny, na, n_views, numImg, numBytesImg, numSingleProj, numBytesSingleProj, numProj;
float da, ai, SO, SD, *angles;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

numImg = nx * ny; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

// detector plane resolutions
if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_views"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of views, defined as iter_para.n_views.\n");


numSingleProj = na;
numBytesSingleProj = numSingleProj * sizeof(float);
numProj = na * n_views;

// detector resolution
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
else{
    da = 1.0f;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}

// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    ai -= (float)na / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - (float)na / 2 + 0.5f;
}

if (mxGetField(GEO_PARA, 0, "SO") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");

if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid projection angles, as iter_para.angles.\n");

float *h_alpha_x, *h_alpha_y, *h_beta_x, *h_beta_y;
h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
h_alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
h_beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
h_beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
cudaMalloc((void**)&d_alpha_x, numBytesImg);
cudaMalloc((void**)&d_alpha_y, numBytesImg);

cudaMalloc((void**)&d_beta_x, numBytesImg);
cudaMalloc((void**)&d_beta_y, numBytesImg);

// host_initial(d_img, nx, ny, 0.0f);
cudaMemcpy(d_alpha_x, h_alpha_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_alpha_y, h_alpha_y, numBytesImg, cudaMemcpyHostToDevice);

cudaMemcpy(d_beta_x, h_beta_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_beta_y, h_beta_y, numBytesImg, cudaMemcpyHostToDevice);

float *volumes, *flows;
volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));

float *d_img, *d_img1, *d_img2, *d_proj;
cudaMalloc((void**)&d_img, numBytesImg);
cudaMalloc((void**)&d_img1, numBytesImg);
cudaMalloc((void**)&d_img2, numBytesImg);
cudaMalloc((void**)&d_proj, numBytesSingleProj);

float *h_proj;
h_proj = (float*)mxGetData(IN_PROJ);

host_initial(d_img, nx, ny, 0.0f);

for (int i_view = 0; i_view < n_views; i_view ++)
{
    cudaMemcpy(d_proj, h_proj + numSingleProj * i_view, numBytesSingleProj, cudaMemcpyHostToDevice);
    host_backprojection(d_img2, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);
    host_deform_invert(d_img1, d_img2, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
    host_add(d_img, d_img1, nx, ny, 1.0);
}

OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[2] = {(mwSize)nx, (mwSize)ny};

mxSetDimensions(OUT_IMG, outDim, 2);
mxSetData(OUT_IMG, mxMalloc(numBytesImg));
float *h_outimg = (float*)mxGetData(OUT_IMG);

cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);

cudaFree(d_proj);
cudaFree(d_img);
cudaFree(d_img1);
cudaFree(d_img2);
cudaDeviceReset();
return;
}

__host__ void host_initial(float *img, int nx, int ny, float value){
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_initial<<<gridSize, blockSize>>>(img, nx, ny, value);
    cudaDeviceSynchronize();

}

__global__ void kernel_initial(float *img, int nx, int ny, float value){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    img[ix + iy * nx] = value;
}


__host__ void host_deform_invert(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    float *mx, *my;
    cudaMalloc((void**)&mx, nx * ny * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, alpha_x, alpha_y, beta_x, beta_y, volume, flow, nx, ny);
    cudaDeviceSynchronize();

    float *mx2, *my2;
    cudaMalloc((void**)&mx2, nx * ny * sizeof(float));
    cudaMalloc((void**)&my2, nx * ny * sizeof(float));
    
    host_invert(mx2, my2, mx, my, nx, ny);
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, 1);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    kernel_deformation<<<gridSize, blockSize>>>(d_img1, tex_img, mx2, my2, nx, ny);
    cudaDeviceSynchronize();
    cudaFree(mx);   
    cudaFree(my);   
    
    cudaFree(mx2);   
    cudaFree(my2);   
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);

}

__host__ void host_invert(float *mx2, float *my2, float *mx, float *my, int nx, int ny)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) mx, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) my, nx * sizeof(float), nx, ny);

    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent = make_cudaExtent(nx, ny, 1);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    copyParams.srcPtr = dp_mx;
    cudaArray *array_mx;
    cudaMalloc3DArray(&array_mx, &channelDesc, extent);
    copyParams.dstArray = array_mx;
    cudaMemcpy3D(&copyParams);   

    copyParams.srcPtr = dp_my;
    cudaArray *array_my;
    cudaMalloc3DArray(&array_my, &channelDesc, extent);
    copyParams.dstArray = array_my;
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

    resDesc.res.array.array = array_mx;
    cudaTextureObject_t tex_mx = 0;
    cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

    resDesc.res.array.array = array_my;
    cudaTextureObject_t tex_my = 0;
    cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

    kernel_invert<<<gridSize, blockSize>>>(mx2, my2, tex_mx, tex_my, nx, ny);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
}
__global__ void kernel_invert(float *mx2, float *my2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float x = 0, y = 0;
    for (int iter = 0; iter < 10; iter ++){
        x = - tex3D<float>(tex_mx, (x + ix + 0.5f), (y + iy + 0.5f), 0.5f);
        y = - tex3D<float>(tex_my, (x + ix + 0.5f), (y + iy + 0.5f), 0.5f);
    }
    mx2[id] = x;
    my2[id] = y;
}


__global__ void kernel_forwardDVF(float *mx, float *my, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y, float volume, float flow, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;    
    mx[id] = alpha_x[id] * volume + beta_x[id] * flow;
    my[id] = alpha_y[id] * volume + beta_y[id] * flow;
}


__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float xi = ix + mx[id];
    float yi = iy + my[id];
    
    img1[id] = tex3D<float>(tex_img, xi + 0.5f, yi + 0.5f, 0.5f);
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

__host__ void host_add(float *img1, float *img, int nx, int ny, float weight){
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_add<<<gridSize, blockSize>>>(img1, img, nx, ny, weight);
    cudaDeviceSynchronize();
}

__global__ void kernel_add(float *img1, float *img, int nx, int ny, float weight){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    img1[id] += img[id] * weight;
}