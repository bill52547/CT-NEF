#include "universal.h"
#include "cu_add.h"
#include "cu_deform.h"
#include "cu_backprojection.h"
#include "forwardTV.h"
__host__ void host_multiply(float *a, float *b, int nx, int ny);
__host__ void host_grad(float *h_out_gx, float *h_out_gy, float *h_img, float *h_gx, float *h_gy, float *h_proj, int nx, int ny, int na, int n_views, int n_iter_inner, float da, float ai, float SO, float SD, float mu, float *volumes, float *flows, float *angles, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define IN_GX prhs[1]
#define IN_GY prhs[2]
#define PROJ prhs[3]
#define GEO_PARA prhs[4]
#define ITER_PARA prhs[5]
#define OUT_GX plhs[0]
#define OUT_GY plhs[1]

// #define OUT_ERR plhs[1]

int nx, ny, na, n_iter_inner, n_views;
float da, ai, SO, SD, mu;
float *volumes, *flows, *angles;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");


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
    ai -= ((float)na / 2 - 0.5f);
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - ((float)na / 2 - 0.5f);
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


if (mxGetField(ITER_PARA, 0, "n_iter_inner") != NULL)
    n_iter_inner = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter_inner")); // number of views in this bin
else{
    n_iter_inner = 1;
    mexPrintf("Automatically set number of iterations to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter_inner manually.\n");
}

if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_views"));
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number bins, which is denoted as iter_para.n_views.\n");
}

if (mxGetField(ITER_PARA, 0, "mu") != NULL)
    mu = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "mu"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.mu.\n");

if (mxGetField(ITER_PARA, 0, "volumes") != NULL)
    volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume in iter_para.volumes.\n");  

if (mxGetField(ITER_PARA, 0, "flows") != NULL)
    flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.flows.\n");    

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");

// load initial guess of image
float *h_img;

h_img = (float*)mxGetData(IN_IMG);

float *h_gx, *h_gy;

h_gx = (float*)mxGetData(IN_GX);
h_gy = (float*)mxGetData(IN_GY);

// load true projection value
float *h_proj;
h_proj = (float*)mxGetData(PROJ);

// setup output images
OUT_GX = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
OUT_GY = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[2] = {(mwSize)nx, (mwSize)ny};
mxSetDimensions(OUT_GX, outDim, 2);
mxSetData(OUT_GX, mxMalloc(nx * ny * sizeof(float)));
mxSetDimensions(OUT_GY, outDim, 2);
mxSetData(OUT_GY, mxMalloc(nx * ny * sizeof(float)));
float *h_out_gx = (float*)mxGetData(OUT_GX);
float *h_out_gy = (float*)mxGetData(OUT_GY);

float *alpha_x, *alpha_y, *beta_x, *beta_y;
alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
host_grad(h_out_gx, h_out_gy, h_img, h_gx, h_gy, h_proj, nx, ny, na, n_views, n_iter_inner, da, ai, SO, SD, mu, volumes, flows, angles, alpha_x, alpha_y, beta_x, beta_y);
    
cudaDeviceReset();
return;
}

__host__ void host_grad(float *h_out_gx, float *h_out_gy, float *h_img, float *h_gx, float *h_gy, float *h_proj, int nx, int ny, int na, int n_views, int n_iter_inner, float da, float ai, float SO, float SD, float mu, float *volumes, float *flows, float *angles, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    int numBytesImg = nx * ny * sizeof(float);
    int numBytesProj = na * sizeof(float);
    float *d_img, *d_img1, *d_bp;
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_img1, numBytesImg);
    cudaMalloc((void**)&d_bp, numBytesImg);

    float *d_gx, *d_gy;
    cudaMalloc((void**)&d_gx, numBytesImg);
    cudaMalloc((void**)&d_gy, numBytesImg);
    
    cudaMemcpy(d_gx, h_gx, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, h_gy, numBytesImg, cudaMemcpyHostToDevice);

    float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
    cudaMalloc((void**)&d_alpha_x, numBytesImg);
    cudaMalloc((void**)&d_alpha_y, numBytesImg);
    cudaMalloc((void**)&d_beta_x, numBytesImg);
    cudaMalloc((void**)&d_beta_y, numBytesImg);
    cudaMemcpy(d_alpha_x, alpha_x, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha_y, alpha_y, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_x, beta_x, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_y, beta_y, numBytesImg, cudaMemcpyHostToDevice);

    float *d_wxy;
    cudaMalloc((void**)&d_wxy, numBytesImg * 2);

    float *d_proj;
    cudaMalloc((void**)&d_proj, numBytesProj);

    for (int i_view = 0; i_view < n_views; i_view ++)
    {
        cudaMemcpy(d_proj, h_proj + i_view * na, numBytesProj, cudaMemcpyHostToDevice);

        host_backprojection(d_bp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

        // host_deform(d_img1, d_img, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);

        // host_wx(d_wxy, d_img1, nx, ny);
        host_wx(d_wxy, d_img, nx, ny);


        host_multiply(d_bp, d_wxy, nx, ny);
        host_add(d_gx, d_bp, nx, ny, -volumes[i_view]);

        host_backprojection(d_bp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

        host_multiply(d_bp, d_wxy + nx * ny, nx, ny);
        host_add(d_gy, d_bp, nx, ny, -volumes[i_view]);

    }
    cudaMemcpy(h_out_gx, d_gx, numBytesImg, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_gy, d_gy, numBytesImg, cudaMemcpyDeviceToHost);
    
    cudaFree(d_img);
    cudaFree(d_img1);
    cudaFree(d_bp);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_alpha_x);
    cudaFree(d_alpha_y);
    cudaFree(d_beta_x);
    cudaFree(d_beta_y);
    cudaFree(d_wxy);
    cudaFree(d_proj);
}

__global__ void kernel_multiply(float* a, float *b, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    a[id] *= b[id];
}
__host__ void host_multiply(float *a, float *b, int nx, int ny)
{

    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    kernel_multiply<<<gridSize, blockSize>>>(a, b, nx, ny);
    
}