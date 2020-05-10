#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) ((x) > 0 ? (x) : -(x))
// Set thread block size
#define GRIDDIM_X 16
#define GRIDDIM_Y 16 
#define GRIDDIM_Z 4

// __global__ void kernel_proj(float *y, float *x, float *phi, float SD, float SO, float scale, float dz, int nx, int ny, int nz, 
//         int nv, int na, float ai, float da, int nb, float bi, float db);

// __global__ void kernel_tex_proj(float *y, cudaTextureObject_t x, float *phi, float SD, float SO, float scale, float dz, int nx, int ny, int nz, 
//         int nv, int na, float ai, float da, int nb, float bi, float db);

__global__ void kernel_tex_proj(float *y, cudaTextureObject_t x, float *phi, float SD, float SO, float scale, float dz, int nx, int ny, int nz, 
        int nv, int na, float ai, float da, int nb, float bi, float db)
{
    int ba=blockIdx.x;
	int bb=blockIdx.y;
    int bv=blockIdx.z;

	int ta=threadIdx.x;
	int tb=threadIdx.y;
	int tv=threadIdx.z;
	int ia=ba*GRIDDIM_X+ta;
    int ib=bb*GRIDDIM_Y+tb;
	int iv=bv*GRIDDIM_Z+tv;

    if (ia < na && ib < nb && iv < nv)
    {
        float cphi, sphi,x1, y1, z1, x20, y20, x2, y2, z2, x2n, y2n, z2n, x2m, y2m, z2m, p2x, p2y, p2z, p2xn, p2yn, p2zn, ptmp;
        float talpha, tgamma, calpha, cgamma, ds, dt, temp, dst, det;
        int is, it;
        cphi = cosf(phi[iv]);
        sphi = sinf(phi[iv]);
        x1 = -SO * cphi;
        y1 = -SO * sphi;
        z1 = 0.0f;
        y[ia + ib * na + iv * na * nb] = 0.0f;
        x20 = SD - SO;
        y20 = (ia + ai - 0.5f) * da;
        x2 = x20 * cphi - y20 * sphi;
        y2 = x20 * sphi + y20 * cphi;
        x20 = SD - SO;
        y20 = (ia + ai + 0.5f) * da;
        x2n = x20 * cphi - y20 * sphi;
        y2n = x20 * sphi + y20 * cphi;
        
        z2 = (bi + ib - 0.5f) * db;
        z2n = (bi + ib + 0.5f) * db;
        x2m = (x2 + x2n) / 2;
        y2m = (y2 + y2n) / 2;
        z2m = (z2 + z2n) / 2;
        if (ABS(x1-x2) > ABS(y1-y2))
        {
            for (int ix = 0; ix < nx; ix++)
            {
                temp = (y2 - y1) / (x2 - x1);
                p2y = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
                temp = (y2n - y1) / (x2n - x1);
                p2yn = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
                temp = (z2 - z1) / (x2m - x1);
                p2z = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (x2m - x1);
                p2zn = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                if (p2y > p2yn)
                {ptmp = p2y; p2y = p2yn; p2yn = ptmp;}
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}
                if (p2yn < 0.0f)
                    continue;
                if (p2y >= (float)ny)
                    continue;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                dst = p2yn - p2y;
                det = p2zn - p2z;

                if (p2y < 0.0f)
                    p2y = 0.0f;
                if (p2yn > ny)
                    p2y = float(ny);
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                talpha = (y2m - y1) / (x2m - x1);
                calpha = 1.0f / ((float)sqrt(1 + talpha * talpha));
                tgamma = (z2m - z1) / (x2m - x1);
                cgamma = 1.0f / ((float)sqrt(1 + tgamma * tgamma));
                for (is = (int)floor(p2y); is < (int)ceil(p2yn); is ++)
                {
                    ds = MIN(p2yn, is + 1) - MAX(is, p2y); ds /= dst;
                    if (is < 0 || is >= ny)
                        continue;
                    for (it = (int)floor(p2z); it < (int)ceil(p2zn); it ++)
                    {
                        dt = MIN(p2zn, it + 1) - MAX(it, p2z); dt /= det;
                        if (it < 0 || it >= nz)
                            continue;
                            
                        y[ia + ib * na + iv * na * nb] += (tex3D<float>(x, ix + 0.5f, (is + 0.5f), (it + 0.5f)) * ds * dt / calpha / cgamma);
                    }
                }
            }
        }
        else
        {
            for (int iy = 0; iy < ny; iy++)
            {
                temp = (x2 - x1) / (y2 - y1);
                p2x = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
                temp = (x2n - x1) / (y2n - y1);
                p2xn = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
                temp = (z2 - z1) / (y2m - y1);
                p2z = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (y2m - y1);
                p2zn = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                if (p2x > p2xn)
                {ptmp = p2x; p2x = p2xn; p2xn = ptmp;}
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}
                if (p2xn < 0.0f)
                    continue;
                if (p2x >= (float)nx)
                    continue;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                dst = p2xn - p2x;
                det = p2zn - p2z;
                if (p2x < 0.0f)
                    p2x = 0.0f;
                if (p2xn > nx)
                    p2x = float(nx);
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                talpha = (x2m - x1) / (y2m - y1);
                calpha = 1.0f / ((float)sqrt(1 + talpha * talpha));
                tgamma = (z2m - z1) / (y2m - y1);
                cgamma = 1.0f / ((float)sqrt(1 + tgamma * tgamma));
                for (is = (int)floor(p2x); is < (int)ceil(p2xn); is ++)
                {
                    ds = MIN(p2xn, is + 1) - MAX(is, p2x); ds /= dst;
                    if (is < 0 || is >= nx)
                        continue;
                    for (it = (int)floor(p2z); it < (int)ceil(p2zn); it ++)
                    {
                        dt = MIN(p2zn, it + 1) - MAX(it, p2z); dt /= det;
                        if (it < 0 || it >= nz) 
                            continue;
                        y[ia + ib * na + iv * na * nb] += (tex3D<float>(x, is + 0.5f, (iy + 0.5f), (it + 0.5f)) * ds * dt / calpha / cgamma);
                    }
                }
            }
        }
    }
}

__global__ void kernel_proj(float *y, float *x, float *phi, float SD, float SO, float scale, float dz, int nx, int ny, int nz, 
        int nv, int na, float ai, float da, int nb, float bi, float db)
{
    int ba=blockIdx.x;
	int bb=blockIdx.y;
    int bv=blockIdx.z;

	int ta=threadIdx.x;
	int tb=threadIdx.y;
	int tv=threadIdx.z;
	int ia=ba*GRIDDIM_X+ta;
    int ib=bb*GRIDDIM_Y+tb;
	int iv=bv*GRIDDIM_Z+tv;
    
    if (ia < na && ib < nb && iv < nv)
    {
        float cphi, sphi,x1, y1, z1, x20, y20, x2, y2, z2, x2n, y2n, z2n, x2m, y2m, z2m, p2x, p2y, p2z, p2xn, p2yn, p2zn, ptmp;
        float talpha, tgamma, calpha, cgamma, ds, dt, temp, dst, det;
        int is, it;
        cphi = cosf(phi[iv]);
        sphi = sinf(phi[iv]);
        x1 = -SO * cphi;
        y1 = -SO * sphi;
        z1 = 0.0f;
        y[ia + ib * na + iv * na * nb] = 0.0f;
        x20 = SD - SO;
        y20 = (ia + ai - 0.5f) * da;
        x2 = x20 * cphi - y20 * sphi;
        y2 = x20 * sphi + y20 * cphi;
        x20 = SD - SO;
        y20 = (ia + ai + 0.5f) * da;
        x2n = x20 * cphi - y20 * sphi;
        y2n = x20 * sphi + y20 * cphi;
        
        z2 = (bi + ib - 0.5f) * db;
        z2n = (bi + ib + 0.5f) * db;
        x2m = (x2 + x2n) / 2;
        y2m = (y2 + y2n) / 2;
        z2m = (z2 + z2n) / 2;
        if (ABS(x1-x2) > ABS(y1-y2))
        {
            for (int ix = 0; ix < nx; ix++)
            {
                temp = (y2 - y1) / (x2 - x1);
                p2y = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
                temp = (y2n - y1) / (x2n - x1);
                p2yn = (ix + 0.5f - nx / 2 - x1) * temp + y1 + ny / 2;
                temp = (z2 - z1) / (x2m - x1);
                p2z = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (x2m - x1);
                p2zn = (ix + 0.5f - nx / 2 - x1) * temp + z1 + nz / 2;
                if (p2y > p2yn)
                {ptmp = p2y; p2y = p2yn; p2yn = ptmp;}
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}
                if (p2yn < 0.0f)
                    continue;
                if (p2y >= (float)ny)
                    continue;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                dst = p2yn - p2y;
                det = p2zn - p2z;

                if (p2y < 0.0f)
                    p2y = 0.0f;
                if (p2yn > ny)
                    p2y = float(ny);
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                talpha = (y2m - y1) / (x2m - x1);
                calpha = 1.0f / ((float)sqrt(1 + talpha * talpha));
                tgamma = (z2m - z1) / (x2m - x1);
                cgamma = 1.0f / ((float)sqrt(1 + tgamma * tgamma));
                for (is = (int)floor(p2y); is < (int)ceil(p2yn); is ++)
                {
                    ds = MIN(p2yn, is + 1) - MAX(is, p2y); ds /= dst;
                    if (is < 0 || is >= ny)
                        continue;
                    for (it = (int)floor(p2z); it < (int)ceil(p2zn); it ++)
                    {
                        dt = MIN(p2zn, it + 1) - MAX(it, p2z); dt /= det;
                        if (it < 0 || it >= nz)
                            continue;
                        y[ia + ib * na + iv * na * nb] += (x[ix + is * nx + it * nx * ny] * ds * dt * scale / calpha / cgamma);
                    }
                }
            }
        }
        else
        {
            for (int iy = 0; iy < ny; iy++)
            {
                temp = (x2 - x1) / (y2 - y1);
                p2x = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
                temp = (x2n - x1) / (y2n - y1);
                p2xn = (iy + 0.5f - ny / 2 - y1) * temp + x1 + nx / 2;
                temp = (z2 - z1) / (y2m - y1);
                p2z = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                temp = (z2n - z1) / (y2m - y1);
                p2zn = (iy + 0.5f - ny / 2 - y1) * temp + z1 + nz / 2;
                if (p2x > p2xn)
                {ptmp = p2x; p2x = p2xn; p2xn = ptmp;}
                if (p2z > p2zn)
                {ptmp = p2z; p2z = p2zn; p2zn = ptmp;}
                if (p2xn < 0.0f)
                    continue;
                if (p2x >= (float)nx)
                    continue;
                if (p2zn < 0.0f)
                    continue;
                if (p2z >= (float)nz)
                    continue;
                dst = p2xn - p2x;
                det = p2zn - p2z;
                if (p2x < 0.0f)
                    p2x = 0.0f;
                if (p2xn > nx)
                    p2x = float(nx);
                if (p2z < 0.0f)
                    p2z = 0.0f;
                if (p2zn > nz)
                    p2zn = float(nz);
                talpha = (x2m - x1) / (y2m - y1);
                calpha = 1.0f / ((float)sqrt(1 + talpha * talpha));
                tgamma = (z2m - z1) / (y2m - y1);
                cgamma = 1.0f / ((float)sqrt(1 + tgamma * tgamma));
                for (is = (int)floor(p2x); is < (int)ceil(p2xn); is ++)
                {
                    ds = MIN(p2xn, is + 1) - MAX(is, p2x); ds /= dst;
                    if (is < 0 || is >= nx)
                        continue;
                    for (it = (int)floor(p2z); it < (int)ceil(p2zn); it ++)
                    {
                        dt = MIN(p2zn, it + 1) - MAX(it, p2z); dt /= det;
                        if (it < 0 || it >= nz) 
                            continue;
                        y[ia + ib * na + iv * na * nb] += (x[is + iy * nx + it * nx * ny] * ds * dt * scale / calpha / cgamma);
                    }
                }
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]){


float time1, time3;
cudaEvent_t start_time, end_time, tol_start_time, tol_end_time;

cudaEventCreate(&start_time);
cudaEventCreate(&end_time);
cudaEventCreate(&tol_start_time);
cudaEventCreate(&tol_end_time);
cudaEventRecord (tol_start_time, 0);
cudaEventRecord (start_time, 0);

// Macros for input/output
#define IMG prhs[0] // input 0, image

#define PARA prhs[1] // input 1, parameters

#define OUT plhs[0] // output 0, projection

#define TIME plhs[1] // elapsed time for each section
   
// Check number of arguments 
if (nrhs != 2 || nlhs > 1){
    mexErrMsgIdAndTxt("MATLAB:badInput","Usage: imageOut = trilinterp(imageIn, Xi, Yi, Zi)\n");
}

// Get dimensions of image volume
const mwSize *imgDims = mxGetDimensions(IMG);
mwSize outDims[3]; // TBD

mxClassID classCheck;
mwSize sizeCheck;

// Check that image input is correct class and dimensionality

classCheck = mxGetClassID(IMG);
if (classCheck != mxSINGLE_CLASS){
mexErrMsgIdAndTxt("MATLAB:badInput","imageIn must be of data type single.\n");
}

sizeCheck = mxGetNumberOfDimensions(IMG);
if (sizeCheck != 3){
mexErrMsgIdAndTxt("MATLAB:badInput","imageIn must be 3D matrices.\n");
}

// 

// load parameter from input structure
size_t nx, ny, nz, nu, nv, nt;
float SI, DI, dx, dy, dz, du, dv, ui, vi, scale;
nx = (size_t)mxGetScalar(mxGetField(PARA, 0, "nx")); 
ny = (size_t)mxGetScalar(mxGetField(PARA, 0, "ny"));
nz = (size_t)mxGetScalar(mxGetField(PARA, 0, "nz"));
nu = (size_t)mxGetScalar(mxGetField(PARA, 0, "nu"));
nv = (size_t)mxGetScalar(mxGetField(PARA, 0, "nv"));
outDims[0] = nu; outDims[1] = nv;
nt = (size_t)mxGetScalar(mxGetField(PARA, 0, "nt")); // by default, nt = 1
outDims[2] = nt;
dx = (float)mxGetScalar(mxGetField(PARA, 0, "dx"));
dy = (float)mxGetScalar(mxGetField(PARA, 0, "dy"));
if (dx != dy){
    mexPrintf("Resolution in x-direction should be equal to that in y-direction.\n");
    mexErrMsgIdAndTxt("Projection:cudaFail","Projection failed.\n");
}
scale = dx;
SI = (float)mxGetScalar(mxGetField(PARA, 0, "SI")) / scale; //distance between source and detector
DI = (float)mxGetScalar(mxGetField(PARA, 0, "DI")) / scale; //distance between source and isocenter
dz = (float)mxGetScalar(mxGetField(PARA, 0, "dz")) / scale;
du = (float)mxGetScalar(mxGetField(PARA, 0, "du")) / scale;
dv = (float)mxGetScalar(mxGetField(PARA, 0, "dv")) / scale;
ui = (float)mxGetScalar(mxGetField(PARA, 0, "ui")) - (float)nu / 2 + 0.5f;
vi = (float)mxGetScalar(mxGetField(PARA, 0, "vi")) - (float)nv / 2 + 0.5f;

float *phi;
phi = (float*)mxGetData(mxGetField(PARA, 0, "angles"));

// cuda error checking setup

cudaError_t cudaStat;    

// Set dimensions of input and output images
size_t imgDimX = imgDims[0];
size_t imgDimY = imgDims[1];
size_t imgDimZ = imgDims[2];

size_t outDimU = outDims[0];
size_t outDimV = outDims[1];
size_t outDimT = outDims[2];

size_t numVoxelsImg = mxGetNumberOfElements(IMG);
size_t numVoxelsOut = outDimU * outDimV * outDimT;	

// Calculate memory allocation sizes
size_t numBytesOut = numVoxelsOut * sizeof(float);
size_t numBytesImg = numVoxelsImg * sizeof(float);

// Get pointer to image data
float* h_img = (float*) mxGetData(IMG);

// Allocate GPU memory for phi and output
float *d_phi, *d_out;
cudaEventRecord (end_time, 0);
cudaEventSynchronize (end_time);
cudaEventElapsedTime(&time3, start_time, end_time);
mexPrintf("Elapsed time before malloc and copy is %f:\n", time3);
cudaEventRecord(start_time, 0);
cudaStat = cudaMalloc((void**)&d_phi, nt * sizeof(float));
if (cudaStat != cudaSuccess) {
    mexPrintf("Device memory allocation for angles failed.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Projection failed.\n");
}

cudaStat = cudaMalloc((void**)&d_out, numBytesOut);
if (cudaStat != cudaSuccess) {
    mexPrintf("Device memory allocation for output prjection failed.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Projection failed.\n");
}

// Copy angles
cudaStat = cudaMemcpy(d_phi, phi, nt * sizeof(float), cudaMemcpyHostToDevice);
if (cudaStat != cudaSuccess) {
    mexPrintf("Failed to copy angles to device.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Projection failed.\n");
}
cudaEventRecord (end_time, 0);
cudaEventSynchronize (end_time);
cudaEventElapsedTime(&time3, start_time, end_time);
mexPrintf("Elapsed time for malloc and copy is %f:\n", time3);



// Launch projection kernels for detector bin
const dim3 blockSize(GRIDDIM_X,GRIDDIM_Y, GRIDDIM_Z);
const dim3 gridSize((outDimU + GRIDDIM_X - 1) / blockSize.x,
(outDimV + GRIDDIM_Y - 1) / blockSize.y, (outDimT + GRIDDIM_Z - 1) / blockSize.z);

// Copy image to device --- texture implememtation

cudaArray *d_img;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(nx, ny, nz);
cudaMalloc3DArray(&d_img, &channelDesc, extent);
cudaPitchedPtr hp_img = make_cudaPitchedPtr((void*) h_img, nx * sizeof(float), nx, ny);

cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hp_img;
copyParams.dstArray = d_img;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaMemcpy3D(&copyParams);

// Create texture objects
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = d_img;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeBorder;
texDesc.addressMode[1] = cudaAddressModeBorder;
texDesc.addressMode[2] = cudaAddressModeBorder;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t tex_img = 0;
cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
// cudaEventRecord (end_time, NULL);
// cudaEventElapsedTime(&time1, start_time, end_time);
// cudaEventRecord (start_time, NULL);
cudaEventRecord(start_time, 0);
kernel_tex_proj<<<gridSize,blockSize>>>(d_out, tex_img, d_phi, SI, DI, scale, dz, nx, ny, nz, nt, nu, ui, du, nv, vi, dv);

// cudaStat = cudaPeekAtLastError();
// if (cudaStat != cudaSuccess) {
//     mexPrintf("Projection kernel launch failure.\n");
//     mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
//         mexErrMsgIdAndTxt("MATLAB:cudaFail","Projection failed.\n");
// }

cudaStat = cudaDeviceSynchronize();
if (cudaStat != cudaSuccess) {
    mexPrintf("Error in Projection kernel.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Projecton failed.\n");
}
cudaEventRecord (end_time, 0);
cudaEventSynchronize (end_time);
cudaEventElapsedTime(&time3, start_time, end_time);
mexPrintf("Elapsed time for kernels is %f:\n", time3);
// cudaEventRecord (end_time, NULL);
// cudaEventSynchronize (end_time);
// cudaEventElapsedTime(&time2, start_time, end_time);
// cudaEventRecord (start_time, NULL);
// Allocate host memory for output projection
OUT = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(OUT, outDims, sizeCheck);
mxSetData(OUT, mxMalloc(numBytesOut));
float * h_out = (float*) mxGetData(OUT);

// Copy output projection to host
cudaStat = cudaMemcpy(h_out, d_out, numBytesOut, cudaMemcpyDeviceToHost);
if (cudaStat != cudaSuccess) {
    mexPrintf("Error copying output projection to host.\n");
    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Projection failed.\n");
}

// Free allocated memory
// cudaEventRecord (end_time, NULL);
// cudaEventElapsedTime(&time3, start_time, end_time);
// mexPrintf("Elapsed time before kernel is %f:\n", time1);
// mexPrintf("Elapsed time of kernel is %f:\n", time2);
// mexPrintf("Elapsed time after kernel is %f:\n", time3);
cudaFreeArray(d_img);
cudaDestroyTextureObject(tex_img);
cudaFree(d_phi);
cudaFree(d_out);

// cudaEventRecord (end_time, 0);
// cudaEventSynchronize (end_time);
// cudaEventElapsedTime(&time3, start_time, end_time);
// mexPrintf("Elapsed time before kernel is %f:\n", time3);
// mexPrintf("Elapsed time of kernel is %f:\n", time2);
// mexPrintf("Total elapsed time after kernel is %f ms.\n", time3);
cudaEventDestroy(start_time); cudaEventDestroy(end_time);

cudaEventRecord (tol_end_time, 0);
cudaEventSynchronize (tol_end_time);
cudaEventElapsedTime(&time1, tol_start_time, tol_end_time);
mexPrintf("Elapsed time of total CUDA program is %f:\n", time1);
cudaEventDestroy(tol_start_time); cudaEventDestroy(tol_end_time);

//Reset device for profiling
cudaDeviceReset();
return;
}

