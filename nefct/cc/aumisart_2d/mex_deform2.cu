__global__ void kernel_deformation2(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz);
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PARA prhs[1]
#define MX prhs[2]
#define MY prhs[3]
#define MZ prhs[4]

#define OUT_IMG plhs[0]

float *h_mx, *h_my, *h_mz, *h_img;
h_img = (float*)mxGetData(IN_IMG);
h_mx = (float*)mxGetData(MX);
h_my = (float*)mxGetData(MY);
h_mz = (float*)mxGetData(MZ);
int nx, ny, nz;
nx = (int)mxGetScalar(mxGetField(PARA, 0, "nx"));
ny = (int)mxGetScalar(mxGetField(PARA, 0, "ny"));
nz = (int)mxGetScalar(mxGetField(PARA, 0, "nz"));

float *d_mx, *d_my, *d_mz, *d_img1, *d_img;
cudaMalloc((void**)&d_mx, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_my, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_mz, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_img1, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_img, nx * ny * nz * sizeof(float));

cudaMemcpy(d_mx, h_mx, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_my, h_my, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_mz, h_mz, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_img, h_img, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mwSize outDim[3] = {(mwSize)nx, (mwSize)ny, (mwSize)nz};
mxSetDimensions(OUT_IMG, outDim, 3);
mxSetData(OUT_IMG, mxMalloc(nx * ny * nz * sizeof(float)));
float *h_outimg = (float*)mxGetData(OUT_IMG);

const dim3 gridSize((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, (nz + 4 - 1) / 4);
const dim3 blockSize(16, 16, 4);
kernel_deformation2<<<gridSize, blockSize>>>(d_img1, d_img, d_mx, d_my, d_mz, nx, ny, nz);
cudaDeviceSynchronize();
cudaMemcpy(h_outimg, d_img1, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_mz);
cudaFree(d_img1);
cudaFree(d_img);

cudaDeviceReset();
return;
}


__global__ void kernel_deformation2(float *img1, float *img, float *mx, float *my, float *mz, int nx, int ny, int nz){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = iy + ix * ny + iz * nx * ny;
    int id2 = id;//ix + iy * nx + iz * nx * ny;

    float dx, dy, dz;
    if (ix == nx - 1)
        dx = 0;
    else
        dx = img[id2 + ny] - img[id2];
        
    if (iy == ny - 1)
        dy = 0;
    else
        dy = img[id2 + 1] - img[id2];

    if (iz == nz - 1)
        dz = 0;
    else
        dz = img[id2 + nx * ny] - img[id2];
    img1[id2] = img[id2] + dy * mx[id2] + dx * my[id2] + dz * mz[id2];
}