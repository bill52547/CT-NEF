#include "cu_division.h"
__host__ void host_division(float *img1, float *img, int nx, int ny)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_division<<<gridSize, blockSize>>>(img1, img, nx, ny);
    cudaDeviceSynchronize();
}


__global__ void kernel_division(float *img1, float *img, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;

    img1[id] /= img[id];
    if (isnan(img1[id]))
        img1[id] = 0.0f;
    if (isinf(img1[id]))
        img1[id] = 0.0f;
}
