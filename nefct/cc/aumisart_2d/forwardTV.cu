#include "forwardTV.h"

__host__ void host_wx(float* d_wx, float *d_x, int nx, int ny)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    kernel_wx<<<gridSize, blockSize>>>(d_wx, d_x, nx, ny);
    cudaDeviceSynchronize();
}

__global__ void kernel_wx(float *wx, float *x, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    
    if (ix == nx - 1)
    {
        wx[id] = 0.0f;
    }
    else
    {
        wx[id] = x[id] - x[id + 1];
    }
        
    if (iy == ny - 1)
    {
        wx[id + nx * ny] = 0.0f;
    }
    else
    {
        wx[id + nx * ny] = x[id] - x[id + nx];
    }
        
}
