#include "invertTV.h"

__host__ void host_wtx(float* d_x, float *d_wx, int nx, int ny)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    kernel_wtx<<<gridSize, blockSize>>>(d_x, d_wx, nx, ny);
    cudaDeviceSynchronize();
}

__global__ void kernel_wtx(float *x, float *wx, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    x[id] = 0.0f;
    if (ix == 0)
    {
        x[id] += wx[id];
    }
    else if (ix == nx - 1)
    {
        x[id] -= wx[id - 2];
    }
    else
    {
        x[id] += (-wx[id - 1] + wx[id]);
    }
        
    if (iy == 0)
    {
        x[id] += wx[id];
    }
    else if (iy == ny - 1)
    {
        x[id] -= wx[id - nx * 2];
    }
    else
    {
        x[id] += (-wx[id - nx] + wx[id]);
    }                
}
