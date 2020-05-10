#include "cu_add.h"
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

__host__ void host_add2(float *img1, float *img, int nx, int ny, float* img0, float weight, int ind){
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_add2<<<gridSize, blockSize>>>(img1, img, nx, ny, img0, weight, ind);
    cudaDeviceSynchronize();
    
}

__global__ void kernel_add2(float *img1, float *img, int nx, int ny, float *img0, float weight, int ind){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float df;
    switch (ind)
    {
        case 1:
            if (ix == nx - 1)
                df = 0.0f;
            else
                df = img0[id + 1] - img0[id];
            break;
        case 2:
            if (iy == ny - 1)
                df = 0.0f;
            else
                df = img0[id + nx] - img0[id];    
            break;
    }
    img1[id] += img[iy + ix * ny] * weight * df;
}