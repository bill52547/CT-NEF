#include "cu_initial.h"
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


__host__ void host_initial2(float *img, int nx, int ny, float *img0, float volume, float flow)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_initial2<<<gridSize, blockSize>>>(img, nx, ny, img0, volume, flow);
    cudaDeviceSynchronize();
}
__global__ void kernel_initial2(float *img, int nx, int ny, float *img0, float volume, float flow)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float dfx, dfy, dfz;
    if (ix == nx - 1)
        dfx = 0.0f;
    else
        dfx = img0[id + 1] - img0[id];
    if (iy == ny - 1)
        dfy = 0.0f;
    else
        dfy = img0[id + nx] - img0[id];
    img[ix + iy * nx] = (dfx + dfy) * (volume + flow);
}