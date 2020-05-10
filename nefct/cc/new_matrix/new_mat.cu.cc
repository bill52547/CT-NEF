#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

// texture<int, 2, cudaReadModeElementType> tex;

__global__ void new_mat_kernel(int nx, int ny, float *dest)
{
    int ix = blockIdx.x * GRIDDIM_X + threadIdx.x;
    int iy = blockIdx.y * GRIDDIM_Y + threadIdx.y;

    if (ix >= nx || iy >= ny)
    {
        return;
    }
    dest[ix + iy * nx] = 1;
}

void new_mat_gpu(const int *grid, float *dest)
{
    int grid_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1]; //number of meshes

    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    new_mat_kernel<<<gridSize, blockSize>>>(nx, ny, dest);
    cudaDeviceSynchronize();
}
#endif
