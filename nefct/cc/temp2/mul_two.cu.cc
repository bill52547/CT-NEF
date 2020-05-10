#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x) < 0 ? (-x) : (x))

const int GRIDDIM_X = 32;
const int GRIDDIM_Y = 32;

__global__ void
multi_two_kernel(const float *src, float *dest)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 10 || iy >= 10)
    {
        return;
    }
    dest[ix + iy * 10] = 2 * src[ix + iy * 10];
}

void multi_two(const float *src, float *dest)
{
    const dim3 gridSize((10 + GRIDDIM_X - 1) / GRIDDIM_X, (10 + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    multi_two_kernel<<<gridSize, blockSize>>>(src, dest);
}

#endif
