#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int BLOCKWIDTH = 16;
const int BLOCKHEIGHT = 16;
const int BLOCKDEPTH = 4;

__device__ void backproject_flat_device(const float *pv_values, const float *angles,
                                        const int nv, const float SID, const float SAD,
                                        const int na, const float da, const float ai,
                                        const float xc, const float yc,
                                        float &image)
{
    const float xs = -SID;
    const float ys = 0.0f;
    for (int iv = 0; iv < nv; iv++)
    {
        const float angle = angles[iv];
        const float cphi = cos(angle);
        const float sphi = sin(angle);
        const float xc0 = xc * cphi + yc * sphi;
        const float yc0 = -xc * sphi + yc * cphi;
        float xd = SAD - SID;
        float yd = SAD / (xc0 - xs) * (yc0 - ys) + ys;
        float a = yd / da + na / 2 - ai - 0.5;
        int ia = (int)floor(a);
        int ia2 = ia + 1;
        float wa2 = a - ia;
        float wa = 1 - wa2;
        if (ia < 0 || ia2 >= na)
        {
            continue;
        }
        image += pv_values[ia + iv * na] * wa + pv_values[ia2 + iv * na] * wa2;
    }
}

__device__ void backproject_cyli_device(const float *pv_values, const float *angles,
                                        const int nv, const float SID, const float SAD,
                                        const int na, const float da, const float ai,
                                        const float xc, const float yc,
                                        float &image)
{
    const float xs = -SID;
    const float ys = 0.0f;
    for (int iv = 0; iv < nv; iv++)
    {
        const float cphi = cos(angles[iv]);
        const float sphi = sin(angles[iv]);
        const float xc0 = xc * cphi + yc * sphi;
        const float yc0 = -xc * sphi + yc * cphi;
        float angle = atan2(yc0, xc0 + SID);
        float a = angle / da + na / 2 - ai - 0.5;
        int ia = (int)floor(a);
        int ia2 = ia + 1;
        float wa2 = a - ia;
        float wa = 1 - wa2;
        if (ia < 0 || ia2 >= na)
        {
            continue;
        }
        image += pv_values[ia + iv * na] * wa + pv_values[ia2 + iv * na] * wa2;
    }
}

__global__ void
BackProjectFlatKernel(const float *pv, const float *angles,
                      const int nx, const int ny,
                      const float cx, const float cy,
                      const float sx, const float sy,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
    {
        return;
    }
    float xc = (ix + 0.5) * sx / nx + cx;
    float yc = (iy + 0.5) * sy / ny + cy;
    const int id = ix + iy * nx;
    backproject_flat_device(pv, angles, nv, SID, SAD, na, da, ai, xc, yc, image[id]);
}

__global__ void
BackProjectCyliKernel(const float *pv, const float *angles,
                      const int nx, const int ny,
                      const float cx, const float cy,
                      const float sx, const float sy,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
    {
        return;
    }
    float xc = (ix + 0.5) * sx / nx + cx;
    float yc = (iy + 0.5) * sy / ny + cy;
    const int id = ix + iy * nx;
    backproject_cyli_device(pv, angles, nv, SID, SAD, na, da, ai, xc, yc, image[id]);
}

void backproject_flat(const float *pv_values, const int *grid, const float *center,
                      const float *size, const float *angles,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
{
    int grid_cpu[2];
    float center_cpu[2];
    float size_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, BLOCKDEPTH);
    BackProjectFlatKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   nx, ny,
                                                   cx, cy,
                                                   sx, sy,
                                                   na, nv,
                                                   SID, SAD, da, ai,
                                                   image);
}

void backproject_cyli(const float *pv_values, const int *grid, const float *center,
                      const float *size, const float *angles,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
{
    int grid_cpu[2];
    float center_cpu[2];
    float size_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, 1);
    BackProjectCyliKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   nx, ny,
                                                   cx, cy,
                                                   sx, sy,
                                                   na, nv,
                                                   SID, SAD, da, ai,
                                                   image);
}

#endif
