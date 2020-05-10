#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

__device__ void backproject_flat_device(const float *pv_values, const float *angles,
                                        const int nv, const float SID, const float SAD,
                                        const int na, const float da, const float ai,
                                        const int nb, const float db, const float bi,
                                        const float xc, const float yc, const float zc,
                                        float &image)
{
    const float xs = -SAD;
    const float ys = 0.0f;
    const float zs = 0.0f;

    for (int iv = 0; iv < nv; iv++)
    {
        const float cphi = cos(angles[iv]);
        const float sphi = sin(angles[iv]);
        const float xc0 = xc * cphi + yc * sphi;
        const float yc0 = -xc * sphi + yc * cphi;
        const float zc0 = zc;

        float xd = SID - SAD;
        float yd = SID / (xc0 - xs) * (yc0 - ys) + ys;
        float zd = SID / (xc0 - xs) * (zc0 - zs) + zs;
        float a = yd / da + na / 2 - ai - 0.5;
        float b = zd / db + nb / 2 - bi - 0.5;
        int ia = (int)floor(a);
        int ia2 = ia + 1;
        float wa2 = a - ia;
        float wa = 1 - wa2;
        int ib = (int)floor(b);
        int ib2 = ib + 1;
        float wb2 = b - ib;
        float wb = 1 - wb2;
        if (ia < 0 || ia2 >= na)
        {
            continue;
        }
        if (ib < 0 || ib2 >= nb)
        {
            continue;
        }
        image += pv_values[ia + ib * na + iv * na * nb] * wa * wb +
        pv_values[ia + ib2 * na + iv * na * nb] * wa * wb2 +
        pv_values[ia2 + ib * na + iv * na * nb] * wa2 * wb +
        pv_values[ia2 + ib2 * na + iv * na * nb] * wa2 * wb2;
    }
}

__device__ void backproject_cyli_device(const float *pv_values, const float *angles,
                                        const int nv, const float SID, const float SAD,
                                        const int na, const float da, const float ai,
                                        const int nb, const float db, const float bi,
                                        const float xc, const float yc, const float zc,
                                        float &image)
{
    const float xs = -SAD;
    const float ys = 0.0f;
    const float zs = 0.0f;

    for (int iv = 0; iv < nv; iv++)
    {
        const float cphi = cos(angles[iv]);
        const float sphi = sin(angles[iv]);
        const float xc0 = xc * cphi + yc * sphi;
        const float yc0 = -xc * sphi + yc * cphi;
        const float zc0 = zc;
        float angle = atan2(yc0, xc0 + SID);
        float a = angle / da + na / 2 - ai - 0.5;
        float zd = SID / (xc0 - xs) * (zc0 - zs) + zs;
        float b = zd / db + nb / 2 - bi - 0.5;
        int ia = (int)floor(a);
        int ia2 = ia + 1;
        float wa2 = a - ia;
        float wa = 1 - wa2;
        int ib = (int)floor(b);
        int ib2 = ib + 1;
        float wb2 = b - ib;
        float wb = 1 - wb2;
        if (ia < 0 || ia2 >= na)
        {
            continue;
        }
        if (ib < 0 || ib2 >= nb)
        {
            continue;
        }
        image += pv_values[ia + ib * na + iv * na * nb] * wa * wb + pv_values[ia + ib2 * na + iv * na * nb] * wa * wb2 + pv_values[ia2 + ib * na + iv * na * nb] * wa2 * wb + pv_values[ia2 + ib2 * na + iv * na * nb] * wa2 * wb2;
    }
}

__global__ void
BackProjectFlatKernel(const float *pv, const float *angles,
                      const int nx, const int ny, const int nz,
                      const float cx, const float cy, const float cz,
                      const int na, const int nb, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      const float db, const float bi,
                      float *image)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;int shape_cpu[3];
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
    {
        return;
    }
    const float xc = (ix + 0.5) + cx;
    const float yc = (iy + 0.5) + cy;
    const float zc = (iz + 0.5) + cz;
    const int id = ix + iy * nx + iz * nx * ny;
    backproject_flat_device(pv, angles, nv, SID, SAD, na, da, ai, nb, db, bi, xc, yc, zc, image[id]);
}

__global__ void
BackProjectCyliKernel(const float *pv, const float *angles,
                      const int nx, const int ny, const int nz,
                      const float cx, const float cy, const float cz,
                      const int na, const int nb, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      const float db, const float bi,
                      float *image)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
    {
        return;
    }
    const float xc = (ix + 0.5) + cx;
    const float yc = (iy + 0.5) + cy;
    const float zc = (iz + 0.5) + cz;
    const int id = ix + iy * nx + iz * nx * ny;
    backproject_cyli_device(pv, angles, nv, SID, SAD, na, da, ai, nb, db, bi, xc, yc, zc, image[id]);
}

void backproject_flat(const float *pv_values, const int *grid, const float *center,
                      const float *angles,
                      const int na, const int nb, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      const float db, const float bi,
                      float *image)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1], nz = grid_cpu[2];         //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center

    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    BackProjectFlatKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   nx, ny, nz,
                                                   cx, cy, cz,
                                                   na, nb, nv,
                                                   SID, SAD, da, ai, db, bi,
                                                   image);
}

void backproject_cyli(const float *pv_values, const int *grid, const float *center,
                      const float *angles,
                      const int na, const int nb, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      const float db, const float bi,
                      float *image)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1], nz = grid_cpu[2];         //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center

    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    BackProjectCyliKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   nx, ny, nz,
                                                   cx, cy, cz,
                                                   na, nb, nv,
                                                   SID, SAD, da, ai, db, bi,
                                                   image);
}

#endif
