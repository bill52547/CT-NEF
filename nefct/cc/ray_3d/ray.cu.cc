#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const float eps_ = 0.01;

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

__device__ void project_device(const float x1_, const float y1_, const float z1_,
                               const float x2_, const float y2_, const float z2_,
                               const int nx, const int ny, const int nz,
                               const float cx, const float cy, const float cz,
                               const float sx, const float sy, const float sz,
                               const float *image, float &vproj)
{
    const float dx_ = sx / nx;
    const float dx = 1.0f;
    const float dy = sy / ny / dx_;
    const float dz = sz / nz / dx_;
    const float x1 = (x1_ - cx) / dx_;
    const float x2 = (x2_ - cx) / dx_;
    const float y1 = (y1_ - cy) / dx_;
    const float y2 = (y2_ - cy) / dx_;
    const float z1 = (z1_ - cz) / dx_;
    const float z2 = (z2_ - cz) / dx_;

    const float xd = x2 - x1;
    const float yd = y2 - y1;
    const float zd = z2 - z1;

    const float L = sqrt(xd * xd + yd * yd + zd * zd);
    vproj = 0.0f;

    for (float alpha = 0.0; alpha < L; alpha += eps_)
    {
        float xc = x1 + alpha / L * xd;
        float yc = y1 + alpha / L * yd;
        float zc = z1 + alpha / L * zd;
        const int ix = (int)floor((xc - cx + sx / 2) / dx);
        const int iy = (int)floor((yc - cy + sy / 2) / dy);
        const int iz = (int)floor((zc - cz + sz / 2) / dz);
        if (ix < 0 || ix >= nx)
        {
            continue;
        }
        if (iy < 0 || iy >= ny)
        {
            continue;
        }
        if (iz < 0 || iz >= nz)
        {
            continue;
        }
        vproj += image[ix + iy * nx + iz * nx * ny] / L / L * eps_ * dx_;
    }
}

__global__ void
ProjectFlatKernel(const float *image, const float *angles,
                  const int gx, const int gy, const int gz,
                  const float cx, const float cy, const float cz,
                  const float sx, const float sy, const float sz,
                  const int na, const int nb, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  const float db, const float bi,
                  float *pv)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = blockIdx.y * blockDim.y + threadIdx.y;
    int iv = blockIdx.z * blockDim.z + threadIdx.z;
    if (ia >= na || ib >= nb || iv >= nv)
    {
        return;
    }
    const float a = (ai + ia) * da - na * da / 2;
    const float b = (bi + ib) * db - nb * db / 2;
    const float angle = angles[iv];
    const float xs = -SID * cosf(angle);
    const float ys = -SID * sinf(angle);
    const float zs = 0.0f;
    float xd0, yd0, xd, yd, zd;
    xd0 = SAD - SID;
    yd0 = a;
    zd = b;

    xd = xd0 * cosf(angle) - yd0 * sinf(angle);
    yd = xd0 * sinf(angle) + yd0 * cosf(angle);
    const int id = ia + ib * na + iv * na * nb;
    project_device(xs, ys, zs, xd, yd, zd, gx, gy, gz, cx, cy, cz, sx, sy, sz, image, pv[id]);
}

__global__ void
ProjectCyliKernel(const float *image, const float *angles,
                  const int gx, const int gy, const int gz,
                  const float cx, const float cy, const float cz,
                  const float sx, const float sy, const float sz,
                  const int na, const int nb, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  const float db, const float bi,
                  float *pv)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = blockIdx.y * blockDim.y + threadIdx.y;
    int iv = blockIdx.z * blockDim.z + threadIdx.z;
    if (ia >= na || ib >= nb || iv >= nv)
    {
        return;
    }
    const float a = (ai + ia) * da - na * da / 2;
    const float b = (bi + ib) * db - nb * db / 2;
    const float angle = angles[iv];
    const float xs = -SID * cosf(angle);
    const float ys = -SID * sinf(angle);
    const float zs = 0.0f;
    float xd0, yd0, xd, yd, zd;
    xd0 = SAD * cosf(a) - SID;
    yd0 = SAD * sinf(a);
    zd = b;
    xd = xd0 * cosf(angle) - yd0 * sinf(angle);
    yd = xd0 * sinf(angle) + yd0 * cosf(angle);
    const int id = ia + ib * na + iv * na * nb;
    project_device(xs, ys, zs, xd, yd, zd, gx, gy, gz, cx, cy, cz, sx, sy, sz, image, pv[id]);
}

void project_flat(const float *image, const int *grid, const float *center,
                  const float *size, const float *angles,
                  const int na, const int nb, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  const float db, const float bi,
                  float *pv_values)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];         //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (nv + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    ProjectFlatKernel<<<gridSize, blockSize>>>(image, angles,
                                               gx, gy, gz,
                                               cx, cy, cz,
                                               sx, sy, sz,
                                               na, nb, nv,
                                               SID, SAD, da, ai, db, bi,
                                               pv_values);
}

void project_cyli(const float *image, const int *grid, const float *center,
                  const float *size, const float *angles,
                  const int na, const int nb, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  const float db, const float bi,
                  float *pv_values)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];         //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (nv + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    ProjectCyliKernel<<<gridSize, blockSize>>>(image, angles,
                                               gx, gy, gz,
                                               cx, cy, cz,
                                               sx, sy, sz,
                                               na, nb, nv,
                                               SID, SAD, da, ai, db, bi,
                                               pv_values);
}

#endif
