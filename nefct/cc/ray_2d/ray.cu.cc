#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int BLOCKWIDTH = 16;
const int BLOCKHEIGHT = 16;
const int BLOCKDEPTH = 4;
const float eps_ = 0.01;

__device__ void project_device(const float x1_, const float y1_,
                               const float x2_, const float y2_,
                               const int nx, const int ny,
                               const float cx, const float cy,
                               const float sx, const float sy,
                               const float *image, float &vproj)
{
    const float dx_ = sx / nx;
    const float dx = 1.0f;
    const float dy = sy / ny / dx_;
    const float x1 = (x1_ - cx) / dx_;
    const float x2 = (x2_ - cx) / dx_;
    const float y1 = (y1_ - cy) / dx_;
    const float y2 = (y2_ - cy) / dx_;

    const float xd = x2 - x1;
    const float yd = y2 - y1;


    const float L = sqrt(xd * xd + yd * yd);
    vproj = 0.0f;

    for (float alpha = 0.0; alpha < L; alpha += eps_)
    {
        float xc = x1 + alpha / L * xd;
        float yc = y1 + alpha / L * yd;
        const int ix = (int)floor((xc - cx + sx / 2) / dx);
        const int iy = (int)floor((yc - cy + sy / 2) / dy);
        if (ix < 0 || ix >= nx)
        {
            continue;
        }
        if (iy < 0 || iy >= ny)
        {
            continue;
        }

        vproj += image[ix + iy * nx] / L / L * eps_ * dx_;
    }
}

__global__ void
ProjectFlatKernel(const float *image, const float *angles,
                  const int gx, const int gy,
                  const float cx, const float cy,
                  const float sx, const float sy,
                  const int na, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  float *pv)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int iv = blockIdx.y * blockDim.y + threadIdx.y;
    if (ia >= na || iv >= nv)
    {
        return;
    }
    const float a = (ai + ia) * da - na * da / 2;
    const float angle = angles[iv];
    const float xs = -SID * cosf(angle);
    const float ys = -SID * sinf(angle);
    float xd0, yd0, xd, yd;
    xd0 = SAD - SID;
    yd0 = a;

    xd = xd0 * cosf(angle) - yd0 * sinf(angle);
    yd = xd0 * sinf(angle) + yd0 * cosf(angle);
    const int id = ia + iv * na;
    project_device(xs, ys, xd, yd, gx, gy, cx, cy, sx, sy, image, pv[id]);
}

__global__ void
ProjectCyliKernel(const float *image, const float *angles,
                  const int gx, const int gy,
                  const float cx, const float cy,
                  const float sx, const float sy,
                  const int na, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  float *pv)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int iv = blockIdx.y * blockDim.y + threadIdx.y;
    if (ia >= na || iv >= nv)
    {
        return;
    }
    const float a = (ai + ia) * da - na * da / 2;
    const float angle = angles[iv];
    const float xs = -SID * cosf(angle);
    const float ys = -SID * sinf(angle);
    float xd0, yd0, xd, yd;
    xd0 = SAD * cosf(a) - SID;
    yd0 = SAD * sinf(a);

    xd = xd0 * cosf(angle) - yd0 * sinf(angle);
    yd = xd0 * sinf(angle) + yd0 * cosf(angle);
    const int id = ia + iv * na;

    project_device(xs, ys, xd, yd, gx, gy, cx, cy, sx, sy, image, pv[id]);
}

void project_flat(const float *image, const int *grid, const float *center,
                  const float *size, const float *angles,
                  const int na, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  float *pv_values)
{
    int grid_cpu[2];
    float center_cpu[2];
    float size_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nv + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, BLOCKDEPTH);
    ProjectFlatKernel<<<gridSize, blockSize>>>(image, angles,
                                               gx, gy,
                                               cx, cy,
                                               sx, sy,
                                               na, nv,
                                               SID, SAD, da, ai,
                                               pv_values);
}

void project_cyli(const float *image, const int *grid, const float *center,
                  const float *size, const float *angles,
                  const int na, const int nv,
                  const float SID, const float SAD,
                  const float da, const float ai,
                  float *pv_values)
{
    int grid_cpu[2];
    float center_cpu[2];
    float size_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nv + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH, BLOCKHEIGHT, 1);
    ProjectCyliKernel<<<gridSize, blockSize>>>(image, angles,
                                               gx, gy,
                                               cx, cy,
                                               sx, sy,
                                               na, nv,
                                               SID, SAD, da, ai,
                                               pv_values);
}

#endif
