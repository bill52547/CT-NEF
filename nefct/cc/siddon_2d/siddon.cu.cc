#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

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

    const float nx2 = nx / 2.0f;
    const float ny2 = ny / 2.0f;

    const float L = sqrt(xd * xd + yd * yd);
    vproj = 0.0f;

    if (abs(xd) >= abs(yd))
    {
        float ky = yd / xd;

        for (int ix = 0; ix < nx; ++ix)
        {
            float xx1 = ix - nx2;
            float xx2 = xx1 + 1.0f;
            float yy1, yy2;

            if (ky >= 0.0f)
            {
                yy1 = (y1 + ky * (xx1 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx2 - x1)) / dy + ny2;
            }
            else
            {
                yy1 = (y1 + ky * (xx2 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx1 - x1)) / dy + ny2;
            }
            int cy1 = (int)floor(yy1);
            int cy2 = (int)floor(yy2);

            if (cy1 == cy2)
            {
                if (0 <= cy1 && cy1 < ny)
                {
                    float weight = sqrt(1 + ky * ky) * dx_ / L / L;
                    vproj += image[ix + cy1 * nx] * weight;
                }
            }
            else
            {
                if (-1 <= cy1 && cy1 < ny)
                {

                    float ry = (cy2 - yy1) / (yy2 - yy1);
                    if (cy1 >= 0)
                    {
                        float weight = ry * sqrt(1 + ky * ky) * dx_ / L / L;
                        vproj += image[ix + cy1 * nx] * weight;
                    }
                    if (cy2 < ny)
                    {
                        float weight = (1 - ry) * sqrt(1 + ky * ky) * dx_ / L / L;
                        vproj += image[ix + cy2 * nx] * weight;
                    }
                }
            }
        }
    }
    else
    {
        float kx = xd / yd;

        for (int iy = 0; iy < ny; ++iy)
        {
            float yy1 = iy - ny2;
            float yy2 = yy1 + 1.0f;
            float xx1, xx2;

            if (kx >= 0.0f)
            {
                xx1 = (x1 + kx * (yy1 - y1)) + nx2;
                xx2 = (x1 + kx * (yy2 - y1)) + nx2;
            }
            else
            {
                xx1 = (x1 + kx * (yy2 - y1)) + nx2;
                xx2 = (x1 + kx * (yy1 - y1)) + nx2;
            }
            int cx1 = (int)floor(xx1);
            int cx2 = (int)floor(xx2);

            if (cx1 == cx2)
            {
                if (0 <= cx1 && cx1 < nx)
                {
                    float weight = sqrt(1 + kx * kx) * dx_ / L / L;
                    vproj += image[cx1 + iy * nx] * weight;
                }
            }
            else
            {
                if (-1 <= cx1 && cx1 < nx)
                {
                    float rx = (cx2 - xx1) / (xx2 - xx1);
                    if (cx1 >= 0)
                    {
                        float weight = rx * sqrt(1 + kx * kx) * dx_ / L / L;
                        vproj += image[cx1 + iy * nx] * weight;
                    }

                    if (cx2 < nx)
                    {
                        float weight = (1 - rx) * sqrt(1 + kx * kx) * dx_ / L / L;
                        vproj += image[cx2 + iy * nx] * weight;
                    }
                }
            }
        }
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

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
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

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    ProjectCyliKernel<<<gridSize, blockSize>>>(image, angles,
                                               gx, gy,
                                               cx, cy,
                                               sx, sy,
                                               na, nv,
                                               SID, SAD, da, ai,
                                               pv_values);
}

__device__ void backproject_device(const float x1_, const float y1_,
                                   const float x2_, const float y2_,
                                   const int nx, const int ny,
                                   const float cx, const float cy,
                                   const float sx, const float sy,
                                   const float vproj, float *image)
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

    const float nx2 = nx / 2.0f;
    const float ny2 = ny / 2.0f;

    const float L = sqrt(xd * xd + yd * yd);

    if (abs(xd) >= abs(yd))
    {
        float ky = yd / xd;

        for (int ix = 0; ix < nx; ++ix)
        {
            float xx1 = ix - nx2;
            float xx2 = xx1 + 1.0f;
            float yy1, yy2;

            if (ky >= 0.0f)
            {
                yy1 = (y1 + ky * (xx1 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx2 - x1)) / dy + ny2;
            }
            else
            {
                yy1 = (y1 + ky * (xx2 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx1 - x1)) / dy + ny2;
            }
            int cy1 = (int)floor(yy1);
            int cy2 = (int)floor(yy2);

            if (cy1 == cy2)
            {
                if (0 <= cy1 && cy1 < ny)
                {
                    float weight = sqrt(1 + ky * ky) * dx_;
                    atomicAdd(image + ix + cy1 * nx, vproj * weight);
                }
            }
            else
            {
                if (-1 <= cy1 && cy1 < ny)
                {

                    float ry = (cy2 - yy1) / (yy2 - yy1);
                    if (cy1 >= 0)
                    {
                        float weight = ry * sqrt(1 + ky * ky) * dx_;
                        atomicAdd(image + ix + cy1 * nx, vproj * weight);
                    }
                    if (cy2 < ny)
                    {
                        float weight = (1 - ry) * sqrt(1 + ky * ky) * dx_;
                        atomicAdd(image + ix + cy2 * nx, vproj * weight);
                    }
                }
            }
        }
    }
    else
    {
        float kx = xd / yd;

        for (int iy = 0; iy < ny; ++iy)
        {
            float yy1 = iy - ny2;
            float yy2 = yy1 + 1.0f;
            float xx1, xx2;

            if (kx >= 0.0f)
            {
                xx1 = (x1 + kx * (yy1 - y1)) + nx2;
                xx2 = (x1 + kx * (yy2 - y1)) + nx2;
            }
            else
            {
                xx1 = (x1 + kx * (yy2 - y1)) + nx2;
                xx2 = (x1 + kx * (yy1 - y1)) + nx2;
            }
            int cx1 = (int)floor(xx1);
            int cx2 = (int)floor(xx2);

            if (cx1 == cx2)
            {
                if (0 <= cx1 && cx1 < nx)
                {
                    float weight = sqrt(1 + kx * kx) * dx_;
                    atomicAdd(image + cx1 + iy * nx, vproj * weight);
                }
            }
            else
            {
                if (-1 <= cx1 && cx1 < nx)
                {
                    float rx = (cx2 - xx1) / (xx2 - xx1);
                    if (cx1 >= 0)
                    {
                        float weight = rx * sqrt(1 + kx * kx) * dx_;
                        atomicAdd(image + cx1 + iy * nx, vproj * weight);
                    }

                    if (cx2 < nx)
                    {
                        float weight = (1 - rx) * sqrt(1 + kx * kx) * dx_;
                        atomicAdd(image + cx2 + iy * nx, vproj * weight);
                    }
                }
            }
        }
    }
}

__global__ void
BackProjectFlatKernel(const float *pv, const float *angles,
                      const int gx, const int gy,
                      const float cx, const float cy,
                      const float sx, const float sy,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
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
    backproject_device(xs, ys, xd, yd, gx, gy, cx, cy, sx, sy, pv[id], image);
}

__global__ void
BackProjectCyliKernel(const float *pv, const float *angles,
                      const int gx, const int gy,
                      const float cx, const float cy,
                      const float sx, const float sy,
                      const int na, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai,
                      float *image)
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

    backproject_device(xs, ys, xd, yd, gx, gy, cx, cy, sx, sy, pv[id], image);
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
    int gx = grid_cpu[0], gy = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    BackProjectFlatKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   gx, gy,
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
    int gx = grid_cpu[0], gy = grid_cpu[1];       //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];

    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    BackProjectCyliKernel<<<gridSize, blockSize>>>(pv_values, angles,
                                                   gx, gy,
                                                   cx, cy,
                                                   sx, sy,
                                                   na, nv,
                                                   SID, SAD, da, ai,
                                                   image);
}

#endif
