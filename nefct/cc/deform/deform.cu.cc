#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x) < 0 ? (-x) : (x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

__global__ void
DeformKernel(const float *img,
             const float *mx, const float *my, const float *mz,
             const int nx, const int ny, const int nz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    int xi = (int)floor(ix + mx[id]);
    int xi2 = xi + 1;
    int yi = (int)floor(iy + my[id]);
    int yi2 = yi + 1;
    int zi = (int)floor(iz + mz[id]);
    int zi2 = zi + 1;
    if (xi <= 0 || xi2 >= nx)
    {
        return;
    }
    if (yi <= 0 || yi2 >= ny)
    {
        return;
    }
    if (zi <= 0 || zi2 >= nz)
    {
        return;
    }
    float wx2 = ix + mx[id] - xi;
    float wx = 1.0f - wx2;
    float wy2 = iy + my[id] - yi;
    float wy = 1.0f - wy2;
    float wz2 = iz + mz[id] - zi;
    float wz = 1.0f - wz2;
    img1[id] = img[xi + yi * nx + zi * nx * ny] * wx * wy * wz +
               img[xi + yi * nx + zi2 * nx * ny] * wx * wy * wz2 +
               img[xi + yi2 * nx + zi * nx * ny] * wx * wy2 * wz +
               img[xi + yi2 * nx + zi2 * nx * ny] * wx * wy2 * wz2 +
               img[xi2 + yi * nx + zi * nx * ny] * wx2 * wy * wz +
               img[xi2 + yi * nx + zi2 * nx * ny] * wx2 * wy * wz2 +
               img[xi2 + yi2 * nx + zi * nx * ny] * wx2 * wy2 * wz +
               img[xi2 + yi2 * nx + zi2 * nx * ny] * wx2 * wy2 * wz2;
}

__global__ void
ShiftKernel(const float *img,
            const float mx, const float my, const float mz,
            const int nx, const int ny, const int nz,
            float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    int xi = (int)floor(ix + mx);
    int xi2 = xi + 1;
    int yi = (int)floor(iy + my);
    int yi2 = yi + 1;
    int zi = (int)floor(iz + mz);
    int zi2 = zi + 1;
    if (xi < 0 || xi2 >= nx)
    {
        return;
    }
    if (yi < 0 || yi2 >= ny)
    {
        return;
    }
    if (zi < 0 || zi2 >= nz)
    {
        return;
    }
    float wx2 = ix + mx - xi;
    float wx = 1.0f - wx2;
    float wy2 = iy + my - yi;
    float wy = 1.0f - wy2;
    float wz2 = iz + mz - zi;
    float wz = 1.0f - wz2;
    img1[id] = img[xi + yi * nx + zi * nx * ny] * wx * wy * wz +
               img[xi + yi * nx + zi2 * nx * ny] * wx * wy * wz2 +
               img[xi + yi2 * nx + zi * nx * ny] * wx * wy2 * wz +
               img[xi + yi2 * nx + zi2 * nx * ny] * wx * wy2 * wz2 +
               img[xi2 + yi * nx + zi * nx * ny] * wx2 * wy * wz +
               img[xi2 + yi * nx + zi2 * nx * ny] * wx2 * wy * wz2 +
               img[xi2 + yi2 * nx + zi * nx * ny] * wx2 * wy2 * wz +
               img[xi2 + yi2 * nx + zi2 * nx * ny] * wx2 * wy2 * wz2;
}

__global__ void
ShiftZKernel(const float *img,
             const float mz,
             const int nx, const int ny, const int nz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    int zi = (int)floor(iz + mz);
    int zi2 = zi + 1;
    float wz2 = iz + mz - zi;
    float wz = 1.0f - wz2;
    if (zi2 == 0)
    {
        img1[id] = img[ix + iy * nx + zi2 * nx * ny] * wz2;
    }
    else if (zi == nz - 1)
    {
        img1[id] = img[ix + iy * nx + zi * nx * ny] * wz;
    }
    else if (0 <= zi && zi2 < nz)
    {
        img1[id] = img[ix + iy * nx + zi * nx * ny] * wz +
                   img[ix + iy * nx + zi2 * nx * ny] * wz2;
    }
    else
    {
        img1[id] = 0.0;
    }
}

__global__ void
RotateKernel(const float *img,
             const float cos_phi, const float sin_phi,
             const int nx, const int ny, const int nz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;

    float mx = (ix + 0.5f - nx / 2) * cos_phi - (iy + 0.5f - ny / 2) * sin_phi + nx / 2;
    float my = (ix + 0.5f - nx / 2) * sin_phi + (iy + 0.5f - ny / 2) * cos_phi + ny / 2;
    int xi = (int)floor(mx);
    int yi = (int)floor(my);
    int xi2 = xi + 1;
    int yi2 = yi + 1;
    if (xi <= 0 || xi2 >= nx)
    {
        return;
    }
    if (yi <= 0 || yi2 >= ny)
    {
        return;
    }
    float wx2 = mx - xi;
    float wx = 1.0f - wx2;
    float wy2 = my - yi;
    float wy = 1.0f - wy2;
    img1[id] = img[xi + yi * nx + iz * nx * ny] * wx * wy +
               img[xi + yi2 * nx + iz * nx * ny] * wx * wy2 +
               img[xi2 + yi * nx + iz * nx * ny] * wx2 * wy +
               img[xi2 + yi2 * nx + iz * nx * ny] * wx2 * wy2;
}

void deform(const float *img,
            const float *mx, const float *my, const float *mz,
            const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];
    const dim3 gridSize((gx + GRIDDIM_X - 1) / GRIDDIM_X, (gy + GRIDDIM_Y - 1) / GRIDDIM_Y, (gz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    DeformKernel<<<gridSize, blockSize>>>(img,
                                          mx, my, mz,
                                          gx, gy, gz,
                                          img1);
}

void shift(const float *img,
           const float mx, const float my, const float mz,
           const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];
    const dim3 gridSize((gx + GRIDDIM_X - 1) / GRIDDIM_X, (gy + GRIDDIM_Y - 1) / GRIDDIM_Y, (gz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    ShiftKernel<<<gridSize, blockSize>>>(img,
                                         mx, my, mz,
                                         gx, gy, gz,
                                         img1);
}

void shift_z(const float *img, const float mz,
             const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];
    const dim3 gridSize((gx + GRIDDIM_X - 1) / GRIDDIM_X, (gy + GRIDDIM_Y - 1) / GRIDDIM_Y, (gz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    ShiftZKernel<<<gridSize, blockSize>>>(img, mz,
                                          gx, gy, gz,
                                          img1);
}

void rotate(const float *img,
            const float cos_phi, const float sin_phi,
            const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];
    const dim3 gridSize((gx + GRIDDIM_X - 1) / GRIDDIM_X, (gy + GRIDDIM_Y - 1) / GRIDDIM_Y, (gz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    RotateKernel<<<gridSize, blockSize>>>(img,
                                          cos_phi, sin_phi,
                                          gx, gy, gz,
                                          img1);
}

#endif
