#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cusparse.h>
#include <iostream>
using namespace std;
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 4
#define EPS 0.0001

void project_gpu(const float *img,
                 const float *data, const int *indices, const int *indptr,
                 const float *angles, const float *distances,
                 const int na, const int nb, const int nv,
                 const int nx, const int ny, const int nz,
                 float *projection);

void back_project_gpu(const float *proj,
                      const float *data, const int *indices, const int *indptr,
                      const float *angles, const float *distances,
                      const int na, const int nb, const int nv,
                      const int nx, const int ny, const int nz,
                      float *image);

__global__ void divide_kernel(float *image1, const float *image, const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }
    if (image[tid] < EPS && image[tid] > -EPS)
    {
        image1[tid] = 0.0f;
    }
    else
    {
        image1[tid] /= image[tid];
    }
}

__global__ void multiply_kernel(float *image1, const float *image, const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    image1[tid] *= image[tid];
}

__global__ void add_inline_kernel(float *image1, const float *image, const int N, const float beta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    image1[tid] += image[tid] * beta;
}

__global__ void initial_kernel(float *image, const int N, const float value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    image[tid] = value;
}
void generate_emap_gpu(const float *data, const int *indices, const int *indptr,
                       const float *data_back, const int *indices_back, const int *indptr_back,
                       const float *angles, const float *distances,
                       const int na, const int nb, const int nv,
                       const int nx, const int ny, const int nz,
                       float *emap)
{
    float *proj, *img1;
    cudaMalloc((void **)&proj, na * nb * nv * sizeof(float));
    cudaMalloc((void **)&img1, nx * ny * nz * sizeof(float));
    initial_kernel<<<nx * ny, nz>>>(img1, nx * ny * nz, 1.0);
    cudaDeviceSynchronize();
    project_gpu(img1, data, indices, indptr, angles, distances, na, nb, nv, nx, ny, nz, proj);
    cudaDeviceSynchronize();
    back_project_gpu(proj, data_back, indices_back, indptr_back, angles, distances, na, nb, nv, nx, ny, nz, emap);
    cudaDeviceSynchronize();
    cudaFree(proj);
    cudaFree(img1);
}

void sart_gpu(const float *img, const float *proj, const float *emap,
              const float *data, const int *indices, const int *indptr,
              const float *data_back, const int *indices_back, const int *indptr_back,
              const float *angles, const float *distances,
              const int na, const int nb, const int nv,
              const int nx, const int ny, const int nz,
              const int n_iter, const float lamb,
              float *img1)
{
    float *proj1, *bproj;
    cudaMalloc((void **)&proj1, na * nb * nv * sizeof(float));
    cudaMalloc((void **)&bproj, nx * ny * nz * sizeof(float));
    add_inline_kernel<<<ny * nz, nx>>>(img1, img, nx * ny * nz, 1);

    for (int iter = 0; iter < n_iter; iter++)
    {
        project_gpu(img1, data, indices, indptr, angles, distances, na, nb, nv, nx, ny, nz, proj1);
        add_inline_kernel<<<nb * nv, na>>>(proj1, proj, na * nb * nv, -1);
        back_project_gpu(proj1, data_back, indices_back, indptr_back, angles, distances, na, nb, nv, nx, ny, nz, bproj);
        divide_kernel<<<ny * nz, nx>>>(bproj, emap, nx * ny * nz);
        add_inline_kernel<<<ny * nz, nx>>>(img1, bproj, nx * ny * nz, -(2 - 1 / lamb));
    }
    cudaFree(proj1);
    cudaFree(bproj);
}
#endif
