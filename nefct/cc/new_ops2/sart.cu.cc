#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"

#include <cusparse.h>
#include <iostream>
//using namespace std;
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 4
#define EPS 0.0001

void project_gpu(const float *image,
                 const float *angles, const float *offsets,
                 const float *ai, const float *bi,
                 const int mode,
                 const int nv,
                 const float SD, const float SO,
                 const int nx, const int ny, const int nz,
                 const float da, const int na,
                 const float db, const int nb,
                 float *pv_values);

void back_project_gpu(const float *pv_values, const float *angles, const float *offsets,
                      const float *ai, const float *bi,
                      const int mode,
                      const int nv,
                      const float SD, const float SO,
                      const int nx, const int ny, const int nz,
                      const float da, const int na,
                      const float db, const int nb,
                      float *image);

__global__ void divide_kernel(float *image1, const float *image, const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }
//    if (image[tid] < EPS && image[tid] > -EPS)
//    {
//        image1[tid] = 0.0f;
//    }
//    else
//    {
//        image1[tid] /= image[tid];
//    }
    image1[tid] /= image[tid];
    if (isnan(image1[tid]))
        image1[tid] = 0.0f;
    if (isinf(image1[tid]))
        image1[tid] = 0.0f;
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

__global__ void initial_kernel(float *emap, const int, const float);
void sart_gpu(const float *img, const float *proj, float *emap,
              const float *angles, const float *offsets,
              const float *ai, const float *bi,
              float *proj1, float *bproj,
              const int n_iter, const float lamb,
              const int mode,
              const int nv,
              const float SD, const float SO,
              const int nx, const int ny, const int nz,
              const float da, const int na,
              const float db, const int nb,
              float *img1)
{
    add_inline_kernel<<<ny * nz, nx>>>(img1, img, nx * ny * nz, 1);

    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int iv = 0; iv < nv; iv++)
        {
            project_gpu(img1,
                    angles + iv, offsets + iv,
                    ai + iv, bi + iv,
                    mode,
                    1,
                    SD, SO,
                    nx, ny, nz,
                    da, na, db, nb,
                    proj1);

            add_inline_kernel<<<nb, na>>>(proj1, proj + iv * na * nb, na * nb, -1);
            back_project_gpu(proj1,
                            angles + iv, offsets + iv,
                            ai + iv, bi + iv,
                            mode,
                            1,
                            SD, SO,
                            nx, ny, nz,
                            da, na, db, nb,
                            bproj);
            initial_kernel<<<ny * nz, nx>>>(emap, nx * ny * nz, 1);
            project_gpu(emap,
                        angles + iv, offsets + iv,
                        ai + iv, bi + iv,
                        mode,
                        1,
                        SD, SO,
                        nx, ny, nz,
                        da, na, db, nb,
                        proj1);
            back_project_gpu(proj1,
                            angles + iv, offsets + iv,
                            ai + iv, bi + iv,
                            mode,
                            1,
                            SD, SO,
                            nx, ny, nz,
                            da, na, db, nb,
                            emap);
            divide_kernel<<<ny * nz, nx>>>(bproj, emap, nx * ny * nz);
            add_inline_kernel<<<ny * nz, nx>>>(img1, bproj, nx * ny * nz, -lamb);
        }
    }
}
#endif
