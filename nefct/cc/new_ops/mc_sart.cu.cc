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

void mc_project_gpu_single(const float *image,
                    const float *angles, const float *offsets,
                    const float *ax, const float *ay, const float *az,
                    const float *bx, const float *by, const float *bz,
                    const float *cx, const float *cy, const float *cz,
                    const float *v_data, const float *f_data,
                    const float *v0_data, const float *f0_data,
                    float *temp_img, float *temp_img1,
                    float *mx, float *my, float *mz,
                    const int mode,
                    const float SD, const float SO,
                    const int nx, const int ny, const int nz,
                    const float da, const float ai, const int na,
                    const float db, const float bi, const int nb,
                    float *pv_values);

void mc_back_project_gpu_single(const float *pv_values,
                         const float *angles, const float *offsets,
                         const float *ax, const float *ay, const float *az,
                         const float *bx, const float *by, const float *bz,
                         const float *cx, const float *cy, const float *cz,
                         const float *v_data, const float *f_data,
                         const float *v0_data, const float *f0_data,
                         float *temp_img, float *temp_img1,
                         float *mx, float *my, float *mz,
                         const int mode,
                         const float SD, const float SO,
                         const int nx, const int ny, const int nz,
                         const float da, const float ai, const int na,
                         const float db, const float bi, const int nb,
                         float *image);

__global__ void divide_kernel(float *image1, const float *image, const int N);

__global__ void multiply_kernel(float *image1, const float *image, const int N);

__global__ void add_inline_kernel(float *image1, const float *image, const int N, const float beta);

__global__ void initial_kernel(float* image, const int N, const float value);

void deform_tex(const float *img,
                const float *mx, const float *my, const float *mz,
                const int nx, const int ny, const int nz,
                const int adding,
                float *img1);

void deform_invert_tex(const float *img,
                       const float *mx, const float *my, const float *mz,
                       const int nx, const int ny, const int nz,
                       const int adding,
                       float *img1);

__global__ void mk_dvf(const float *ax, const float *ay, const float *az,
                       const float *bx, const float *by, const float *bz,
                       const float *cx, const float *cy, const float *cz,
                       const float *v_data, const float *f_data,
                       const int nx, const int ny, const int nz,
                       float *mx, float *my, float *mz);

void mc_sart_gpu(const float *img, const float *proj, float *emap,
                 const float *angles, const float *offsets,
                 const float *ax, const float *ay, const float *az,
                 const float *bx, const float *by, const float *bz,
                 const float *cx, const float *cy, const float *cz,
                 const float *v_data, const float *f_data,
                 const float *v0_data, const float *f0_data, const int *num_in_bin,
                 float *temp_img, float *temp_img1,
                 float *proj1, float *bproj,
                 float *mx, float *my, float *mz,
                 const int n_iter, const float lamb,
                 const int mode,
                 const int nbin,
                 const int out_iter,
                 const float SD, const float SO,
                 const int nx, const int ny, const int nz,
                 const float da, const float ai, const int na,
                 const float db, const float bi, const int nb,
                 float *img1)
{
    int num_in_bin_host[nbin + 1];
    cudaMemcpy(num_in_bin_host, num_in_bin, (nbin + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for (int ibin = 0; ibin < nbin; ibin ++)
    {
        if (out_iter > 0)
        {
            if (ibin == 0)
            {
                initial_kernel<<<ny * nz, nx>>>(temp_img, nx * ny * nz, 0);
                mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v0_data + nbin - 1, f0_data + nbin - 1,
                            nx, ny, nz,
                            mx, my, mz);
                deform_invert_tex(img + (nbin - 1) * nx * ny * nz, mx, my, mz, nx, ny, nz, 0, temp_img);

                mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                                        bx, by, bz,
                                        cx, cy, cz,
                                        v0_data + ibin, f0_data + ibin,
                                        nx, ny, nz,
                                        mx, my, mz);
                deform_tex(temp_img, mx, my, mz, nx, ny, nz, 0, img1 + ibin * nx * ny * nz);
            }
            else
            {
                initial_kernel<<<ny * nz, nx>>>(temp_img, nx * ny * nz, 0);
                mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v0_data + ibin - 1, f0_data + ibin - 1,
                            nx, ny, nz,
                            mx, my, mz);
                deform_invert_tex(img1 + (ibin - 1) * nx * ny * nz, mx, my, mz, nx, ny, nz, 0, temp_img);

                mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                                        bx, by, bz,
                                        cx, cy, cz,
                                        v0_data + ibin, f0_data + ibin,
                                        nx, ny, nz,
                                        mx, my, mz);
                deform_tex(temp_img, mx, my, mz, nx, ny, nz, 0, img1 + ibin * nx * ny * nz);
            }
        }
        for (int iter = 0; iter < n_iter; iter++)
        {
            for (int iv = num_in_bin_host[ibin]; iv < num_in_bin_host[ibin + 1]; iv++)
            {
                mc_project_gpu_single(img1 + ibin * nx * ny * nz,
                           angles + iv, offsets + iv,
                           ax, ay, az,
                           bx, by, bz,
                           cx, cy, cz,
                           v_data + iv, f_data + iv,
                           v0_data + ibin, f0_data + ibin,
                           temp_img, temp_img1,
                           mx, my, mz,
                           mode,
                           SD, SO,
                           nx, ny, nz,
                           da, ai, na, db, bi, nb,
                           proj1);

                add_inline_kernel<<<nb, na>>>(proj1, proj + iv * na * nb, na * nb, -1);
                mc_back_project_gpu_single(proj1,
                                    angles + iv, offsets + iv,
                                    ax, ay, az,
                                    bx, by, bz,
                                    cx, cy, cz,
                                    v_data + iv, f_data + iv,
                                    v0_data + ibin, f0_data + ibin,
                                    temp_img, temp_img1,
                                    mx, my, mz,
                                    mode,
                                    SD, SO,
                                    nx, ny, nz,
                                    da, ai, na, db, bi, nb,
                                    bproj);
                initial_kernel<<<ny * nz, nx>>>(emap, nx * ny * nz, 1);
                mc_project_gpu_single(emap,
                               angles + iv, offsets + iv,
                               ax, ay, az,
                               bx, by, bz,
                               cx, cy, cz,
                               v_data + iv, f_data + iv,
                               v0_data + ibin, f0_data + ibin,
                               temp_img, temp_img1,
                               mx, my, mz,
                               mode,
                               SD, SO,
                               nx, ny, nz,
                               da, ai, na, db, bi, nb,
                               proj1);
               mc_back_project_gpu_single(proj1,
                                   angles + iv, offsets + iv,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx, cy, cz,
                                   v_data + iv, f_data + iv,
                                   v0_data + ibin, f0_data + ibin,
                                   temp_img, temp_img1,
                                   mx, my, mz,
                                   mode,
                                   SD, SO,
                                   nx, ny, nz,
                                   da, ai, na, db, bi, nb,
                                   emap);
               divide_kernel<<<ny * nz, nx>>>(bproj, emap, nx * ny * nz);
               add_inline_kernel<<<ny * nz, nx>>>(img1 + ibin * nx * ny * nz,
                                                   bproj, nx * ny * nz, -lamb);
            }
        }
        // if (out_iter > 0)
        // {
        //     if (ibin == nbin - 1)
        //     {
        //         initial_kernel<<<ny * nz, nx>>>(temp_img, nx * ny * nz, 0);
        //         mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
        //                                 bx, by, bz,
        //                                 cx, cy, cz,
        //                                 v0_data + ibin, f0_data + ibin,
        //                                 nx, ny, nz,
        //                                 mx, my, mz);
        //         deform_invert_tex(img1 + ibin * nx * ny * nz, mx, my, mz, nx, ny, nz, 0, temp_img);

        //         initial_kernel<<<ny * nz, nx>>>(img1, nx * ny * nz, 0);
        //         mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
        //                                 bx, by, bz,
        //                                 cx, cy, cz,
        //                                 v0_data, f0_data,
        //                                 nx, ny, nz,
        //                                 mx, my, mz);
        //         deform_tex(temp_img, mx, my, mz, nx, ny, nz, 0, img1);
        //     }
        // }
    }
}
#endif
