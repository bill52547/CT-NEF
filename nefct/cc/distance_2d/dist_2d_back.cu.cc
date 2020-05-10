#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

//#define MAX4(a, b, c, d) (((((a) > (b)) ? (a) : (b)) > (((c) > (d)) ? (c) : (d))) > (((a) > (b)) ? (a) : (b)) : (((c) > (d)) ? (c) : (d)))
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#define ABS(x) ((x) > 0 ? x : -(x))
#define PI 3.141592653589793f
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;
const float eps_ = 0.01;

__global__ void kernel_backprojection(const float *vproj,
                                      const float *angles,
                                      const int mode,
                                      const float SD, const float SO,
                                      const int nx, const int ny,
                                      const float da, const float ai, const int na,
                                      float *image)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;

    int id = ix + iy * nx;

    float angle = angles[0];
    float sphi = __sinf(angle);
    float cphi = __cosf(angle);
    // float dd_voxel[3];
    float xc, yc, zc;
    xc = (float)ix - nx / 2 + 0.5f;
    yc = (float)iy - ny / 2 + 0.5f;

    // voxel boundary coordinates
    float xll, yll, xlr, ylr, xrl, yrl, xrr, yrr, xt, yt;
    xll = +xc * cphi + yc * sphi - 0.5f;
    yll = -xc * sphi + yc * cphi - 0.5f;
    xrr = +xc * cphi + yc * sphi + 0.5f;
    yrr = -xc * sphi + yc * cphi + 0.5f;
    xrl = +xc * cphi + yc * sphi + 0.5f;
    yrl = -xc * sphi + yc * cphi - 0.5f;
    xlr = +xc * cphi + yc * sphi - 0.5f;
    ylr = -xc * sphi + yc * cphi + 0.5f;

    // the coordinates of source and detector plane here are after rotation
    float ratio, all, alr, arl, arr, at, ab, a_max, a_min;
    // calculate a value for each boundary coordinates

    // the a and b here are all absolute positions from isocenter, which are on detector planes
    if (mode == 0)
    {
        ratio = SD / (xll + SO);
        all = ratio * yll;
        ratio = SD / (xrr + SO);
        arr = ratio * yrr;
        ratio = SD / (xlr + SO);
        alr = ratio * ylr;
        ratio = SD / (xrl + SO);
        arl = ratio * yrl;
    }
    else
    {
        all = (float)atan2(yll, xll + SO);
        ratio = SD * cosf(all) / (xll + SO);
        arr = (float)atan2(yrr, xrr + SO);
        ratio = SD * cosf(arr) / (xrr + SO);
        alr = (float)atan2(ylr, xlr + SO);
        ratio = SD * cosf(alr) / (xlr + SO);
        arl = (float)atan2(yrl, xrl + SO);
        ratio = SD * cosf(arl) / (xrl + SO);
    }

    a_max = MAX4(all, arr, alr, arl);
    a_min = MIN4(all, arr, alr, arl);

    // the related positions on detector plane from start points
    a_max = (a_max - ai) / da + na / 2; //  now they are the detector coordinates
    a_min = (a_min - ai) / da + na / 2;
    int a_ind_max = (int)floorf(a_max);
    int a_ind_min = (int)floorf(a_min);

    float bin_bound_1, bin_bound_2, wa;
    for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia++)
    {
        bin_bound_1 = ia + 0.0f;
        bin_bound_2 = ia + 1.0f;

        wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min); wa /= a_max - a_min;

        image[id] += wa * vproj[ia];
    }
}

//__global__ void kernel_backprojection2(const float *vproj,
//                                       const float *angles, const float *offsets,
//                                       const int mode,
//                                       const float SD, const float SO,
//                                       const int nx, const int ny, const int nz,
//                                       const float da, const float ai, const int na,
//                                       const float db, const float bi, const int nb,
//                                       float *image)
//{
//    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
//    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
//    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
//    if (ix >= nx || iy >= ny || iz >= nz)
//        return;
//
//    int id = ix + iy * nx + iz * nx * ny;
//
//    float angle = angles[0];
//    float sphi = __sinf(angle);
//    float cphi = __cosf(angle);
//    // float dd_voxel[3];
//    float xc, yc, zc;
//    xc = (float)ix - nx / 2 + 0.5f;
//    yc = (float)iy - ny / 2 + 0.5f;
//    zc = (float)iz - nz / 2 + 0.5f - offsets[0];
//
//    // voxel boundary coordinates
//    float xc1, yc1, zc1;
//    xc1 = +xc * cphi + yc * sphi;
//    yc1 = -xc * sphi + yc * cphi;
//    zc1 = zc;
//
//    // the coordinates of source and detector plane here are after rotation
//    float ratio, a, b;
//    // calculate a value for each boundary coordinates
//
//    // the a and b here are all absolute positions from isocenter, which are on detector planes
//    if (mode == 0)
//    {
//        ratio = SD / (xc1 + SO);
//        a = ratio * yc1;
//        b = ratio * zc1;
//    }
//    else
//    {
//        a = (float)atan2(yc1, xc1 + SO);
//        ratio = SD * cosf(a) / (xc1 + SO);
//        b = ratio * zc1;
//    }
//
//    int ia1, ia2, ib1, ib2;
//    float wa1, wa2, wb1, wb2;
//    float ta = (a - ai) / da + na / 2;
//    float tb = (b - bi) / db + nb / 2;
//    ia1 = (int)floorf(ta);
//    ia2 = ia1 + 1;
//    ib1 = (int)floorf(tb);
//    ib2 = ib1 + 1;
//    wa2 = ta - ia1;
//    wa1 = 1 - wa2;
//    wb2 = tb - ib1;
//    wb1 = 1 - wb2;
//    if (0 <= ia1 && ia2 < na && 0 <= ib1 && ib2 < nb)
//    {
//        image[id] += wa1 * wb1 * vproj[ia1 + ib1 * na] +
//                     wa2 * wb1 * vproj[ia2 + ib1 * na] +
//                     wa1 * wb2 * vproj[ia1 + ib2 * na] +
//                     wa2 * wb2 * vproj[ia2 + ib2 * na];
//    }
//}

__global__ void initial_kernel(float *image1, const int N, const float beta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    image1[tid] = beta;
}

void back_project_gpu(const float *pv_values, const float *angles,
                      const int mode,
                      const int nv,
                      const float SD, const float SO,
                      const int nx, const int ny,
                      const float da, const float ai, const int na,
                      float *image)
{
    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                         (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    initial_kernel<<<ny, nx>>>(image, nx * ny, 0.0f);
    for (int iv = 0; iv < nv; iv++)
    {
        kernel_backprojection<<<shapeSize, blockSize>>>(pv_values + iv * na,
                                                        angles + iv,
                                                        mode,
                                                        SD, SO,
                                                        nx, ny,
                                                        da, ai, na,
                                                        image);
        cudaDeviceSynchronize();
    }
}

void back_project_flat_gpu(const float *pv_values, const int *shape,
						   const float *angles, const int nv,
						   const float SD, const float SO,
						   const float da, const float ai, const int na,
						   float *image)
{
    int shape_cpu[2];
    cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0];
    int ny = shape_cpu[1];
    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                         (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    initial_kernel<<<ny, nx>>>(image, nx * ny, 0.0f);
    for (int iv = 0; iv < nv; iv++)
    {
        kernel_backprojection<<<shapeSize, blockSize>>>(pv_values + iv * na,
                                                        angles + iv,
                                                        0,
                                                        SD, SO,
                                                        nx, ny,
                                                        da, ai, na,
                                                        image);
        cudaDeviceSynchronize();
    }
}

void back_project_cyli_gpu(const float *pv_values, const int *shape,
						   const float *angles, const int nv,
						   const float SD, const float SO,
						   const float da, const float ai, const int na,
						   float *image)
{
    int shape_cpu[2];
    cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0];
    int ny = shape_cpu[1];

    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                         (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    initial_kernel<<<ny, nx>>>(image, nx * ny, 0.0f);

    for (int iv = 0; iv < nv; iv++)
    {
        kernel_backprojection<<<shapeSize, blockSize>>>(pv_values + iv * na,
                                                        angles + iv,
                                                        1,
                                                        SD, SO,
                                                        nx, ny,
                                                        da, ai, na,
                                                        image);
        cudaDeviceSynchronize();
    }
}
#endif
