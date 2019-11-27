#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define ABS(x) ((x > 0) ? x : -(x))
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;
const float eps_ = 0.01;

// mode 0 for flat, mode 1 for cylin
__global__ void project_kernel(const int mode, const float *angles, const int nv, const float SID, const float SAD,
                               const int nx, const int ny, const float da, const float ai, const int na, const float *image,
                               float *vproj)
{
    int ba = blockIdx.x;
    int bv = blockIdx.y;

    int ta = threadIdx.x;
    int tv = threadIdx.y;

    int ia = ba * GRIDDIM_X + ta;
    int iv = bv * GRIDDIM_Y + tv;

    if (ia >= na || iv >= nv)
        return;
    float x20, y20, x2, y2, x2n, y2n, x2m, y2m, p2x, p2y, p2xn, p2yn, ptmp;
    float talpha, calpha, ds, temp, dst;
    const float cphi = cosf(angles[iv]);
    const float sphi = sinf(angles[iv]);

    const float x1 = -SAD * cphi;
    const float y1 = -SAD * sphi;

    const int id = ia + iv * na;

    vproj[id] = 0.0f;

    if (mode == 0)
    {
        x20 = SID - SAD;
        y20 = (ia + 0.5) * da + ai - na * da / 2;
    }
    else
    {
        float ang = (ia + 0.5) * da + ai - na * da / 2;
        x20 = SID * cosf(ang) - SAD;
        y20 = SID * sinf(ang);
    }
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;

    float x21, y21; // offset between SADurce and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;

    if (ABS(x21) > ABS(y21))
    {
        // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2;
        int Yi1, Yi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++)
        {
            // calculate y indices of intersecting voxel candidates
            float xl, xr, yl, yr, ratio;
            float cyll, cylr, cyrl, cyrr, xc;
            if (mode == 0)
            {
                x20 = SID - SAD;
                y20 = ia * da + ai - na * da / 2;
            }
            else
            {
                float ang = ia * da + ai - na * da / 2;
                x20 = SID * cosf(ang) - SAD;
                y20 = SID * sinf(ang);
            }
            xl = x20 * cphi - y20 * sphi - x1;
            yl = x20 * sphi + y20 * cphi - y1;

            if (mode == 0)
            {
                x20 = SID - SAD;
                y20 = (ia + 1) * da + ai - na * da / 2;
            }
            else
            {
                float ang = (ia + 1) * da + ai - na * da / 2;
                x20 = SID * cosf(ang) - SAD;
                y20 = SID * sinf(ang);
            }
            xr = x20 * cphi - y20 * sphi - x1;
            yr = x20 * sphi + y20 * cphi - y1;

            // xl = x21 - da / 2 * sphi;
            // xr = x21 + da / 2 * sphi;
            // yl = y21 - da / 2 * cphi;
            // yr = y21 + da / 2 * cphi;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;

            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2;

            yi1 = MIN(cyll, cyrr);
            Yi1 = (int)floorf(yi1);
            yi2 = MAX(cyll, cyrr);
            Yi2 = (int)floorf(yi2);

            xc = (float)ix + 0.5f - (float)nx / 2 - x1;

            float wy;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1);
                wy /= (yi2 - yi1);
                vproj[id] += image[ix + iy * nx] * wy / ABS(x21) * sqrt(x21 * x21 + y21 * y21);
            }
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)
    else
    {
        float xi1, xi2;
        int Xi1, Xi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++)
        {
            // calculate y indices of intersecting voxel candidates
            float yl, yr, xl, xr, ratio;
            float cxll, cxlr, cxrl, cxrr, yc;
            if (mode == 0)
            {
                x20 = SID - SAD;
                y20 = ia * da + ai - na * da / 2;
            }
            else
            {
                float ang = ia * da + ai - na * da / 2;
                x20 = SID * cosf(ang) - SAD;
                y20 = SID * sinf(ang);
            }
            xl = x20 * cphi - y20 * sphi - x1;
            yl = x20 * sphi + y20 * cphi - y1;

            if (mode == 0)
            {
                x20 = SID - SAD;
                y20 = (ia + 1) * da + ai - na * da / 2;
            }
            else
            {
                float ang = (ia + 1) * da + ai - na * da / 2;
                x20 = SID * cosf(ang) - SAD;
                y20 = SID * sinf(ang);
            }
            xr = x20 * cphi - y20 * sphi - x1;
            yr = x20 * sphi + y20 * cphi - y1;

            // yl = y21 - da / 2 * cphi;
            // yr = y21 + da / 2 * cphi;
            // xl = x21 - da / 2 * sphi;
            // xr = x21 + da / 2 * sphi;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2;

            xi1 = MIN(cxll, cxrr);
            Xi1 = (int)floorf(xi1);
            xi2 = MAX(cxll, cxrr);
            Xi2 = (int)floorf(xi2);

            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            float wx;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1);
                wx /= (xi2 - xi1);
                vproj[id] += image[ix + iy * nx] * wx / ABS(y21) * sqrt(x21 * x21 + y21 * y21);
            }
        }
    }
}

// // mode 0 for flat, mode 1 for cylin
// __global__ void back_project_kernel(const int mode, const float *angles, const int nv, const float SID, const float SAD,
//                                     const int nx, const int ny, const float da, const float ai, const int na, const float *vproj,
//                                     float *image)
// {
//     int ba = blockIdx.x;
//     int bv = blockIdx.y;

//     int ta = threadIdx.x;
//     int tv = threadIdx.y;

//     int ia = ba * GRIDDIM_X + ta;
//     int iv = bv * GRIDDIM_Y + tv;

//     if (ia >= na || iv >= nv)
//         return;
//     float x20, y20, x2, y2, x2n, y2n, x2m, y2m, p2x, p2y, p2xn, p2yn, ptmp;
//     float talpha, calpha, ds, temp, dst;
//     const float cphi = cosf(angles[iv]);
//     const float sphi = sinf(angles[iv]);

//     const float x1 = -SAD * cphi;
//     const float y1 = -SAD * sphi;

//     const int id = ia + iv * na;

//     if (mode == 0)
//     {
//         x20 = SID;
//         y20 = (ia + 0.5) * da + ai - na * da / 2;
//     }
//     else
//     {
//         float ang = (ia + 0.5) * da + ai - na * da / 2;
//         x20 = SID * cosf(ang) - SAD;
//         y20 = SID * sinf(ang);
//     }
//     x2 = x20 * cphi - y20 * sphi;
//     y2 = x20 * sphi + y20 * cphi;

//     float x21, y21; // offset between SADurce and detector center
//     x21 = x2 - x1;
//     y21 = y2 - y1;

//     if (ABS(x21) > ABS(y21))
//     {
//         // if (ABS(cphi) > ABS(sphi)){
//         float yi1, yi2;
//         int Yi1, Yi2;
//         // for each y - z plane, we calculate and add the contribution of related pixels
//         for (int ix = 0; ix < nx; ix++)
//         {
//             // calculate y indices of intersecting voxel candidates
//             float xl, xr, yl, yr, ratio;
//             float cyll, cylr, cyrl, cyrr, xc;
//             if (mode == 0)
//             {
//                 x20 = SID;
//                 y20 = ia * da + ai - na * da / 2;
//             }
//             else
//             {
//                 float ang = ia * da + ai - na * da / 2;
//                 x20 = SID * cosf(ang) - SAD;
//                 y20 = SID * sinf(ang);
//             }
//             xl = x20 * cphi - y20 * sphi - x1;
//             yl = x20 * sphi + y20 * cphi - y1;

//             if (mode == 0)
//             {
//                 x20 = SID;
//                 y20 = (ia + 1) * da + ai - na * da / 2;
//             }
//             else
//             {
//                 float ang = (ia + 1) * da + ai - na * da / 2;
//                 x20 = SID * cosf(ang) - SAD;
//                 y20 = SID * sinf(ang);
//             }
//             xr = x20 * cphi - y20 * sphi - x1;
//             yr = x20 * sphi + y20 * cphi - y1;

//             // xl = x21 - da / 2 * sphi;
//             // xr = x21 + da / 2 * sphi;
//             // yl = y21 - da / 2 * cphi;
//             // yr = y21 + da / 2 * cphi;
//             xc = (float)ix + 0.5f - (float)nx / 2 - x1;

//             ratio = yl / xl;
//             cyll = ratio * xc + y1 + ny / 2;
//             ratio = yr / xr;
//             cyrr = ratio * xc + y1 + ny / 2;

//             yi1 = MIN(cyll, cyrr);
//             Yi1 = (int)floorf(yi1);
//             yi2 = MAX(cyll, cyrr);
//             Yi2 = (int)floorf(yi2);

//             xc = (float)ix + 0.5f - (float)nx / 2 - x1;

//             float wy;

//             for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
//             {
//                 wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1);
//                 wy /= (yi2 - yi1);
//                 atomicAdd(image + ix + iy * nx, vproj[id] * wy / ABS(x21) * sqrt(x21 * x21 + y21 * y21));
//             }
//         }
//     }
//     // x - z plane, where ABS(x21) <= ABS(y21)
//     else
//     {
//         float xi1, xi2;
//         int Xi1, Xi2;
//         // for each y - z plane, we calculate and add the contribution of related pixels
//         for (int iy = 0; iy < ny; iy++)
//         {
//             // calculate y indices of intersecting voxel candidates
//             float yl, yr, xl, xr, ratio;
//             float cxll, cxlr, cxrl, cxrr, yc;
//             if (mode == 0)
//             {
//                 x20 = SID;
//                 y20 = ia * da + ai - na * da / 2;
//             }
//             else
//             {
//                 float ang = ia * da + ai - na * da / 2;
//                 x20 = SID * cosf(ang) - SAD;
//                 y20 = SID * sinf(ang);
//             }
//             xl = x20 * cphi - y20 * sphi - x1;
//             yl = x20 * sphi + y20 * cphi - y1;

//             if (mode == 0)
//             {
//                 x20 = SID;
//                 y20 = (ia + 1) * da + ai - na * da / 2;
//             }
//             else
//             {
//                 float ang = (ia + 1) * da + ai - na * da / 2;
//                 x20 = SID * cosf(ang) - SAD;
//                 y20 = SID * sinf(ang);
//             }
//             xr = x20 * cphi - y20 * sphi - x1;
//             yr = x20 * sphi + y20 * cphi - y1;

//             // yl = y21 - da / 2 * cphi;
//             // yr = y21 + da / 2 * cphi;
//             // xl = x21 - da / 2 * sphi;
//             // xr = x21 + da / 2 * sphi;
//             yc = (float)iy + 0.5f - (float)ny / 2 - y1;

//             ratio = xl / yl;
//             cxll = ratio * yc + x1 + nx / 2;
//             ratio = xr / yr;
//             cxrr = ratio * yc + x1 + nx / 2;

//             xi1 = MIN(cxll, cxrr);
//             Xi1 = (int)floorf(xi1);
//             xi2 = MAX(cxll, cxrr);
//             Xi2 = (int)floorf(xi2);

//             yc = (float)iy + 0.5f - (float)ny / 2 - y1;

//             float wx;

//             for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
//             {
//                 wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1);
//                 wx /= (xi2 - xi1);
//                 atomicAdd(image + ix + iy * nx, vproj[id] * wx / ABS(y21) * sqrt(x21 * x21 + y21 * y21));
//             }
//         }
//     }
// }

void project_flat_gpu(const float *image, const int *shape,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      float *pv_values)
{
    int shape_cpu[2];
    cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    project_kernel<<<shapeSize, blockSize>>>(0, angles, nv,
                                             SID, SAD,
                                             nx, ny,
                                             da, ai, na,
                                             image, pv_values);
}

void project_cyli_gpu(const float *image, const int *shape,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      float *pv_values)
{
    int shape_cpu[2];
    cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    project_kernel<<<shapeSize, blockSize>>>(1, angles, nv,
                                             SID, SAD,
                                             nx, ny,
                                             da, ai, na,
                                             image, pv_values);
}

// void back_project_flat_gpu(const float *pv_values, const int *shape, const float *center,
//                            const float *angles, const int nv,
//                            const float SID, const float SAD,
//                            const float da, const float ai, const int na,
//                            float *image)
// {
//     int shape_cpu[2];
//     cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
//     int nx = shape_cpu[0], ny = shape_cpu[1]; //number of meshes

//     const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
//     const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
//     back_project_kernel<<<shapeSize, blockSize>>>(0, angles, nv,
//                                                   SID, SAD,
//                                                   nx, ny,
//                                                   da, ai, na,
//                                                   pv_values, image);
// }

// void back_project_cyli_gpu(const float *pv_values, const int *shape, const float *center,
//                            const float *angles, const int nv,
//                            const float SID, const float SAD,
//                            const float da, const float ai, const int na,
//                            float *image)
// {
//     int shape_cpu[2];
//     cudaMemcpy(shape_cpu, shape, 2 * sizeof(int), cudaMemcpyDeviceToHost);
//     int nx = shape_cpu[0], ny = shape_cpu[1]; //number of meshes

//     const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nv + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
//     const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

//     back_project_kernel<<<shapeSize, blockSize>>>(1, angles, nv,
//                                                   SID, SAD,
//                                                   nx, ny,
//                                                   da, ai, na,
//                                                   pv_values, image);
// }

#endif
