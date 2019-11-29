#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define ABS(x) ((x > 0) ? x : -(x))
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

// mode 0 for flat, mode 1 for cylin
__global__ void kernel_back_count(const int mode,
                                  const float SD, const float SO,
                                  const int nx, const int ny, const int nz,
                                  const float da, const float ai, const int na,
                                  const float db, const float bi, const int nb,
                                  int *n_in_row)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int id = ix + iy * nx + iz * nx * ny;
    n_in_row[id] = 0;
    float xc, yc, zc;
    xc = (float)ix - nx / 2 + 0.5f;
    yc = (float)iy - ny / 2 + 0.5f;
    zc = (float)iz - nz / 2 + 0.5f;
    // voxel boundary coordinates
    float xll, yll, zll, xlr, ylr, zlr, xrl, yrl, zrl, xrr, yrr, zrr, xt, yt, zt, xb, yb, zb;
    xll = xc - 0.5f;
    yll = yc - 0.5f;
    xrr = xc + 0.5f;
    yrr = yc + 0.5f;
    zll = zc;
    zrr = zc;
    xrl = xc + 0.5f;
    yrl = yc - 0.5f;
    xlr = xc - 0.5f;
    ylr = yc + 0.5f;
    zrl = zc;
    zlr = zc;
    xt = xc;
    yt = yc;
    zt = zc + 0.5f;
    xb = xc;
    yb = yc;
    zb = zc - 0.5f;

    // the coordinates of source and detector plane here are after rotation
    float ratio, all, bll, alr, blr, arl, brl, arr, brr, at, bt, ab, bb, a_max, a_min, b_max, b_min;
    // calculate a value for each boundary coordinates

    // the a and b here are all absolute positions from isocenter, which are on detector planes
    if (mode == 0)
    {
        ratio = SD / (xll + SO);
        all = ratio * yll;
        bll = ratio * zll;
        ratio = SD / (xrr + SO);
        arr = ratio * yrr;
        brr = ratio * zrr;
        ratio = SD / (xlr + SO);
        alr = ratio * ylr;
        blr = ratio * zlr;
        ratio = SD / (xrl + SO);
        arl = ratio * yrl;
        brl = ratio * zrl;
        ratio = SD / (xt + SO);
        at = ratio * yt;
        bt = ratio * zt;
        ratio = SD / (xb + SO);
        ab = ratio * yb;
        bb = ratio * zb;
    }
    else
    {
        all = (float)atan2(yll, xll);
        ratio = SD * cosf(all) / (xll + SO);
        bll = ratio * zll;
        arr = (float)atan2(yrr, xrr);
        ratio = SD * cosf(arr) / (xrr + SO);
        brr = ratio * zrr;
        ratio = SD * cosf(arr) / (xrr + SO);
        brr = ratio * zrr;
        alr = (float)atan2(ylr, xlr);
        ratio = SD * cosf(alr) / (xlr + SO);
        blr = ratio * zlr;
        arl = (float)atan2(yrl, xrl);
        ratio = SD * cosf(arl) / (xrl + SO);
        brl = ratio * zrl;
        at = (float)atan2(yt, xt);
        ratio = SD * cosf(at) / (xt + SO);
        bt = ratio * zt;
        ab = (float)atan2(yb, xb);
        ratio = SD * cosf(ab) / (xb + SO);
        bb = ratio * zb;
    }

    a_max = MAX6(all, arr, alr, arl, at, ab);
    a_min = MIN6(all, arr, alr, arl, at, ab);
    b_max = MAX6(bll, brr, blr, brl, bt, bb);
    b_min = MIN6(bll, brr, blr, brl, bt, bb);

    // the related positions on detector plane from start points
    a_max = (a_max - ai) / da + na / 2; //  now they are the detector coordinates
    a_min = (a_min - ai) / da + na / 2;
    b_max = (b_max - bi) / db + nb / 2;
    b_min = (b_min - bi) / db + nb / 2;
    int a_ind_max = (int)floorf(a_max);
    int a_ind_min = (int)floorf(a_min);
    int b_ind_max = (int)floorf(b_max);
    int b_ind_min = (int)floorf(b_min);
    for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia++)
    {
        for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib++)
        {
            n_in_row[id] += 1;
        }
    }
}

void count_flat_back_gpu(const int *shape,
                         const float SD, const float SO,
                         const float da, const float ai, const int na,
                         const float db, const float bi, const int nb,
                         int *n_in_row)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    kernel_back_count<<<shapeSize, blockSize>>>(0,
                                                SD, SO,
                                                nx, ny, nz,
                                                da, ai, na,
                                                db, bi, nb,
                                                n_in_row);
}

void count_cyli_back_gpu(const int *shape,
                         const float SD, const float SO,
                         const float da, const float ai, const int na,
                         const float db, const float bi, const int nb,
                         int *n_in_row)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    kernel_back_count<<<shapeSize, blockSize>>>(1,
                                                SD, SO,
                                                nx, ny, nz,
                                                da, ai, na,
                                                db, bi, nb,
                                                n_in_row);
}
#endif
