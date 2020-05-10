#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define MAX(a,b) (((a) > (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

//#define MAX4(a, b, c, d) (((((a) > (b)) ? (a) : (b)) > (((c) > (d)) ? (c) : (d))) > (((a) > (b)) ? (a) : (b)) : (((c) > (d)) ? (c) : (d)))
#define MIN(a,b) (((a) < (b)) ? a : b)
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#define ABS(x) ((x) > 0 ? x : -(x))
#define PI 3.141592653589793f
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;
const float eps_ = 0.01;

__global__ void kernel_backprojection(const int mode, const float *angles, const float *offsets, const int nv,
                        const float SID, const float SAD,
                        const int nx, const int ny, const int nz,
                        const float da, const float ai, const int na,
                        const float db, const float bi, const int nb,
                        const float *vproj,
                        float *image){
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int id = ix + iy * nx + iz * nx * ny;
	// angle += 3.141592653589793;

    image[id] = 0.0f;
	// float sphi = __sinf(angle);
	// float cphi = __cosf(angle);
	for (int iv = 0; iv < nv; iv++)
	{
	    float angle = angles[iv];
        float sphi = __sinf(angle);
        float cphi = __cosf(angle);
        // float dd_voxel[3];
        float xc, yc, zc;
        xc = (float)ix - nx / 2 + 0.5f;
        yc = (float)iy - ny / 2 + 0.5f;
        zc = (float)iz - nz / 2 + 0.5f;

        // voxel boundary coordinates
        float xll, yll, zll, xlr, ylr, zlr, xrl, yrl, zrl, xrr, yrr, zrr, xt, yt, zt, xb, yb, zb;
        xll = +xc * cphi + yc * sphi - 0.5f;
        yll = -xc * sphi + yc * cphi - 0.5f;
        xrr = +xc * cphi + yc * sphi + 0.5f;
        yrr = -xc * sphi + yc * cphi + 0.5f;
        zll = zc; zrr = zc;
        xrl = +xc * cphi + yc * sphi + 0.5f;
        yrl = -xc * sphi + yc * cphi - 0.5f;
        xlr = +xc * cphi + yc * sphi - 0.5f;
        ylr = -xc * sphi + yc * cphi + 0.5f;
        zrl = zc; zlr = zc;
        xt = xc * cphi + yc * sphi;
        yt = -xc * sphi + yc * cphi;
        zt = zc + 0.5f;
        xb = xc * cphi + yc * sphi;
        yb = -xc * sphi + yc * cphi;
        zb = zc - 0.5f;

        // the coordinates of source and detector plane here are after rotation
        float ratio, all, bll, alr, blr, arl, brl, arr, brr, at, bt, ab, bb, a_max, a_min, b_max, b_min;
        // calculate a value for each boundary coordinates


        // the a and b here are all absolute positions from isocenter, which are on detector planes
        if (mode == 0)
        {
            ratio = SID / (xll + SAD);
            all = ratio * yll;
            bll = ratio * zll;
            ratio = SID / (xrr + SAD);
            arr = ratio * yrr;
            brr = ratio * zrr;
            ratio = SID / (xlr + SAD);
            alr = ratio * ylr;
            blr = ratio * zlr;
            ratio = SID / (xrl + SAD);
            arl = ratio * yrl;
            brl = ratio * zrl;
            ratio = SID / (xt + SAD);
            at = ratio * yt;
            bt = ratio * zt;
            ratio = SID / (xb + SAD);
            ab = ratio * yb;
            bb = ratio * zb;
        }
        else
        {                        
            all = (float)atan2(yll, xll);
            ratio = SID * cosf(all) / (xll + SAD);
            bll = ratio * zll;
            arr = (float)atan2(yrr, xrr);
            ratio = SID * cosf(arr) / (xrr + SAD);
            brr = ratio * zrr;
            alr = (float)atan2(ylr, xlr);
            ratio = SID * cosf(alr) / (xlr + SAD);
            blr = ratio * zlr;
            arl = (float)atan2(yrl, xrl);
            ratio = SID * cosf(arl) / (xrl + SAD);
            brl = ratio * zrl;
            at = (float)atan2(yt, xt);
            ratio = SID * cosf(at) / (xt + SAD);
            bt = ratio * zt;
            ab = (float)atan2(yb, xb);
            ratio = SID * cosf(ab) / (xb + SAD);
            bb = ratio * zb;
        }

        a_max = MAX6(all ,arr, alr, arl, at, ab);
        a_min = MIN6(all ,arr, alr, arl, at, ab);
        b_max = MAX6(bll ,brr, blr, brl, bt, bb);
        b_min = MIN6(bll ,brr, blr, brl, bt, bb);

        // the related positions on detector plane from start points
        a_max = (a_max - ai) / da + na / 2; //  now they are the detector coordinates
        a_min = (a_min - ai) / da + na / 2;
        b_max = (b_max - bi) / db + nb / 2;
        b_min = (b_min - bi) / db + nb / 2;
        int a_ind_max = (int)floorf(a_max);
        int a_ind_min = (int)floorf(a_min);
        int b_ind_max = (int)floorf(b_max);
        int b_ind_min = (int)floorf(b_min);

        float bin_bound_1, bin_bound_2, wa, wb;
        for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia ++){
            bin_bound_1 = ia + 0.0f;
            bin_bound_2 = ia + 1.0f;

            wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);// wa /= a_max - a_min;

            for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib ++){
                bin_bound_1 = ib + 0.0f;
                bin_bound_2 = ib + 1.0f;
                wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= b_max - b_min;

                image[id] += wa * wb * vproj[ia + ib * na + iv * na * nb];
            }
        }
    }
}


void back_project_flat_gpu(const float *pv_values, const int *shape, const float *offsets,
                           const float *angles, const int nv,
                           const float SID, const float SAD,
                           const float da, const float ai, const int na,
                           const float db, const float bi, const int nb,
                           float *image)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    kernel_backprojection<<<shapeSize, blockSize>>>(0, angles, offsets, nv,
                                                  SID, SAD,
                                                  nx, ny, nz,
                                                  da, ai, na,
                                                  db, bi, nb,
                                                  pv_values, image);
}

void back_project_cyli_gpu(const float *pv_values, const int *shape, const float *offsets,
                           const float *angles, const int nv,
                           const float SID, const float SAD,
                           const float da, const float ai, const int na,
                           const float db, const float bi, const int nb,
                           float *image)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (nv + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    kernel_backprojection<<<shapeSize, blockSize>>>(1, angles, offsets, nv,
                                                  SID, SAD,
                                                  nx, ny, nz,
                                                  da, ai, na,
                                                  db, bi, nb,
                                                  pv_values, image);
}

#endif
