#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define ABS(x) ((x > 0) ? x : -(x))
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;

// mode 0 for flat, mode 1 for cylin
__global__ void kernel_count(const int mode,
                             const float SD, const float SO,
                             const int nx, const int ny, const int nz,
                             const float da, const float ai, const int na,
                             const float db, const float bi, const int nb,
                             int *n_in_row)
{
    int ba = blockIdx.x;
    int bb = blockIdx.y;

    int ta = threadIdx.x;
    int tb = threadIdx.y;

    int ia = ba * GRIDDIM_X + ta;
    int ib = bb * GRIDDIM_Y + tb;

    if (ia >= na || ib >= nb)
        return;
    int id = ia + ib * na;
    n_in_row[id] = 0;
    float x1, y1, z1, x2, y2, z2;
    x1 = -SO;
    y1 = 0.0f;
    z1 = 0.0f;
    if (mode == 0)
    {
        x2 = SD - SO;
        y2 = (ia + 0.5) * da + ai - na * da / 2;
        z2 = (ib + 0.5) * db + bi - nb * db / 2;
    }
    else
    {
        float ang = (ia + 0.5) * da + ai - na * da / 2;
        x2 = SD * cosf(ang) - SO;
        y2 = SD * sinf(ang);
        z2 = (ib + 0.5) * db + bi - nb * db / 2;
    }
    float x21, y21, z21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;
    z21 = z2 - z1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21))
    {
        // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2, zi1, zi2;
        int Yi1, Yi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++)
        {
            // calculate y indices of intersecting voxel candidates
            float xl, xr, yl, yr, ratio;
            float cyll, cyrr, xc;
            xl = x21;
            xr = x21;
            yl = y21 - da / 2;
            yr = y21 + da / 2;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;

            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2;

            yi1 = MIN(cyll, cyrr);
            Yi1 = (int)floorf(yi1);
            yi2 = MAX(cyll, cyrr);
            Yi2 = (int)floorf(yi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;

            ratio = zl / x21;
            czl = ratio * xc + z1 + nz / 2;
            ratio = zr / x21;
            czr = ratio * xc + z1 + nz / 2;

            zi1 = MIN(czl, czr);
            Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr);
            Zi2 = (int)floorf(zi2);

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    n_in_row[id] += 1;
                }
            }
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)
    else
    {
        float xi1, xi2, zi1, zi2;
        int Xi1, Xi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++)
        {
            // calculate y indices of intersecting voxel candidates
            float yl, yr, xl, xr, ratio;
            float cxll, cxrr, yc;
            yl = y21 - da / 2;
            yr = y21 + da / 2;
            xl = x21;
            xr = x21;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2;

            xi1 = MIN(cxll, cxrr);
            Xi1 = (int)floorf(xi1);
            xi2 = MAX(cxll, cxrr);
            Xi2 = (int)floorf(xi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            ratio = zl / y21;
            czl = ratio * yc + z1 + nz / 2;
            ratio = zr / y21;
            czr = ratio * yc + z1 + nz / 2;

            zi1 = MIN(czl, czr);
            Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr);
            Zi2 = (int)floorf(zi2);

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    n_in_row[id] += 1;
                }
            }
        }
    }
}

void count_flat_gpu(const int *shape,
                    const float SD, const float SO,
                    const float da, const float ai, const int na,
                    const float db, const float bi, const int nb,
                    int *n_in_row)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    kernel_count<<<shapeSize, blockSize>>>(0,
                                           SD, SO,
                                           nx, ny, nz,
                                           da, ai, na,
                                           db, bi, nb,
                                           n_in_row);
}

void count_cyli_gpu(const int *shape,
                    const float SD, const float SO,
                    const float da, const float ai, const int na,
                    const float db, const float bi, const int nb,
                    int *n_in_row)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    kernel_count<<<shapeSize, blockSize>>>(1,
                                           SD, SO,
                                           nx, ny, nz,
                                           da, ai, na,
                                           db, bi, nb,
                                           n_in_row);
}
#endif
