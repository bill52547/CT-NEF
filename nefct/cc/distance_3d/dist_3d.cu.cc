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
__global__ void kernel_projection(const int mode, const float *angles, const float *offsets,
                               const int nv, const float SID, const float SAD,
                               const int nx, const int ny, const int nz,
                               const float da, const float ai, const int na,
                               const float db, const float bi, const int nb,
                               const float *image,
                               float *vproj)
{
    int ba = blockIdx.x;
    int bb = blockIdx.y;
    int bv = blockIdx.z;

    int ta = threadIdx.x;
    int tb = threadIdx.y;
    int tv = threadIdx.z;

    int ia = ba * GRIDDIM_X + ta;
    int ib = bb * GRIDDIM_Y + tb;
    int iv = bv * GRIDDIM_Z + tv;

    if (ia >= na || ib >= nb || iv >= nv)
        return;
    int id = ia + ib * na + iv * na * nb;
    vproj[id] = 0.0f;
    float x1, y1, z1, x2, y2, z2, x20, y20, cphi, sphi, z20;
    float angle = angles[iv];
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SAD * cphi;
    y1 = -SAD * sphi;
    z1 = 0.0f;
    if (mode == 0)
    {
        x20 = SID - SAD;
        y20 = (ia + 0.5) * da + ai - na * da / 2;
        z20 = (ib + 0.5) * db + bi - nb * db / 2;
    }
    else
    {
        float ang = (ia + 0.5) * da + ai - na * da / 2;
        x20 = SID * cosf(ang) - SAD;
        y20 = SID * sinf(ang);
        z20 = (ib + 0.5) * db + bi - nb * db / 2;
    }
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    z2 = z20;
    float x21, y21, z21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;
    z21 = z2 - z1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21)){
    // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2, zi1, zi2;
        int Yi1, Yi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++){
            // calculate y indices of intersecting voxel candidates
            float xl, xr, yl, yr, ratio;
            float cyll, cylr, cyrl, cyrr, xc;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;
            
            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2;
            ratio = yl / xr;
            cylr = ratio * xc + y1 + ny / 2;
            ratio = yr / xl;
            cyrl = ratio * xc + y1 + ny / 2;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2;

            yi1 = MIN4(cyll, cylr, cyrl, cyrr); Yi1 = (int)floorf(yi1);
            yi2 = MAX4(cyll, cylr, cyrl, cyrr); Yi2 = (int)floorf(yi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1 ;

            ratio = zl / x21;
            czl = ratio * xc + z1 + nz / 2;
            ratio = zr / x21;
            czr = ratio * xc + z1 + nz / 2;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2);

            float wy, wz;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1); wy /= (yi2 - yi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2) - MAX(iz + 0.0f, zi1); wz /= (zi2 - zi1);
                    vproj[id] += image[ix + iy * nx + iz * nx * ny] * wy * wz;
                }                
            }        
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)    
    else{
        float xi1, xi2, zi1, zi2;
        int Xi1, Xi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++){
            // calculate y indices of intersecting voxel candidates
            float yl, yr, xl, xr, ratio;
            float cxll, cxlr, cxrl, cxrr, yc;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;
            
            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2;
            ratio = xl / yr;
            cxlr = ratio * yc + x1 + nx / 2;
            ratio = xr / yl;
            cxrl = ratio * yc + x1 + nx / 2;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2;

            xi1 = MIN4(cxll, cxlr, cxrl, cxrr); Xi1 = (int)floorf(xi1);
            xi2 = MAX4(cxll, cxlr, cxrl, cxrr); Xi2 = (int)floorf(xi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            ratio = zl / y21;
            czl = ratio * yc + z1 + nz / 2;
            ratio = zr / y21;
            czr = ratio * yc + z1 + nz / 2;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2);

            float wx, wz;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1); wx /= (xi2 - xi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2) - MAX(iz + 0.0f, zi1); wz /= (zi2 - zi1);
                    vproj[id] += image[ix + iy * nx + iz * nx * ny] * wx * wz;
                }                
            }        
        }            
    }
}

void project_flat_gpu(const float *image, const int *shape, const float *offsets,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      const float db, const float bi, const int nb,
                      float *pv_values)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (nv + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    kernel_projection<<<shapeSize, blockSize>>>(0, angles, offsets, nv,
                                             SID, SAD,
                                             nx, ny, nz,
                                             da, ai, na,
                                             db, bi, nb,
                                             image, pv_values);
}

void project_cyli_gpu(const float *image, const int *shape, const float *offsets,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      const float db, const float bi, const int nb,
                      float *pv_values)
{
    int shape_cpu[3];
    cudaMemcpy(shape_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = shape_cpu[0], ny = shape_cpu[1], nz = shape_cpu[2]; //number of meshes

    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (nv + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    kernel_projection<<<shapeSize, blockSize>>>(1, angles, offsets, nv,
                                             SID, SAD,
                                             nx, ny, nz,
                                             da, ai, na,
                                             db, bi, nb,
                                             image, pv_values);
}
#endif
