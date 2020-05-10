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
const float eps_ = 0.01;

// mode 0 for flat, mode 1 for cylin
__global__ void kernel_projection(const float *image,
                                  const float *angles,
                                  const int mode,
                                  const float SD, const float SO,
                                  const int nx, const int ny,
                                  const float da, const float ai, const int na,
                                  float *vproj)
{
    int ia = blockIdx.x;

    if (ia >= na)
        return;
    int id = ia;
    vproj[id] = 0.0f;
    float x1, y1, x2, y2, x20, y20, cphi, sphi;
    float angle = angles[0];
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    if (mode == 0)
    {
        x20 = SD - SO;
        y20 = (ia + 0.5) * da + ai - na * da / 2;
    }
    else
    {
        float ang = (ia + 0.5) * da + ai - na * da / 2;
        x20 = SD * cosf(ang) - SO;
        y20 = SD * sinf(ang);
    }
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    float x21, y21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;

    // y - z plane, where ABS(x21) > ABS(y21)
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

            yi1 = MIN4(cyll, cylr, cyrl, cyrr);
            Yi1 = (int)floorf(yi1);
            yi2 = MAX4(cyll, cylr, cyrl, cyrr);
            Yi2 = (int)floorf(yi2);

            float wy;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1);
//                wy /= (yi2 - yi1);
                    vproj[id] += image[ix + iy * nx] * wy;
                }
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

            xi1 = MIN4(cxll, cxlr, cxrl, cxrr);
            Xi1 = (int)floorf(xi1);
            xi2 = MAX4(cxll, cxlr, cxrl, cxrr);
            Xi2 = (int)floorf(xi2);

            float wx;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1);
                    vproj[id] += image[ix + iy * nx] * wx;
                }
            }
        }
    }
}

void project_gpu(const float *image,
                 const float *angles,
                 const int mode,
                 const int nv,
                 const float SD, const float SO,
                 const int nx, const int ny,
                 const float da, const float ai, const int na,
                 float *pv_values)
{
    for (int iv = 0; iv < nv; iv++)
    {
        kernel_projection<<<na, 1>>>(image,
                                      angles + iv,
                                      mode,
                                      SD, SO,
                                      nx, ny,
                                      da, ai, na,
                                      pv_values + iv * na);
    }
}

#endif
