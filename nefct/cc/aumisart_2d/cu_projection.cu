#include "cu_projection.h"
__host__ void host2_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny)
{
    
}

__host__ void host_projection(float *d_proj, float *d_img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny)
{
    const dim3 gridSize_singleProj((na + BLOCKSIZE_X - 1) / BLOCKSIZE_X, 1, 1);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, 1);
    kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img, angle, SO, SD, da, na, ai, nx, ny);
    cudaDeviceSynchronize();
}

__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny){
    int ia = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    if (ia >= na)
        return;
    int id = ia;
    proj[id] = 0.0f;
    float x1, y1, x2, y2, x20, y20, cphi, sphi;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    x20 = SD - SO;
    y20 = (ia + ai) * da; // locate the detector cell center before any rotation
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    float x21, y21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21)){
    // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2;
        int Yi1, Yi2;
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

            xc = (float)ix + 0.5f - (float)nx / 2 - x1 ;

            float wy;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1); wy /= (yi2 - yi1);
                proj[id] += img[ix + iy * nx] * wy / ABS(x21) * sqrt(x21 * x21 + y21 * y21);                
            }        
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)    
    else{
        float xi1, xi2;
        int Xi1, Xi2;
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

            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            float wx;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1); wx /= (xi2 - xi1);
                proj[id] += img[ix + iy * nx] * wx / ABS(y21) * sqrt(x21 * x21 + y21 * y21);                
            }        
        }            
    }
}