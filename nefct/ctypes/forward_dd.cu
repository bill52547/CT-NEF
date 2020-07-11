#include "cuda.h"
#include "cuda_runtime.h"
#include <malloc.h>
#include <iostream>
#define ABS(x) ((x > 0) ? x : -(x))
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;
const float eps_ = 0.01;

using namespace std;


extern "C" void project(float *proj_values, float *image,
             float *angles, float *off_a, float *off_b, 
             int nx, int ny, int nz,
             float dx, float dy, float dz,
             float sid, const float sad,
             float da, int na,
             float db, int nb,
             int num_angles);

__global__ void kernel_projection(float *vproj, const float *image,
                                  const float *angles, const float *off_a, const float *off_b,
                                  const int nx, const int ny, const int nz, 
                                  const float dx, const float dy, const float dz,
                                  const float SD, const float SO,
                                  const float da, const int na, const float db, const int nb, const int nv)
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
    float x1, y1, z1, x2, y2, z2, x20, y20, cphi, sphi, z20;
    float ai = off_a[iv];
    float bi = off_b[iv];
    float phi = angles[iv];
    cphi = (float)__cosf(phi);
    sphi = (float)__sinf(phi);
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    z1 = 0.0f;
    x20 = SD - SO;
    y20 = (ia + 0.5) * da + ai - na * da / 2;
    z20 = (ib + 0.5) * db + bi - nb * db / 2;
    
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
            xc = ((float)ix + 0.5f - (float)nx / 2) * dx - x1;
            
            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2 * dy;
            ratio = yl / xr;
            cylr = ratio * xc + y1 + ny / 2 * dy; 
            ratio = yr / xl;
            cyrl = ratio * xc + y1 + ny / 2 * dy;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2 * dy;

            yi1 = MIN4(cyll, cylr, cyrl, cyrr); Yi1 = (int)floorf(yi1 / dy);
            yi2 = MAX4(cyll, cylr, cyrl, cyrr); Yi2 = (int)floorf(yi2 / dy);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            xc = ((float)ix + 0.5f - (float)nx / 2) * dx - x1;

            ratio = zl / x21;
            czl = ratio * xc + z1 + nz / 2 * dz;
            ratio = zr / x21;
            czr = ratio * xc + z1 + nz / 2 * dz;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1 / dz);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2 / dz);

            float wy, wz;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2 / dy) - MAX(iy + 0.0f, yi1 / dy); wy /= (yi2 - yi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2 / dz) - MAX(iz + 0.0f, zi1 / dz); wz /= (zi2 - zi1);
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
            yc = ((float)iy + 0.5f - (float)ny / 2) * dy - y1;
            
            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2 * dx;
            ratio = xl / yr;
            cxlr = ratio * yc + x1 + nx / 2 * dx;
            ratio = xr / yl;
            cxrl = ratio * yc + x1 + nx / 2 * dx;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2 * dx;

            xi1 = MIN4(cxll, cxlr, cxrl, cxrr); Xi1 = (int)floorf(xi1 / dx);
            xi2 = MAX4(cxll, cxlr, cxrl, cxrr); Xi2 = (int)floorf(xi2 / dx);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            yc = ((float)iy + 0.5f - (float)ny / 2) * dy - y1;

            ratio = zl / y21;
            czl = ratio * yc + z1 + nz / 2 * dz;
            ratio = zr / y21;
            czr = ratio * yc + z1 + nz / 2 * dz;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1 / dz);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2 / dz);

            float wx, wz;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2 / dx) - MAX(ix + 0.0f, xi1 / dx); wx /= (xi2 - xi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2 / dz) - MAX(iz + 0.0f, zi1 / dz); wz /= (zi2 - zi1);
                    vproj[id] += image[ix + iy * nx + iz * nx * ny] * wx * wz;
                }                
            }        
        }            
    }
}

void project(float *proj_values, float *image,
             float *angles, float *off_a, float *off_b, 
             int nx, int ny, int nz,
             float dx, float dy, float dz,
             float SID, const float SAD,
             float da, int na,
             float db, int nb,
             int num_angles)
{
    float *d_proj_values, *d_image, *d_angles, *d_off_a, *d_off_b;
    const int size_image = nx * ny * nz * sizeof(float);
    const int size_proj_values = na * nb * num_angles * sizeof(float);
    const int size_angles = num_angles * sizeof(float);

    cudaMalloc((void**)&d_proj_values, size_proj_values);
    cudaMemcpy(d_proj_values, proj_values, size_proj_values, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_image, size_image);
    cudaMemcpy(d_image, image, size_image, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_angles, size_angles);
    cudaMemcpy(d_angles, angles, size_angles, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_off_a, size_angles);
    cudaMemcpy(d_off_a, off_a, size_angles, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_off_b, size_angles);
    cudaMemcpy(d_off_b, off_b, size_angles, cudaMemcpyHostToDevice);


    const dim3 shapeSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, (num_angles + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    kernel_projection<<<shapeSize, blockSize>>>(d_proj_values, 
                                                d_image, 
                                                d_angles, d_off_a, d_off_b,
                                                nx, ny, nz, 
                                                dx, dy, dz,
                                                SID, SAD, 
                                                da, na, db, nb, num_angles);
    cudaMemcpy(proj_values, d_proj_values, size_proj_values, cudaMemcpyDeviceToHost);
    cudaFree(d_proj_values);
    cudaFree(d_image);
    cudaFree(d_angles);
    cudaFree(d_off_a);
    cudaFree(d_off_b);
}


