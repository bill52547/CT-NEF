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

__constant__ int const_image_shape[3];


__global__ void
DeformKernel(cudaTextureObject_t tex_img,
             const float *ax, const float *ay, const float *az,
             const float *bx, const float *by, const float *bz,
             const float *cx, const float *cy, const float *cz,
             const float *v_data, const float *f_data,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    float mx = cx[id] + v_data[0] * ax[id] + f_data[0] * bx[id];
    float my = cy[id] + v_data[0] * ay[id] + f_data[0] * by[id];
    float mz = cz[id] + v_data[0] * az[id] + f_data[0] * bz[id];
    img1[id] = tex3D<float>(tex_img, ix + mx + 0.5f, iy + my + 0.5f, iz + mz + 0.5f);
}

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
    float x1, y1, z1, x2, y2, z2, x20, y20, cphi, sphi;
    float angle = angles[iv];
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SAD * cphi;
    y1 = -SAD * sphi;
    z1 = 0.0f;
    x20 = SID - SAD;
    y20 = (ia + 0.5 - na / 2) * da + ai; // locate the detector cell center before any rotation
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    z2 = (ib + 0.5 - nb / 2) * db + bi;
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
                      const float *ax, const float *ay, const float *az,
                      const float *bx, const float *by, const float *bz,
                      const float *cx, const float *cy, const float *cz,
                      const float *v_data, const float *f_data,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      const float db, const float bi, const int nb,
                      float *pv_values)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);

    const dim3 gridSize_image((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_image = make_cudaPitchedPtr((void *)image, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_image = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_image;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_image;
    cudaArray *array_image;
    cudaMalloc3DArray(&array_image, &channelDesc, extent_image);
    copyParams.dstArray = array_image;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    resDesc.res.array.array = array_image;
    cudaTextureObject_t tex_image = 0;
    cudaCreateTextureObject(&tex_image, &resDesc, &texDesc, NULL);

    float *image_deform;
    cudaMalloc(&image_deform, nx * ny * nz * sizeof(float));
    for (int i = 0; i < nv; i++)
    {
        DeformKernel<<<gridSize_image, blockSize>>>(tex_image,
                                              ax, ay, az,
                                              bx, by, bz,
                                              cx, cy, cz,
                                              v_data + i, f_data + i,
                                              image_deform);
        kernel_projection<<<gridSize, blockSize>>>(0, angles + i, offsets + i, 1,
                                                 SID, SAD,
                                                 nx, ny, nz,
                                                 da, ai, na,
                                                 db, bi, nb,
                                                 image_deform, pv_values + i * na * nb);
    }

    cudaFree(image_deform);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_image);
    cudaFreeArray(array_image);

}


void project_cyli_gpu(const float *image, const int *shape, const float *offsets,
                      const float *ax, const float *ay, const float *az,
                      const float *bx, const float *by, const float *bz,
                      const float *cx, const float *cy, const float *cz,
                      const float *v_data, const float *f_data,
                      const float *angles, const int nv,
                      const float SID, const float SAD,
                      const float da, const float ai, const int na,
                      const float db, const float bi, const int nb,
                      float *pv_values)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_image = make_cudaPitchedPtr((void *)image, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_image = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_image;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_image;
    cudaArray *array_image;
    cudaMalloc3DArray(&array_image, &channelDesc, extent_image);
    copyParams.dstArray = array_image;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    resDesc.res.array.array = array_image;
    cudaTextureObject_t tex_image = 0;
    cudaCreateTextureObject(&tex_image, &resDesc, &texDesc, NULL);

    float *image_deform;
    cudaMalloc(&image_deform, nx * ny * nz * sizeof(float));
    for (int i = 0; i < nv; i++)
    {
        DeformKernel<<<gridSize, blockSize>>>(tex_image,
                                              ax, ay, az,
                                              bx, by, bz,
                                              cx, cy, cz,
                                              v_data + i, f_data + i,
                                              image_deform);
        kernel_projection<<<gridSize, blockSize>>>(1, angles + i, offsets + i, 1,
                                                 SID, SAD,
                                                 nx, ny, nz,
                                                 da, ai, na,
                                                 db, bi, nb,
                                                 image_deform, pv_values + i * na * nb);
    }

    cudaFree(image_deform);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_image);
    cudaFreeArray(array_image);

}


#endif
