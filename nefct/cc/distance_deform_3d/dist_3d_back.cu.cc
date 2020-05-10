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

__constant__ int const_image_shape[3];


__device__ float interp3d(const float *mx,
                          const float x, const float y, const float z,
                          const int nx, const int ny, const int nz)
{
    if (x < 1 || x > nx - 2) {return 0;}
    if (y < 1 || y > ny - 2) {return 0;}
    if (z < 1 || z > nz - 2) {return 0;}
    int ix1 = (int)floorf(x); int ix2 = 1 + ix1;
    int iy1 = (int)floorf(y); int iy2 = 1 + iy1;
    int iz1 = (int)floorf(z); int iz2 = 1 + iz1;
    float wx2 = x - ix1; float wx1 = 1 - wx2;
    float wy2 = y - iy1; float wy1 = 1 - wy2;
    float wz2 = z - iz1; float wz1 = 1 - wz2;
    return mx[ix1 + iy1 * nx + iz1 * nx * ny] * wx1 * wy1 * wz1 +
           mx[ix2 + iy1 * nx + iz1 * nx * ny] * wx2 * wy1 * wz1 +
           mx[ix1 + iy2 * nx + iz1 * nx * ny] * wx1 * wy2 * wz1 +
           mx[ix2 + iy2 * nx + iz1 * nx * ny] * wx2 * wy2 * wz1 +
           mx[ix1 + iy1 * nx + iz2 * nx * ny] * wx1 * wy1 * wz2 +
           mx[ix2 + iy1 * nx + iz2 * nx * ny] * wx2 * wy1 * wz2 +
           mx[ix1 + iy2 * nx + iz2 * nx * ny] * wx1 * wy2 * wz2 +
           mx[ix2 + iy2 * nx + iz2 * nx * ny] * wx2 * wy2 * wz2;
}

__global__ void
DeformInvertKernel2(const float *mx, const float *my, const float *mz,
                   const int nx, const int ny, const int nz,
                   float *mx2, float *my2, float *mz2)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    float x_, y_, z_;
    for (int iter = 0; iter < 10; iter++)
    {
        x = -interp3d(mx, x + ix, y + iy, z + iz, nx, ny, nz);
        y = -interp3d(my, x + ix, y + iy, z + iz, nx, ny, nz);
        z = -interp3d(mz, x + ix, y + iy, z + iz, nx, ny, nz);
//        x = x_; y = y_; z = z_;
    }
    mx2[id] = x;
    my2[id] = y;
    mz2[id] = z;
}

__global__ void
DeformInvertKernel(cudaTextureObject_t tex_mx,
                   cudaTextureObject_t tex_my,
                   cudaTextureObject_t tex_mz,
                   const int nx, const int ny, const int nz,
                   float *mx, float *my, float *mz)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    mx[id] = tex3D<float>(tex_mx, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
    return;

    for (int iter = 0; iter < 10; iter++)
    {
        x = -tex3D<float>(tex_mx, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        y = -tex3D<float>(tex_my, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        z = -tex3D<float>(tex_mz, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
    }
    mx[id] = x;
    my[id] = y;
    mz[id] = z;
}

__global__ void AddToDvf(const float *ax, const float *bx, const float *cx,
                         const float *v_data, const float *f_data,
                         const int nx, const int ny, const int nz,
                         float *mx)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    mx[id] = ax[id] * v_data[0] + bx[id] * f_data[0] + cx[id];
//    mx[id] = v_data[0] + f_data[0];
}

__global__ void
DeformKernel(cudaTextureObject_t tex_img,
             const float *mx, const float *my, const float *mz,
             const int nx, const int ny, const int nz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img1[id] += tex3D<float>(tex_img, ix + mx[id] + 0.5f, iy + my[id] + 0.5f, iz + mz[id] + 0.5f);
}


__global__ void
DeformKernel2(const float *img,
             const float *mx, const float *my, const float *mz,
             const int nx, const int ny, const int nz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img1[id] += interp3d(img, mx[id] + ix, my[id] + iy, mz[id] + iz, nx, ny, nz);
}


__global__ void
InitialKernel(float *img, const int nx, const int ny, const int nz)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    img[id] = 0.0f;
}

__host__ void invert(const float *mx, const float *my, const float *mz,
                     const int nx, const int ny, const int nz,
                     float *mx2, float *my2, float *mz2)
{
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void *)mx, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void *)my, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void *)mz, nx * sizeof(float), nx, ny);

    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    copyParams.srcPtr = dp_mx;
    cudaArray *array_mx;
    cudaMalloc3DArray(&array_mx, &channelDesc, extent);
    copyParams.dstArray = array_mx;
    cudaMemcpy3D(&copyParams);

    copyParams.srcPtr = dp_my;
    cudaArray *array_my;
    cudaMalloc3DArray(&array_my, &channelDesc, extent);
    copyParams.dstArray = array_my;
    cudaMemcpy3D(&copyParams);

    copyParams.srcPtr = dp_mz;
    cudaArray *array_mz;
    cudaMalloc3DArray(&array_mz, &channelDesc, extent);
    copyParams.dstArray = array_mz;
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

    resDesc.res.array.array = array_mx;
    cudaTextureObject_t tex_mx = 0;
    cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

    resDesc.res.array.array = array_my;
    cudaTextureObject_t tex_my = 0;
    cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

    resDesc.res.array.array = array_mz;
    cudaTextureObject_t tex_mz = 0;
    cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

    DeformInvertKernel<<<gridSize, blockSize>>>(tex_mx, tex_my, tex_mz, nx, ny, nz, mx2, my2, mz2);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
    cudaDestroyTextureObject(tex_mz);
    cudaFreeArray(array_mz);
}


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

        // get the max and min values of all boundary projectors of voxel boundaries on detector plane
        // a_max = MAX4(al ,ar, at, ab);
        // a_min = MIN4(al ,ar, at, ab);
        // b_max = MAX4(bl ,br, bt, bb);
        // b_min = MIN4(bl ,br, bt, bb);
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
            // bin_bound_1 = ((float)ia + ai) * da;
            // bin_bound_2 = ((float)ia + ai + 1.0f) * da;
            bin_bound_1 = ia + 0.0f;
            bin_bound_2 = ia + 1.0f;

            wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);// wa /= a_max - a_min;

            for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib ++){
                // bin_bound_1 = ((float)ib + bi) * db;
                // bin_bound_2 = ((float)ib + bi + 1.0f) * db;
                bin_bound_1 = ib + 0.0f;
                bin_bound_2 = ib + 1.0f;
                // wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= db;
                wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= b_max - b_min;


                image[id] += wa * wb * vproj[ia + ib * na + iv * na * nb];
            }
        }
    }
}


void back_project_flat_gpu(const float *pv_values, const int *shape, const float *offsets,
                           const float *ax, const float *ay, const float *az,
                           const float *bx, const float *by, const float *bz,
                           const float *cx, const float *cy, const float *cz,
                           const float *v_data, const float *f_data,
                           const float *angles, const int nv,
                           const float SID, const float SAD,
                           const float da, const float ai, const int na,
                           const float db, const float bi, const int nb,
                           float *image)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1], nz = grid_cpu[2]; //number of meshes
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    const dim3 gridSize_image((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                    (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                    (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

//    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//    cudaArray *array_image;
//    cudaPitchedPtr dp_image;

//    cudaArray *array_mx;
//    cudaArray *array_my;
//    cudaArray *array_mz;
//
//    cudaTextureObject_t tex_mx = 0;
//    cudaTextureObject_t tex_my = 0;
//    cudaTextureObject_t tex_mz = 0;


    float *image_deform, *mx, *my, *mz, *mx2, *my2, *mz2;
    cudaMalloc(&image_deform, nx * ny * nz * sizeof(float));
    cudaMalloc(&mx, nx * ny * nz * sizeof(float));
    cudaMalloc(&my, nx * ny * nz * sizeof(float));
    cudaMalloc(&mz, nx * ny * nz * sizeof(float));
    cudaMalloc(&mx2, nx * ny * nz * sizeof(float));
    cudaMalloc(&my2, nx * ny * nz * sizeof(float));
    cudaMalloc(&mz2, nx * ny * nz * sizeof(float));
    for (int i = 0; i < nv; i++)
    {
        InitialKernel<<<gridSize_image, blockSize>>>(image_deform, nx, ny, nz);
        kernel_backprojection<<<gridSize_image, blockSize>>>(0, angles + i, offsets + i, 1,
                                              SID, SAD,
                                              nx, ny, nz,
                                              da, ai, na,
                                              db, bi, nb,
                                              pv_values + i * na * nb, image_deform);
//        dp_image = make_cudaPitchedPtr((void *)image_deform, nx * sizeof(float), nx, ny);
//        cudaMemcpy3DParms copyParams = {0};
//        struct cudaExtent extent_image = make_cudaExtent(nx, ny, nz);
//        copyParams.extent = extent_image;
//        copyParams.kind = cudaMemcpyDeviceToDevice;
//        copyParams.srcPtr = dp_image;
//
//        cudaMalloc3DArray(&array_image, &channelDesc, extent_image);
//        copyParams.dstArray = array_image;
//        cudaMemcpy3D(&copyParams);
//
//        cudaResourceDesc resDesc;
//        cudaTextureDesc texDesc;
//        memset(&resDesc, 0, sizeof(resDesc));
//        resDesc.resType = cudaResourceTypeArray;
//        memset(&texDesc, 0, sizeof(texDesc));
//        texDesc.addressMode[0] = cudaAddressModeClamp;
//        texDesc.addressMode[1] = cudaAddressModeClamp;
//        texDesc.addressMode[2] = cudaAddressModeClamp;
//        texDesc.filterMode = cudaFilterModeLinear;
//        texDesc.readMode = cudaReadModeElementType;
//        texDesc.normalizedCoords = 0;
//        resDesc.res.array.array = array_image;
//        cudaTextureObject_t tex_image = 0;
//        cudaCreateTextureObject(&tex_image, &resDesc, &texDesc, NULL);

        AddToDvf<<<gridSize_image, blockSize>>>(ax, bx, cx, v_data + i, f_data + i, nx, ny, nz, mx);
        AddToDvf<<<gridSize_image, blockSize>>>(ay, by, cy, v_data + i, f_data + i, nx, ny, nz, my);
        AddToDvf<<<gridSize_image, blockSize>>>(az, bz, cz, v_data + i, f_data + i, nx, ny, nz, mz);

//        cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void *)mx, nx * sizeof(float), nx, ny);
//        copyParams.srcPtr = dp_mx;
//        cudaMalloc3DArray(&array_mx, &channelDesc, extent_image);
//        copyParams.dstArray = array_mx;
//        cudaMemcpy3D(&copyParams);
//
//        cudaPitchedPtr dp_my = make_cudaPitchedPtr((void *)my, nx * sizeof(float), nx, ny);
//        copyParams.srcPtr = dp_my;
//        cudaMalloc3DArray(&array_my, &channelDesc, extent_image);
//        copyParams.dstArray = array_my;
//        cudaMemcpy3D(&copyParams);
//
//        cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void *)mz, nx * sizeof(float), nx, ny);
//        copyParams.srcPtr = dp_mz;
//        cudaMalloc3DArray(&array_mz, &channelDesc, extent_image);
//        copyParams.dstArray = array_mz;
//        cudaMemcpy3D(&copyParams);
//
//        resDesc.res.array.array = array_mx;
//        cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);
//
//        resDesc.res.array.array = array_my;
//        cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);
//
//        resDesc.res.array.array = array_mz;
//        cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);
//

//        DeformInvertKernel<<<gridSize_image, blockSize>>>(tex_mx, tex_my, tex_mz, nx, ny, nz, image,
//        my2, mz2);
//        cudaDeviceSynchronize();
        DeformInvertKernel2<<<gridSize_image, blockSize>>>(mx, my, mz, nx, ny, nz, mx2, my2, mz2);

//        invert(mx, my, mz, nx, ny, nz, image, my2, mz2);break;
//        DeformKernel<<<gridSize_image, blockSize>>>(tex_image, mx2, my2, mz2, nx, ny, nz, image);
        DeformKernel2<<<gridSize_image, blockSize>>>(image_deform, mx2, my2, mz2, nx, ny, nz, image);
        cudaDeviceSynchronize();
//        cudaDestroyTextureObject(tex_image);
    }
    cudaFree(image_deform);

//    cudaDestroyTextureObject(tex_image);
//    cudaFreeArray(array_image);
    cudaFree(mx);
    cudaFree(my);
    cudaFree(mz);
    cudaFree(mx2);
    cudaFree(my2);
    cudaFree(mz2);
//
//    cudaDestroyTextureObject(tex_mx);
//    cudaFreeArray(array_mx);
//    cudaDestroyTextureObject(tex_my);
//    cudaFreeArray(array_my);
//    cudaDestroyTextureObject(tex_mz);
//    cudaFreeArray(array_mz);
}

void back_project_cyli_gpu(const float *pv_values, const int *shape, const float *offsets,
                           const float *ax, const float *ay, const float *az,
                           const float *bx, const float *by, const float *bz,
                           const float *cx, const float *cy, const float *cz,
                           const float *v_data, const float *f_data,
                           const float *angles, const int nv,
                           const float SID, const float SAD,
                           const float da, const float ai, const int na,
                           const float db, const float bi, const int nb,
                           float *image)
{
     int grid_cpu[3];
    cudaMemcpy(grid_cpu, shape, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int nx = grid_cpu[0], ny = grid_cpu[1], nz = grid_cpu[2]; //number of meshes
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((na + GRIDDIM_X - 1) / GRIDDIM_X, (nb + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    const dim3 gridSize_image((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                    (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                    (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaTextureObject_t tex_image = 0;
    cudaArray *array_image;
    cudaPitchedPtr dp_image;
    float *image_deform, *mx, *my, *mz, *mx2, *my2, *mz2;
    cudaMalloc(&image_deform, nx * ny * nz * sizeof(float));
    cudaMalloc(&mx, nx * ny * nz * sizeof(float));
    cudaMalloc(&my, nx * ny * nz * sizeof(float));
    cudaMalloc(&mz, nx * ny * nz * sizeof(float));
    cudaMalloc(&mx2, nx * ny * nz * sizeof(float));
    cudaMalloc(&my2, nx * ny * nz * sizeof(float));
    cudaMalloc(&mz2, nx * ny * nz * sizeof(float));
    for (int i = 0; i < nv; i++)
    {
        InitialKernel<<<gridSize_image, blockSize>>>(image_deform, nx, ny, nz);
        kernel_backprojection<<<gridSize, blockSize>>>(1, angles + i, offsets + i, 1,
                                              SID, SAD,
                                              nx, ny, nz,
                                              da, ai, na,
                                              db, bi, nb,
                                              pv_values + i * na * nb, image_deform);
        dp_image = make_cudaPitchedPtr((void *)image_deform, nx * sizeof(float), nx, ny);
        cudaMemcpy3DParms copyParams = {0};
        struct cudaExtent extent_image = make_cudaExtent(nx, ny, nz);
        copyParams.extent = extent_image;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.srcPtr = dp_image;

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

        cudaCreateTextureObject(&tex_image, &resDesc, &texDesc, NULL);
        AddToDvf<<<gridSize_image, blockSize>>>(ax, bx, cx, v_data + i, f_data + i, nx, ny, nz, mx);
        AddToDvf<<<gridSize_image, blockSize>>>(ay, by, cy, v_data + i, f_data + i, nx, ny, nz, my);
        AddToDvf<<<gridSize_image, blockSize>>>(az, bz, cz, v_data + i, f_data + i, nx, ny, nz, mz);

        invert(mx, my, mz, nx, ny, nz, mx2, my2, mz2);

        DeformKernel<<<gridSize_image, blockSize>>>(tex_image, mx2, my2, mz2, nx, ny, nz, image);
        cudaDeviceSynchronize();
    }
    cudaFree(image_deform);

    cudaDestroyTextureObject(tex_image);
    cudaFreeArray(array_image);
    cudaFree(mx);
    cudaFree(my);
    cudaFree(mz);
    cudaFree(mx2);
    cudaFree(my2);
    cudaFree(mz2);
}

#endif
