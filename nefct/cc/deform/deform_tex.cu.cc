#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x) < 0 ? (-x) : (x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;

__constant__ int const_image_shape[3];

__global__ void
DeformKernel(cudaTextureObject_t tex_img,
             const float *mx, const float *my, const float *mz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    img1[id] = tex3D<float>(tex_img, ix + mx[id] + 0.5f, iy + my[id] + 0.5f, iz + mz[id] + 0.5f);
}

void deform_tex(const float *img,
                const float *mx, const float *my, const float *mz,
                const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    DeformKernel<<<gridSize, blockSize>>>(tex_img, mx, my, mz, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

__global__ void
DeformInvertKernel(cudaTextureObject_t tex_mx,
                   cudaTextureObject_t tex_my,
                   cudaTextureObject_t tex_mz,
                   float *mx, float *my, float *mz)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    float x = 0, y = 0, z = 0;
    for (int iter = 0; iter < 30; iter++)
    {
        x = -tex3D<float>(tex_mx, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        y = -tex3D<float>(tex_my, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        z = -tex3D<float>(tex_mz, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
    }
    mx[id] = x;
    my[id] = y;
    mz[id] = z;
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

    DeformInvertKernel<<<gridSize, blockSize>>>(tex_mx, tex_my, tex_mz, mx2, my2, mz2);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
    cudaDestroyTextureObject(tex_mz);
    cudaFreeArray(array_mz);
}

void deform_invert_tex(const float *img,
                       const float *mx, const float *my, const float *mz,
                       const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    float *mx2, *my2, *mz2;
    cudaMalloc((void **)&mx2, nx * ny * nz * sizeof(float));
    cudaMalloc((void **)&my2, nx * ny * nz * sizeof(float));
    cudaMalloc((void **)&mz2, nx * ny * nz * sizeof(float));
    invert(mx, my, mz, nx, ny, nz, mx2, my2, mz2);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    DeformKernel<<<gridSize, blockSize>>>(tex_img, mx2, my2, mz2, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
    cudaFree(mx2);
    cudaFree(my2);
    cudaFree(mz2);
}

__global__ void
ShiftKernel(cudaTextureObject_t tex_img,
            const float mx, const float my, const float mz,
            float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    img1[id] = tex3D<float>(tex_img, ix - mx + 0.5f, iy - my + 0.5f, iz - mz + 0.5f);
}

void shift_tex(const float *img,
               const float mx, const float my, const float mz,
               const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    ShiftKernel<<<gridSize, blockSize>>>(tex_img, mx, my, mz, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

__global__ void
ShiftZKernel(cudaTextureObject_t tex_img,
             const float mz,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    img1[id] = tex3D<float>(tex_img, ix + 0.5f, iy + 0.5f, iz - mz + 0.5f);
}


void shift_z_tex(const float *img,
                 const float mz,
                 const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    ShiftZKernel<<<gridSize, blockSize>>>(tex_img, mz, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

__global__ void
RotateKernel(cudaTextureObject_t tex_img,
             const float cos_phi, const float sin_phi,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    int nx = const_image_shape[0];
    int ny = const_image_shape[1];
    float mx = (ix + 0.5f - nx / 2) * cos_phi - (iy + 0.5f - ny / 2) * sin_phi + nx / 2;
    float my = (ix + 0.5f - nx / 2) * sin_phi + (iy + 0.5f - ny / 2) * cos_phi + ny / 2;
    img1[id] = tex3D<float>(tex_img, mx, my, iz + 0.5f);
}

void rotate_tex(const float *img,
                const float cos_phi, const float sin_phi,
                const int *grid, float *img1)
{
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                        (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_img;
    cudaArray *array_img;
    cudaMalloc3DArray(&array_img, &channelDesc, extent_img);
    copyParams.dstArray = array_img;
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
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img = 0;
    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    RotateKernel<<<gridSize, blockSize>>>(tex_img, cos_phi, sin_phi, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

#endif
