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
RotateKernel(cudaTextureObject_t tex_img,
             const float cphi, const float sphi,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    const int nx = const_image_shape[0], ny = const_image_shape[1], nz = const_image_shape[2];
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    float ix_ = (ix + 0.5 - nx / 2) * cphi + (iy + 0.5 - ny / 2) * sphi + nx / 2 - 0.5;
    float iy_ = -(ix + 0.5 - nx / 2) * sphi + (iy + 0.5 - ny / 2) * cphi + ny / 2 - 0.5;
    img1[id] = tex3D<float>(tex_img, ix_ + 0.5f, iy_ + 0.5f, iz + 0.5f);
}

__global__ void
ShiftKernel(cudaTextureObject_t tex_img,
            const float distance,
            float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    const int nx = const_image_shape[0], ny = const_image_shape[1], nz = const_image_shape[2];
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1] || iz >= const_image_shape[2])
        return;
    int id = ix + iy * const_image_shape[0] + iz * const_image_shape[0] * const_image_shape[1];
    img1[id] = tex3D<float>(tex_img, ix + 0.5f, iy + 0.5f, iz + 0.5f - distance);
}

void rotate_tex(const float *img,
                const int *grid, const float angle,
                float *img1)
{
    const float cphi = cosf(angle);
    const float sphi = sinf(angle);
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
    RotateKernel<<<gridSize, blockSize>>>(tex_img, cphi, sphi, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

void shift_tex(const float *img,
               const int *grid, const float distance,
               float *img1)
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
    ShiftKernel<<<gridSize, blockSize>>>(tex_img, distance, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

// rotate and shift
void rs_tex(const float *img,
            const int *grid, const float angle, const float distance,
            float *img2)
{
    const float cphi = cosf(angle);
    const float sphi = sinf(angle);
    int grid_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    const int nz = grid_cpu[2];
    float * img1;
    cudaMalloc((void**)&img1, nx * ny * nz * sizeof(float));
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
    RotateKernel<<<gridSize, blockSize>>>(tex_img, cphi, sphi, img1);
    cudaDeviceSynchronize();

    cudaPitchedPtr dp_img1 = make_cudaPitchedPtr((void *)img1, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_img1;
    cudaMemcpy3D(&copyParams);
    resDesc.res.array.array = array_img;
    cudaTextureObject_t tex_img1 = 0;
    cudaCreateTextureObject(&tex_img1, &resDesc, &texDesc, NULL);
    ShiftKernel<<<gridSize, blockSize>>>(tex_img1, distance, img2);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(tex_img);
    cudaDestroyTextureObject(tex_img1);
    cudaFreeArray(array_img);
    cudaFree(img1);
}

#endif
