#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x) < 0 ? (-x) : (x))

const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;

__constant__ int const_image_shape[2];

__global__ void
DeformKernel(cudaTextureObject_t tex_img,
             const float *mx, const float *my,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1])
        return;
    int id = ix + iy * const_image_shape[0];
    img1[id] = tex3D<float>(tex_img, ix + mx[id] + 0.5f, iy + my[id] + 0.5f, 0.5f);
}

void deform_2d_tex(const float *img,
                const float *mx, const float *my,
                const int *grid, float *img1)
{
    int grid_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, 1);
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
    DeformKernel<<<gridSize, blockSize>>>(tex_img, mx, my, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

__global__ void
DeformInvertKernel(cudaTextureObject_t tex_mx,
                   cudaTextureObject_t tex_my,
                   float *mx, float *my)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    if (ix >= const_image_shape[0] || iy >= const_image_shape[1])
        return;
    int id = ix + iy * const_image_shape[0];
    float x = 0, y = 0;
    for (int iter = 0; iter < 30; iter++)
    {
        x = -tex3D<float>(tex_mx, x + ix + 0.5f, y + iy + 0.5f, 0.5f);
        y = -tex3D<float>(tex_my, x + ix + 0.5f, y + iy + 0.5f, 0.5f);
    }
    mx[id] = x;
    my[id] = y;
}

__host__ void invert(const float *mx, const float *my,
                     const int nx, const int ny,
                     float *mx2, float *my2)
{
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X, (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void *)mx, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void *)my, nx * sizeof(float), nx, ny);

    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent = make_cudaExtent(nx, ny, 1);
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

    DeformInvertKernel<<<gridSize, blockSize>>>(tex_mx, tex_my, mx2, my2);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
}

void deform_invert_2d_tex(const float *img,
                       const float *mx, const float *my,
                       const int *grid, float *img1)
{
    int grid_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    const int nx = grid_cpu[0];
    const int ny = grid_cpu[1];
    cudaMemcpyToSymbol(const_image_shape, &grid_cpu, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
    const dim3 gridSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                        (ny + GRIDDIM_Y - 1) / GRIDDIM_Y, 1);

    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, 1);

    float *mx2, *my2;
    cudaMalloc((void **)&mx2, nx * ny * sizeof(float));
    cudaMalloc((void **)&my2, nx * ny * sizeof(float));
    invert(mx, my, nx, ny, mx2, my2);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)img, nx * sizeof(float), nx, ny);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, 1);
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
    DeformKernel<<<gridSize, blockSize>>>(tex_img, mx2, my2, img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
    cudaFree(mx2);
    cudaFree(my2);
}

#endif
