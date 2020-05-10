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

__global__ void
DeformKernel(cudaTextureObject_t tex_img,
             const float *mx, const float *my, const float *mz,
             const int nx, const int ny, const int nz,
             const int adding,
             float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    if (adding == 0)
    {
        img1[id] = tex3D<float>(tex_img, ix + mx[id] + 0.5f, iy + my[id] + 0.5f, iz + mz[id] + 0.5f);
    }
    else
    {
        img1[id] += tex3D<float>(tex_img, ix + mx[id] + 0.5f, iy + my[id] + 0.5f, iz + mz[id] + 0.5f);
    }
}

void deform_tex(const float *img,
                const float *mx, const float *my, const float *mz,
                const int nx, const int ny, const int nz,
                const int adding,
                float *img1)
{
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
    DeformKernel<<<gridSize, blockSize>>>(tex_img,
                                          mx, my, mz,
                                          nx, ny, nz,
                                          adding,
                                          img1);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
}

__global__ void
DeformInvertKernel(cudaTextureObject_t tex_img,
                   cudaTextureObject_t tex_mx,
                   cudaTextureObject_t tex_my,
                   cudaTextureObject_t tex_mz,
                   const int nx, const int ny, const int nz,
                   const int adding,
                   float *img1)
{
    int ix = GRIDDIM_X * blockIdx.x + threadIdx.x;
    int iy = GRIDDIM_Y * blockIdx.y + threadIdx.y;
    int iz = GRIDDIM_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0, x_, y_, z_;
    for (int iter = 0; iter < 5; iter++)
    {
        x_ = -tex3D<float>(tex_mx, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        y_ = -tex3D<float>(tex_my, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        z_ = -tex3D<float>(tex_mz, x + ix + 0.5f, y + iy + 0.5f, z + iz + 0.5f);
        x = x_;
        y = y_;
        z = z_;
    }
    if (adding == 0)
    {
        img1[id] = tex3D<float>(tex_img, ix + x + 0.5f, iy + y + 0.5f, iz + z + 0.5f);
    }
    else
    {
        img1[id] += tex3D<float>(tex_img, ix + x + 0.5f, iy + y + 0.5f, iz + z + 0.5f);
    }
}

void deform_invert_tex(const float *img,
                       const float *mx, const float *my, const float *mz,
                       const int nx, const int ny, const int nz,
                       const int adding,
                       float *img1)
{
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

    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void *)mx, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_mx;
    cudaArray *array_mx;
    cudaMalloc3DArray(&array_mx, &channelDesc, extent_img);
    copyParams.dstArray = array_mx;
    cudaMemcpy3D(&copyParams);
    resDesc.res.array.array = array_mx;
    cudaTextureObject_t tex_mx = 0;
    cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void *)my, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_my;
    cudaArray *array_my;
    cudaMalloc3DArray(&array_my, &channelDesc, extent_img);
    copyParams.dstArray = array_my;
    cudaMemcpy3D(&copyParams);
    resDesc.res.array.array = array_my;
    cudaTextureObject_t tex_my = 0;
    cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

    cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void *)mz, nx * sizeof(float), nx, ny);
    copyParams.srcPtr = dp_mz;
    cudaArray *array_mz;
    cudaMalloc3DArray(&array_mz, &channelDesc, extent_img);
    copyParams.dstArray = array_mz;
    cudaMemcpy3D(&copyParams);
    resDesc.res.array.array = array_mz;
    cudaTextureObject_t tex_mz = 0;
    cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

    DeformInvertKernel<<<gridSize, blockSize>>>(tex_img, tex_mx, tex_my, tex_mz,
                                                nx, ny, nz,
                                                adding, img1);

    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
    cudaDestroyTextureObject(tex_mz);
    cudaFreeArray(array_mz);
}
#endif
