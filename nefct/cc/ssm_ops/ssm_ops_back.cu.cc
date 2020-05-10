#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cusparse.h>
#define abs(x) ((x) < 0 ? (-x) : (x))

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 4
__global__ void KernelBackProj(const float *proj,
                               const float *data, const int *indices, const int *indptr,
                               const int na, const int nb,
                               const int nx, const int ny, const int nz,
                               float *image)
{
    const int ix = threadIdx.x + blockIdx.x * BLOCKDIM_X;
    const int iy = threadIdx.y + blockIdx.y * BLOCKDIM_Y;
    const int iz = threadIdx.z + blockIdx.z * BLOCKDIM_Z;
    if (ix >= nx || iy >= ny || iz >= nz)
    {
        return;
    }
    const int ivoxel = ix + iy * nx + iz * nx * ny;
    image[ivoxel] = 0.0f;

    int ia, ib;

    for (int i = indptr[ivoxel]; i < indptr[ivoxel + 1]; i++)
    {
        ia = indices[i] % na;
        ib = indices[i] / na;
        image[ivoxel] += proj[ia + ib * na] * data[i];
    }
}

__global__ void deform_add(cudaTextureObject_t tex_img,
                           const float cphi, const float sphi, const float distance,
                           const int nx, const int ny, const int nz,
                           float *image1)
{
    const int ix = threadIdx.x + blockIdx.x * BLOCKDIM_X;
    const int iy = threadIdx.y + blockIdx.y * BLOCKDIM_Y;
    const int iz = threadIdx.z + blockIdx.z * BLOCKDIM_Z;
    if (ix >= nx || iy >= ny || iz >= nz)
    {
        return;
    }
    const int ix_ = (ix - nx / 2 + 0.5) * cphi + (iy - ny / 2 + 0.5) * sphi + nx / 2 - 0.5;
    const int iy_ = -(ix - nx / 2 + 0.5) * sphi + (iy - ny / 2 + 0.5) * cphi + ny / 2 - 0.5;
    const int iz_ = iz - distance;
    image1[ix + iy * nx + iz * nx * ny] += tex3D<float>(tex_img, ix_ + 0.5, iy_ + 0.5, iz_ + 0.5);
}

void back_project_gpu(const float *proj,
                      const float *data, const int *indices, const int *indptr,
                      const float *angles, const float *distances,
                      const int na, const int nb, const int nv,
                      const int nx, const int ny, const int nz,
                      float *image)
{
    const dim3 gridSize((nx + BLOCKDIM_X - 1) / BLOCKDIM_X,
                        (ny + BLOCKDIM_Y - 1) / BLOCKDIM_Y,
                        (nz + BLOCKDIM_Z - 1) / BLOCKDIM_Z);

    const dim3 blockSize(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
    float host_angles[nv], host_distances[nv], *image1;
    cudaMemcpy(host_angles, angles, nv * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_distances, distances, nv * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMalloc((void **)&image1, nx * ny * nz * sizeof(float));
    for (int iv = 0; iv < nv; iv++)
    {
        KernelBackProj<<<gridSize, blockSize>>>(proj + na * nb * iv,
                                                data, indices, indptr,
                                                na, nb,
                                                nx, ny, nz,
                                                image1);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaPitchedPtr dp_img = make_cudaPitchedPtr((void *)(image1), nx * sizeof(float), nx, ny);
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
        deform_add<<<gridSize, blockSize>>>(tex_img,
                                            cos(host_angles[iv]), sin(host_angles[iv]), host_distances[iv],
                                            nx, ny, nz,
                                            image);
        cudaDestroyTextureObject(tex_img);
        cudaFreeArray(array_img);
    }
    cudaFree(image1);
}
#endif
