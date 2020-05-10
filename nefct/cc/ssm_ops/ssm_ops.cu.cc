#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cusparse.h>
#define abs(x) ((x) < 0 ? (-x) : (x))

#define BLOCKDIM 1024

__global__ void SparseMatVecMulFull(cudaTextureObject_t tex_img,
                                    const float *data, const int *indices, const int *indptr,
                                    const float *cos_phi, const float *sin_phi, const float *distances,
                                    const int na, const int nb, const int nv,
                                    const int nx, const int ny, const int nz,
                                    float *projection)
{
    int idet = blockIdx.x;
    int iv = threadIdx.x;
    if (idet >= na * nb || iv >= nv)
    {
        return;
    }
    int id = idet + iv * na * nb;
    projection[id] = 0.0f;
    const float cphi = cos_phi[iv], sphi = sin_phi[iv];
    float ix_, iy_, iz_, ix, iy, iz;

    for (int i = indptr[idet]; i < indptr[idet + 1]; i++)
    {
        iz = indices[i] / nx / ny;
        iy = (indices[i] / nx) % ny;
        ix = indices[i] % nx;
        ix_ = (ix + 0.5 - nx / 2) * cphi - (iy + 0.5 - ny / 2) * sphi + nx / 2 - 0.5;
        iy_ = (ix + 0.5 - nx / 2) * sphi + (iy + 0.5 - ny / 2) * cphi + ny / 2 - 0.5;
        iz_ = iz + distances[iv];
        projection[id] += tex3D<float>(tex_img, ix_ + 0.5, iy_ + 0.5, iz_ + 0.5) * data[i];
    }
}

__global__ void calculate_triangles(const float *angles, const int nv,
                                    float *cos_phi, float *sin_phi)
{
    int iv = blockIdx.x;
    cos_phi[iv] = cosf(angles[iv]);
    sin_phi[iv] = sinf(angles[iv]);
}

void project_gpu(const float *img,
                 const float *data, const int *indices, const int *indptr,
                 const float *angles, const float *distances,
                 const int na, const int nb, const int nv,
                 const int nx, const int ny, const int nz,
                 float *projection)
{
    // SparseMatVecMul<<<gridSize, blockSize>>>(img, data, indices, indptr, n_det, projection + n_det * ind);
    // cudaDeviceSynchronize();
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

    float *cos_phi, *sin_phi;
    cudaMalloc((void **)&cos_phi, nv * sizeof(float));
    cudaMalloc((void **)&sin_phi, nv * sizeof(float));
    calculate_triangles<<<nv, 1>>>(angles, nv, cos_phi, sin_phi);

    if (nv <= BLOCKDIM)
    {
        SparseMatVecMulFull<<<na * nb, nv>>>(tex_img,
                                             data, indices, indptr,
                                             cos_phi, sin_phi, distances,
                                             na, nb, nv,
                                             nx, ny, nz,
                                             projection);
    }
    else
    {
        int num = (nv + BLOCKDIM - 1) / BLOCKDIM;
        int nv0 = BLOCKDIM, accu_nv = 0;
        for (int i = 0; i < num; i++)
        {
            if (i == num - 1)
            {
                nv0 = nv % BLOCKDIM;
            }
            SparseMatVecMulFull<<<na * nb, BLOCKDIM>>>(tex_img,
                                                       data, indices, indptr,
                                                       cos_phi + accu_nv, sin_phi + accu_nv, distances + accu_nv,
                                                       na, nb, nv0, nx, ny, nz,
                                                       projection + accu_nv * na * nb);
            accu_nv += nv0;
        }
    }

    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);
    cudaFree(cos_phi);
    cudaFree(sin_phi);
}

#endif
