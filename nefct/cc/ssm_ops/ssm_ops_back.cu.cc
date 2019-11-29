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
__global__ void KernelBackProj(cudaTextureObject_t tex_proj,
                               const float *data, const int *indices, const int *indptr,
                               const float *angles, const float *distances,
                               const int na, const int nb, const int nv,
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
    float ix_, iy_, iz_, wx1, wx2, wy1, wy2, wz1, wz2;
    int ix1, iy1, iz1, ix2, iy2, iz2, ivoxel1;
    for (int iv = 0; iv < nv; iv++)
    {
        const float cphi = cosf(angles[iv]), sphi = sinf(angles[iv]);
        ix_ = (ix + 0.5 - nx / 2) * cphi + (iy + 0.5 - ny / 2) * sphi + nx / 2 - 0.5;
        iy_ = -(ix + 0.5 - nx / 2) * sphi + (iy + 0.5 - ny / 2) * cphi + ny / 2 - 0.5;
        iz_ = iz + distances[iv];
        ix1 = (int)floorf(ix_);
        ix2 = ix1 + 1;
        iy1 = (int)floorf(iy_);
        iy2 = iy1 + 1;
        iz1 = (int)floorf(iz_);
        iz2 = iz1 + 1;
        if (ix1 >= nx - 1 || ix1 < 0 || iy1 >= ny - 1 || iy1 < 0 || iz1 >= nz - 1 || iz1 < 0)
        {
            continue;
        }

        wx2 = ix_ - ix1;
        wx1 = 1.0 - wx2;
        wy2 = iy_ - iy1;
        wy1 = 1.0 - wy2;
        wz2 = iz_ - iz1;
        wz1 = 1.0 - wz2;
        int ia, ib;

        ivoxel1 = ix1 + iy1 * nx + iz1 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx1 * wy1 * wz1 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix2 + iy1 * nx + iz1 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx2 * wy1 * wz1 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix1 + iy2 * nx + iz1 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx1 * wy2 * wz1 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix2 + iy2 * nx + iz1 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx2 * wy2 * wz1 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix1 + iy1 * nx + iz2 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx1 * wy1 * wz2 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix2 + iy1 * nx + iz2 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx2 * wy1 * wz2 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix1 + iy2 * nx + iz2 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx1 * wy2 * wz2 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }

        ivoxel1 = ix2 + iy2 * nx + iz2 * nx * ny;
        for (int i = indptr[ivoxel1]; i < indptr[ivoxel1 + 1]; i++)
        {
            ia = indices[i] % na;
            ib = indices[i] / na;
            image[ivoxel] += wx2 * wy2 * wz2 * tex3D<float>(tex_proj, ia, ib, iv + 0.5) * data[i];
        }
    }
}

void back_project_gpu(const float *proj,
                      const float *data, const int *indices, const int *indptr,
                      const float *angles, const float *distances,
                      const int na, const int nb, const int nv,
                      const int nx, const int ny, const int nz,
                      float *image)
{
    // SparseMatVecMul<<<gridSize, blockSize>>>(img, data, indices, indptr, n_det, projection + n_det * ind);
    // cudaDeviceSynchronize();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_proj = make_cudaPitchedPtr((void *)proj, na * sizeof(float), na, nb);
    cudaMemcpy3DParms copyParams = {0};
    struct cudaExtent extent_proj = make_cudaExtent(na, nb, nv);
    copyParams.extent = extent_proj;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.srcPtr = dp_proj;
    cudaArray *array_proj;
    cudaMalloc3DArray(&array_proj, &channelDesc, extent_proj);
    copyParams.dstArray = array_proj;
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
    resDesc.res.array.array = array_proj;
    cudaTextureObject_t tex_proj = 0;
    cudaCreateTextureObject(&tex_proj, &resDesc, &texDesc, NULL);

    const dim3 gridSize((nx + BLOCKDIM_X - 1) / BLOCKDIM_X,
                        (ny + BLOCKDIM_Y - 1) / BLOCKDIM_Y,
                        (nz + BLOCKDIM_Z - 1) / BLOCKDIM_Z);

    const dim3 blockSize(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
    KernelBackProj<<<gridSize, blockSize>>>(tex_proj,
                                            data, indices, indptr,
                                            angles, distances,
                                            na, nb, nv,
                                            nx, ny, nz,
                                            image);

    cudaDestroyTextureObject(tex_proj);
    cudaFreeArray(array_proj);
}
#endif