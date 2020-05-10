#include "cu_deform.h"

__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    float *mx, *my;
    cudaMalloc((void**)&mx, nx * ny * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, alpha_x, alpha_y, beta_x, beta_y, volume, flow, nx, ny);
    cudaDeviceSynchronize();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
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
    kernel_deformation<<<gridSize, blockSize>>>(d_img1, tex_img, mx, my, nx, ny);
    cudaDeviceSynchronize();
    cudaFree(mx);   
    cudaFree(my);   
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);

}

__host__ void host_deform_invert(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    float *mx, *my;
    cudaMalloc((void**)&mx, nx * ny * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, alpha_x, alpha_y, beta_x, beta_y, volume, flow, nx, ny);
    cudaDeviceSynchronize();

    float *mx2, *my2;
    cudaMalloc((void**)&mx2, nx * ny * sizeof(float));
    cudaMalloc((void**)&my2, nx * ny * sizeof(float));
    
    host_invert(mx2, my2, mx, my, nx, ny);
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
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
    kernel_deformation<<<gridSize, blockSize>>>(d_img1, tex_img, mx2, my2, nx, ny);
    cudaDeviceSynchronize();
    cudaFree(mx);   
    cudaFree(my);   
    
    cudaFree(mx2);   
    cudaFree(my2);   
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(array_img);

}

__host__ void host_invert(float *mx2, float *my2, float *mx, float *my, int nx, int ny)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) mx, nx * sizeof(float), nx, ny);
    cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) my, nx * sizeof(float), nx, ny);

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

    kernel_invert<<<gridSize, blockSize>>>(mx2, my2, tex_mx, tex_my, nx, ny);
    cudaDeviceSynchronize();
    cudaDestroyTextureObject(tex_mx);
    cudaFreeArray(array_mx);
    cudaDestroyTextureObject(tex_my);
    cudaFreeArray(array_my);
}
__global__ void kernel_invert(float *mx2, float *my2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float x = 0, y = 0;
    for (int iter = 0; iter < 10; iter ++){
        x = - tex3D<float>(tex_mx, (x + ix + 0.5f), (y + iy + 0.5f), 0.5f);
        y = - tex3D<float>(tex_my, (x + ix + 0.5f), (y + iy + 0.5f), 0.5f);
    }
    mx2[id] = x;
    my2[id] = y;
}

__host__ void host_deform2(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    float *mx, *my;
    cudaMalloc((void**)&mx, nx * ny * sizeof(float));
    cudaMalloc((void**)&my, nx * ny * sizeof(float));
    kernel_forwardDVF<<<gridSize, blockSize>>>(mx, my, alpha_x, alpha_y, beta_x, beta_y, volume, flow, nx, ny);
    cudaDeviceSynchronize();
    kernel_deformation2<<<gridSize, blockSize>>>(d_img1, d_img, mx, my, nx, ny);
    cudaDeviceSynchronize();
    cudaFree(mx);
    cudaFree(my);
}

__global__ void kernel_forwardDVF(float *mx, float *my, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y, float volume, float flow, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;    
    mx[id] = alpha_x[id] * volume + beta_x[id] * flow;
    my[id] = alpha_y[id] * volume + beta_y[id] * flow;
}

__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    float xi = ix + mx[id];
    float yi = iy + my[id];
    
    img1[id] = tex3D<float>(tex_img, xi + 0.5f, yi + 0.5f, 0.5f);
}

__global__ void kernel_deformation2(float *img1, float *img, float *mx, float *my, int nx, int ny){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = iy + ix * ny;
    int id2 = ix + iy * nx;

    float dx, dy;
    if (ix == nx - 1)
        dx = 0;
    else
        dx = img[id2 + 1] - img[id2];
        
    if (iy == ny - 1)
        dy = 0;
    else
        dy = img[id2 + nx] - img[id2];


    img1[id2] = img[id2] + dy * mx[id2] + dx * my[id2];
}