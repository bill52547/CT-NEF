#include "cu_vector_multiply.h"

float cu_vector_multiply(float *a, float *b, int nx, int ny)
{
    
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
    float *d_ab;
    cudaMalloc((void**)&d_ab, nx * ny * sizeof(float)); 
    
    kernel_multiply<<<gridSize, blockSize>>>(d_ab, a, b, nx, ny);
    float *d_sum1, *d_sum2;
    cudaMalloc((void**)&d_sum1, sizeof(float)); 
    // cudaMalloc((void**)&d_sum2, nx * sizeof(float)); 
    
    host_initial(d_sum1, 1, 1, 0.0f);
    // host_initial(d_sum2, nx, 1, 0.0f);

    kernel_sum<<<nx * ny, 1>>>(d_sum1, d_ab);
    // kernel_sum<<<gridSize, blockSize>>>(d_sum1, d_sum2, nx, 1);

    cudaFree(d_ab);
    cudaFree(d_sum1);
    // cudaFree(d_sum2);
    return *d_sum1;
}

__global__ void kernel_multiply(float *c, float *a, float *b, int nx, int ny)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    int id = ix + iy * nx;
    c[id] = a[ix + iy * nx] * b[ix + iy * nx];
}

// __global__ void kernel_sum(float *sum, float *in, int nx, int ny)
// {
//     int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
//     int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
//     if (ix >= nx || iy >= ny)
//         return;
//     atomicAdd(&sum[iy], in[ix + iy * nx]);

// }
__global__ void kernel_sum(float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2) 
    {
        if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
        }
       __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}