#include "cu_dot_product.h"

__host__ void host_dot(float c, float *a, float *b, int nx, int ny)
{
    int N = nx * ny;
    kernel_wtx<<<N / BLOCKSIZE_X, BLOCKSIZE_X>>>(c, a, b, N);
    cudaDeviceSynchronize();
}

__global__ void dot(float c, float* a, float* b, int N) 
{
	__shared__ float cache[BLOCKSIZE_X];
	int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= N)
        return;
	int cacheIndex = threadIdx.x;
	
	float temp = 0.0f;
	while (id < N){
		temp += a[id] * b[id];
		id += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}
