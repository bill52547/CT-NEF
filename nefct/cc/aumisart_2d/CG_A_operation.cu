#include "CG_A_operation.h"

__host__ void A_operation(float *d_xp, float *d_img, int nx, int ny, int na, int n_views, float da, float ai, float SO, float SD, float mu, float* volumes,float* flows, float* angles, float* d_alpha_x, float* d_alpha_y, float* d_beta_x, float* d_beta_y)
{
    float *d_wx, *d_wtx, *d_img1, *d_img2, *d_proj, *d_bproj;
    int numBytesImg = nx * ny * sizeof(float);
    int numBytesSingleProj = na * sizeof(float);
    int numBytesProj = numBytesSingleProj * n_views;
    cudaMalloc((void**)&d_wx, numBytesImg * 2);
    cudaMalloc((void**)&d_wtx, numBytesImg);
    cudaMalloc((void**)&d_img1, numBytesImg);
    cudaMalloc((void**)&d_img2, numBytesImg);

    cudaMalloc((void**)&d_proj, numBytesSingleProj);
    cudaMalloc((void**)&d_bproj, numBytesImg);
    host_wx(d_wx, d_img, nx, ny);
    host_wtx(d_wtx, d_wx, nx, ny);

    const dim3 gridSize_singleProj((na + BLOCKSIZE_X - 1) / BLOCKSIZE_X);
    const dim3 blockSize(BLOCKSIZE_X);
    host_initial(d_bproj, nx, ny, 0.0f);

    for (int i_view = 0; i_view < n_views; i_view ++)
    {   
        host_deform(d_img1, d_img, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);

        kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img1, angles[i_view], SO, SD, da, na, ai, nx, ny);
        cudaDeviceSynchronize();

        host_backprojection(d_img1, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

        host_deform_invert(d_img2, d_img1, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
        host_add(d_bproj, d_img2, nx, ny, mu);
    }

    host_add(d_bproj, d_wtx, nx, ny, mu);
    cudaMemcpy(d_xp, d_bproj, numBytesImg, cudaMemcpyDeviceToDevice);
    cudaFree(d_wx);
    cudaFree(d_wtx);
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_proj);
    cudaFree(d_bproj);
}