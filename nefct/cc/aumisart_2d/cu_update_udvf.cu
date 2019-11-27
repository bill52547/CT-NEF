#include "cu_update_udvf.h"

__host__ void host_update_udvf(float *alpha_x, float* alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float *img, float *img0, float volume_diff, float flow_diff, int nx, int ny, int nz, int iter)
{
    const dim3 gridSize((nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (nz + 1 - 1) / 1);
    const dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    kernel_update_udvf<<<gridSize, blockSize>>>(alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, img, img0, volume_diff, flow_diff, nx, ny, nz, iter);
    cudaDeviceSynchronize();
}

__global__ void kernel_update_udvf(float *alpha_x, float* alpha_y, float *alpha_z, float *beta_x, float *beta_y, float *beta_z, float *img, float *img0, float volume_diff, float flow_diff, int nx, int ny, int nz, int iter)
{
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = 1 * blockIdx.z + threadIdx.z;
    
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float dfI = img0[id] - img[id];
    float dfx, dfy, dfz;
    if (ix == nx - 1)
        dfx = 0.0f;
    else
        dfx = img[id + 1] - img[id];
    if (iy == ny - 1)
        dfy = 0.0f;
    else
        dfy = img[id + nx] - img[id];
    if (iz == nz - 1)
        dfz = 0.0f;
    else
        dfz = img[id + nx * ny] - img[id];
    // float dfx = diff_x(img, ix, iy, iz, nx, ny, nz);
    // float dfy = diff_y(img, ix, iy, iz, nx, ny, nz);
    // float dfz = diff_z(img, ix, iy, iz, nx, ny, nz);
    float cax = dfx * volume_diff;
    float cay = dfy * volume_diff;
    float caz = dfz * volume_diff;
    float cbx = dfx * flow_diff;
    float cby = dfy * flow_diff;
    float cbz = dfz * flow_diff;
    float w = cax * cax + cay * cay + caz * caz + cbx * cbx + cby * cby + cbz * cbz;

    float ax = alpha_y[id];
    float ay = alpha_x[id];
    float az = alpha_z[id];
    float bx = beta_y[id];
    float by = beta_x[id];
    float bz = beta_z[id];
    float b = dfI - cax * ax - cay * ay - caz * az - cbx * bx - cby * by - cbz * bz;
    if (w == 0)
        return;
    b /= w;
    alpha_x[id] += 1.0f / 6 * b * cax;
    alpha_y[id] += 1.0f / 6 * b * cay;
    alpha_z[id] += 1.0f / 6 * b * caz;
    beta_x[id] += 1.0f / 6 * b * cbx;
    beta_y[id] += 1.0f / 6 * b * cby;
    beta_z[id] += 1.0f / 6 * b * cbz;

}

// __device__ float diff_x(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
//     if (ix == nx - 1)
//         return 0.0f;
//     else
//     {
//         int id = ix + iy * nx + iz * nx * ny;
//         int id1 = id + 1;
//         return img[id1] - img[id];
//     }
// }

// __device__ float diff_y(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
//     if (iy == ny - 1)
//         return 0.0f;
//     else
//     {
//         int id = ix + iy * nx + iz * nx * ny;
//         int id1 = id + nx;
//         return img[id1] - img[id];
//     }
// }

// __device__ float diff_z(float *img, int ix, int iy, int iz, int nx, int ny, int nz){
//     if (iz == nz - 1)
//         return 0.0f;
//     else
//     {
//         int id = ix + iy * nx + iz * nx * ny;
//         int id1 = id + nx * ny;
//         return img[id1] - img[id];
//     }
// }