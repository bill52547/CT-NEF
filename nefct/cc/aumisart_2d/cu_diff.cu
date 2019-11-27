#include "cu_diff.h"
__device__ float diff_x(float *img, int ix, int iy, int nx, int ny){
    if (ix == nx - 1)
        return 0.0f;
    else
    {
        int id = ix + iy * nx;
        int id1 = id + 1;
        return img[id1] - img[id];
    }
}

__device__ float diff_y(float *img, int ix, int iy, int nx, int ny){
    if (iy == ny - 1)
        return 0.0f;
    else
    {
        int id = ix + iy * nx;
        int id1 = id + nx;
        return img[id1] - img[id];
    }
}

