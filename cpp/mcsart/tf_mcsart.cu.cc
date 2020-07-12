#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x)<0 ? (-x) : (x))

const int GRIDDIM = 32;
const int BLOCKDIM = 1024;

__device__ void project_device(const float x1_, const float y1_, const float z1_,
                               const float x2_, const float y2_, const float z2_,
                               const int nx, const int ny, const int nz,
                               const float cx, const float cy, const float cz,
                               const float sx, const float sy, const float sz,
                               const float *image, float *vproj)
{
    const float dx_ = sx / nx;
    const float dx = 1.0f;
    const float dy = sy / ny / dx_;
    const float dz = sz / nz / dx_;
    const float x1 = (x1_ - cx) / dx_;
    const float x2 = (x2_ - cx) / dx_;
    const float y1 = (y1_ - cy) / dx_;
    const float y2 = (y2_ - cy) / dx_;
    const float z1 = (z1_ - cz) / dx_;
    const float z2 = (z2_ - cz) / dx_;

    const float xd = x2 - x1;
    const float yd = y2 - y1;
    const float zd = z2 - z1;

    if (sqrt(xd * xd + yd * yd) < 10.0f) {return;}

    const float nx2 = nx / 2.0f;
    const float ny2 = ny / 2.0f;
    const float nz2 = nz / 2.0f;

    const float L = sqrt(xd * xd + yd * yd + zd * zd);
    vproj[0] = 0.0f;

    if (abs(xd) > abs(yd))
    {
        float ky = yd / xd;
        float kz = zd / xd;

        for (int ix = 0; ix < nx; ++ix)
        {
            float xx1 = ix - nx2;
            float xx2 = xx1 + 1.0f;
            float yy1, yy2, zz1, zz2;

            if (ky >= 0.0f)
            {
                yy1 = (y1 + ky * (xx1 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx2 - x1)) / dy + ny2;

            }
            else
            {
                yy1 = (y1 + ky * (xx2 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx1 - x1)) / dy + ny2;
            }
            int cy1 = (int)floor(yy1);
            int cy2 = (int)floor(yy2);

            if (kz >= 0.0f)
            {
                zz1 = (z1 + kz * (xx1 - x1)) / dz + nz2;
                zz2 = (z1 + kz * (xx2 - x1)) / dz + nz2;
            }
            else
            {
                zz1 = (z1 + kz * (xx2 - x1)) / dz + nz2;
                zz2 = (z1 + kz * (xx1 - x1)) / dz + nz2;
            }
            int cz1 = (int)floor(zz1);
            int cz2 = (int)floor(zz2);

            if (cy1 == cy2)
            {
                if (0 <= cy1 && cy1 < ny)
                {
                    if (cz1 == cz2)
                    {
                        if (0 <= cz1 && cz1 < nz)
                        {
                            float weight = sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                            vproj[0] += image[ix + cy1 * nx + cz1 * nx * ny] * weight;
                        }
                    }
                    else
                    {
                        if (-1 <= cz1 and cz1 < nz)
                        {
                            float rz = (cz2 - zz1) / (zz2 - zz1);
                            if (cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz1 * nx * ny] * weight;
                            }

                            if (cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz2 * nx * ny] * weight;
                            }
                        }
                    }
                }
            }
            else
            {
                if (-1 <= cy1 && cy1 < ny)
                {
                    if (cz1 == cz2)
                    {
                         if (0 <= cz1 and cz1 < nz)
                         {
                            float ry = (cy2 - yy1) / (yy2 - yy1);
                            if (cy1 >= 0)
                            {
                                float weight = ry * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz1 * nx * ny] * weight;
                            }


                            if (cy2 < ny)
                            {
                                float weight = (1 - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy2 * nx + cz1 * nx * ny] * weight;
                            }
                         }
                    }
                    else if (-1 <= cz1 and cz1 < nz)
                    {
                        float ry = (cy2 - yy1) / (yy2 - yy1);
                        float rz = (cz2 - zz1) / (zz2 - zz1);
                        if (ry > rz)
                        {
                            if (cy1 >= 0 && cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz1 * nx * ny] * weight;

                            }

                            if (cy1 >= 0 && cz2 < nz)
                            {
                                float weight = (ry - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz2 * nx * ny] * weight;
                            }

                            if (cy2 < ny && cz2 < nz)
                            {
                                float weight = (1 - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy2 * nx + cz2 * nx * ny] * weight;
                            }
                        }
                        else
                        {
                            if (cy1 >= 0 && cz1 >= 0)
                            {
                                float weight = ry * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy1 * nx + cz1 * nx * ny] * weight;
                            }

                            if (cy2 < ny && cz1 >= 0)
                            {
                                float weight = (rz - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy2 * nx + cz1 * nx * ny] * weight;
                            }

                            if (cy2 < ny && cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ / L / L;
                                vproj[0] += image[ix + cy2 * nx + cz2 * nx * ny] * weight;
                            }
                        }
                    }
                }

            }
        }
    }
    else
    {
        float kx = xd / yd;
        float kz = zd / yd;

        for (int iy = 0; iy < ny; ++iy)
        {
            float yy1 = iy - ny2;
            float yy2 = yy1 + 1.0f;
            float xx1, xx2, zz1, zz2;

            if (kx >= 0.0f)
            {
                xx1 = (x1 + kx * (yy1 - y1)) + nx2;
                xx2 = (x1 + kx * (yy2 - y1)) + nx2;
            }
            else
            {
                xx1 = (x1 + kx * (yy2 - y1)) + nx2;
                xx2 = (x1 + kx * (yy1 - y1)) + nx2;
            }
            int cx1 = (int)floor(xx1);
            int cx2 = (int)floor(xx2);

            if (kz >= 0.0f)
            {
                zz1 = (z1 + kz * (yy1 - y1)) / dz + nz2;
                zz2 = (z1 + kz * (yy2 - y1)) / dz + nz2;
            }
            else
            {
                zz1 = (z1 + kz * (yy2 - y1)) / dz + nz2;
                zz2 = (z1 + kz * (yy1 - y1)) / dz + nz2;
            }
            int cz1 = (int)floor(zz1);
            int cz2 = (int)floor(zz2);

            if (cx1 == cx2)
            {
                if (0 <= cx1 && cx1 < nx)
                {
                    if (cz1 == cz2)
                    {
                        if (0 <= cz1 && cz1 < nz)
                        {
                            float weight = sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                            vproj[0] += image[cx1 + iy * nx + cz1 * nx * ny] * weight;
                        }
                    }
                    else
                    {
                        if (-1 <= cz1 and cz1 < nz)
                        {
                            float rz = (cz2 - zz1) / (zz2 - zz1);
                            if (cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx1 + iy * nx + cz1 * nx * ny] * weight;
                            }

                            if (cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx1 + iy * nx + cz2 * nx * ny] * weight;
                            }
                        }
                    }
                }

            }
            else
            {
                if (-1 <= cx1 && cx1 < nx)
                {
                    if (cz1 == cz2)
                    {
                         if (0 <= cz1 and cz1 < nz)
                         {
                            float rx = (cx2 - xx1) / (xx2 - xx1);
                            if (cx1 >= 0)
                            {
                                float weight = rx * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx1 + iy * nx + cz1 * nx * ny] * weight;
                            }

                            if (cx2 < nx)
                            {
                                float weight = (1 - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx2 + iy * nx + cz1 * nx * ny] * weight;
                            }
                         }
                    }
                    else if (-1 <= cz1 and cz1 < nz)
                    {
                        float rx = (cx2 - xx1) / (xx2 - xx1);
                        float rz = (cz2 - zz1) / (zz2 - zz1);
                        if (rx > rz)
                        {
                            if (cx1 >= 0 && cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx1 + iy * nx + cz1 * nx * ny] * weight;
                            }

                            if (cx1 >= 0 && cz2 < nz)
                            {
                                float weight = (rx - rz) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L /
                                L;
                                vproj[0] += image[cx1 + iy * nx + cz2 * nx * ny] * weight;
                            }

                            if (cx2 < nx && cz2 < nz)
                            {
                                float weight = (1 - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx2 + iy * nx + cz2 * nx * ny] * weight;
                            }
                        }
                        else
                        {
                            if (cx1 >= 0 && cz1 >= 0)
                            {
                                float weight = rx * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx1 + iy * nx + cz1 * nx * ny] * weight;
                            }

                            if (cx2 < nx && cz1 >= 0)
                            {
                                float weight = (rz - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L /
                                L;
                                vproj[0]+= image[cx2 + iy * nx + cz1 * nx * ny] * weight;
                            }

                            if (cx2 < nx && cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + kx * kx + kz * kz) * dx_ / L / L;
                                vproj[0] += image[cx2 + iy * nx + cz2 * nx * ny] * weight;
                            }
                        }
                    }
                }

            }
        }
    }
//    if (vproj[0] < 0.00000001f) {vproj[0] = 100000000.0f;}

}



__device__ void backproject_device(const float x1_, const float y1_, const float z1_,
                                   const float x2_, const float y2_, const float z2_,
                                   const int nx, const int ny, const int nz,
                                   const float cx, const float cy, const float cz,
                                   const float sx, const float sy, const float sz,
                                   const float vproj, float *image)
{
    const float dx_ = sx / nx;
    const float dx = 1.0f;
    const float dy = sy / ny / dx_;
    const float dz = sz / nz / dx_;
    const float x1 = (x1_ - cx) / dx_;
    const float x2 = (x2_ - cx) / dx_;
    const float y1 = (y1_ - cy) / dx_;
    const float y2 = (y2_ - cy) / dx_;
    const float z1 = (z1_ - cz) / dx_;
    const float z2 = (z2_ - cz) / dx_;

    const float xd = x2 - x1;
    const float yd = y2 - y1;
    const float zd = z2 - z1;

    if (sqrt(xd * xd + yd * yd) < 10.0f) {return;}

    const float nx2 = nx / 2.0f;
    const float ny2 = ny / 2.0f;
    const float nz2 = nz / 2.0f;

    const float L = sqrt(xd * xd + yd * yd + zd * zd);

    if (abs(xd) > abs(yd))
    {
        float ky = yd / xd;
        float kz = zd / xd;

        for (int ix = 0; ix < nx; ++ix)
        {
            float xx1 = ix - nx2;
            float xx2 = xx1 + 1.0f;
            float yy1, yy2, zz1, zz2;

            if (ky >= 0.0f)
            {
                yy1 = (y1 + ky * (xx1 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx2 - x1)) / dy + ny2;

            }
            else
            {
                yy1 = (y1 + ky * (xx2 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx1 - x1)) / dy + ny2;
            }
            int cy1 = (int)floor(yy1);
            int cy2 = (int)floor(yy2);

            if (kz >= 0.0f)
            {
                zz1 = (z1 + kz * (xx1 - x1)) / dz + nz2;
                zz2 = (z1 + kz * (xx2 - x1)) / dz + nz2;
            }
            else
            {
                zz1 = (z1 + kz * (xx2 - x1)) / dz + nz2;
                zz2 = (z1 + kz * (xx1 - x1)) / dz + nz2;
            }
            int cz1 = (int)floor(zz1);
            int cz2 = (int)floor(zz2);

            if (cy1 == cy2)
            {
                if (0 <= cy1 && cy1 < ny)
                {
                    if (cz1 == cz2)
                    {
                        if (0 <= cz1 && cz1 < nz)
                        {
                            float weight = sqrt(1 + ky * ky + kz * kz) * dx_ ;
                            atomicAdd(image + ix + cy1 * nx + cz1 * nx * ny, vproj * weight);
                        }
                    }
                    else
                    {
                        if (-1 <= cz1 and cz1 < nz)
                        {
                            float rz = (cz2 - zz1) / (zz2 - zz1);
                            if (cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                    }
                }
            }
            else
            {
                if (-1 <= cy1 && cy1 < ny)
                {
                    if (cz1 == cz2)
                    {
                         if (0 <= cz1 and cz1 < nz)
                         {
                            float ry = (cy2 - yy1) / (yy2 - yy1);
                            if (cy1 >= 0)
                            {
                                float weight = ry * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz1 * nx * ny, vproj * weight);
                            }


                            if (cy2 < ny)
                            {
                                float weight = (1 - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy2 * nx + cz1 * nx * ny, vproj * weight);
                            }
                         }
                    }
                    else if (-1 <= cz1 and cz1 < nz)
                    {
                        float ry = (cy2 - yy1) / (yy2 - yy1);
                        float rz = (cz2 - zz1) / (zz2 - zz1);
                        if (ry > rz)
                        {
                            if (cy1 >= 0 && cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cy1 >= 0 && cz2 < nz)
                            {
                                float weight = (ry - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz2 * nx * ny, vproj * weight);
                            }

                            if (cy2 < ny && cz2 < nz)
                            {
                                float weight = (1 - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy2 * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                        else
                        {
                            if (cy1 >= 0 && cz1 >= 0)
                            {
                                float weight = ry * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy1 * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cy2 < ny && cz1 >= 0)
                            {
                                float weight = (rz - ry) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy2 * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cy2 < ny && cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + ky * ky + kz * kz) * dx_ ;
                                atomicAdd(image + ix + cy2 * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                    }
                }

            }
        }
    }
    else
    {
        float kx = xd / yd;
        float kz = zd / yd;

        for (int iy = 0; iy < ny; ++iy)
        {
            float yy1 = iy - ny2;
            float yy2 = yy1 + 1.0f;
            float xx1, xx2, zz1, zz2;

            if (kx >= 0.0f)
            {
                xx1 = (x1 + kx * (yy1 - y1)) + nx2;
                xx2 = (x1 + kx * (yy2 - y1)) + nx2;
            }
            else
            {
                xx1 = (x1 + kx * (yy2 - y1)) + nx2;
                xx2 = (x1 + kx * (yy1 - y1)) + nx2;
            }
            int cx1 = (int)floor(xx1);
            int cx2 = (int)floor(xx2);

            if (kz >= 0.0f)
            {
                zz1 = (z1 + kz * (yy1 - y1)) / dz + nz2;
                zz2 = (z1 + kz * (yy2 - y1)) / dz + nz2;
            }
            else
            {
                zz1 = (z1 + kz * (yy2 - y1)) / dz + nz2;
                zz2 = (z1 + kz * (yy1 - y1)) / dz + nz2;
            }
            int cz1 = (int)floor(zz1);
            int cz2 = (int)floor(zz2);

            if (cx1 == cx2)
            {
                if (0 <= cx1 && cx1 < nx)
                {
                    if (cz1 == cz2)
                    {
                        if (0 <= cz1 && cz1 < nz)
                        {
                            float weight = sqrt(1 + kx * kx + kz * kz) * dx_ ;
                            atomicAdd(image + cx1 + iy * nx + cz1 * nx * ny, vproj * weight);
                        }
                    }
                    else
                    {
                        if (-1 <= cz1 and cz1 < nz)
                        {
                            float rz = (cz2 - zz1) / (zz2 - zz1);
                            if (cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                    }
                }
            }
            else
            {
                if (-1 <= cx1 && cx1 < nx)
                {
                    if (cz1 == cz2)
                    {
                         if (0 <= cz1 and cz1 < nz)
                         {
                            float rx = (cx2 - xx1) / (xx2 - xx1);
                            if (cx1 >= 0)
                            {
                                float weight = rx * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cx2 < nx)
                            {
                                float weight = (1 - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx2 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }
                         }
                    }
                    else if (-1 <= cz1 and cz1 < nz)
                    {
                        float rx = (cx2 - xx1) / (xx2 - xx1);
                        float rz = (cz2 - zz1) / (zz2 - zz1);
                        if (rx > rz)
                        {
                            if (cx1 >= 0 && cz1 >= 0)
                            {
                                float weight = rz * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cx1 >= 0 && cz2 < nz)
                            {
                                float weight = (rx - rz) * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz2 * nx * ny, vproj * weight);
                            }

                            if (cx2 < nx && cz2 < nz)
                            {
                                float weight = (1 - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx2 + iy * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                        else
                        {
                            if (cx1 >= 0 && cz1 >= 0)
                            {
                                float weight = rx * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx1 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cx2 < nx && cz1 >= 0)
                            {
                                float weight = (rz - rx) * sqrt(1 + kx * kx + kz * kz) * dx_ ;
                                atomicAdd(image + cx2 + iy * nx + cz1 * nx * ny, vproj * weight);
                            }

                            if (cx2 < nx && cz2 < nz)
                            {
                                float weight = (1 - rz) * sqrt(1 + kx * kx + kz * kz) * dx_  ;
                                atomicAdd(image + cx2 + iy * nx + cz2 * nx * ny, vproj * weight);
                            }
                        }
                    }
                }

            }
        }
    }
}

__global__ void
ProjectKernel(const float *x1, const float *y1, const float *z1,
             const float *x2, const float *y2, const float *z2,
             const int gx, const int gy, const int gz,
             const float cx, const float cy, const float cz,
             const float sx, const float sy, const float sz,
             const int num_events,
             const float *image_data, float *projection_value)
{
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid >= num_events) {return;}
        project_device(x1[tid], y1[tid], z1[tid],
                       x2[tid], y2[tid], z2[tid],
                       gx, gy, gz,
                       cx, cy, cz,
                       sx, sy, sz,
                       image_data, projection_value + tid);
    }
}

__global__ void
BackProjectKernel(const float *x1, const float *y1, const float *z1,
                    const float *x2, const float *y2, const float *z2,
                    const int gx, const int gy, const int gz,
                    const float cx, const float cy, const float cz,
                    const float sx, const float sy, const float sz,
                    const int num_events,
                    const float *projection_value, float *image_data)
{
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid >= num_events) {return;}

        backproject_device(x1[tid], y1[tid], z1[tid],
                           x2[tid], y2[tid], z2[tid],
                           gx, gy, gz,
                           cx, cy, cz,
                           sx, sy, sz,
                           projection_value[tid], image_data);

    }
}

void projection(const float *x1, const float *y1, const float *z1,
                const float *x2, const float *y2, const float *z2,
                float *vproj,
                const int *grid, const float *center, const float *size,
                const float *image, const int num_events)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    ProjectKernel<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                         x2, y2, z2,
                                         gx, gy, gz,
                                         cx, cy, cz,
                                         sx, sy, sz,
                                         num_events,
                                         image, vproj);
}


void backprojection(const float *x1, const float *y1, const float *z1,
                    const float *x2, const float *y2, const float *z2,
                    const float *vproj,
                    const int *grid, const float *center, const float *size,
                    float *image, const int num_events)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    BackProjectKernel<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                             x2, y2, z2,
                                             gx, gy, gz,
                                             cx, cy, cz,
                                             sx, sy, sz,
                                             num_events,
                                             vproj, image);
}

void mcsart(const int nx, const int ny, const int nz, const int na, const int nb, 
            const float dx, const float da, const float db, const float ais, const int bis,
            const float SO, const float SD, const int n_iter, 
            const int *n_views, const int numProj, 
            const float *d_alpha_x, const float *d_alpha_y, const float *d_alpha_z, 
            const float *d_beta_x, const float *d_beta_y, const float *d_beta_z, 
            const float *volumes, const float *flows, const float* ref_volumes, const float *ref_flows, 
            const float *angles, const int outIter, 
            const float *h_img, const float* h_proj, 
            )
{
    const int numImg = nx * ny * nz; // size of image
    const int numBytesImg = numImg * sizeof(float); // number of bytes in image
    const int numSingleProj = na * nb;
    const int numBytesSingleProj = numSingleProj * sizeof(float);
    const int n_iter_invertDVF = 10;
    const int n_bin = 8;
    const int N_view = n_views[n_bin];
    const int numProj = numSingleProj * N_view;
    const int numBytesProj = numProj * sizeof(float);
    // load initial guess of image
    float *h_img;
    h_img = (float*)mxGetData(IN_IMG);

    // load true projection value
    float *h_proj;
    h_proj = (float*)mxGetData(PROJ);

    // define thread distributions
    const dim3 gridSize_img((nx + BLOCKWIDTH - 1) / BLOCKWIDTH, (ny + BLOCKHEIGHT - 1) / BLOCKHEIGHT, (nz + BLOCKDEPTH - 1) / BLOCKDEPTH);
    const dim3 gridSize_singleProj((na + BLOCKWIDTH - 1) / BLOCKWIDTH, (nb + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1);
    const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT, BLOCKDEPTH);

    // CUDA 3DArray Malloc parameters
    struct cudaExtent extent_img = make_cudaExtent(nx, ny, nz);
    struct cudaExtent extent_singleProj = make_cudaExtent(na, nb, 1);

    //Allocate CUDA array in device memory of 5DCT matrices: alpha and beta
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaError_t cudaStat;

    // Get pitched pointer to alpha and beta in host memory
    cudaPitchedPtr hp_alpha_x = make_cudaPitchedPtr((void*) d_alpha_x, nx * sizeof(float), nx, ny);
    cudaPitchedPtr hp_alpha_y = make_cudaPitchedPtr((void*) d_alpha_y, nx * sizeof(float), nx, ny);
    cudaPitchedPtr hp_alpha_z = make_cudaPitchedPtr((void*) d_alpha_z, nx * sizeof(float), nx, ny);
    cudaPitchedPtr hp_beta_x = make_cudaPitchedPtr((void*) d_beta_x, nx * sizeof(float), nx, ny);
    cudaPitchedPtr hp_beta_y = make_cudaPitchedPtr((void*) d_beta_y, nx * sizeof(float), nx, ny);
    cudaPitchedPtr hp_beta_z = make_cudaPitchedPtr((void*) d_beta_z, nx * sizeof(float), nx, ny);

    // Copy alpha and beta to texture memory from pitched pointer
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = extent_img;
    copyParams.kind = cudaMemcpyHostToDevice;

    //alpha_x
    copyParams.srcPtr = hp_alpha_x;
    copyParams.dstArray = d_alpha_x;
    cudaStat = cudaMemcpy3D(&copyParams);

    //alpha_y
    copyParams.srcPtr = hp_alpha_y;
    copyParams.dstArray = d_alpha_y;
    cudaStat = cudaMemcpy3D(&copyParams);

    //alpha_z
    copyParams.srcPtr = hp_alpha_z;
    copyParams.dstArray = d_alpha_z;
    cudaStat = cudaMemcpy3D(&copyParams);

    //beta_x
    copyParams.srcPtr = hp_beta_x;
    copyParams.dstArray = d_beta_x;
    cudaStat = cudaMemcpy3D(&copyParams);

    //beta_y
    copyParams.srcPtr = hp_beta_y;
    copyParams.dstArray = d_beta_y;
    cudaStat = cudaMemcpy3D(&copyParams);

    //beta_z
    copyParams.srcPtr = hp_beta_z;
    copyParams.dstArray = d_beta_z;
    cudaStat = cudaMemcpy3D(&copyParams);


    // create texture object alpha and beta
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc, texDesc2;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    memset(&texDesc2, 0, sizeof(texDesc2));
    texDesc2.addressMode[0] = cudaAddressModeClamp;
    texDesc2.addressMode[1] = cudaAddressModeClamp;
    texDesc2.addressMode[2] = cudaAddressModeClamp;
    texDesc2.filterMode = cudaFilterModePoint;
    texDesc2.readMode = cudaReadModeElementType;
    texDesc2.normalizedCoords = 0;

    // alpha_x
    resDesc.res.array.array = d_alpha_x;
    cudaTextureObject_t tex_alpha_x = 0;
    cudaCreateTextureObject(&tex_alpha_x, &resDesc, &texDesc, NULL);

    // alpha_y
    resDesc.res.array.array = d_alpha_y;
    cudaTextureObject_t tex_alpha_y = 0;
    cudaCreateTextureObject(&tex_alpha_y, &resDesc, &texDesc, NULL);

    // alpha_z
    resDesc.res.array.array = d_alpha_z;
    cudaTextureObject_t tex_alpha_z = 0;
    cudaCreateTextureObject(&tex_alpha_z, &resDesc, &texDesc, NULL);

    // beta_x
    resDesc.res.array.array = d_beta_x;
    cudaTextureObject_t tex_beta_x = 0;
    cudaCreateTextureObject(&tex_beta_x, &resDesc, &texDesc, NULL);

    // beta_y
    resDesc.res.array.array = d_beta_y;
    cudaTextureObject_t tex_beta_y = 0;
    cudaCreateTextureObject(&tex_beta_y, &resDesc, &texDesc, NULL);

    // beta_z
    resDesc.res.array.array = d_beta_z;
    cudaTextureObject_t tex_beta_z = 0;
    cudaCreateTextureObject(&tex_beta_z, &resDesc, &texDesc, NULL);

    // malloc in device: projection of the whole bin
    float *d_proj;
    cudaMalloc((void**)&d_proj, numBytesSingleProj);

    // copy to device: projection of the whole bin
    // cudaMemcpy(d_proj, h_proj, numBytesProj, cudaMemcpyHostToDevice);

    // malloc in device: another projection pointer, with single view size
    float *d_singleViewProj2;
    cudaMalloc((void**)&d_singleViewProj2, numBytesSingleProj);

    // malloc in device: projection of the whole bin
    float *d_img ,*d_img1;
    cudaArray *array_img;
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMalloc((void**)&d_img1, numBytesImg);
    cudaStat = cudaMalloc3DArray(&array_img, &channelDesc, extent_img);

    // malloc in device: another image pointer, for single view 
    float *d_singleViewImg1, *d_imgOnes;
    cudaMalloc(&d_singleViewImg1, numBytesImg);

    cudaMalloc(&d_imgOnes, numBytesImg);
    float angle, volume, flow, ai, bi;

    //Malloc forward and inverted DVFs in device
    float *d_mx, *d_my, *d_mz, *d_mx2, *d_my2, *d_mz2;
    cudaMalloc(&d_mx, numBytesImg);
    cudaMalloc(&d_my, numBytesImg);
    cudaMalloc(&d_mz, numBytesImg);
    cudaMalloc(&d_mx2, numBytesImg);
    cudaMalloc(&d_my2, numBytesImg);
    cudaMalloc(&d_mz2, numBytesImg);


    // Alloc forward and inverted DVFs in device, in form of array memory
    cudaArray *array_mx, *array_my, *array_mz, *array_mx2, *array_my2, *array_mz2;
    cudaStat = cudaMalloc3DArray(&array_mx, &channelDesc, extent_img);
    cudaStat = cudaMalloc3DArray(&array_my, &channelDesc, extent_img);
    cudaStat = cudaMalloc3DArray(&array_mz, &channelDesc, extent_img);
    cudaStat = cudaMalloc3DArray(&array_mx2, &channelDesc, extent_img);
    cudaStat = cudaMalloc3DArray(&array_my2, &channelDesc, extent_img);
    cudaStat = cudaMalloc3DArray(&array_mz2, &channelDesc, extent_img);

    // define tex_mx etc
    cudaTextureObject_t tex_mx = 0, tex_my = 0, tex_mz = 0, tex_mx2 = 0, tex_my2 = 0, tex_mz2 = 0, tex_img = 0;


    // setup output images
    OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
    const mwSize outDim[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)n_bin};
    mxSetDimensions(OUT_IMG, outDim, 4);
    mxSetData(OUT_IMG, mxMalloc(numBytesImg * n_bin));
    float *h_outimg = (float*)mxGetData(OUT_IMG);

    plhs[1] = mxCreateNumericMatrix(n_bin * n_iter, 1, mxSINGLE_CLASS, mxREAL);
    float *h_outnorm = (float*)mxGetData(plhs[1]);
    for (int i = 0; i < n_bin * n_iter; i++)
        h_outnorm[i] = 0.0f;

    float tempNorm;

    copyParams.kind = cudaMemcpyDeviceToDevice;

    for (int ibin = 0; ibin < n_bin; ibin++){
        if (outIter == 0)
        {
            cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
        }
        else{
            if (ibin < 1){
                cudaMemcpy(d_img1, h_img + (n_bin - 1) * numImg, numBytesImg, cudaMemcpyHostToDevice);
            }
            if (ibin == 0){
                volume = ref_volumes[0] - ref_volumes[n_bin - 1];
                flow = ref_flows[0] - ref_flows[n_bin - 1];
            }
            else{
                volume = ref_volumes[ibin] - ref_volumes[ibin - 1];
                flow = ref_flows[ibin] - ref_flows[ibin - 1];
            }
            kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
            cudaDeviceSynchronize();

            // copy img to pitched pointer and bind it to a texture object
            cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img1, nx * sizeof(float), nx, ny);
            copyParams.srcPtr = dp_img;
            copyParams.dstArray = array_img;
            cudaStat = cudaMemcpy3D(&copyParams);   
            resDesc.res.array.array = array_img;
            cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);

            kernel_deformation<<<gridSize_img, blockSize>>>(d_img, tex_img, d_mx, d_my, d_mz, nx, ny, nz);
            cudaDeviceSynchronize();
        }

        for (int iter = 0; iter < n_iter; iter++){ // iteration
            processBar(ibin, n_bin, iter, n_iter);
            for (int i_view = n_views[ibin]; i_view < n_views[ibin + 1]; i_view++){ // view
                ai = ais[i_view];
                bi = bis[i_view];
                angle = angles[i_view];
                volume = ref_volumes[ibin] - volumes[i_view];
                flow = ref_flows[ibin] - flows[i_view];
                
                // generate forwards DVFs: d_mx, d_my, d_mz and inverted DVFs: d_mx2, d_my2, d_mz2
                kernel_forwardDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_alpha_x, tex_alpha_y, tex_alpha_z, tex_beta_x, tex_beta_y, tex_beta_z, volume, flow, nx, ny, nz);
                cudaDeviceSynchronize();
                
                // copy mx etc to pitched pointer and bind it to a texture object
                cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) d_mx, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_mx;
                copyParams.dstArray = array_mx;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_mx;
                cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

                cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) d_my, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_my;
                copyParams.dstArray = array_my;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_my;
                cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

                cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) d_mz, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_mz;
                copyParams.dstArray = array_mz;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_mz;
                cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);

                kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx2, d_my2, d_mz2, tex_mx, tex_my, tex_mz, nx, ny, nz, n_iter_invertDVF);
                cudaDeviceSynchronize();        
                
                // copy mx2 etc to pitched pointer and bind it to a texture object
                cudaPitchedPtr dp_mx2 = make_cudaPitchedPtr((void*) d_mx2, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_mx2;
                copyParams.dstArray = array_mx2;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_mx2;
                cudaCreateTextureObject(&tex_mx2, &resDesc, &texDesc, NULL);

                cudaPitchedPtr dp_my2 = make_cudaPitchedPtr((void*) d_my2, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_my2;
                copyParams.dstArray = array_my2;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_my2;
                cudaCreateTextureObject(&tex_my2, &resDesc, &texDesc, NULL);

                cudaPitchedPtr dp_mz2 = make_cudaPitchedPtr((void*) d_mz2, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_mz2;
                copyParams.dstArray = array_mz2;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_mz2;
                cudaCreateTextureObject(&tex_mz2, &resDesc, &texDesc, NULL);

                // copy img to pitched pointer and bind it to a texture object
                cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) d_img, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_img;
                copyParams.dstArray = array_img;
                cudaStat = cudaMemcpy3D(&copyParams);   
                resDesc.res.array.array = array_img;
                cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);

                // deformed image for i_view, from reference image of the bin          
                
                kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg1, tex_img, d_mx2, d_my2, d_mz2, nx, ny, nz);
                cudaDeviceSynchronize();

                // projection of deformed image from initial guess
                kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_singleViewImg1, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz); // TBD
                cudaDeviceSynchronize();

                // difference between true projection and projection from initial guess
                // update d_singleViewProj2 instead of malloc a new one
                cudaMemcpy(d_proj, h_proj + i_view * numSingleProj, numBytesSingleProj, cudaMemcpyHostToDevice);
                // mexPrintf("i_view = %d.\n", i_view);mexEvalString("drawnow;");

                kernel_add<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_proj, 0, na, nb, -1);
                cudaDeviceSynchronize();
                // cublasSnrm2_v2(handle, na * nb, d_singleViewProj2, 1, temp_err);
                // h_outerr[iter] += temp_err[0];

                // backprojecting the difference of projections
                // print parameters              
                kernel_backprojection(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

                // calculate the ones backprojection data
                kernel_initial<<<gridSize_img, blockSize>>>(d_imgOnes, nx, ny, nz, 1);
                cudaDeviceSynchronize();
                kernel_projection<<<gridSize_singleProj, blockSize>>>(d_singleViewProj2, d_imgOnes, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                cudaDeviceSynchronize();

                // kernel_backprojection<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
                // cudaDeviceSynchronize();

                kernel_backprojection(d_imgOnes, d_singleViewProj2, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);

                // weighting
                kernel_division<<<gridSize_img, blockSize>>>(d_singleViewImg1, d_imgOnes, nx, ny, nz);
                cudaDeviceSynchronize();

                // copy img to pitched pointer and bind it to a texture object
                dp_img = make_cudaPitchedPtr((void*) d_singleViewImg1, nx * sizeof(float), nx, ny);
                copyParams.srcPtr = dp_img;
                copyParams.dstArray = array_img;
                cudaStat = cudaMemcpy3D(&copyParams);   
                if (cudaStat != cudaSuccess) {
                    mexPrintf("Failed to copy dp_img to array memory array_img.\n");
                    mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
                        mexErrMsgIdAndTxt("MATLAB:cudaFail","SART failed.\n");
                }
                resDesc.res.array.array = array_img;
                cudaCreateTextureObject(&tex_img, &resDesc, &texDesc2, NULL);


                kernel_invertDVF<<<gridSize_img, blockSize>>>(d_mx, d_my, d_mz, tex_mx2, tex_my2, tex_mz2, nx, ny, nz, n_iter_invertDVF);
                cudaDeviceSynchronize();  
                // deform backprojection back to the bin
                kernel_deformation<<<gridSize_img, blockSize>>>(d_singleViewImg1, tex_img, d_mx, d_my, d_mz, nx, ny, nz);
                cudaDeviceSynchronize();


                // updating
                kernel_update<<<gridSize_img, blockSize>>>(d_img, d_singleViewImg1, nx, ny, nz, lambda);
                cudaDeviceSynchronize();          
                // mexPrintf("13");mexEvalString("drawnow;");
            }  
        }
        cudaMemcpy(d_img1, d_img, numBytesImg, cudaMemcpyDeviceToDevice);
        cudaMemcpy(h_outimg + ibin * numImg, d_img, numBytesImg, cudaMemcpyDeviceToHost);    
    }




    cudaDestroyTextureObject(tex_alpha_x);
    cudaDestroyTextureObject(tex_alpha_y);
    cudaDestroyTextureObject(tex_alpha_z);
    cudaDestroyTextureObject(tex_beta_x);
    cudaDestroyTextureObject(tex_beta_y);
    cudaDestroyTextureObject(tex_beta_z);
    cudaDestroyTextureObject(tex_img);
    cudaDestroyTextureObject(tex_mx);
    cudaDestroyTextureObject(tex_my);
    cudaDestroyTextureObject(tex_mz);
    cudaDestroyTextureObject(tex_mx2);
    cudaDestroyTextureObject(tex_my2);
    cudaDestroyTextureObject(tex_mz2);

    cudaFreeArray(d_alpha_x);
    cudaFreeArray(d_alpha_y);
    cudaFreeArray(d_alpha_z);
    cudaFreeArray(d_beta_x);
    cudaFreeArray(d_beta_y);
    cudaFreeArray(d_beta_z);
    // cudaFreeArray(d_img);
    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_mz);
    cudaFree(d_mx2);
    cudaFree(d_my2);
    cudaFree(d_mz2);
    cudaFreeArray(array_mx);
    cudaFreeArray(array_my);
    cudaFreeArray(array_mz);
    cudaFreeArray(array_mx2);
    cudaFreeArray(array_my2);
    cudaFreeArray(array_mz2);
    cudaFree(d_proj);
    cudaFree(d_singleViewImg1);
    cudaFree(d_imgOnes);
    cudaFree(d_singleViewProj2);

    cudaFree(d_img);
    cudaFree(d_img1);

    cudaDeviceReset();
    }
}

#endif
