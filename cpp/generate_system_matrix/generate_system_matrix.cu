#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x)<0 ? (-x) : (x))

const int GRIDDIM = 32;
const int BLOCKDIM = 1024;

__device__ int project_device(const float x1_, const float y1_, const float z1_,
                               const float x2_, const float y2_, const float z2_,
                               const int nx, const int ny, const int nz,
                               const float cx, const float cy, const float cz,
                               const float sx, const float sy, const float sz,
                               const float *image)
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