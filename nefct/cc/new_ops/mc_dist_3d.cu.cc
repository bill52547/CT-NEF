#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define ABS(x) ((x > 0) ? x : -(x))
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
const float eps_ = 0.01;

__global__ void mk_dvf(const float *ax, const float *ay, const float *az,
                       const float *bx, const float *by, const float *bz,
                       const float *cx, const float *cy, const float *cz,
                       const float *v_data, const float *f_data,
                       const int nx, const int ny, const int nz,
                       float *mx, float *my, float *mz)
{
    int iyz = blockIdx.x;
    int ix = threadIdx.x;
    if (ix >= nx || iyz >= ny * nz)
    {
        return;
    }
    const int id = ix + iyz * nx;
    mx[id] = ax[id] * v_data[0] + bx[id] * f_data[0] + cx[id];
    my[id] = ay[id] * v_data[0] + by[id] * f_data[0] + cy[id];
    mz[id] = az[id] * v_data[0] + bz[id] * f_data[0] + cz[id];
}

__global__ void mk_dvf2(const float *ax, const float *ay, const float *az,
                        const float *bx, const float *by, const float *bz,
                        const float *cx, const float *cy, const float *cz,
                        const float v_data, const float f_data,
                        const int nx, const int ny, const int nz,
                        float *mx, float *my, float *mz)
{
    int iyz = blockIdx.x;
    int ix = threadIdx.x;
    if (ix >= nx || iyz >= ny * nz)
    {
        return;
    }
    const int id = ix + iyz * nx;
    mx[id] = ax[id] * v_data + bx[id] * f_data + cx[id];
    my[id] = ay[id] * v_data + by[id] * f_data + cy[id];
    mz[id] = az[id] * v_data + bz[id] * f_data + cz[id];
}

__global__ void kernel_projection(const float *image,
                                  const float *angles, const float *offsets,
                                  const int mode,
                                  const float SD, const float SO,
                                  const int nx, const int ny, const int nz,
                                  const float da, const float ai, const int na,
                                  const float db, const float bi, const int nb,
                                  float *vproj);

void deform_tex(const float *img,
                const float *mx, const float *my, const float *mz,
                const int nx, const int ny, const int nz,
                const int adding,
                float *img1);

void deform_invert_tex(const float *img,
                       const float *mx, const float *my, const float *mz,
                       const int nx, const int ny, const int nz,
                       const int adding,
                       float *img1);

void mc_project_gpu(const float *image,
                    const float *angles, const float *offsets,
                    const float *ax, const float *ay, const float *az,
                    const float *bx, const float *by, const float *bz,
                    const float *cx, const float *cy, const float *cz,
                    const float *v_data, const float *f_data,
                    float *temp_img, float *temp_img1,
                    float *mx, float *my, float *mz,
                    const int mode,
                    const int nv,
                    const float SD, const float SO,
                    const int nx, const int ny, const int nz,
                    const float da, const float ai, const int na,
                    const float db, const float bi, const int nb,
                    float *pv_values)
{
    for (int iv = 0; iv < nv; iv++)
    {
        mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                                bx, by, bz,
                                cx, cy, cz,
                                v_data + iv, f_data + iv,
                                nx, ny, nz,
                                mx, my, mz);
        deform_tex(image, mx, my, mz, nx, ny, nz, 0, temp_img1);

        kernel_projection<<<na, nb>>>(temp_img1,
                                      angles + iv, offsets + iv,
                                      mode,
                                      SD, SO,
                                      nx, ny, nz,
                                      da, ai, na,
                                      db, bi, nb,
                                      pv_values + iv * na * nb);
    }
}

void mc_project_gpu_single(const float *image,
                    const float *angles, const float *offsets,
                    const float *ax, const float *ay, const float *az,
                    const float *bx, const float *by, const float *bz,
                    const float *cx, const float *cy, const float *cz,
                    const float *v_data, const float *f_data,
                    const float *v0_data, const float *f0_data, 
                    float *temp_img, float *temp_img1,
                    float *mx, float *my, float *mz,
                    const int mode,
                    const float SD, const float SO,
                    const int nx, const int ny, const int nz,
                    const float da, const float ai, const int na,
                    const float db, const float bi, const int nb,
                    float *pv_values)
{
    mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v0_data, f0_data,
                            nx, ny, nz,
                            mx, my, mz);
    deform_invert_tex(image, mx, my, mz, nx, ny, nz, 0, temp_img);
    mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v_data, f_data,
                            nx, ny, nz,
                            mx, my, mz);
    deform_tex(temp_img, mx, my, mz, nx, ny, nz, 0, temp_img1);

    kernel_projection<<<na, nb>>>(temp_img1,
                                    angles, offsets,
                                    mode,
                                    SD, SO,
                                    nx, ny, nz,
                                    da, ai, na,
                                    db, bi, nb,
                                    pv_values);
}

#endif
