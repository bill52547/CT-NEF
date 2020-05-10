#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define MAX(a, b) (((a) > (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

//#define MAX4(a, b, c, d) (((((a) > (b)) ? (a) : (b)) > (((c) > (d)) ? (c) : (d))) > (((a) > (b)) ? (a) : (b)) : (((c) > (d)) ? (c) : (d)))
#define MIN(a, b) (((a) < (b)) ? a : b)
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#define ABS(x) ((x) > 0 ? x : -(x))
#define PI 3.141592653589793f
const int GRIDDIM_X = 16;
const int GRIDDIM_Y = 16;
const int GRIDDIM_Z = 4;
const float eps_ = 0.01;

__global__ void kernel_backprojection(const float *vproj,
                                      const float *angles, const float *offsets,
                                      const float *ai, const float *bi,
                                      const int mode,
                                      const float SD, const float SO,
                                      const int nx, const int ny, const int nz,
                                      const float da,  const int na,
                                      const float db, const int nb,
                                      float *image);

__global__ void initial_kernel(float *image, const int N, const float value);

__global__ void mk_dvf(const float *ax, const float *ay, const float *az,
                       const float *bx, const float *by, const float *bz,
                       const float *cx, const float *cy, const float *cz,
                       const float *v_data, const float *f_data,
                       const int nx, const int ny, const int nz,
                       float *mx, float *my, float *mz);

__global__ void mk_dvf2(const float *ax, const float *ay, const float *az,
                        const float *bx, const float *by, const float *bz,
                        const float *cx, const float *cy, const float *cz,
                        const float v_data, const float f_data,
                        const int nx, const int ny, const int nz,
                        float *mx, float *my, float *mz);

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

void mc_back_project_gpu(const float *pv_values,
                         const float *angles, const float *offsets,
                         const float *ai, const float *bi,
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
                         const float da, const int na,
                         const float db, const int nb,
                         float *image)
{
    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                         (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                         (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    initial_kernel<<<ny * nz, nx>>>(image, nx * ny * nz, 0.0f);
    initial_kernel<<<ny * nz, nx>>>(temp_img, nx * ny * nz, 0.0f);
    for (int iv = 0; iv < nv; iv++)
    {

        kernel_backprojection<<<shapeSize, blockSize>>>(pv_values + iv * na * nb,
                                                        angles + iv, offsets + iv,
                                                        ai + iv, bi + iv,
                                                        mode,
                                                        SD, SO,
                                                        nx, ny, nz,
                                                        da, na,
                                                        db, nb,
                                                        temp_img);
        mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                                bx, by, bz,
                                cx, cy, cz,
                                v_data + iv, f_data + iv,
                                nx, ny, nz,
                                mx, my, mz);
        deform_invert_tex(temp_img, mx, my, mz, nx, ny, nz, 1, image);
    }
}

void mc_back_project_gpu_single(const float *pv_values,
                         const float *angles, const float *offsets,
                         const float *ai, const float *bi,
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
                         const float da, const int na,
                         const float db, const int nb,
                         float *image)
{
    const dim3 shapeSize((nx + GRIDDIM_X - 1) / GRIDDIM_X,
                         (ny + GRIDDIM_Y - 1) / GRIDDIM_Y,
                         (nz + GRIDDIM_Z - 1) / GRIDDIM_Z);
    const dim3 blockSize(GRIDDIM_X, GRIDDIM_Y, GRIDDIM_Z);
    initial_kernel<<<ny * nz, nx>>>(image, nx * ny * nz, 0.0f);
    initial_kernel<<<ny * nz, nx>>>(temp_img, nx * ny * nz, 0.0f);
    initial_kernel<<<ny * nz, nx>>>(temp_img1, nx * ny * nz, 0.0f);
    kernel_backprojection<<<shapeSize, blockSize>>>(pv_values,
                                                    angles, offsets, ai, bi,
                                                    mode,
                                                    SD, SO,
                                                    nx, ny, nz,
                                                    da, na,
                                                    db, nb,
                                                    temp_img);
    mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v_data, f_data,
                            nx, ny, nz,
                            mx, my, mz);
    deform_invert_tex(temp_img, mx, my, mz, nx, ny, nz, 0, temp_img1);
    mk_dvf<<<ny * nz, nx>>>(ax, ay, az,
                            bx, by, bz,
                            cx, cy, cz,
                            v0_data, f0_data,
                            nx, ny, nz,
                            mx, my, mz);
    deform_tex(temp_img1, mx, my, mz, nx, ny, nz, 0, image);
}

#endif
