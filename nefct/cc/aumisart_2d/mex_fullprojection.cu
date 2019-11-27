#include "mex.h"
__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_deform_invert(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_invert(float *mx2, float *my2, float *mx, float *my, int nx, int ny);
__global__ void kernel_invert(float *mx2, float *my2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, int nx, int ny);
__global__ void kernel_forwardDVF(float *mx, float *my, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y, float volume, float flow, int nx, int ny);
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny);
__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny);
#define MAX(a,b) (((a) > (b)) ? a : b)
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

#define MIN(a,b) (((a) < (b)) ? a : b)
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#define ABS(x) ((x) > 0 ? x : -(x))
#define PI 3.141592653589793f

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16 
#define BLOCKSIZE_Z 1
__global__ void kernel_projection(float *proj, float *img, float *angle, float SO, float SD, float da, int na, float ai, int nx, int ny, int n_views);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define GEO_PARA prhs[1]
#define ITER_PARA prhs[2]
#define OUT_PROJ plhs[0]

int nx, ny, na, n_views, numImg, numBytesImg, numSingleProj, numBytesSingleProj, numProj, numBytesProj;
float da, ai, SO, SD, *angles;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

numImg = nx * ny; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

// detector plane resolutions
if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_views"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of views, defined as iter_para.n_views.\n");


numSingleProj = na;
numBytesSingleProj = numSingleProj * sizeof(float);
numProj = na * n_views;
numBytesProj = numProj * sizeof(float);

// detector resolution
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
else{
    da = 1.0f;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}

// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    ai -= (float)na / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - (float)na / 2 + 0.5f;
}

if (mxGetField(GEO_PARA, 0, "SO") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");

if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid projection angles, as iter_para.angles.\n");

float *h_alpha_x, *h_alpha_y, *h_beta_x, *h_beta_y;
h_alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
h_alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
h_beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
h_beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
cudaMalloc((void**)&d_alpha_x, numBytesImg);
cudaMalloc((void**)&d_alpha_y, numBytesImg);

cudaMalloc((void**)&d_beta_x, numBytesImg);
cudaMalloc((void**)&d_beta_y, numBytesImg);

// host_initial(d_img, nx, ny, 0.0f);
cudaMemcpy(d_alpha_x, h_alpha_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_alpha_y, h_alpha_y, numBytesImg, cudaMemcpyHostToDevice);

cudaMemcpy(d_beta_x, h_beta_x, numBytesImg, cudaMemcpyHostToDevice);
cudaMemcpy(d_beta_y, h_beta_y, numBytesImg, cudaMemcpyHostToDevice);

float *volumes, *flows;
volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));

float *d_img, *d_img1, *d_proj;
cudaMalloc((void**)&d_img, nx * ny * sizeof(float));
cudaMalloc((void**)&d_img1, nx * ny * sizeof(float));

float *h_img;
h_img = (float*)mxGetData(IN_IMG);
cudaMemcpy(d_img, h_img, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
cudaMalloc((void**)&d_proj, na * sizeof(float));

const dim3 gridSize_singleProj((na + 16 - 1) / 16);
const dim3 blockSize(16);

OUT_PROJ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[2] = {(mwSize)na, (mwSize)n_views};

mxSetDimensions(OUT_PROJ, outDim, 2);
mxSetData(OUT_PROJ, mxMalloc(numBytesProj));
float *h_outproj = (float*)mxGetData(OUT_PROJ);

for (int i_view = 0; i_view < n_views; i_view ++)
{   
    host_deform(d_img1, d_img, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);

    kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img1, angles[i_view], SO, SD, da, na, ai, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(h_outproj + i_view * na, d_proj, numBytesSingleProj, cudaMemcpyDeviceToHost);
}

cudaFree(d_proj);

cudaFree(d_img);
cudaFree(d_img1);

cudaDeviceReset();
return;
}


__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny){
    int ia = 16 * blockIdx.x + threadIdx.x;
    if (ia >= na)
        return;
    int id = ia;
    proj[id] = 0.0f;
    float x1, y1, x2, y2, x20, y20, cphi, sphi;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    x20 = SD - SO;
    y20 = (ia + ai) * da; // locate the detector cell center before any rotation
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    float x21, y21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21)){
    // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2;
        int Yi1, Yi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++){
            // calculate y indices of intersecting voxel candidates
            float xl, xr, yl, yr, ratio;
            float cyll, cylr, cyrl, cyrr, xc;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;
            
            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2;
            ratio = yl / xr;
            cylr = ratio * xc + y1 + ny / 2;
            ratio = yr / xl;
            cyrl = ratio * xc + y1 + ny / 2;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2;

            yi1 = MIN4(cyll, cylr, cyrl, cyrr); Yi1 = (int)floorf(yi1);
            yi2 = MAX4(cyll, cylr, cyrl, cyrr); Yi2 = (int)floorf(yi2);

            xc = (float)ix + 0.5f - (float)nx / 2 - x1 ;

            float wy;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1); wy /= (yi2 - yi1);
                proj[id] += img[ix + iy * nx] * wy / ABS(x21) * sqrt(x21 * x21 + y21 * y21);                
            }        
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)    
    else{
        float xi1, xi2;
        int Xi1, Xi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++){
            // calculate y indices of intersecting voxel candidates
            float yl, yr, xl, xr, ratio;
            float cxll, cxlr, cxrl, cxrr, yc;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;
            
            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2;
            ratio = xl / yr;
            cxlr = ratio * yc + x1 + nx / 2;
            ratio = xr / yl;
            cxrl = ratio * yc + x1 + nx / 2;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2;

            xi1 = MIN4(cxll, cxlr, cxrl, cxrr); Xi1 = (int)floorf(xi1);
            xi2 = MAX4(cxll, cxlr, cxrl, cxrr); Xi2 = (int)floorf(xi2);

            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            float wx;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1); wx /= (xi2 - xi1);
                proj[id] += img[ix + iy * nx] * wx / ABS(y21) * sqrt(x21 * x21 + y21 * y21);
            }        
        }            
    }
}


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
