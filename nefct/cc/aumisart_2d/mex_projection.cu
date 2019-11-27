#include "mex.h"
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
#define BLOCKSIZE_Z 4
__global__ void kernel_projection(float *proj, float *img, float angle, float SO, float SD, float da, int na, float ai, int nx, int ny);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define GEO_PARA prhs[1]
#define OUT_PROJ plhs[0]

int nx, ny, na, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, ai, SO, SD, angle;

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

numSingleProj = na;
numBytesSingleProj = numSingleProj * sizeof(float);

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

if (mxGetField(GEO_PARA, 0, "angle") != NULL)
    angle = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "angle"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid projection angle, which is denoted with para.angle.\n");

float *d_img, *d_proj;
cudaMalloc((void**)&d_img, nx * ny * sizeof(float));

float *h_img;
h_img = (float*)mxGetData(IN_IMG);
cudaMemcpy(d_img, h_img, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void**)&d_proj, na * sizeof(float));
const dim3 gridSize_singleProj((na + 16 - 1) / 16, 1, 1);
const dim3 blockSize(16, 1, 1);

kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img, angle, SO, SD, da, na, ai, nx, ny);
cudaDeviceSynchronize();

OUT_PROJ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[1] = {(mwSize)na};

mxSetDimensions(OUT_PROJ, outDim, 1);
mxSetData(OUT_PROJ, mxMalloc(na * sizeof(float)));
float *h_outproj = (float*)mxGetData(OUT_PROJ);

cudaMemcpy(h_outproj, d_proj, numBytesSingleProj, cudaMemcpyDeviceToHost);

cudaFree(d_proj);

cudaFree(d_img);
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