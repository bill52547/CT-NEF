#include "mex_ConjGrad.h" // consists all required package and functions
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]
// #define OUT_ERR plhs[1]

int nx, ny, na, n_iter_inner, n_views;
float da, ai, SO, SD, mu;
float *volumes, *flows, *angles;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");


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
    ai -= ((float)na / 2 - 0.5f);
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - ((float)na / 2 - 0.5f);
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


if (mxGetField(ITER_PARA, 0, "n_iter_inner") != NULL)
    n_iter_inner = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter_inner")); // number of views in this bin
else{
    n_iter_inner = 1;
    mexPrintf("Automatically set number of iterations to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter_inner manually.\n");
}

if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_views"));
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number bins, which is denoted as iter_para.n_views.\n");
}

if (mxGetField(ITER_PARA, 0, "mu") != NULL)
    mu = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "mu"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.mu.\n");

if (mxGetField(ITER_PARA, 0, "volumes") != NULL)
    volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume in iter_para.volumes.\n");  

if (mxGetField(ITER_PARA, 0, "flows") != NULL)
    flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.flows.\n");    

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

// load true projection value
float *h_b;
h_b = (float*)mxGetData(PROJ);

// setup output images
const mwSize outDim[2] = {(mwSize)nx, (mwSize)ny};
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mxSetDimensions(OUT_IMG, outDim, 2);
mxSetData(OUT_IMG, mxMalloc(nx * ny * sizeof(float)));
float *h_outimg = (float*)mxGetData(OUT_IMG);
// for (int i = 0; i < nx * ny; i++)
//     h_outimg[i] = 0.0f;
// return;
float *alpha_x, *alpha_y, *beta_x, *beta_y;
alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
host_CG(h_outimg, h_img, h_b, nx, ny, na, n_views, n_iter_inner, da, ai, SO, SD, mu, volumes, flows, angles, alpha_x, alpha_y, beta_x, beta_y);
cudaDeviceReset();
return;
}

__host__ void host_CG(float *h_outimg, float *h_img, float *h_b, int nx, int ny, int na, int n_views, int n_iter_inner, float da, float ai, float SO, float SD, float mu, float *volumes, float *flows, float *angles, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y)
{
    float *d_img, *d_img1, *d_Ap;
    int numBytesImg = nx * ny * sizeof(float);
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMalloc((void**)&d_img1, numBytesImg);
    cudaMalloc((void**)&d_Ap, numBytesImg);

    cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);

    float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
    cudaMalloc((void**)&d_alpha_x, numBytesImg);
    cudaMalloc((void**)&d_alpha_y, numBytesImg);
    
    cudaMalloc((void**)&d_beta_x, numBytesImg);
    cudaMalloc((void**)&d_beta_y, numBytesImg);

    cudaMemcpy(d_alpha_x, alpha_x, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha_y, alpha_y, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_x, beta_x, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_y, beta_y, numBytesImg, cudaMemcpyHostToDevice);

    float *d_res, *d_p;
    cudaMalloc((void**)&d_res, numBytesImg);
    cudaMalloc((void**)&d_p, numBytesImg);

    cudaMemcpy(d_res, h_b, numBytesImg, cudaMemcpyHostToDevice);
    A_operation(d_img1, d_img, nx, ny, na, n_views, da, ai, SO, SD, mu, volumes, flows, angles, d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
    host_add(d_res, d_img1, nx, ny, -1.0f);
    cudaMemcpy(d_p, d_res, numBytesImg, cudaMemcpyDeviceToDevice);
    // for (int i = 0; i < nx * ny; i++)
    //     rsold += d_res[i] * d_res[i];
    float rsold, rsnew, temp;
    float *h_temp1 = (float*)malloc(nx * ny * sizeof(float));
    float *h_temp2 = (float*)malloc(nx * ny * sizeof(float));
    // rsold = cu_vector_multiply(d_res, d_res, nx, ny);
    cudaMemcpy(h_temp1, d_res, numBytesImg, cudaMemcpyDeviceToHost);
    rsold = 0.0f;
    for (int i = 0; i < nx * ny; i++)
        rsold += h_temp1[i] * h_temp1[i];

    float errp = 1e20f, err;
    mexPrintf("Print something.\n");

    for (int iter = 0; iter < n_iter_inner; iter ++)
    {
        A_operation(d_Ap, d_p, nx, ny, na, n_views, da, ai, SO, SD, mu, volumes, flows, angles, d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
        float alpha;
        // for (int i = 0; i < nx * ny; i++)
        //     temp += d_p[i] * d_Ap[i];
        // cu_vector_multiply(temp, d_p, d_Ap, nx, ny);
        cudaMemcpy(h_temp1, d_p, numBytesImg, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_temp2, d_Ap, numBytesImg, cudaMemcpyDeviceToHost);
        temp = 0.0f;
        for (int i = 0; i < nx * ny; i++)
            temp += h_temp1[i] * h_temp2[i];

        // temp = cu_vector_multiply(d_p, d_Ap, nx, ny);
        alpha = rsold / temp;
        host_add(d_img, d_p, nx, ny, alpha);
        host_add(d_res, d_Ap, nx, ny, -alpha);
        // for (int i = 0; i < nx * ny; i++)
        //     rsnew += d_res[i] * d_res[i];
        cudaMemcpy(h_temp1, d_res, numBytesImg, cudaMemcpyDeviceToHost);
        rsnew = 0.0f;
        for (int i = 0; i < nx * ny; i++)
            rsnew += h_temp1[i] * h_temp1[i];
        // rsnew = cu_vector_multiply(d_res, d_res, nx, ny);
        err = rsnew;
        if (err > rsold && rsold > errp)
        {
            mexPrintf("break at iteration %d\n", iter);
            break;
        }
        cudaMemcpy(d_Ap, d_p, numBytesImg, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_p, d_res, numBytesImg, cudaMemcpyDeviceToDevice);
        host_add(d_p, d_Ap, nx, ny, rsnew / rsold);
        errp = rsold;
        rsold = rsnew;
    }
    cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_img1);
    cudaFree(d_Ap);
    cudaFree(d_p);
    cudaFree(d_res);
    cudaFree(d_alpha_x);
    cudaFree(d_alpha_y);
    cudaFree(d_beta_x);
    cudaFree(d_beta_y);
}

