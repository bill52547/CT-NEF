#include "AUMISART.h" // consists all required package and functions

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PROJ prhs[1]
#define GEO_PARA prhs[2]
#define ITER_PARA prhs[3]
#define OUT_IMG plhs[0]
// #define OUT_ERR plhs[1]

int nx, ny, na, outIter, n_iter, *op_iter, n_views;
float da, ai, SO, SD, dx, lambda;
float *volumes, *flows, *err_weights, *angles;

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

// voxel resolution dx, which is also the scaling factor of the whole system
if (mxGetField(GEO_PARA, 0, "dx") != NULL)
    dx = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "dx"));
else{
    dx = 1.0f;
    mexPrintf("Automatically set voxel size dx to 1. \n");
    mexPrintf("If don't want that default value, please set para.dx manually.\n");
}

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


if (mxGetField(ITER_PARA, 0, "n_iter") != NULL)
    n_iter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_iter")); // number of views in this bin
else{
    n_iter = 1;
    mexPrintf("Automatically set number of iterations to 1. \n");
    mexPrintf("If don't want that default value, please set iter_para.n_iter manually.\n");
}

if (mxGetField(ITER_PARA, 0, "op_iter") != NULL)
    op_iter = (int*)mxGetData(mxGetField(ITER_PARA, 0, "op_iter")); // number of views in this bin
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid iteration distribution as iter_para.op_iter.\n");
}

if (mxGetField(ITER_PARA, 0, "n_views") != NULL)
    n_views = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "n_views"));
else{
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number bins, which is denoted as iter_para.n_views.\n");
}

if (mxGetField(ITER_PARA, 0, "lambda") != NULL)
    lambda = (float)mxGetScalar(mxGetField(ITER_PARA, 0, "lambda"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.lambda.\n");

if (mxGetField(ITER_PARA, 0, "outIter") != NULL)
    outIter = (int)mxGetScalar(mxGetField(ITER_PARA, 0, "outIter"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid coefficience iter_para.outIter.\n");


if (mxGetField(ITER_PARA, 0, "volumes") != NULL)
    volumes = (float*)mxGetData(mxGetField(ITER_PARA, 0, "volumes"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume in iter_para.volumes.\n");  

if (mxGetField(ITER_PARA, 0, "flows") != NULL)
    flows = (float*)mxGetData(mxGetField(ITER_PARA, 0, "flows"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.flows.\n");    

if (mxGetField(ITER_PARA, 0, "err_weights") != NULL)
    err_weights = (float*)mxGetData(mxGetField(ITER_PARA, 0, "err_weights"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid flow in iter_para.err_weights.\n");    

if (mxGetField(ITER_PARA, 0, "angles") != NULL)
    angles = (float*)mxGetData(mxGetField(ITER_PARA, 0, "angles"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid angles iter_para.angles.\n");

// load initial guess of image
float *h_img;
h_img = (float*)mxGetData(IN_IMG);

// load true projection value
float *h_proj;
h_proj = (float*)mxGetData(PROJ);


// setup output images
OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[2] = {(mwSize)nx, (mwSize)ny};
mxSetDimensions(OUT_IMG, outDim, 2);
mxSetData(OUT_IMG, mxMalloc(nx * ny * sizeof(float)));
float *h_outimg = (float*)mxGetData(OUT_IMG);

plhs[1] = mxCreateNumericMatrix(n_iter * n_views, 1, mxSINGLE_CLASS, mxREAL);
float *h_outnorm = (float*)mxGetData(plhs[1]);

plhs[2] = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mxSetDimensions(plhs[2], outDim, 2);
mxSetData(plhs[2], mxMalloc(nx * ny * sizeof(float)));
float *h_outalphax = (float*)mxGetData(plhs[2]);

const mwSize outDim2[2] = {(mwSize)na, (mwSize)n_views};
plhs[3] = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mxSetDimensions(plhs[3], outDim2, 2);
mxSetData(plhs[3], mxMalloc(na * n_views * sizeof(float)));
float *h_outproj = (float*)mxGetData(plhs[3]);

mexPrintf("Start main body of AUMISART. \n");
if (mxGetField(ITER_PARA, 0, "alpha_x") == NULL)
{
    host_AUMISART(h_outimg, h_outproj, h_outnorm, h_outalphax, h_img, h_proj, nx, ny, na, outIter, n_views, n_iter, op_iter, da, ai, SO, SD, dx, lambda, volumes, flows, err_weights, angles);
}
else
{
    float *alpha_x, *alpha_y, *beta_x, *beta_y;
    alpha_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_x"));
    alpha_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "alpha_y"));
    beta_x = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_x"));
    beta_y = (float*)mxGetData(mxGetField(ITER_PARA, 0, "beta_y"));
    host_AUMISART(h_outimg, h_outproj, h_outnorm, h_outalphax, h_img, h_proj, nx, ny, na, outIter, n_views, n_iter, op_iter, da, ai, SO, SD, dx, lambda, volumes, flows, err_weights, angles, alpha_x, alpha_y, beta_x, beta_y);
}
    
return;
}

