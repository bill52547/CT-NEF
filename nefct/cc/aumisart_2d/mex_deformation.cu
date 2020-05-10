#include "mex.h"
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define PARA prhs[1]
#define MX prhs[2]
#define MY prhs[3]

#define OUT_IMG plhs[0]

float *h_mx, *h_my, *h_img;
h_img = (float*)mxGetData(IN_IMG);
h_mx = (float*)mxGetData(MX);
h_my = (float*)mxGetData(MY);
int nx, ny;
nx = (int)mxGetScalar(mxGetField(PARA, 0, "nx"));
ny = (int)mxGetScalar(mxGetField(PARA, 0, "ny"));

float *d_mx, *d_my, *d_img1;
cudaMalloc((void**)&d_mx, nx * ny * sizeof(float));
cudaMalloc((void**)&d_my, nx * ny * sizeof(float));
cudaMalloc((void**)&d_img1, nx * ny * sizeof(float));

cudaMemcpy(d_mx, h_mx, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_my, h_my, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

OUT_IMG = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mwSize outDim[2] = {(mwSize)nx, (mwSize)ny};
mxSetDimensions(OUT_IMG, outDim, 2);
mxSetData(OUT_IMG, mxMalloc(nx * ny * sizeof(float)));
float *h_outimg = (float*)mxGetData(OUT_IMG);


cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaPitchedPtr dp_img = make_cudaPitchedPtr((void*) h_img, nx * sizeof(float), nx, ny);
cudaMemcpy3DParms copyParams = {0};
struct cudaExtent extent_img = make_cudaExtent(nx, ny, 1);
copyParams.extent = extent_img;
copyParams.kind = cudaMemcpyHostToDevice;
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


const dim3 gridSize_img((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, 1);
const dim3 blockSize(16, 16, 4);
kernel_deformation<<<gridSize_img, blockSize>>>(d_img1, tex_img, d_mx, d_my, nx, ny);
cudaDeviceSynchronize();

cudaMemcpy(h_outimg, d_img1, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
cudaFreeArray(array_img);
cudaFree(d_mx);
cudaFree(d_my);
cudaFree(d_img1);

cudaDeviceReset();
return;
}




__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx, float *my, int nx, int ny){
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny)
        return;
    // int id = iy + ix * ny;
    // float xi = iy + my[id];
    // float yi = ix + mx[id];
    int id = ix + iy * nx;
    float xi = ix + mx[id];
    float yi = iy + my[id];
    
    img1[id] = tex3D<float>(tex_img, xi + 0.5f, yi + 0.5f, 0.5f);
}
