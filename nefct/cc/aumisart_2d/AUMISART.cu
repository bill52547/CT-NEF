#include "AUMISART.h" // consists all required package and functions

__host__ void host_AUMISART(float *h_outimg, float *h_outproj, float *h_outnorm, float *h_outalphax, float *h_img, float *h_proj, int nx, int ny, int na, int outIter, int n_views, int n_iter, int *op_iter, float da, float ai, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles)
{
    float *d_img, *d_img0, *d_img_temp, *d_proj, *d_proj_temp, *d_img_ones, *d_proj_ones;
    int numBytesImg = nx * ny * sizeof(float);
    int numBytesProj = na * sizeof(float);
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMalloc((void**)&d_img0, numBytesImg);
    cudaMalloc((void**)&d_img_temp, numBytesImg);
    cudaMalloc((void**)&d_img_ones, numBytesImg);
    cudaMalloc((void**)&d_proj, numBytesProj);
    cudaMalloc((void**)&d_proj_temp, numBytesProj);
    cudaMalloc((void**)&d_proj_ones, numBytesProj);

    float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
    cudaMalloc((void**)&d_alpha_x, numBytesImg);
    cudaMalloc((void**)&d_alpha_y, numBytesImg);
    
    cudaMalloc((void**)&d_beta_x, numBytesImg);
    cudaMalloc((void**)&d_beta_y, numBytesImg);
    
    cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
    // host_initial(d_img, nx, ny, 0.0f);
    host_initial(d_alpha_x, nx, ny, 0.0f);
    host_initial(d_alpha_y, nx, ny, 0.0f);
    
    host_initial(d_beta_x, nx, ny, 0.0f);
    host_initial(d_beta_y, nx, ny, 0.0f);
    
    mexPrintf("Start iteration\n");

    float tempNorm, tempNorm0;

    for (int i_iter = 0; i_iter < n_iter; i_iter ++)
    {
        if (op_iter[i_iter] == 1)
        {   
            for (int i_view = 0; i_view < n_views; i_view ++)
            {   
                mexPrintf("iIter = %d / %d, and iView = %d / %d.\n", i_iter + 1, n_iter, i_view + 1, n_views); mexEvalString("drawnow;");

                host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, nx, ny);
                cudaMemcpy(d_proj, h_proj + na * i_view, numBytesProj, cudaMemcpyHostToDevice);

                host_add(d_proj, d_proj_temp, na, 1, -1.0);
                cudaMemcpy(h_outproj + na * i_view, d_proj, numBytesProj, cudaMemcpyDeviceToHost);
                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_initial(d_img_ones, nx, ny, 1.0f);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_division(d_img_temp, d_img_ones, nx, ny);

                host_add(d_img, d_img_temp, nx, ny, lambda);
            }
            cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
        }
        else
        {   
            if (i_iter == 0)
                cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
            else
                cudaMemcpy(d_img, h_outimg, numBytesImg, cudaMemcpyHostToDevice);
            for (int i_view = 1; i_view < n_views; i_view += 10)
            {   
                mexPrintf("iIter = %d / %d, and iView = %d / %d.", i_iter + 1, n_iter, i_view + 1, n_views); 

                host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, nx, ny);
                cudaMemcpy(d_proj, h_proj + na * i_view, numBytesProj, cudaMemcpyHostToDevice);
                float vd = volumes[i_view] - volumes[i_view - 1];
                float fd = flows[i_view] - flows[i_view - 1];
                host_add(d_proj, d_proj_temp, na, 1, -1.0f); // new b
                host_initial(d_img0, nx, ny, 0.0f);
                host_add2(d_img0, d_alpha_y, nx, ny, d_img, vd, 1);
                host_add2(d_img0, d_alpha_x, nx, ny, d_img, vd, 2);

                host_add2(d_img0, d_beta_y, nx, ny, d_img, fd, 1);
                host_add2(d_img0, d_beta_x, nx, ny, d_img, fd, 2);

                host_projection(d_proj_temp, d_img0, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_add(d_proj, d_proj_temp, na, 1, 1.0f); // new b

                h_outnorm[i_iter * n_views + i_view] = tempNorm / tempNorm0;
                mexPrintf("error on projection = %f\n", tempNorm / tempNorm0);mexEvalString("drawnow;");
                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_initial2(d_img_ones, nx, ny, d_img, -vd, -fd);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_division(d_img_temp, d_img_ones, nx, ny);

                host_add2(d_alpha_y, d_img_temp, nx, ny, d_img, volumes[i_view - 1] - volumes[i_view], 1);
                host_add2(d_alpha_x, d_img_temp, nx, ny, d_img, volumes[i_view - 1] - volumes[i_view], 2);
                host_add2(d_beta_y, d_img_temp, nx, ny, d_img, flows[i_view - 1] - flows[i_view], 1);
                host_add2(d_beta_x, d_img_temp, nx, ny, d_img, flows[i_view - 1] - flows[i_view], 2);
            }

        }
    }
    cudaMemcpy(h_outalphax, d_alpha_x, numBytesImg, cudaMemcpyDeviceToHost);
            
    cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);   

    cudaFree(d_img);
    cudaFree(d_img);
    cudaFree(d_img_temp);
    cudaFree(d_proj);
    cudaFree(d_proj_temp);
    cudaFree(d_img_ones);
    cudaFree(d_proj_ones);
    cudaFree(d_alpha_x);
    cudaFree(d_alpha_y);
    cudaFree(d_beta_x);
    cudaFree(d_beta_y);

    cudaDeviceReset();
}

__host__ void host_AUMISART(float *h_outimg, float *h_outproj, float *h_outnorm, float *h_outalphax, float *h_img, float *h_proj, int nx, int ny, int na,  int outIter, int n_views, int n_iter, int *op_iter, float da, float ai, float SO, float SD, float dx, float lambda, float* volumes, float* flows, float* err_weights, float* angles, float *ax, float *ay, float *bx, float *by)
{
    float *d_img, *d_img1, *d_img0, *d_img_temp, *d_proj, *d_proj_temp, *d_img_ones, *d_proj_ones;
    int numBytesImg = nx * ny * sizeof(float);
    int numBytesProj = na * sizeof(float);
    cudaMalloc((void**)&d_img, numBytesImg);
    cudaMalloc((void**)&d_img1, numBytesImg);
    cudaMalloc((void**)&d_img0, numBytesImg);
    cudaMalloc((void**)&d_img_temp, numBytesImg);
    cudaMalloc((void**)&d_img_ones, numBytesImg);
    cudaMalloc((void**)&d_proj, numBytesProj);
    cudaMalloc((void**)&d_proj_temp, numBytesProj);
    cudaMalloc((void**)&d_proj_ones, numBytesProj);

    cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);

    float *d_alpha_x, *d_alpha_y, *d_beta_x, *d_beta_y;
    cudaMalloc((void**)&d_alpha_x, numBytesImg);
    cudaMalloc((void**)&d_alpha_y, numBytesImg);
    
    cudaMalloc((void**)&d_beta_x, numBytesImg);
    cudaMalloc((void**)&d_beta_y, numBytesImg);
    
    // host_initial(d_img, nx, ny, 0.0f);
    cudaMemcpy(d_alpha_x, ax, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha_y, ay, numBytesImg, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_beta_x, bx, numBytesImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_y, by, numBytesImg, cudaMemcpyHostToDevice);
    
    mexPrintf("Start iteration\n");


    float tempNorm, tempNorm0;
    int P = 1;
    for (int i_iter = 0; i_iter < n_iter; i_iter ++)
    {
        if (op_iter[i_iter] == 1)
        {   
            for (int i_view = 0; i_view < n_views; i_view ++)
            {   
                int iv1, iv2;
                iv1 = i_view - i_view % P;
                iv2 = i_view % P;
                if (i_view == 0)
                    mexPrintf("iIter = %d / %d, and iView = %d / %d.\n", i_iter + 1, n_iter, i_view + 1, n_views); mexEvalString("drawnow;");
                
                if (i_view % P == 0)
                {
                    host_deform(d_img1, d_img, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
                    host_projection(d_proj_temp, d_img1, angles[i_view], SO, SD, da, na, ai, nx, ny);
                }
                else 
                {
                    host_deform(d_img1, d_img, nx, ny, volumes[iv1], flows[iv1], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
                    host_deform2(d_img_temp, d_img1, nx, ny, volumes[i_view] - volumes[iv1], flows[i_view] - flows[iv1], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);   
                    host_projection(d_proj_temp, d_img_temp, angles[i_view], SO, SD, da, na, ai, nx, ny);
                }
                    
                cudaMemcpy(d_proj, h_proj + na * i_view, numBytesProj, cudaMemcpyHostToDevice);

                host_add(d_proj, d_proj_temp, na, 1, -1.0);
                // stat = cublasSnrm2(handle, nx * ny * nz, d_img1, 1, &tempNorm);

                cudaMemcpy(h_outproj + na * i_view, d_proj, numBytesProj, cudaMemcpyDeviceToHost);
                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_initial(d_img_ones, nx, ny, 1.0f);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_division(d_img_temp, d_img_ones, nx, ny);

                if (i_view % P == 0)
                {
                    host_deform_invert(d_img1, d_img_temp, nx, ny, volumes[i_view], flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
                    host_add(d_img, d_img1, nx, ny, lambda);
                }
                else
                {
                    host_deform2(d_img1, d_img_temp, nx, ny, volumes[iv1] - volumes[i_view], flows[iv1] - flows[i_view], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
                    host_deform_invert(d_img_temp, d_img1, nx, ny, volumes[iv1], flows[iv1], d_alpha_x, d_alpha_y, d_beta_x, d_beta_y);
                    host_add(d_img, d_img_temp, nx, ny, lambda);
                }
            }
            cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);
        }
        else
        {   
            if (i_iter == 0)
                cudaMemcpy(d_img, h_img, numBytesImg, cudaMemcpyHostToDevice);
            else
                cudaMemcpy(d_img, h_outimg, numBytesImg, cudaMemcpyHostToDevice);
            for (int i_view = 1; i_view < n_views; i_view += P)
            {   
                mexPrintf("iIter = %d / %d, and iView = %d / %d.", i_iter + 1, n_iter, i_view + 1, n_views); 

                host_projection(d_proj_temp, d_img, angles[i_view], SO, SD, da, na, ai, nx, ny);
                cudaMemcpy(d_proj, h_proj + na * i_view, numBytesProj, cudaMemcpyHostToDevice);
                float vd = volumes[i_view] - volumes[i_view - 1];
                float fd = flows[i_view] - flows[i_view - 1];
                host_add(d_proj, d_proj_temp, na, 1, -1.0f); // new b
                host_initial(d_img0, nx, ny, 0.0f);
                host_add2(d_img0, d_alpha_y, nx, ny, d_img, vd, 1);
                host_add2(d_img0, d_alpha_x, nx, ny, d_img, vd, 2);
                host_add2(d_img0, d_beta_y, nx, ny, d_img, fd, 1);
                host_add2(d_img0, d_beta_x, nx, ny, d_img, fd, 2);
                
                host_projection(d_proj_temp, d_img0, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_add(d_proj, d_proj_temp, na, 1, 1.0f); // new b

                host_backprojection(d_img_temp, d_proj, angles[i_view], SO, SD, da, na, ai, nx, ny);

                host_initial2(d_img_ones, nx, ny, d_img, -vd, -fd);
                host_projection(d_proj_ones, d_img_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_backprojection(d_img_ones, d_proj_ones, angles[i_view], SO, SD, da, na, ai, nx, ny);
                host_division(d_img_temp, d_img_ones, nx, ny);

                host_add2(d_alpha_y, d_img_temp, nx, ny, d_img, volumes[i_view - 1] - volumes[i_view], 1);
                host_add2(d_alpha_x, d_img_temp, nx, ny, d_img, volumes[i_view - 1] - volumes[i_view], 2);
                host_add2(d_beta_y, d_img_temp, nx, ny, d_img, flows[i_view - 1] - flows[i_view], 1);
                host_add2(d_beta_x, d_img_temp, nx, ny, d_img, flows[i_view - 1] - flows[i_view], 2);
            }

        }
    }
    cudaMemcpy(h_outalphax, d_alpha_x, numBytesImg, cudaMemcpyDeviceToHost);
            
    // cudaMemcpy(h_outimg, d_img, numBytesImg, cudaMemcpyDeviceToHost);   

    cudaFree(d_img);
    cudaFree(d_img1);
    cudaFree(d_img0);

    cudaFree(d_img_temp);
    cudaFree(d_proj);
    cudaFree(d_proj_temp);
    cudaFree(d_img_ones);
    cudaFree(d_proj_ones);
    cudaFree(d_alpha_x);
    cudaFree(d_alpha_y);
    cudaFree(d_beta_x);
    cudaFree(d_beta_y);

    cudaDeviceReset();
}
