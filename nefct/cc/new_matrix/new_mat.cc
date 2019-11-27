#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("NewMat")
    .Input("grid: int32")
    .Output("dest: float");

void new_mat_gpu(const int *grid, float *dest);

class NewMatOp : public OpKernel
{
public:
    explicit NewMatOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor

        const Tensor &grid = context->input(0);
        auto grid_flat = grid.flat<int>();

        // define the shape of output tensors.
        Tensor *dest_out = NULL;
        int grid_cpu[2];
        cudaMemcpy(grid_cpu, grid_flat.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
        OP_REQUIRES_OK(context, context->allocate_output(0, {grid_cpu[0], grid_cpu[1]}, &dest_out));
        auto dest_flat = dest_out->flat<float>();
        cudaMemset(dest_flat.data(), 0, sizeof(float) * dest_flat.size());

        new_mat_gpu(grid_flat.data(), dest_flat.data());
    }
};

#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("NewMat", NewMatOp);

#undef REGISTER_GPU_KERNEL
