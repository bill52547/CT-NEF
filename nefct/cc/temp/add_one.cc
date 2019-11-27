#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("AddOne")
    .Input("src: float")
    .Input("grid: int32")
    .Output("dest: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void add_one(const float *src, const int *grid, const float *dest);

class AddOne : public OpKernel
{
public:
    explicit AddOne(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor

        const Tensor &image = context->input(0);
        const Tensor &grid = context->input(1);
        auto image_flat = image.flat<float>();
        auto grid_flat = grid.flat<int>();

        // define the shape of output tensors.
        Tensor *dest_out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &dest_out));
        auto dest_flat = dest_out->flat<float>();
        cudaMemset(dest_flat.data(), 0, sizeof(float) * dest_flat.size());

        add_one(image_flat.data(), grid_flat.data(), dest_flat.data());
    }
};

#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("AddOne", AddOne);

#undef REGISTER_GPU_KERNEL
