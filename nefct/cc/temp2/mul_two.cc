#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("MulTwo")
    .Input("src: float")
    .Output("dest: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        //set the size of backpro_image the same as the input image.
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void multi_two(const float *src, float *dest);

class MulTwo : public OpKernel
{
public:
    explicit MulTwo(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {

        // Grab the geometries of an image.
        const Tensor &src = context->input(0);
        auto src_flat = src.flat<float>();

        // Create an output backprojected image
        Tensor *dest = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, src.shape(),
                                                         &dest));
        //set the initial backprojection image value to zero.
        auto dest_flat = dest->flat<float>();
        cudaMemset(dest_flat.data(), 0, sizeof(float) * dest_flat.size());

        multi_two(src_flat.data(), dest_flat.data());
    }
};

#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("MulTwo", MulTwo);

#undef REGISTER_GPU_KERNEL
