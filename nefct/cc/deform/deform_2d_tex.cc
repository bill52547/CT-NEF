#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

#include <ctime>

using namespace tensorflow;

REGISTER_OP("DeformTwo")
		.Input("image: float")
		.Input("mx: float")
		.Input("my: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

REGISTER_OP("DeformInvertTwo")
		.Input("image: float")
		.Input("mx: float")
		.Input("my: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

void deform_2d_tex(const float *img,
                const float *mx, const float *my,
                const int *grid, float *img1);

void deform_invert_2d_tex(const float *img,
                       const float *mx, const float *my,
                       const int *grid, float *img1);

class DeformTexTwoOp : public OpKernel
{
public:
	explicit DeformTexTwoOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &mx_tensor = context->input(1);
		const Tensor &my_tensor = context->input(2);
		const Tensor &grid_tensor = context->input(3);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto mx_flat = mx_tensor.flat<float>();
		auto my_flat = my_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		deform_2d_tex(image_flat.data(),
						mx_flat.data(), my_flat.data(),
						grid_flat.data(),
						image_out_flat.data());
	}
};

class DeformInvertTwoTexOp : public OpKernel
{
public:
	explicit DeformInvertTwoTexOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &mx_tensor = context->input(1);
		const Tensor &my_tensor = context->input(2);
		const Tensor &grid_tensor = context->input(3);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto mx_flat = mx_tensor.flat<float>();
		auto my_flat = my_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		deform_invert_2d_tex(image_flat.data(),
								mx_flat.data(), my_flat.data(),
								grid_flat.data(),
								image_out_flat.data());
	}
};


#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(            \
			Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("DeformTwo", DeformTexTwoOp);
REGISTER_GPU_KERNEL("DeformInvertTwo", DeformInvertTwoTexOp);


#undef REGISTER_GPU_KERNEL
