#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

#include <ctime>

using namespace tensorflow;

REGISTER_OP("RotateAlone")
	.Input("image: float")
	.Input("grid: int32")
	.Output("deformed_image: float")
	.Attr("angle: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

REGISTER_OP("Shift")
	.Input("image: float")
	.Input("grid: int32")
	.Output("deformed_image: float")
	.Attr("distance: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

REGISTER_OP("RotateShift")
	.Input("image: float")
	.Input("grid: int32")
	.Output("deformed_image: float")
	.Attr("angle: float")
	.Attr("distance: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

void rotate_tex(const float *img,
				const int *grid, const float angle,
				float *img1);

void shift_tex(const float *img,
			   const int *grid, const float distance,
			   float *img1);

void rs_tex(const float *img,
			const int *grid, const float angle, const float distance,
			float *img1);

class RotateTexOp : public OpKernel
{
public:
	explicit RotateTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("angle", &angle));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &grid_tensor = context->input(1);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		rotate_tex(image_flat.data(),
				   grid_flat.data(), angle,
				   image_out_flat.data());
	}

private:
	float angle;
};

class ShiftTexOp : public OpKernel
{
public:
	explicit ShiftTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("distance", &distance));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &grid_tensor = context->input(1);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		shift_tex(image_flat.data(),
				  grid_flat.data(), distance,
				  image_out_flat.data());
	}

private:
	float distance;
};

class RotateShiftTexOp : public OpKernel
{
public:
	explicit RotateShiftTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("angle", &angle));
		OP_REQUIRES_OK(context, context->GetAttr("distance", &distance));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &grid_tensor = context->input(1);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		rs_tex(image_flat.data(),
			   grid_flat.data(), angle, distance,
			   image_out_flat.data());
	}

private:
	float angle;
	float distance;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("RotateAlone", RotateTexOp);
REGISTER_GPU_KERNEL("Shift", ShiftTexOp);
REGISTER_GPU_KERNEL("RotateShift", RotateShiftTexOp);

#undef REGISTER_GPU_KERNEL
