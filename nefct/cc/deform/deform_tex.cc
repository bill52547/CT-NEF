#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

#include <ctime>

using namespace tensorflow;

REGISTER_OP("Deform")
		.Input("image: float")
		.Input("mx: float")
		.Input("my: float")
		.Input("mz: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

REGISTER_OP("DeformInvert")
		.Input("image: float")
		.Input("mx: float")
		.Input("my: float")
		.Input("mz: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

REGISTER_OP("Shift")
		.Input("image: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.Attr("mx: float")
		.Attr("my: float")
		.Attr("mz: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

REGISTER_OP("ShiftZ")
		.Input("image: float")
		.Input("grid: int32")
		.Output("deformed_image: float")
		.Attr("mz: float")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(0));
			return Status::OK();
		});

REGISTER_OP("Rotate")
	.Input("image: float")
	.Input("grid: int32")
	.Output("deformed_image: float")
	.Attr("cos_phi: float")
	.Attr("sin_phi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
	return Status::OK();
	});

void deform_tex(const float *img,
                const float *mx, const float *my, const float *mz,
                const int *grid, float *img1);

void deform_invert_tex(const float *img,
                       const float *mx, const float *my, const float *mz,
                       const int *grid, float *img1);

void shift_tex(const float *img,
               const float mx, const float my, const float mz,
               const int *grid, float *img1);

void shift_z_tex(const float *img, const float mz,
                 const int *grid, float *img1);

void rotate_tex(const float *img,
                const float cos_phi, const float sin_phi,
                const int *grid, float *img1);

class DeformTexOp : public OpKernel
{
public:
	explicit DeformTexOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &mx_tensor = context->input(1);
		const Tensor &my_tensor = context->input(2);
		const Tensor &mz_tensor = context->input(3);
		const Tensor &grid_tensor = context->input(4);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto mx_flat = mx_tensor.flat<float>();
		auto my_flat = my_tensor.flat<float>();
		auto mz_flat = mz_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		deform_tex(image_flat.data(),
							 mx_flat.data(), my_flat.data(), mz_flat.data(),
							 grid_flat.data(),
							 image_out_flat.data());
	}
};

class DeformInvertTexOp : public OpKernel
{
public:
	explicit DeformInvertTexOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &mx_tensor = context->input(1);
		const Tensor &my_tensor = context->input(2);
		const Tensor &mz_tensor = context->input(3);
		const Tensor &grid_tensor = context->input(4);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto mx_flat = mx_tensor.flat<float>();
		auto my_flat = my_tensor.flat<float>();
		auto mz_flat = mz_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();

		deform_invert_tex(image_flat.data(),
											mx_flat.data(), my_flat.data(), mz_flat.data(),
											grid_flat.data(),
											image_out_flat.data());
	}
};

class ShiftTexOp : public OpKernel
{
public:
	explicit ShiftTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mx", &mx));
		OP_REQUIRES_OK(context, context->GetAttr("my", &my));
		OP_REQUIRES_OK(context, context->GetAttr("mz", &mz));
	}

	void Compute(OpKernelContext *context) override
	{

		// Grab the geometries of an image.
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
							mx, my, mz,
							grid_flat.data(),
							image_out_flat.data());
	}

private:
	// string model;
	float mx;
	float my;
	float mz;
};

class ShiftZTexOp : public OpKernel
{
public:
	explicit ShiftZTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mz", &mz));
	}

	void Compute(OpKernelContext *context) override
	{

		// Grab the geometries of an image.
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

		shift_z_tex(image_flat.data(), mz,
								grid_flat.data(),
								image_out_flat.data());
	}

private:
	// string model;
	float mz;
};



class RotateTexOp : public OpKernel
{
public:
	explicit RotateTexOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("cos_phi", &cos_phi));
		OP_REQUIRES_OK(context, context->GetAttr("sin_phi", &sin_phi));
	}

	void Compute(OpKernelContext *context) override
	{

		// Grab the geometries of an image.
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
		           cos_phi, sin_phi,
		           grid_flat.data(),
		           image_out_flat.data());
	}

private:
	float cos_phi;
	float sin_phi;
};


#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(            \
			Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("Deform", DeformTexOp);
REGISTER_GPU_KERNEL("DeformInvert", DeformInvertTexOp);
REGISTER_GPU_KERNEL("Shift", ShiftTexOp);
REGISTER_GPU_KERNEL("ShiftZ", ShiftZTexOp);
REGISTER_GPU_KERNEL("Rotate", RotateTexOp);


#undef REGISTER_GPU_KERNEL
