#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("ProjectFlatTwo")
	.Input("image: float")
	.Input("shape: int32")
	.Input("angles: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

REGISTER_OP("ProjectCyliTwo")
	.Input("image: float")
	.Input("shape: int32")
	.Input("angles: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

REGISTER_OP("BackProjectFlatTwo")
	.Input("projection_value: float")
	.Input("shape: int32")
	.Input("angles: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

REGISTER_OP("BackProjectCyliTwo")
	.Input("projection_value: float")
	.Input("shape: int32")
	.Input("angles: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

void project_flat_gpu(const float *image, const int *shape,
					  const float *angles, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai, const int na,
					  float *pv_values);

void project_cyli_gpu(const float *image, const int *shape,
					  const float *angles, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai, const int na,
					  float *pv_values);

void back_project_flat_gpu(const float *pv_values, const int *shape,
						   const float *angles, const int nv,
						   const float SID, const float SAD,
						   const float da, const float ai, const int na,
						   float *image);

void back_project_cyli_gpu(const float *pv_values, const int *shape,
						   const float *angles, const int nv,
						   const float SID, const float SAD,
						   const float da, const float ai, const int na,
						   float *image);

class ProjectFlatOp : public OpKernel
{
public:
	explicit ProjectFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
		OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &image = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &angles = context->input(2);

		auto image_flat = image.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_flat_gpu(image_flat.data(), shape_flat.data(),
						 angles_flat.data(), nv,
						 SID, SAD, da, ai, na,
						 projection_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
};

class ProjectCyliOp : public OpKernel
{
public:
	explicit ProjectCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
		OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &image = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &angles = context->input(2);

		auto image_flat = image.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_cyli_gpu(image_flat.data(), shape_flat.data(),
						 angles_flat.data(), nv,
						 SID, SAD, da, ai, na,
						 projection_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
};

class BackProjectFlatOp : public OpKernel
{
public:
	explicit BackProjectFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
		OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &angles = context->input(2);

		auto projection_value_flat = projection_value.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[2];
		cudaMemcpy(shape_cpu, shape_flat.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[1], shape_cpu[0]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_flat_gpu(projection_value_flat.data(), shape_flat.data(),
							  angles_flat.data(), nv,
							  SID, SAD, da, ai, na,
							  image_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
};

class BackProjectCyliOp : public OpKernel
{
public:
	explicit BackProjectCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
		OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &angles = context->input(2);

		auto projection_value_flat = projection_value.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[2];
		cudaMemcpy(shape_cpu, shape_flat.data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[1], shape_cpu[0]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_cyli_gpu(projection_value_flat.data(), shape_flat.data(),
							  angles_flat.data(), nv,
							  SID, SAD, da, ai, na,
							  image_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ProjectFlatTwo", ProjectFlatOp);
REGISTER_GPU_KERNEL("ProjectCyliTwo", ProjectCyliOp);
REGISTER_GPU_KERNEL("BackProjectFlatTwo", BackProjectFlatOp);
REGISTER_GPU_KERNEL("BackProjectCyliTwo", BackProjectCyliOp);

#undef REGISTER_GPU_KERNEL
