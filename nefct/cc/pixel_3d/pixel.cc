#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("BackProjectFlat")
	.Input("projection_value: float")
	.Input("shape: int32")
	.Input("offsets: float")
	.Input("angles: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

REGISTER_OP("BackProjectCyli")
	.Input("projection_value: float")
	.Input("shape: int32")
	.Input("offsets: float")
	.Input("angles: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

void backproject_flat(const float *pv_values, const int *shape, const float *offsets,
					  const float *angles,
					  const int na, const int nb, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai,
					  const float db, const float bi,
					  float *image);

void backproject_cyli(const float *pv_values, const int *shape, const float *offsets,
					  const float *angles,
					  const int na, const int nb, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai,
					  const float db, const float bi,
					  float *image);

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
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("db", &db));
		OP_REQUIRES_OK(context, context->GetAttr("bi", &bi));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &offsets = context->input(2);
		const Tensor &angles = context->input(3);

		auto projection_value_flat = projection_value.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto offsets_flat = offsets.flat<float>();
		auto angle_flat = angles.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *image_value = NULL;
		int shape_cpu[3];
		cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image_value));
        auto image_value_flat = image_value->flat<float>();
		cudaMemset(image_value_flat.data(), 0, sizeof(float) * image_value_flat.size());

		backproject_flat(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
						 angle_flat.data(),
						 na, nb, nv,
						 SID, SAD, da, ai, db, bi,
						 image_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
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
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("db", &db));
		OP_REQUIRES_OK(context, context->GetAttr("bi", &bi));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &offsets = context->input(2);
		const Tensor &angles = context->input(3);

		auto projection_value_flat = projection_value.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto offsets_flat = offsets.flat<float>();
		auto angle_flat = angles.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *image_value = NULL;
		int shape_cpu[3];
        cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image_value));
		auto image_value_flat = image_value->flat<float>();
		cudaMemset(image_value_flat.data(), 0, sizeof(float) * image_value_flat.size());

		backproject_cyli(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
						 angle_flat.data(),
						 na, nb, nv,
						 SID, SAD, da, ai, db, bi,
						 image_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("BackProjectFlat", BackProjectFlatOp);
REGISTER_GPU_KERNEL("BackProjectCyli", BackProjectCyliOp);

#undef REGISTER_GPU_KERNEL
