#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("ProjectFlat")
	.Input("image: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("angle: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

REGISTER_OP("ProjectCyli")
	.Input("image: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("angle: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float");

void project_flat(const float *image, const int *shape, const float *center,
				  const float *size, const float *angles,
				  const int na, const int nv,
				  const float SID, const float SAD,
				  const float da, const float ai,
				  float *pv_values);

void project_cyli(const float *image, const int *shape, const float *center,
				  const float *size, const float *angles,
				  const int na, const int nv,
				  const float SID, const float SAD,
				  const float da, const float ai,
				  float *pv_values);

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
		const Tensor &grid = context->input(1);
		const Tensor &center = context->input(2);
		const Tensor &size = context->input(3);
		const Tensor &angle = context->input(4);

		auto image_flat = image.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
		auto angle_flat = angle.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_flat(image_flat.data(), grid_flat.data(), center_flat.data(),
					 size_flat.data(), angle_flat.data(),
					 na, nv,
					 SID, SAD, da, ai,
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
		const Tensor &grid = context->input(1);
		const Tensor &center = context->input(2);
		const Tensor &size = context->input(3);
		const Tensor &angle = context->input(4);

		auto image_flat = image.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
		auto angle_flat = angle.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_cyli(image_flat.data(), grid_flat.data(), center_flat.data(),
					 size_flat.data(), angle_flat.data(),
					 na, nv,
					 SID, SAD, da, ai,
					 projection_value_flat.data());
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

REGISTER_GPU_KERNEL("ProjectFlat", ProjectFlatOp);
REGISTER_GPU_KERNEL("ProjectCyli", ProjectCyliOp);

#undef REGISTER_GPU_KERNEL
