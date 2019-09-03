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

REGISTER_OP("BackProjectFlat")
	.Input("projection_value: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("angle: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nx: int")
	.Attr("ny: int");

REGISTER_OP("BackProjectCyli")
	.Input("projection_value: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("angle: float")
	.Output("image: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nx: int")
	.Attr("ny: int");

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

void backproject_flat(const float *pv_values, const int *grid, const float *center,
					  const float *size, const float *angles,
					  const int na, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai,
					  float *image);

void backproject_cyli(const float *pv_values, const int *grid, const float *center,
					  const float *size, const float *angles,
					  const int na, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai,
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
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &grid = context->input(1);
		const Tensor &center = context->input(2);
		const Tensor &size = context->input(3);
		const Tensor &angle = context->input(4);

		auto projection_value_flat = projection_value.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
		auto angle_flat = angle.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *image_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nx, ny}, &image_value));
		auto image_value_flat = image_value->flat<float>();
		cudaMemset(image_value_flat.data(), 0, sizeof(float) * image_value_flat.size());

		backproject_flat(projection_value_flat.data(), grid_flat.data(), center_flat.data(),
						 size_flat.data(), angle_flat.data(),
						 na, nv,
						 SID, SAD, da, ai,
						 image_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
	int nx;
	int ny;
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
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection_value = context->input(0);
		const Tensor &grid = context->input(1);
		const Tensor &center = context->input(2);
		const Tensor &size = context->input(3);
		const Tensor &angle = context->input(4);

		auto projection_value_flat = projection_value.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
		auto angle_flat = angle.flat<float>();
		unsigned int nv = angle_flat.size();

		// define the shape of output tensors.
		Tensor *image_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nx, ny}, &image_value));
		auto image_value_flat = image_value->flat<float>();
		cudaMemset(image_value_flat.data(), 0, sizeof(float) * image_value_flat.size());

		backproject_cyli(projection_value_flat.data(), grid_flat.data(), center_flat.data(),
						 size_flat.data(), angle_flat.data(),
						 na, nv,
						 SID, SAD, da, ai,
						 image_value_flat.data());
	}

private:
	float SID;
	float SAD;
	int na;
	float da;
	float ai;
	int nx;
	int ny;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ProjectFlat", ProjectFlatOp);
REGISTER_GPU_KERNEL("ProjectCyli", ProjectCyliOp);
REGISTER_GPU_KERNEL("BackProjectFlat", BackProjectFlatOp);
REGISTER_GPU_KERNEL("BackProjectCyli", BackProjectCyliOp);

#undef REGISTER_GPU_KERNEL
