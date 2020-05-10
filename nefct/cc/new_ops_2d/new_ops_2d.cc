#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("Project")
	.Input("image: float")
	.Input("angles: float")
	.Input("offsets: float")
	.Output("projection: float")
	.Attr("mode: int")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("nx: int")
	.Attr("ny: int")
	.Attr("nz: int")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

REGISTER_OP("BackProject")
	.Input("projection: float")
	.Input("angles: float")
	.Input("offsets: float")
	.Output("image: float")
	.Attr("mode: int")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("nx: int")
	.Attr("ny: int")
	.Attr("nz: int")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

void project_gpu(const float *image,
				 const float *angles,
				 const int mode,
				 const int nv,
				 const float SD, const float SO,
				 const int nx, const int ny,
				 const float da, const float ai, const int na,
				 float *pv_values);

void back_project_gpu(const float *pv_values,
					  const float *angles,
					  const int mode,
					  const int nv,
					  const float SD, const float SO,
					  const int nx, const int ny,
					  const float da, const float ai, const int na,

class ProjectOp : public OpKernel
{
public:
	explicit ProjectOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("SD", &SD));
		OP_REQUIRES_OK(context, context->GetAttr("SO", &SO));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &image = context->input(0);
		const Tensor &angles = context->input(1);

		auto image_flat = image.flat<float>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *projection = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, na}, &projection));
		auto projection_flat = projection->flat<float>();
		cudaMemset(projection_flat.data(), 0, sizeof(float) * projection_flat.size());

		project_gpu(image_flat.data(),
					angles_flat.data(),
					mode,
					nv,
					SD, SO,
					nx, ny,
					da, ai, na,
					projection_flat.data());
	}

private:
	int mode;
	float SD;
	float SO;
	int nx;
	int ny;
	int na;
	float da;
	float ai;
};

class BackProjectOp : public OpKernel
{
public:
	explicit BackProjectOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("SD", &SD));
		OP_REQUIRES_OK(context, context->GetAttr("SO", &SO));
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("da", &da));
		OP_REQUIRES_OK(context, context->GetAttr("ai", &ai));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &projection = context->input(0);
		const Tensor &angles = context->input(1);

		auto projection_flat = projection.flat<float>();
		auto offsets_flat = offsets.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {ny, nx}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_gpu(projection_flat.data(),
						 angles_flat.data(), offsets_flat.data(),
						 mode,
						 nv,
						 SD, SO,
						 nx, ny,
						 da, ai, na,
						 image_flat.data());
	}

private:
	int mode;
	float SD;
	float SO;
	int nx;
	int ny;
	int na;
	float da;
	float ai;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("Project", ProjectOp);
REGISTER_GPU_KERNEL("BackProject", BackProjectOp);

#undef REGISTER_GPU_KERNEL
