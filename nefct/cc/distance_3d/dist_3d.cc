#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("ProjectFlatThree")
	.Input("image: float")
	.Input("shape: int32")
	.Input("offsets: float")
	.Input("angles: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

REGISTER_OP("ProjectCyliThree")
	.Input("image: float")
	.Input("shape: int32")
	.Input("offsets: float")
	.Input("angles: float")
	.Output("projection_value: float")
	.Attr("SID: float")
	.Attr("SAD: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float");

REGISTER_OP("BackProjectFlatThree")
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

REGISTER_OP("BackProjectCyliThree")
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

REGISTER_OP("BackProjectFlatFdkThree")
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

REGISTER_OP("BackProjectCyliFdkThree")
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

void project_flat_gpu(const float *image, const int *shape, const float *offsets,
					  const float *angles, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai, const int na,
					  const float db, const float bi, const int nb,
					  float *pv_values);

void project_cyli_gpu(const float *image, const int *shape, const float *offsets,
					  const float *angles, const int nv,
					  const float SID, const float SAD,
					  const float da, const float ai, const int na,
					  const float db, const float bi, const int nb,
					  float *pv_values);

void back_project_flat_gpu(const float *pv_values, const int *shape, const float *offsets,
						   const float *angles, const int nv,
						   const float SID, const float SAD,
						   const float da, const float ai, const int na,
						   const float db, const float bi, const int nb,
						   float *image);

void back_project_cyli_gpu(const float *pv_values, const int *shape, const float *offsets,
						   const float *angles, const int nv,
						   const float SID, const float SAD,
						   const float da, const float ai, const int na,
						   const float db, const float bi, const int nb,
						   float *image);

void back_project_flat_fdk_gpu(const float *pv_values, const int *shape, const float *offsets,
							   const float *angles, const int nv,
							   const float SID, const float SAD,
							   const float da, const float ai, const int na,
							   const float db, const float bi, const int nb,
							   float *image);

void back_project_cyli_fdk_gpu(const float *pv_values, const int *shape, const float *offsets,
							   const float *angles, const int nv,
							   const float SID, const float SAD,
							   const float da, const float ai, const int na,
							   const float db, const float bi, const int nb,
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
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("db", &db));
		OP_REQUIRES_OK(context, context->GetAttr("bi", &bi));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &image = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &offsets = context->input(2);
		const Tensor &angles = context->input(3);

		auto image_flat = image.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto offsets_flat = offsets.flat<float>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, nb, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_flat_gpu(image_flat.data(), shape_flat.data(), offsets_flat.data(),
						 angles_flat.data(), nv,
						 SID, SAD, da, ai, na, db, bi, nb,
						 projection_value_flat.data());
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
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("db", &db));
		OP_REQUIRES_OK(context, context->GetAttr("bi", &bi));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor

		const Tensor &image = context->input(0);
		const Tensor &shape = context->input(1);
		const Tensor &offsets = context->input(2);
		const Tensor &angles = context->input(3);

		auto image_flat = image.flat<float>();
		auto shape_flat = shape.flat<int>();
		auto offsets_flat = offsets.flat<float>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *projection_value = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nv, nb, na}, &projection_value));
		auto projection_value_flat = projection_value->flat<float>();
		cudaMemset(projection_value_flat.data(), 0, sizeof(float) * projection_value_flat.size());

		project_cyli_gpu(image_flat.data(), shape_flat.data(), offsets_flat.data(),
						 angles_flat.data(), nv,
						 SID, SAD, da, ai, na, db, bi, nb,
						 projection_value_flat.data());
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
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[3];
		cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_flat_gpu(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
							  angles_flat.data(), nv,
							  SID, SAD, da, ai, na, db, bi, nb,
							  image_flat.data());
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
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[3];
		cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_cyli_gpu(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
							  angles_flat.data(), nv,
							  SID, SAD, da, ai, na, db, bi, nb,
							  image_flat.data());
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

class BackProjectFlatFdkOp : public OpKernel
{
public:
	explicit BackProjectFlatFdkOp(OpKernelConstruction *context) : OpKernel(context)
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
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[3];
		cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_flat_fdk_gpu(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
								  angles_flat.data(), nv,
								  SID, SAD, da, ai, na, db, bi, nb,
								  image_flat.data());
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

class BackProjectCyliFdkOp : public OpKernel
{
public:
	explicit BackProjectCyliFdkOp(OpKernelConstruction *context) : OpKernel(context)
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
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		int shape_cpu[3];
		cudaMemcpy(shape_cpu, shape_flat.data(), 3 * sizeof(int), cudaMemcpyDeviceToHost);
		OP_REQUIRES_OK(context, context->allocate_output(0, {shape_cpu[0], shape_cpu[1], shape_cpu[2]}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_cyli_fdk_gpu(projection_value_flat.data(), shape_flat.data(), offsets_flat.data(),
								  angles_flat.data(), nv,
								  SID, SAD, da, ai, na, db, bi, nb,
								  image_flat.data());
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

REGISTER_GPU_KERNEL("ProjectFlatThree", ProjectFlatOp);
REGISTER_GPU_KERNEL("ProjectCyliThree", ProjectCyliOp);
REGISTER_GPU_KERNEL("BackProjectFlatThree", BackProjectFlatOp);
REGISTER_GPU_KERNEL("BackProjectCyliThree", BackProjectCyliOp);
REGISTER_GPU_KERNEL("BackProjectFlatFdkThree", BackProjectFlatFdkOp);
REGISTER_GPU_KERNEL("BackProjectCyliFdkThree", BackProjectCyliFdkOp);

#undef REGISTER_GPU_KERNEL
