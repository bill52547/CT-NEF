#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

#include <ctime>

using namespace tensorflow;

REGISTER_OP("ProjectAlone")
	.Input("image: float")
	.Input("data: float")
	.Input("indices: int32")
	.Input("indptr: int32")
	.Input("angles: float")
	.Input("distances: float")
	.Input("template: float")
	.Output("projection: float")
	.Attr("na: int")
	.Attr("nb: int")
	.Attr("nv: int")
	.Attr("nx: int")
	.Attr("ny: int")
	.Attr("nz: int")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(6));
		return Status::OK();
	});

void project_gpu(const float *img,
				 const float *data, const int *indices, const int *indptr,
				 const float *angles, const float *distances,
				 const int na, const int nb, const int nv,
				 const int nx, const int ny, const int nz,
				 float *projection);

class ProjectAloneOp : public OpKernel
{
public:
	explicit ProjectAloneOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("nv", &nv));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("nz", &nz));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &data_tensor = context->input(1);
		const Tensor &indices_tensor = context->input(2);
		const Tensor &indptr_tensor = context->input(3);
		const Tensor &angles_tensor = context->input(4);
		const Tensor &distances_tensor = context->input(5);
		const Tensor &template_tensor = context->input(6);

		// Create an output tensor
		Tensor *projection = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, template_tensor.shape(), &projection));

		auto projection_flat = projection->flat<float>();
		cudaMemset(projection_flat.data(), 0, sizeof(float) * projection_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto data_flat = data_tensor.flat<float>();
		auto indices_flat = indices_tensor.flat<int>();
		auto indptr_flat = indptr_tensor.flat<int>();
		auto angles_flat = angles_tensor.flat<float>();
		auto distances_flat = distances_tensor.flat<float>();

		project_gpu(image_flat.data(),
					data_flat.data(), indices_flat.data(), indptr_flat.data(),
					angles_flat.data(), distances_flat.data(),
					na, nb, nv,
					nx, ny, nz,
					projection_flat.data());
	}

private:
	int nx;
	int ny;
	int nz;
	int na;
	int nb;
	int nv;
};

REGISTER_OP("BackProjectAlone")
	.Input("projection: float")
	.Input("data: float")
	.Input("indices: int32")
	.Input("indptr: int32")
	.Input("angles: float")
	.Input("distances: float")
	.Input("template: float")
	.Output("image: float")
	.Attr("na: int")
	.Attr("nb: int")
	.Attr("nv: int")
	.Attr("nx: int")
	.Attr("ny: int")
	.Attr("nz: int")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(6));
		return Status::OK();
	});

void back_project_gpu(const float *proj,
					  const float *data, const int *indices, const int *indptr,
					  const float *angles, const float *distances,
					  const int na, const int nb, const int nv,
					  const int nx, const int ny, const int nz,
					  float *image);

class BackProjectAloneOp : public OpKernel
{
public:
	explicit BackProjectAloneOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("nv", &nv));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("nz", &nz));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &proj_tensor = context->input(0);
		const Tensor &data_tensor = context->input(1);
		const Tensor &indices_tensor = context->input(2);
		const Tensor &indptr_tensor = context->input(3);
		const Tensor &angles_tensor = context->input(4);
		const Tensor &distances_tensor = context->input(5);
		const Tensor &template_tensor = context->input(6);

		// Create an output tensor
		Tensor *image = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, template_tensor.shape(), &image));

		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		auto proj_flat = proj_tensor.flat<float>();
		auto data_flat = data_tensor.flat<float>();
		auto indices_flat = indices_tensor.flat<int>();
		auto indptr_flat = indptr_tensor.flat<int>();
		auto angles_flat = angles_tensor.flat<float>();
		auto distances_flat = distances_tensor.flat<float>();

		back_project_gpu(proj_flat.data(),
						 data_flat.data(), indices_flat.data(), indptr_flat.data(),
						 angles_flat.data(), distances_flat.data(),
						 na, nb, nv, nx, ny, nz,
						 image_flat.data());
	}

private:
	int na;
	int nb;
	int nv;
	int nx;
	int ny;
	int nz;
};

REGISTER_OP("SartAlone")
	.Input("image: float")
	.Input("projection: float")
	.Input("data: float")
	.Input("indices: int32")
	.Input("indptr: int32")
	.Input("data_back: float")
	.Input("indices_back: int32")
	.Input("indptr_back: int32")
	.Input("angles: float")
	.Input("distances: float")
	.Input("emap: float")
	.Output("image1: float")
	.Attr("na: int")
	.Attr("nb: int")
	.Attr("nv: int")
	.Attr("nx: int")
	.Attr("ny: int")
	.Attr("nz: int")
	.Attr("n_iter: int")
	.Attr("lamb: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

void sart_gpu(const float *img, const float *proj, const float *emap,
			  const float *data, const int *indices, const int *indptr,
			  const float *data_back, const int *indices_back, const int *indptr_back,
			  const float *angles, const float *distances,
			  const int na, const int nb, const int nv,
			  const int nx, const int ny, const int nz,
			  const int n_iter, const float lamb,
			  float *img1);

class SartAloneOp : public OpKernel
{
public:
	explicit SartAloneOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("na", &na));
		OP_REQUIRES_OK(context, context->GetAttr("nb", &nb));
		OP_REQUIRES_OK(context, context->GetAttr("nv", &nv));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("nz", &nz));
		OP_REQUIRES_OK(context, context->GetAttr("n_iter", &n_iter));
		OP_REQUIRES_OK(context, context->GetAttr("lamb", &lamb));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &proj_tensor = context->input(1);
		const Tensor &data_tensor = context->input(2);
		const Tensor &indices_tensor = context->input(3);
		const Tensor &indptr_tensor = context->input(4);
		const Tensor &data_back_tensor = context->input(5);
		const Tensor &indices_back_tensor = context->input(6);
		const Tensor &indptr_back_tensor = context->input(7);
		const Tensor &angles_tensor = context->input(8);
		const Tensor &distances_tensor = context->input(9);
		const Tensor &emap_tensor = context->input(10);

		// Create an output tensor
		Tensor *image_out = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, image_tensor.shape(), &image_out));

		auto image_out_flat = image_out->flat<float>();
		cudaMemset(image_out_flat.data(), 0, sizeof(float) * image_out_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto proj_flat = proj_tensor.flat<float>();
		auto data_flat = data_tensor.flat<float>();
		auto indices_flat = indices_tensor.flat<int>();
		auto indptr_flat = indptr_tensor.flat<int>();
		auto data_back_flat = data_back_tensor.flat<float>();
		auto indices_back_flat = indices_back_tensor.flat<int>();
		auto indptr_back_flat = indptr_back_tensor.flat<int>();
		auto angles_flat = angles_tensor.flat<float>();
		auto distances_flat = distances_tensor.flat<float>();
		auto emap_flat = emap_tensor.flat<float>();
		sart_gpu(image_flat.data(), proj_flat.data(), emap_flat.data(),
				 data_flat.data(), indices_flat.data(), indptr_flat.data(),
				 data_back_flat.data(), indices_back_flat.data(), indptr_back_flat.data(),
				 angles_flat.data(), distances_flat.data(),
				 na, nb, nv, nx, ny, nz, n_iter, lamb,
				 image_out_flat.data());
	}

private:
	int na;
	int nb;
	int nv;
	int nx;
	int ny;
	int nz;
	int n_iter;
	float lamb;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ProjectAlone", ProjectAloneOp);
REGISTER_GPU_KERNEL("BackProjectAlone", BackProjectAloneOp);
REGISTER_GPU_KERNEL("SartAlone", SartAloneOp);

#undef REGISTER_GPU_KERNEL
