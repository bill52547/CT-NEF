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
	.Input("grid: int32")
	.Input("data: float")
	.Input("indices: int32")
	.Input("indptr: int32")
	.Input("angles: float")
	.Input("distances: float")
	.Input("template: float")
	.Output("projection: float")
	.Attr("n_view: int")
	.Attr("n_det: int")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(7));
		return Status::OK();
	});

void project_gpu(const float *img, const int *grid,
				 const float *data, const int *indices, const int *indptr,
				 const float *angles, const float *distances,
				 const int n_view, const int n_det,
				 float *projection);

class ProjectAloneOp : public OpKernel
{
public:
	explicit ProjectAloneOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("n_view", &n_view));
		OP_REQUIRES_OK(context, context->GetAttr("n_det", &n_det));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &image_tensor = context->input(0);
		const Tensor &grid_tensor = context->input(1);
		const Tensor &data_tensor = context->input(2);
		const Tensor &indices_tensor = context->input(3);
		const Tensor &indptr_tensor = context->input(4);
		const Tensor &angles_tensor = context->input(5);
		const Tensor &distances_tensor = context->input(6);
		const Tensor &template_tensor = context->input(7);

		// Create an output tensor
		Tensor *projection = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, template_tensor.shape(), &projection));

		auto projection_flat = projection->flat<float>();
		cudaMemset(projection_flat.data(), 0, sizeof(float) * projection_flat.size());

		auto image_flat = image_tensor.flat<float>();
		auto grid_flat = grid_tensor.flat<int>();
		auto data_flat = data_tensor.flat<float>();
		auto indices_flat = indices_tensor.flat<int>();
		auto indptr_flat = indptr_tensor.flat<int>();
		auto angles_flat = angles_tensor.flat<float>();
		auto distances_flat = distances_tensor.flat<float>();

		project_gpu(image_flat.data(), grid_flat.data(),
					data_flat.data(), indices_flat.data(), indptr_flat.data(),
					angles_flat.data(), distances_flat.data(),
					n_view, n_det,
					projection_flat.data());
	}

private:
	int n_view;
	int n_det;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ProjectAlone", ProjectAloneOp);

#undef REGISTER_GPU_KERNEL
