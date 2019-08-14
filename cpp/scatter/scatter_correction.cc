//
// Created by Minghao on 5/9/2019.
//


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

#include <ctime>

using namespace tensorflow;

REGISTER_OP("ScatterOverLors")
.Input("pos: float")
.Input("crystal_pos: float")
.Input("emission: float")
.Input("atten: float")
.Input("u_map: float")
.Input("unit_size: int32")
.Input("lors_inds: int32")
.Input("shape: int32")
.Output("out: float") // with first half scatter_ab and second half scale_ab
.Output("scale_ab: float")
.Attr("n_pos: int")
.Attr("low_win: float")
.Attr("high_win: float")
.Attr("energy_res: float")
.Attr("nb_blocks_per_ring: int")
.Attr("unit_area: float")
.Attr("n_lors: int")
.SetShapeFn([](
::tensorflow::shape_inference::InferenceContext *c
) {
c->set_output(0, c->input(6));
return

Status::OK();

});


void scatter_over_lors(const float *pos, const int n_pos, const float *crystal_pos,
                       const float low_win, const float high_win, const float energy_res,
                       const float *emission, const float *atten, const int nb_blocks_per_ring,
                       const float *u_map, const float *unit_size, const float unit_area,
                       const int *lors_inds, const int n_lors, const int *shape,
                       float *scatter_ab, float *scale_ab);

class ScatterOverLorsOp : public OpKernel {
public:
	explicit ScatterOverLorsOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("n_pos", &n_pos));
		OP_REQUIRES_OK(context, context->GetAttr("low_win", &low_win));
		OP_REQUIRES_OK(context, context->GetAttr("high_win", &high_win));
		OP_REQUIRES_OK(context, context->GetAttr("energy_res", &energy_res));
		OP_REQUIRES_OK(context, context->GetAttr("nb_blocks_per_ring", &nb_blocks_per_ring));
		OP_REQUIRES_OK(context, context->GetAttr("unit_area", &unit_area));
		OP_REQUIRES_OK(context, context->GetAttr("n_lors", &n_lors));
	}

	void Compute(OpKernelContext *context) override {
		// Grab the input tensor
		const Tensor &pos_tensor = context->input(0);
		const Tensor &crystal_pos_tensor = context->input(1);
		const Tensor &emission_tensor = context->input(2);
		const Tensor &atten_tensor = context->input(3);
		const Tensor &u_map_tensor = context->input(4);
		const Tensor &unit_size_tensor = context->input(5);
		const Tensor &lors_inds_tensor = context->input(6);
		const Tensor &shape_tensor = context->input(7);

		// Create an output tensor
		Tensor *out_tensor = NULL;

		// define the shape of output tensors.
		OP_REQUIRES_OK(context, context->allocate_output(0, lors_inds_tensor.shape(), &out_tensor));

		auto out_flat = out_tensor->flat<float>();
		cudaMemset(out_flat.data(), 0, sizeof(float) * out_flat.size());

		auto pos_flat = pos_tensor.flat<float>();
		auto crystal_pos_flat = crystal_pos_tensor.flat<float>();
		auto emission_flat = emission_tensor.flat<float>();
		auto atten_flat = atten_tensor.flat<float>();
		auto u_map_flat = u_map_tensor.flat<float>();
		auto unit_size_flat = unit_size_tensor.flat<float>();
		auto lors_inds_flat = lors_inds_tensor.flat<int>();
		auto shape_flat = shape_tensor.flat<int>();

		scatter_over_lors(pos_flat.data(), n_pos, crystal_pos_flat.data(), low_win, high_win, energy_res,
		                  emission_flat.data(), atten_flat.data(),
		                  nb_blocks_per_ring, u_map_flat.data(), unit_size_flat.data(), unit_area,
		                  lors_inds_flat.data(), n_lors, shape_flat.data(),
		                  out_flat.data(), out_flat.data() + n_lors);
	}


private:
	int n_pos;
	float low_win;
	float high_win;
	float energy_res;
	int nb_blocks_per_ring;
	float unit_area;
	int n_lors;

};


#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)
REGISTER_GPU_KERNEL("ScatterOverLors", ScatterOverLorsOp);

#undef REGISTER_GPU_KERNEL