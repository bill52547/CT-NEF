#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("BackProjectFdk")
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

REGISTER_OP("McBackProjectFdk")
	.Input("projection: float")
	.Input("angles: float")
	.Input("offsets: float")
	.Input("ax: float")
	.Input("ay: float")
	.Input("az: float")
	.Input("bx: float")
	.Input("by: float")
	.Input("bz: float")
	.Input("cx: float")
	.Input("cy: float")
	.Input("cz: float")
	.Input("v_data: float")
	.Input("f_data: float")
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

void back_project_gpu(const float *pv_values,
					  const float *angles, const float *offsets,
					  const int mode,
					  const int nv,
					  const float SD, const float SO,
					  const int nx, const int ny, const int nz,
					  const float da, const float ai, const int na,
					  const float db, const float bi, const int nb,
					  float *image);

void mc_back_project_gpu(const float *pv_values,
						 const float *angles, const float *offsets,
						 const float *ax, const float *ay, const float *az,
						 const float *bx, const float *by, const float *bz,
						 const float *cx, const float *cy, const float *cz,
						 const float *v_data, const float *f_data,
						 float *temp_img, float *temp_img1,
						 float *mx, float *my, float *mz,
						 const int mode,
						 const int nv,
						 const float SD, const float SO,
						 const int nx, const int ny, const int nz,
						 const float da, const float ai, const int na,
						 const float db, const float bi, const int nb,
						 float *image);

class BackProjectFdkOp : public OpKernel
{
public:
	explicit BackProjectFdkOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("nz", &nz));
		OP_REQUIRES_OK(context, context->GetAttr("SD", &SD));
		OP_REQUIRES_OK(context, context->GetAttr("SO", &SO));
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

		const Tensor &projection = context->input(0);
		const Tensor &angles = context->input(1);
		const Tensor &offsets = context->input(2);

		auto projection_flat = projection.flat<float>();
		auto offsets_flat = offsets.flat<float>();
		auto angles_flat = angles.flat<float>();
		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nz, ny, nx}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		back_project_gpu(projection_flat.data(),
						 angles_flat.data(), offsets_flat.data(),
						 mode,
						 nv,
						 SD, SO,
						 nx, ny, nz,
						 da, ai, na, db, bi, nb,
						 image_flat.data());
	}

private:
	int mode;
	float SD;
	float SO;
	int nx;
	int ny;
	int nz;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class McBackProjectFdkOp : public OpKernel
{
public:
	explicit McBackProjectFdkOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("mode", &mode));
		OP_REQUIRES_OK(context, context->GetAttr("nx", &nx));
		OP_REQUIRES_OK(context, context->GetAttr("ny", &ny));
		OP_REQUIRES_OK(context, context->GetAttr("nz", &nz));
		OP_REQUIRES_OK(context, context->GetAttr("SD", &SD));
		OP_REQUIRES_OK(context, context->GetAttr("SO", &SO));
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

		const Tensor &projection = context->input(0);
		const Tensor &angles = context->input(1);
		const Tensor &offsets = context->input(2);

		const Tensor &ax = context->input(3);
		const Tensor &ay = context->input(4);
		const Tensor &az = context->input(5);
		const Tensor &bx = context->input(6);
		const Tensor &by = context->input(7);
		const Tensor &bz = context->input(8);
		const Tensor &cx = context->input(9);
		const Tensor &cy = context->input(10);
		const Tensor &cz = context->input(11);

		const Tensor &v_data = context->input(12);
		const Tensor &f_data = context->input(13);

		auto projection_flat = projection.flat<float>();
		auto offsets_flat = offsets.flat<float>();
		auto angles_flat = angles.flat<float>();

		auto ax_flat = ax.flat<float>();
		auto ay_flat = ay.flat<float>();
		auto az_flat = az.flat<float>();
		auto bx_flat = bx.flat<float>();
		auto by_flat = by.flat<float>();
		auto bz_flat = bz.flat<float>();
		auto cx_flat = cx.flat<float>();
		auto cy_flat = cy.flat<float>();
		auto cz_flat = cz.flat<float>();

		auto v_data_flat = v_data.flat<float>();
		auto f_data_flat = f_data.flat<float>();

		unsigned int nv = angles_flat.size();

		// define the shape of output tensors.
		Tensor *image = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, {nz, ny, nx}, &image));
		auto image_flat = image->flat<float>();
		cudaMemset(image_flat.data(), 0, sizeof(float) * image_flat.size());

		Tensor temp_img;
		OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {nz, ny, nx}, &temp_img));
		auto temp_img_flat = temp_img.flat<float>();
		cudaMemset(temp_img_flat.data(), 0, sizeof(float) * temp_img_flat.size());

		Tensor temp_img2;
		OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {nz, ny, nx}, &temp_img2));
		auto temp_img2_flat = temp_img2.flat<float>();
		cudaMemset(temp_img2_flat.data(), 0, sizeof(float) * temp_img2_flat.size());

		Tensor mx;
		OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {nz, ny, nx}, &mx));
		auto mx_flat = mx.flat<float>();
		cudaMemset(mx_flat.data(), 0, sizeof(float) * mx_flat.size());

		Tensor my;
		OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {nz, ny, nx}, &my));
		auto my_flat = my.flat<float>();
		cudaMemset(my_flat.data(), 0, sizeof(float) * my_flat.size());

		Tensor mz;
		OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {nz, ny, nx}, &mz));
		auto mz_flat = mz.flat<float>();
		cudaMemset(mz_flat.data(), 0, sizeof(float) * mz_flat.size());

		mc_back_project_gpu(projection_flat.data(),
							angles_flat.data(), offsets_flat.data(),
							ax_flat.data(), ay_flat.data(), az_flat.data(),
							bx_flat.data(), by_flat.data(), bz_flat.data(),
							cx_flat.data(), cy_flat.data(), cz_flat.data(),
							v_data_flat.data(), f_data_flat.data(),
							temp_img_flat.data(), temp_img2_flat.data(),
							mx_flat.data(), my_flat.data(), mz_flat.data(),
							mode,
							nv,
							SD, SO,
							nx, ny, nz,
							da, ai, na, db, bi, nb,
							image_flat.data());
	}

private:
	int mode;
	float SD;
	float SO;
	int nx;
	int ny;
	int nz;
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

REGISTER_GPU_KERNEL("BackProjectFdk", BackProjectFdkOp);
REGISTER_GPU_KERNEL("McBackProjectFdk", McBackProjectFdkOp);

#undef REGISTER_GPU_KERNEL
