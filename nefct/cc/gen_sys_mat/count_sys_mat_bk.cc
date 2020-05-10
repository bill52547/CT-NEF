#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("CountFlat")
	.Input("shape: int32")
	.Input("template: int32")
	.Output("n_in_row: int32")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(1));
		return Status::OK();
	});

REGISTER_OP("CountCyli")
	.Input("shape: int32")
	.Input("template: int32")
	.Output("n_in_row: int32")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(1));
		return Status::OK();
	});

REGISTER_OP("GenerateFlat")
	.Input("shape: int32")
	.Input("row: int32")
	.Input("template: int32")
	.Output("col: int32")
	.Output("data: float")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(2));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

REGISTER_OP("GenerateCyli")
	.Input("shape: int32")
	.Input("row: int32")
	.Input("template: int32")
	.Output("col: int32")
	.Output("data: float")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(2));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

void count_flat_gpu(const int *shape,
					const float SD, const float SO,
					const float da, const float ai, const int na,
					const float db, const float bi, const int nb,
					int *n_in_row);

void count_cyli_gpu(const int *shape,
					const float SD, const float SO,
					const float da, const float ai, const int na,
					const float db, const float bi, const int nb,
					int *n_in_row);

void generate_flat_gpu(const int *shape,
					   const float SD, const float SO,
					   const float da, const float ai, const int na,
					   const float db, const float bi, const int nb,
					   const int *row, int *col, float *data);

void generate_cyli_gpu(const int *shape,
					   const float SD, const float SO,
					   const float da, const float ai, const int na,
					   const float db, const float bi, const int nb,
					   const int *row, int *col, float *data);

class CountFlatOp : public OpKernel
{
public:
	explicit CountFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &temp_ = context->input(1);

		auto shape_flat = shape.flat<int>();

		// define the shape of output tensors.
		Tensor *n_in_row = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &n_in_row));
		auto n_in_row_flat = n_in_row->flat<int>();
		cudaMemset(n_in_row_flat.data(), 0, sizeof(int) * n_in_row_flat.size());

		count_flat_gpu(shape_flat.data(),
					   SD, SO, da, ai, na, db, bi, nb,
					   n_in_row_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class CountCyliOp : public OpKernel
{
public:
	explicit CountCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &temp_ = context->input(1);

		auto shape_flat = shape.flat<int>();

		// define the shape of output tensors.
		Tensor *n_in_row = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &n_in_row));
		auto n_in_row_flat = n_in_row->flat<int>();
		cudaMemset(n_in_row_flat.data(), 0, sizeof(int) * n_in_row_flat.size());

		count_flat_gpu(shape_flat.data(),
					   SD, SO, da, ai, na, db, bi, nb,
					   n_in_row_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class GenerateFlatOp : public OpKernel
{
public:
	explicit GenerateFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &row = context->input(1);
		const Tensor &temp_ = context->input(2);

		auto shape_flat = shape.flat<int>();
		auto row_flat = row.flat<int>();

		// define the shape of output tensors.
		Tensor *col = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &col));
		auto col_flat = col->flat<int>();
		cudaMemset(col_flat.data(), 0, sizeof(int) * col_flat.size());

		Tensor *data = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, temp_.shape(), &data));
		auto data_flat = data->flat<float>();
		cudaMemset(data_flat.data(), 0, sizeof(float) * data_flat.size());

		generate_flat_gpu(shape_flat.data(),
						  SD, SO, da, ai, na, db, bi, nb,
						  row_flat.data(), col_flat.data(), data_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class GenerateCyliOp : public OpKernel
{
public:
	explicit GenerateCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &row = context->input(1);
		const Tensor &temp_ = context->input(2);

		auto shape_flat = shape.flat<int>();
		auto row_flat = row.flat<int>();

		// define the shape of output tensors.
		Tensor *col = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &col));
		auto col_flat = col->flat<int>();
		cudaMemset(col_flat.data(), 0, sizeof(int) * col_flat.size());

		Tensor *data = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, temp_.shape(), &data));
		auto data_flat = data->flat<float>();
		cudaMemset(data_flat.data(), 0, sizeof(float) * data_flat.size());

		generate_cyli_gpu(shape_flat.data(),
						  SD, SO, da, ai, na, db, bi, nb,
						  row_flat.data(), col_flat.data(), data_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

REGISTER_OP("CountBackFlat")
	.Input("shape: int32")
	.Input("template: int32")
	.Output("n_in_row: int32")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(1));
		return Status::OK();
	});

REGISTER_OP("CountBackCyli")
	.Input("shape: int32")
	.Input("template: int32")
	.Output("n_in_row: int32")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(1));
		return Status::OK();
	});

REGISTER_OP("GenerateBackFlat")
	.Input("shape: int32")
	.Input("row: int32")
	.Input("template: int32")
	.Output("col: int32")
	.Output("data: float")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(2));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

REGISTER_OP("GenerateBackCyli")
	.Input("shape: int32")
	.Input("row: int32")
	.Input("template: int32")
	.Output("col: int32")
	.Output("data: float")
	.Attr("SD: float")
	.Attr("SO: float")
	.Attr("na: int")
	.Attr("da: float")
	.Attr("ai: float")
	.Attr("nb: int")
	.Attr("db: float")
	.Attr("bi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->input(2));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

void count_flat_back_gpu(const int *shape,
						 const float SD, const float SO,
						 const float da, const float ai, const int na,
						 const float db, const float bi, const int nb,
						 int *n_in_row);

void count_cyli_back_gpu(const int *shape,
						 const float SD, const float SO,
						 const float da, const float ai, const int na,
						 const float db, const float bi, const int nb,
						 int *n_in_row);

void generate_flat_back_gpu(const int *shape,
							const float SD, const float SO,
							const float da, const float ai, const int na,
							const float db, const float bi, const int nb,
							const int *row, int *col, float *data);

void generate_cyli_back_gpu(const int *shape,
							const float SD, const float SO,
							const float da, const float ai, const int na,
							const float db, const float bi, const int nb,
							const int *row, int *col, float *data);

class CountBackFlatOp : public OpKernel
{
public:
	explicit CountBackFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &temp_ = context->input(1);

		auto shape_flat = shape.flat<int>();

		// define the shape of output tensors.
		Tensor *n_in_row = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &n_in_row));
		auto n_in_row_flat = n_in_row->flat<int>();
		cudaMemset(n_in_row_flat.data(), 0, sizeof(int) * n_in_row_flat.size());

		count_flat_back_gpu(shape_flat.data(),
							SD, SO, da, ai, na, db, bi, nb,
							n_in_row_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class CountBackCyliOp : public OpKernel
{
public:
	explicit CountBackCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &temp_ = context->input(1);

		auto shape_flat = shape.flat<int>();

		// define the shape of output tensors.
		Tensor *n_in_row = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &n_in_row));
		auto n_in_row_flat = n_in_row->flat<int>();
		cudaMemset(n_in_row_flat.data(), 0, sizeof(int) * n_in_row_flat.size());

		count_cyli_back_gpu(shape_flat.data(),
							SD, SO, da, ai, na, db, bi, nb,
							n_in_row_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class GenerateBackFlatOp : public OpKernel
{
public:
	explicit GenerateBackFlatOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &row = context->input(1);
		const Tensor &temp_ = context->input(2);

		auto shape_flat = shape.flat<int>();
		auto row_flat = row.flat<int>();

		// define the shape of output tensors.
		Tensor *col = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &col));
		auto col_flat = col->flat<int>();
		cudaMemset(col_flat.data(), 0, sizeof(int) * col_flat.size());

		Tensor *data = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, temp_.shape(), &data));
		auto data_flat = data->flat<float>();
		cudaMemset(data_flat.data(), 0, sizeof(float) * data_flat.size());

		generate_flat_back_gpu(shape_flat.data(),
							   SD, SO, da, ai, na, db, bi, nb,
							   row_flat.data(), col_flat.data(), data_flat.data());
	}

private:
	float SD;
	float SO;
	int na;
	float da;
	float ai;
	int nb;
	float db;
	float bi;
};

class GenerateBackCyliOp : public OpKernel
{
public:
	explicit GenerateBackCyliOp(OpKernelConstruction *context) : OpKernel(context)
	{
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

		const Tensor &shape = context->input(0);
		const Tensor &row = context->input(1);
		const Tensor &temp_ = context->input(2);

		auto shape_flat = shape.flat<int>();
		auto row_flat = row.flat<int>();

		// define the shape of output tensors.
		Tensor *col = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, temp_.shape(), &col));
		auto col_flat = col->flat<int>();
		cudaMemset(col_flat.data(), 0, sizeof(int) * col_flat.size());

		Tensor *data = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, temp_.shape(), &data));
		auto data_flat = data->flat<float>();
		cudaMemset(data_flat.data(), 0, sizeof(float) * data_flat.size());

		generate_cyli_back_gpu(shape_flat.data(),
							   SD, SO, da, ai, na, db, bi, nb,
							   row_flat.data(), col_flat.data(), data_flat.data());
	}

private:
	float SD;
	float SO;
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

REGISTER_GPU_KERNEL("CountFlat", CountFlatOp);
REGISTER_GPU_KERNEL("CountCyli", CountCyliOp);
REGISTER_GPU_KERNEL("GenerateFlat", GenerateFlatOp);
REGISTER_GPU_KERNEL("GenerateCyli", GenerateCyliOp);
REGISTER_GPU_KERNEL("CountBackFlat", CountBackFlatOp);
REGISTER_GPU_KERNEL("CountBackCyli", CountBackCyliOp);
REGISTER_GPU_KERNEL("GenerateBackFlat", GenerateBackFlatOp);
REGISTER_GPU_KERNEL("GenerateBackCyli", GenerateBackCyliOp);

#undef REGISTER_GPU_KERNEL
