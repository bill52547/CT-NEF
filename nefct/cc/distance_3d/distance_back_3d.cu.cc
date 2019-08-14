#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x > 0) ? x : -(x))

const int BLOCKWIDTH = 16;
const int BLOCKHEIGHT = 16;
const int BLOCKDEPTH = 4;

__device__ void backproject_device(const float *proj,
								   const int ix, const int iy, const int iz,
								   const int nx, const int ny, const int nz,
								   const float cx, const float cy, const float cz,
								   const float sx, const float sy, const float sz,
								   const float *angles, const float SO, const float SD,
								   const int nv,
								   const int na, const float da, const float ai,
								   const int nb, const float db, const float bi,
								   float *img)
{
	int id = ix + iy * nx + iz * nx * ny;

	// float dd_voxel[3];
	float xc, yc, zc;
	float usx = sx / nx, usy = sy / ny, usz = sz / nz;
	xc = ((float)ix - nx / 2 + 0.5f) * usx + cx;
	yc = ((float)iy - ny / 2 + 0.5f) * usy + cy;
	zc = ((float)iz - nz / 2 + 0.5f) * usz + cz;

	for (int iv = 0; iv < nv; iv++)
	{
		float sphi = __sinf(angles[iv]);
		float cphi = __cosf(angles[iv]);

		float xll, yll, zll, xlr, ylr, zlr, xrl, yrl, zrl, xrr, yrr, zrr, xt, yt, zt, xb, yb, zb;

		xll = +xc * cphi + yc * sphi - 0.5f * usx;
		yll = -xc * sphi + yc * cphi - 0.5f * usy;
		xrr = +xc * cphi + yc * sphi + 0.5f * usx;
		yrr = -xc * sphi + yc * cphi + 0.5f * usy;
		zll = zc;
		zrr = zc;
		xrl = +xc * cphi + yc * sphi + 0.5f * usx;
		yrl = -xc * sphi + yc * cphi - 0.5f * usy;
		xlr = +xc * cphi + yc * sphi - 0.5f * usx;
		ylr = -xc * sphi + yc * cphi + 0.5f * usy;
		zrl = zc;
		zlr = zc;
		xt = xc * cphi + yc * sphi;
		yt = -xc * sphi + yc * cphi;
		zt = zc + 0.5f * usz;
		xb = xc * cphi + yc * sphi;
		yb = -xc * sphi + yc * cphi;
		zb = zc - 0.5f * usz;

		// the coordinates of source and detector plane here are after rotation
		float ratio, all, bll, alr, blr, arl, brl, arr, brr, at, bt, ab, bb, a_max, a_min, b_max, b_min;
		// calculate a value for each boundary coordinates

		// the a and b here are all absolute positions from isocenter, which are on detector planes
		ratio = SD / (xll + SO);
		all = ratio * yll;
		bll = ratio * zll;
		ratio = SD / (xrr + SO);
		arr = ratio * yrr;
		brr = ratio * zrr;
		ratio = SD / (xlr + SO);
		alr = ratio * ylr;
		blr = ratio * zlr;
		ratio = SD / (xrl + SO);
		arl = ratio * yrl;
		brl = ratio * zrl;
		ratio = SD / (xt + SO);
		at = ratio * yt;
		bt = ratio * zt;
		ratio = SD / (xb + SO);
		ab = ratio * yb;
		bb = ratio * zb;

		// get the max and min values of all boundary projectors of voxel boundaries on detector plane
		a_max = MAX6(all, arr, alr, arl, at, ab);
		a_min = MIN6(all, arr, alr, arl, at, ab);
		b_max = MAX6(bll, brr, blr, brl, bt, bb);
		b_min = MIN6(bll, brr, blr, brl, bt, bb);

		// the related positions on detector plane from start points
		a_max = a_max / da - ai + 0.5f; //  now they are the detector coordinates
		a_min = a_min / da - ai + 0.5f;
		b_max = b_max / db - bi + 0.5f;
		b_min = b_min / db - bi + 0.5f;
		int a_ind_max = (int)floorf(a_max);
		int a_ind_min = (int)floorf(a_min);
		int b_ind_max = (int)floorf(b_max);
		int b_ind_min = (int)floorf(b_min);

		float bin_bound_1, bin_bound_2, wa, wb;
		for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia++)
		{
			bin_bound_1 = ia + 0.0f;
			bin_bound_2 = ia + 1.0f;

			wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);

			for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib++)
			{
				bin_bound_1 = ib + 0.0f;
				bin_bound_2 = ib + 1.0f;
				wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);

				img[id] += wa * wb * proj[ia + ib * na + iv * na * nb];
			}
		}
	}
}

__global__ void kernel_backprojection(float *img, cudaTextureObject_t tex_proj, float angle, float SO, float SD, int na, int nb, float da, float db, float ai, float bi, int nx, int ny, int nz)
{
	int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
	int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
	int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
	if (ix >= nx || iy >= ny || iz >= nz)
		return;

	int id = ix + iy * nx + iz * nx * ny;

	float sphi = __sinf(angle);
	float cphi = __cosf(angle);
	// float dd_voxel[3];
	float xc, yc, zc;
	xc = (float)ix - nx / 2 + 0.5f;
	yc = (float)iy - ny / 2 + 0.5f;
	zc = (float)iz - nz / 2 + 0.5f;

	// voxel boundary coordinates
	float xll, yll, zll, xlr, ylr, zlr, xrl, yrl, zrl, xrr, yrr, zrr, xt, yt, zt, xb, yb, zb;

	xll = +xc * cphi + yc * sphi - 0.5f;
	yll = -xc * sphi + yc * cphi - 0.5f;
	xrr = +xc * cphi + yc * sphi + 0.5f;
	yrr = -xc * sphi + yc * cphi + 0.5f;
	zll = zc;
	zrr = zc;
	xrl = +xc * cphi + yc * sphi + 0.5f;
	yrl = -xc * sphi + yc * cphi - 0.5f;
	xlr = +xc * cphi + yc * sphi - 0.5f;
	ylr = -xc * sphi + yc * cphi + 0.5f;
	zrl = zc;
	zlr = zc;
	xt = xc * cphi + yc * sphi;
	yt = -xc * sphi + yc * cphi;
	zt = zc + 0.5f;
	xb = xc * cphi + yc * sphi;
	yb = -xc * sphi + yc * cphi;
	zb = zc - 0.5f;

	// the coordinates of source and detector plane here are after rotation
	float ratio, all, bll, alr, blr, arl, brl, arr, brr, at, bt, ab, bb, a_max, a_min, b_max, b_min;
	// calculate a value for each boundary coordinates

	// the a and b here are all absolute positions from isocenter, which are on detector planes
	ratio = SD / (xll + SO);
	all = ratio * yll;
	bll = ratio * zll;
	ratio = SD / (xrr + SO);
	arr = ratio * yrr;
	brr = ratio * zrr;
	ratio = SD / (xlr + SO);
	alr = ratio * ylr;
	blr = ratio * zlr;
	ratio = SD / (xrl + SO);
	arl = ratio * yrl;
	brl = ratio * zrl;
	ratio = SD / (xt + SO);
	at = ratio * yt;
	bt = ratio * zt;
	ratio = SD / (xb + SO);
	ab = ratio * yb;
	bb = ratio * zb;

	// get the max and min values of all boundary projectors of voxel boundaries on detector plane
	// a_max = MAX4(al ,ar, at, ab);
	// a_min = MIN4(al ,ar, at, ab);
	// b_max = MAX4(bl ,br, bt, bb);
	// b_min = MIN4(bl ,br, bt, bb);
	a_max = MAX6(all, arr, alr, arl, at, ab);
	a_min = MIN6(all, arr, alr, arl, at, ab);
	b_max = MAX6(bll, brr, blr, brl, bt, bb);
	b_min = MIN6(bll, brr, blr, brl, bt, bb);

	// the related positions on detector plane from start points
	a_max = a_max / da - ai + 0.5f; //  now they are the detector coordinates
	a_min = a_min / da - ai + 0.5f;
	b_max = b_max / db - bi + 0.5f;
	b_min = b_min / db - bi + 0.5f;
	int a_ind_max = (int)floorf(a_max);
	int a_ind_min = (int)floorf(a_min);
	int b_ind_max = (int)floorf(b_max);
	int b_ind_min = (int)floorf(b_min);

	// int a_ind_max = (int)floorf(a_max / da - ai);
	// int a_ind_min = (int)floorf(a_min / da - ai);
	// int b_ind_max = (int)floorf(b_max / db - bi);
	// int b_ind_min = (int)floorf(b_min / db - bi);

	float bin_bound_1, bin_bound_2, wa, wb;
	for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia++)
	{
		// bin_bound_1 = ((float)ia + ai) * da;
		// bin_bound_2 = ((float)ia + ai + 1.0f) * da;
		bin_bound_1 = ia + 0.0f;
		bin_bound_2 = ia + 1.0f;

		wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min); // wa /= a_max - a_min;

		for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib++)
		{
			// bin_bound_1 = ((float)ib + bi) * db;
			// bin_bound_2 = ((float)ib + bi + 1.0f) * db;
			bin_bound_1 = ib + 0.0f;
			bin_bound_2 = ib + 1.0f;
			// wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= db;
			wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min); // wb /= b_max - b_min;

			img[id] += wa * wb * tex3D<float>(tex_proj, (ia + 0.5f), (ib + 0.5f), 0.5f);
		}
	}
}
