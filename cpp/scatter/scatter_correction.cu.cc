#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"

#define abs(x) ((x)<0 ? (-x) : (x))

const int BLOCKDIM = 1024;
const float PI = 3.1415926;

float eff_without_scatter(const float low_win, const float high_win, const float energy_res) {
	float eff = 0.0f;
	for (float energy = low_win; energy < high_win; energy += 5.0) {
		eff += expf(-(energy - 511.0) * (energy - 511.0) / 2.0 / energy_res / energy_res);
	}
	return eff;
}

__device__ float
eff_with_scatter(const float low_win, const float high_win, const float energy_res, const float scattered_energy) {
	float eff = 0.0f;
	for (float energy = low_win; energy < high_win; energy += 5.0) {
		eff += expf(-(energy - scattered_energy) * (energy - scattered_energy) / 2.0 / energy_res / energy_res);
	}
	return eff;
}

__device__ float distance_a2b(const float x1, const float y1, const float z1,
                              const float x2, const float y2, const float z2) {
	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__device__ float get_angle(const int nb_detectors_per_ring, const float *p) {
	float dtheta = PI / nb_detectors_per_ring;
	float theta = atan2f(p[1], p[0]) + dtheta / 2;
	return floorf(theta / dtheta) * dtheta;
}

__device__ float
project_area(const float *pa, const float *pb, const int nb_detectors_per_ring, const float unit_area) {
	float theta_o2b = get_angle(nb_detectors_per_ring, pb);
	float x_a2b = pa[0] - pb[0];
	float y_a2b = pa[1] - pb[1];
	float z_a2b = pa[2] - pb[2];
	return unit_area * abs(x_a2b * cosf(theta_o2b) + y_a2b * sinf(theta_o2b) + z_a2b) /
	       sqrtf(x_a2b * x_a2b + y_a2b * y_a2b + z_a2b * z_a2b);
}

__device__ float get_scatter_cos_theta(const float *s, const float *a, const float *b) {
	return -((s[0] - a[0]) * (s[0] - b[0]) + (s[1] - a[1]) * (s[1] - b[1]) + (s[2] - a[2]) * (s[2] - b[2])) /
	       sqrtf((s[0] - a[0]) * (s[0] - a[0]) + (s[1] - a[1]) * (s[1] - a[1]) + (s[2] - a[2]) * (s[2] - a[2])) /
	       sqrtf((s[0] - b[0]) * (s[0] - b[0]) + (s[1] - b[1]) * (s[1] - b[1]) + (s[2] - b[2]) * (s[2] - b[2]));

}

__device__ float
fkn(const float *s, const float *a, const float *b, const float *umap, const float *unit_size, const int *shape) {
	float cos_asb = get_scatter_cos_theta(s, a, b);
	int s_ix = (int) floorf(s[0] / unit_size[0] + 0.5);
	int s_iy = (int) floorf(s[1] / unit_size[1] + 0.5);
	int s_iz = (int) floorf(s[2] / unit_size[2] + 0.5);

	float diff_val = (0.5 / (2.0 - cos_asb) / (2.0 - cos_asb) *
	                  (1.0 + cos_asb * cos_asb + (1.0 - cos_asb) * (1.0 - cos_asb) / (2 - cos_asb)));
	float Z = umap[s_ix + s_iy * shape[0] + s_iz * shape[0] * shape[1]] * 1000 + 0.4;
	float sigma = PI * Z * (40 / 9 - 3 * logf(3));
	return diff_val / sigma * umap[s_ix + s_iy * shape[0] + s_iz * shape[0] * shape[1]];
}


__device__ float get_scatter_ab(const float *pos, const int n_pos, const float *pa, const float *pb,
                                const float low_win, const float high_win, const float energy_res,
                                const float *emission_s2a, const float *emission_s2b,
                                const float *atten_s2a, const float *atten_s2b, const int nb_blocks_per_ring,
                                const float *u_map, const float *unit_size, const float unit_area, const int *shape) {
	float scatter_ab = 0.0f;
	for (int i = 0; i < n_pos; i++) {
		float cos_theta = get_scatter_cos_theta(pos + i * 3, pa, pb);
		float scatter_energy = 511.0 / (2.0 - cos_theta);
		float Ia = atten_s2a[i] * expf(logf(atten_s2b[i]) * (2.0 - cos_theta)) * emission_s2a[i];
		float Ib = atten_s2b[i] * expf(logf(atten_s2a[i]) * (2.0 - cos_theta)) * emission_s2b[i];
		scatter_ab += project_area(pos + i * 3, pa, nb_blocks_per_ring, unit_area) *
		              project_area(pos + i * 3, pb, nb_blocks_per_ring, unit_area) *
		              eff_with_scatter(low_win, high_win, scatter_energy, energy_res) *
		              fkn(pos + i * 3, pa, pb, u_map, unit_size, shape) * (Ia + Ib) /
		              ((pa[0] - pos[i * 3]) * (pa[0] - pos[i * 3]) +
		               (pa[1] - pos[i * 3 + 1]) * (pa[1] - pos[i * 3 + 1]) +
		               (pa[2] - pos[i * 3 + 2]) * (pa[2] - pos[i * 3 + 2])) /
		              ((pb[0] - pos[i * 3]) * (pb[0] - pos[i * 3]) +
		               (pb[1] - pos[i * 3 + 1]) * (pb[1] - pos[i * 3 + 1]) +
		               (pb[2] - pos[i * 3 + 2]) * (pb[2] - pos[i * 3 + 2]));
	}
	return scatter_ab;
}


__device__ float get_scale(const float *pa, const float *pb, const int nb_blocks_per_ring, float unit_area) {
	float area = project_area(pa, pb, nb_blocks_per_ring, unit_area);
	return ((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]) + (pa[2] - pb[2]) * (pa[2] - pb[2])) /
	       area / area;
}


__global__ void kernel(const float *pos, const int n_pos, const float *crystal_pos,
                       const float low_win, const float high_win, const float energy_res,
                       const float *emission, const float *atten, const int nb_blocks_per_ring,
                       const float *u_map, const float *unit_size, const float unit_area,
                       const int *lors_inds, const int n_lors, const int *shape,
                       float *scatter_ab, float *scale_ab) {
	int i = BLOCKDIM * blockIdx.x + threadIdx.x;
	if (i >= n_lors) { return; }
	int ia = lors_inds[i * 2];
	int ib = lors_inds[i * 2 + 1];
	scatter_ab[i] = get_scatter_ab(pos, n_pos, crystal_pos + 3 * ia, crystal_pos + 3 * ib, low_win, high_win, energy_res,
	                               emission + ia * n_pos, emission + ib * n_pos, atten + ia * n_pos, atten + ib * n_pos,
	                               nb_blocks_per_ring, u_map, unit_size, unit_area, shape);
	scale_ab[i] = get_scale(crystal_pos + 3 * ia, crystal_pos + 3 * ib, nb_blocks_per_ring, unit_area);

}

void scatter_over_lors(const float *pos, const int n_pos, const float *crystal_pos,
                       const float low_win, const float high_win, const float energy_res,
                       const float *emission, const float *atten, const int nb_blocks_per_ring,
                       const float *u_map, const float *unit_size, const float unit_area,
                       const int *lors_inds, const int n_lors, const int *shape,
                       float *scatter_ab, float *scale_ab) {
	int GRIDDIM = 1 + int(n_lors / BLOCKDIM);
	kernel <<< GRIDDIM, BLOCKDIM >>>
	                     (pos, n_pos, crystal_pos, low_win, high_win, energy_res, emission, atten, nb_blocks_per_ring,
			                     u_map, unit_size, unit_area, lors_inds, n_lors, shape, scatter_ab, scale_ab);
}

#endif