/** @file   testbed_nerf.cu
 *  @author Yiming Wang <w752531540@gmail.com>
 */

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <curand_kernel.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>


#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN


#define NORMAL_VECTORS_NORMALIZED 0
#define SDF_GRID 0

#define BENT_DIR 1

#define rotation_reprensentation 1 // 1 refers to rotation 6d, 0 refers to quaternion

#define PRINT_PART_TIME 0

inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.2f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
#if SDF_GRID
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 30.0f; }
#else
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.1f; }
#endif

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

template <typename T>
__global__ void transform_mesh_with_quaternion(
	const uint32_t  n_elements,
	const T* rotation,
	const T* transition,
	float* mesh_vex
)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float qx = rotation[0] , qy = rotation[1], qz = rotation[2], qw = - rotation[3];
	float tx = transition[0], ty = transition[1], tz = transition[2];
	float vx = mesh_vex[i * 3 + 0], vy = mesh_vex[i * 3 + 1], vz = mesh_vex[i * 3 + 2];

	vx -= tx;
	vy -= ty;
	vz -= tz;

	mesh_vex[i * 3 + 0] = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx;
	mesh_vex[i * 3 + 1] = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy;
	mesh_vex[i * 3 + 2] = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz;

}

template <typename T>
__global__ void transform_mesh_with_6d(
	const uint32_t  n_elements,
	const T* rotation_6d,
	const T* transition,
	float* mesh_vex
)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float vx = mesh_vex[i * 3 + 0], vy = mesh_vex[i * 3 + 1], vz = mesh_vex[i * 3 + 2];
	Eigen::Vector3f input_vector(vx,vy,vz);

	Eigen::Matrix3f rotation_matrix;
	rotation_matrix << (float)rotation_6d[0], (float)rotation_6d[1], (float)rotation_6d[2],
					   (float)rotation_6d[3], (float)rotation_6d[4], (float)rotation_6d[5],
					   (float)rotation_6d[6], (float)rotation_6d[7], (float)rotation_6d[8];


	Eigen::Vector3f transition_vector(transition[0],transition[1],transition[2]);

	input_vector -= transition_vector;

	input_vector = rotation_matrix.inverse() * input_vector;

	mesh_vex[i * 3 + 0] = input_vector[0];
	mesh_vex[i * 3 + 1] = input_vector[1];
	mesh_vex[i * 3 + 2] = input_vector[2];

}


Testbed::NetworkDims Testbed::network_dims_nerf() const {
	NetworkDims dims;
	dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
	dims.n_output = 16;
	dims.n_pos = sizeof(NerfPosition) / sizeof(float);
	return dims;
}

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant) {
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();
	
	return cone_angle_constant;
}

inline __host__ __device__ float calc_dt(float t, float cone_angle) {
	return tcnn::clamp(t*cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());
}

__device__ float single_variance_sigmoid(float val, float s) {
	return 1.0f / (1.0f + __expf(-s * val));
}

template <typename T>
__device__ void global_movement_with_rotation_quaternion(
	const T* rotation,
	const T* transition,
	Eigen::Vector3f & ray_o,
	Eigen::Vector3f & ray_d
)
{
	float qx = rotation[0], qy = rotation[1], qz = rotation[2], qw = rotation[3];
	float vx = ray_o[0], vy = ray_o[1], vz = ray_o[2];
	float tx = transition[0], ty = transition[1], tz = transition[2];

	ray_o[0] = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx + tx;
	ray_o[1] = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy + ty;
	ray_o[2] = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz + tz;


	vx = ray_d[0], vy = ray_d[1], vz = ray_d[2];
	ray_d[0] = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx;
	ray_d[1] = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy;
	ray_d[2] = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz;
}

template <typename T>
__device__ void global_movement_with_rotation_6d(
	const T* rotation_6d,
	const T* transition,
	Eigen::Vector3f & ray_o,
	Eigen::Vector3f & ray_d
)
{

	Eigen::Matrix3f rotation_matrix;
	rotation_matrix << (float)rotation_6d[0], (float)rotation_6d[1], (float)rotation_6d[2],
					   (float)rotation_6d[3], (float)rotation_6d[4], (float)rotation_6d[5],
					   (float)rotation_6d[6], (float)rotation_6d[7], (float)rotation_6d[8];

	Eigen::Vector3f transition_vector(transition[0],transition[1],transition[2]);

	ray_o = rotation_matrix * ray_o + transition_vector;

	ray_d = rotation_matrix * ray_d;

}

template <typename T>
__device__ void local_movement_with_rotation_6d(
	const T* rotation_6d,
	Eigen::Vector3f & ray_o,
	Eigen::Vector3f & ray_d
)
{
	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<T>(rotation_6d, rotation_matrix);

	ray_o = rotation_matrix * ray_o;

	ray_d = rotation_matrix * ray_d;

}

__device__ float single_variance_sigmoid_derivative(float val, float s) {
	float sigmoid_val = single_variance_sigmoid(val, s);
	return s * sigmoid_val * (1.0f - sigmoid_val);
}

__device__ float single_variance_sigmoid_second_derivative(float val, float s) {
	float sigmoid_val = single_variance_sigmoid(val, s);
	return s * s *  sigmoid_val * (1.0f - sigmoid_val) * (1.0f - 2 * sigmoid_val);
}

__device__ Array3f network_to_pos_gradient(const tcnn::vector_t<tcnn::network_precision_t, 16>& local_network_output, ENerfActivation activation) {
	Vector3f raw_gradient{
		float(local_network_output[4]),
		float(local_network_output[5]),
		float(local_network_output[6])
	};
	return {
		raw_gradient.x(),
		raw_gradient.y(),
		raw_gradient.z()
	};
}

__device__ float network_to_pos_gradient(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return val;
	case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic: return tcnn::logistic(val);
	case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
	default: assert(false);
	}
	return 0.0f;
}

struct LossAndGradient {
	float loss;
	Eigen::Array4f gradient;

	__host__ __device__ LossAndGradient operator*(float scalar) {
		return {loss * scalar, gradient * scalar};
	}

	__host__ __device__ LossAndGradient operator/(float scalar) {
		return {loss / scalar, gradient / scalar};
	}
};

inline __device__ Array4f copysign(const Array4f& a, const Array4f& b) {
	return {
		copysignf(a.x(), b.x()),
		copysignf(a.y(), b.y()),
		copysignf(a.z(), b.z()),
		copysignf(a.w(), b.w())
	};
}



inline __device__ LossAndGradient mse_loss(const Array4f& target, const Array4f& prediction) {
	Array4f difference = prediction - target;
	float loss = difference.x() * difference.x() + difference.y() * difference.y() + difference.z() * difference.z() + difference.w() * difference.w();
    Array4f gradient = 2*difference;

    return {
        loss,
        gradient
    };
}

inline __device__ LossAndGradient l1_loss(const Array4f& target, const Array4f& prediction) {
	Array4f difference = prediction - target;
	Array4f diff_abs = difference.abs();
	return {
		diff_abs.x()+diff_abs.y()+diff_abs.z()+diff_abs.w(),
		copysign(Array4f::Ones(), difference),
	};
}


inline __device__ float distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) { // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do {
		t += calc_dt(t, cone_angle);
	} while (t < t_target);
	return t;
}

// Begin: Activation function.
__device__ float activation_function(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return val;
	case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic: return tcnn::logistic(val);
	case ENerfActivation::Exponential: return __expf(val);
	default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

// __device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 8>& local_network_output, ENerfActivation activation) {
__device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 16>& local_network_output, ENerfActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

__device__ Vector3f warp_position(const Vector3f& pos, const BoundingBox& aabb) {
	return aabb.relative_pos(pos);

}

__device__ Vector3f unwarp_position(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}

__device__ Vector3f unwarp_position_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ Vector3f warp_position_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	return unwarp_position_derivative(pos, aabb).cwiseInverse();
}

__host__ __device__ Vector3f warp_direction(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ Vector3f unwarp_direction(const Vector3f& dir) {
	return dir * 2.0f - Vector3f::Ones();
}

__device__ Vector3f warp_direction_derivative(const Vector3f& dir) {
	return Vector3f::Constant(0.5f);
}

__device__ Vector3f unwarp_direction_derivative(const Vector3f& dir) {
	return Vector3f::Constant(2.0f);
}

__device__ float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	// printf("pos:%f,%f,%f\n",pos[0],pos[1],pos[2]);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	if (i.x() < -1 || i.x() > NERF_GRIDSIZE() || i.y() < -1 || i.y() > NERF_GRIDSIZE() || i.z() < -1 || i.z() > NERF_GRIDSIZE()) {
		printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
	}

	uint32_t idx = tcnn::morton3D(
		tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE()-1)
	);

	return idx;
}

__device__ bool density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	// return true;
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

__device__ float cascaded_grid_at(Vector3f pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__device__ float& cascaded_grid_at(Vector3f pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__global__ void extract_srgb_with_activation(const uint32_t n_elements,	const uint32_t rgb_stride, const float* __restrict__ rgbd, float* __restrict__ rgb, ENerfActivation rgb_activation, bool from_linear) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	float c = network_to_rgb(rgbd[elem_idx*7 + dim_idx], rgb_activation);
	if (from_linear) {
		c = linear_to_srgb(c);
	}

	rgb[elem_idx*rgb_stride + dim_idx] = c;
}

__global__ void mark_untrained_density_grid(const uint32_t n_elements,  float* __restrict__ grid_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	bool clear_visible_voxels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t level = i / (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
	uint32_t pos_idx = i % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);



	Vector3f pos = ((Vector3f{(float)x+0.5f, (float)y+0.5f, (float)z+0.5f}) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);
	float voxel_radius = 0.5f*SQRT3()*scalbnf(1.0f, level) / NERF_GRIDSIZE();
	int count=0;
	for (uint32_t j=0; j < n_training_images; ++j) {
		if (metadata[j].camera_distortion.mode == ECameraDistortionMode::FTheta) {
			// not supported for now
			count++;
			break;
		}
		float half_resx = metadata[j].resolution.x() * 0.5f;
		float half_resy = metadata[j].resolution.y() * 0.5f;
		Matrix<float, 3, 4> xform = training_xforms[j].start;
		Vector3f ploc = pos - xform.col(3);
		float x = ploc.dot(xform.col(0));
		float y = ploc.dot(xform.col(1));
		float z = ploc.dot(xform.col(2));
		if (z > 0.f) {
			auto focal = metadata[j].focal_length;
			// TODO - add a box / plane intersection to stop thomas from murdering me
			if (fabsf(x) - voxel_radius < z / focal.x() * half_resx && fabsf(y) - voxel_radius < z / focal.y() * half_resy) {
				count++;
				if (count > 0) break;
			}
		}
	}

	if (clear_visible_voxels || (grid_out[i] < 0) != (count <= 0)) {
		grid_out[i] = (count > 0) ? 0.f : -1.f;
	}
}

__global__ void generate_grid_samples_nerf_uniform(Eigen::Vector3i res_3d, const uint32_t step, BoundingBox render_aabb, BoundingBox train_aabb, NerfPosition* __restrict__ out) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	Vector3f pos = Array3f{(float)x, (float)y, (float)z} * Array3f{1.f/res_3d.x(),1.f/res_3d.y(),1.f/res_3d.z()};
	pos = pos.cwiseProduct(render_aabb.max - render_aabb.min) + render_aabb.min;
	// printf("pos:%f,%f,%f\n",pos[0],pos[1],pos[2]);
	out[i] = { warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE()) };
}

// generate samples for uniform grid including constant ray direction
__global__ void generate_grid_samples_nerf_uniform_dir(Eigen::Vector3i res_3d, const uint32_t step, BoundingBox render_aabb, BoundingBox train_aabb, Eigen::Vector3f ray_dir, NerfCoordinate* __restrict__ network_input) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
	Vector3f pos = Array3f{(float)x, (float)y, (float)z} * Array3f{1.f/res_3d.x(),1.f/res_3d.y(),1.f/res_3d.z()};
	pos = pos.cwiseProduct(render_aabb.max - render_aabb.min) + render_aabb.min;
	network_input[i] = { warp_position(pos, train_aabb), warp_direction(ray_dir), warp_dt(MIN_CONE_STEPSIZE()) };
}

inline __device__ int mip_from_pos(const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(max_cascade, max(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int mip = mip_from_pos(pos, max_cascade);
	dt *= 2*NERF_GRIDSIZE();
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(max_cascade, max(exponent, mip));
}

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
		idx += level * NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE();
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	Vector3f pos = ((Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);

	out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements, const uint32_t* __restrict__ indices, const tcnn::network_precision_t* network_output, float* __restrict__ grid_out, ENerfActivation rgb_activation, ENerfActivation density_activation) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	// uint32_t level = 0;//local_idx / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

	// float mlp = network_to_density(float(network_output[i]), density_activation);
	float mlp = float(network_output[i]);

	// float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);
	float optical_thickness = mlp;

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void grid_samples_half_to_float(const uint32_t n_elements, BoundingBox aabb, float* dst, const tcnn::network_precision_t* network_output, ENerfActivation density_activation, const NerfPosition* __restrict__ coords_in, const float* __restrict__ grid_in, uint32_t max_cascade) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
	//float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float mlp = float(network_output[i]);

	if (grid_in) {
		Vector3f pos = unwarp_position(coords_in[i].p, aabb);
		float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos, max_cascade));
		// if (grid_density < NERF_MIN_OPTICAL_THICKNESS()) {
		// 	mlp = -10000.f;
		// }
	}
	dst[i] = mlp;
}

__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
	float decay,
	const uint32_t count,
	float* __restrict__ grid_out,
	const float* __restrict__ grid_in
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows splat_grid_samples_nerf_max_nearest_neighborcapture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	// float val;
	// if (prev_val < 0.f || importance < 1e-6f){
	// 	val = prev_val;  // does not change
	// }
	// else{
	// 	val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	// }
	grid_out[i] = val;
}

__global__ void decay_sharpness_grid_nerf(const uint32_t n_elements, float decay, float* __restrict__ grid) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	grid[i] *= decay;
}

__global__ void grid_to_bitfield(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = tcnn::morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = tcnn::morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = tcnn::morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[tcnn::morton3D(x, y, z)] |= bits;
}

__global__ void advance_pos_nerf(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	Vector3f camera_fwd,
	Vector2f focal_length,
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Vector3f origin = payload.origin;
	Vector3f dir = payload.dir;
	Vector3f idir = dir.cwiseInverse();

	float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;
	float dt = calc_dt(t, cone_angle);
	t += ld_random_val(sample_index, i * 786433) * dt;
	Vector3f pos;

	while (1) {
		if (!render_aabb.contains(pos = origin + dir * t)) {
			payload.alive = false;
			break;
		}

		dt = calc_dt(t, cone_angle);
		uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

		if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
			break;
		}

		uint32_t res = NERF_GRIDSIZE()>>mip;
		t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
	}

	payload.t = t;
}

__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements, BoundingBox aabb, const Vector3f* __restrict__ pos, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Vector3f dir=(pos[i]-Vector3f::Constant(0.5f)).normalized(); // choose outward pointing directions, for want of a better choice
	network_input(i)->set_with_optional_extra_dims(warp_position(pos[i], aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__global__ void generate_nerf_network_inputs_at_current_position(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Vector3f dir = payloads[i].dir;
	network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__global__ void compute_nerf_density(const uint32_t n_elements, Array4f* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Array4f rgba = network_output[i];
	rgba.w() = tcnn::clamp(1.f - __expf(-network_to_density(rgba.w(), density_activation) / 100.0f), 0.0f, 1.0f);
	rgba.x() = network_to_rgb(rgba.x(), rgb_activation) * rgba.w();
	rgba.y() = network_to_rgb(rgba.y(), rgb_activation) * rgba.w();
	rgba.z() = network_to_rgb(rgba.z(), rgb_activation) * rgba.w();

	network_output[i] = rgba;
}

__global__ void generate_next_nerf_network_inputs(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	BoundingBox train_aabb,
	Vector2f focal_length,
	Vector3f camera_fwd,
	NerfPayload* __restrict__ payloads,
	PitchedPtr<NerfCoordinate> network_input,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	float cone_angle_constant,
	const float* extra_dims
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Vector3f origin = payload.origin;
	Vector3f dir = payload.dir;
	Vector3f idir = dir.cwiseInverse();

	float cone_angle = calc_cone_angle(dir.dot(camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;

	for (uint32_t j = 0; j < n_steps; ++j) {
		Vector3f pos;
		float dt = 0.0f;
		while (1) {
			if (!render_aabb.contains(pos = origin + dir * t)) {
				payload.n_steps = j;
				return;
			}

			dt = calc_dt(t, cone_angle);
			uint32_t mip = max(min_mip, mip_from_dt(dt, pos));

			if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
				break;
			}

			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
		}

		network_input(i + j * n_elements)->set_with_optional_extra_dims(warp_position(pos, train_aabb), warp_direction(dir), warp_dt(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
		t += dt;
	}

	payload.t = t;
	payload.n_steps = n_steps;
}

__global__ void composite_kernel_nerf(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t current_step,
	BoundingBox aabb,
	float glow_y_cutoff,
	int glow_mode,
	const uint32_t n_training_images,
	const TrainingXForm* __restrict__ training_xforms,
	Matrix<float, 3, 4> camera_matrix,
	Vector2f focal_length,
	float depth_scale,
	Array4f* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* payloads,
	PitchedPtr<NerfCoordinate> network_input,
	const tcnn::network_precision_t* __restrict__ network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* __restrict__ density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance,
	float variance,
	uint32_t training_step,
	const float cos_anneal_ratio
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	Array4f local_rgba = rgba[i];
	float local_depth = depth[i];
	Vector3f origin = payload.origin;
	Vector3f cam_fwd = camera_matrix.col(2);
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;
	uint32_t j = 0;

	// const float cos_anneal_ratio = 1.0f;

	Vector3f prev_deformed_pos = origin;
	Vector3f dir;
	const Vector3f const_offset = Vector3f{1e-8f,1e-8f,1e-8f};
	for (; j < actual_n_steps; ++j) {
		tcnn::vector_t<tcnn::network_precision_t, 16> local_network_output;
		local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
		local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
		local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
		local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
		local_network_output[4] = network_output[i + j * n_elements + 4 * stride];
		local_network_output[5] = network_output[i + j * n_elements + 5 * stride];
		local_network_output[6] = network_output[i + j * n_elements + 6 * stride];
		local_network_output[7] = network_output[i + j * n_elements + 7 * stride];
		local_network_output[8] = network_output[i + j * n_elements + 8 * stride];
		local_network_output[9] = network_output[i + j * n_elements + 9 * stride];
		local_network_output[10] = network_output[i + j * n_elements + 10 * stride];
		const NerfCoordinate* input = network_input(i + j * n_elements);
		Vector3f warped_pos = input->pos.p;
		Vector3f pos = unwarp_position(warped_pos, aabb);

		float T = 1.f - local_rgba.w();
		float dt = unwarp_dt(input->dt);
		dir = unwarp_direction(input->dir.d);

		float inv_s = __expf((tcnn::network_precision_t)10 * local_network_output[7]);
		#if BENT_DIR
			Vector3f deformed_viewdir{float(local_network_output[8]),float(local_network_output[9]),float(local_network_output[10])};
			dir = unwarp_direction(deformed_viewdir).normalized();
		#endif 
		// Neus rendering

		float sdf_value = float(local_network_output[3]);
		Array3f pos_gradient = network_to_pos_gradient(local_network_output, ENerfActivation::None);
		#if NORMAL_VECTORS_NORMALIZED
			float gradient_norm_for_dir = std::sqrt(pos_gradient[0]*pos_gradient[0] + pos_gradient[1]*pos_gradient[1] + pos_gradient[2]*pos_gradient[2] + 1e-5);
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2])/ gradient_norm_for_dir ;
		#else
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2]);
		#endif
		float iter_cos = -(activation_function(-true_cos * 0.5 + 0.5, ENerfActivation::ReLU) * (1.0 - cos_anneal_ratio) + \
			activation_function(-true_cos, ENerfActivation::ReLU) * cos_anneal_ratio);

		float estimated_next_sdf = sdf_value + iter_cos * dt * 0.5;
		float estimated_prev_sdf = sdf_value - iter_cos * dt * 0.5;

		float next_cdf = activation_function(estimated_next_sdf * inv_s, ENerfActivation::Logistic);
		float prev_cdf = activation_function(estimated_prev_sdf * inv_s, ENerfActivation::Logistic);

		float p = prev_cdf - next_cdf;
		float c = prev_cdf;
		float p_div_c = (p + 1e-5f) / (c + 1e-5f);
		float alpha = tcnn::clamp(p_div_c, 0.0f, 1.0f); // alpha = 1.f - sigmoid(next_sdf / s) / sigmoid(prev_sdf / s);

		if (show_accel >= 0) {
			alpha = 1.f;
		}
		float weight = alpha * T;

		Array3f rgb = network_to_rgb(local_network_output, rgb_activation);

		if (glow_mode) { // random grid visualizations ftw!
			float glow = 0.f;

			bool green_grid = glow_mode & 1;
			bool green_cutline = glow_mode & 2;
			bool mask_to_alpha = glow_mode & 4;

			// less used?
			bool radial_mode = glow_mode & 8;
			bool grid_mode = glow_mode & 16; // makes object rgb go black!

			{
				float dist;
				if (radial_mode) {
					dist = (pos - camera_matrix.col(3)).norm();
					dist = min(dist, (4.5f - pos.y()) * 0.333f);
				} else {
					dist = pos.y();
				}

				if (grid_mode) {
					glow = 1.f / max(1.f, dist);
				} else {
					float y = glow_y_cutoff - dist; // - (ii*0.005f);
					float mask = 0.f;
					if (y > 0.f) {
						y *= 80.f;
						mask = min(1.f, y);
						{
							if (green_cutline) {
								glow += max(0.f, 1.f - abs(1.f -y)) * 4.f;
							}

							if (y>1.f) {
								y = 1.f - (y - 1.f) * 0.05f;
							}

							if (green_grid) {
								glow += max(0.f, y / max(1.f, dist));
							}
						}
					}
					if (mask_to_alpha) {
						weight *= mask;
					}
				}
			}

			if (glow > 0.f) {
				float line;
				line  = max(0.f, cosf(pos.y() * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x() * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z() * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y() * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x() * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z() * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y() * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x() * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z() * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y() * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x() * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z() * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				if (grid_mode) {
					glow = /*glow*glow*0.75f + */ glow * line * 15.f;
					rgb.y() = glow;
					rgb.z() = glow * 0.5f;
					rgb.x() = glow * 0.25f;
				} else {
					glow = glow * glow * 0.25f + glow * line * 15.f;
					rgb.y() += glow;
					rgb.z() += glow * 0.5f;
					rgb.x() += glow * 0.25f;
				}
			}
		} // glow

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			Vector3f normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normal.normalized().array();
		} else if (render_mode == ERenderMode::Positions) {
			if (show_accel >= 0) {
				uint32_t mip = max(show_accel, mip_from_pos(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x()*(res);
				int iy = pos.y()*(res);
				int iz = pos.z()*(res);
				default_rng_t rng(ix+iy*232323+iz*727272);
				rgb.x() = 1.f-mip*(1.f/(NERF_CASCADES()-1));
				rgb.y() = rng.next_float();
				rgb.z() = rng.next_float();
			} else {
				rgb = (pos.array() - Array3f::Constant(0.5f)) / 2.0f + Array3f::Constant(0.5f);
			}
		} else if (render_mode == ERenderMode::EncodingVis) {
			rgb = warped_pos.array();
		} else if (render_mode == ERenderMode::Depth) {
			rgb = Array3f::Constant(cam_fwd.dot(pos - origin) * depth_scale);
		} else if (render_mode == ERenderMode::AO) {
			rgb = Array3f::Constant(alpha);
		}

		local_rgba.head<3>() += rgb * weight;
		local_rgba.w() += weight;
		if (weight > payload.max_weight) {
			payload.max_weight = weight;
			local_depth = cam_fwd.dot(pos - camera_matrix.col(3));
		}

		if (local_rgba.w() > (1.0f - min_transmittance)) {
			local_rgba /= local_rgba.w();
			break;
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	rgba[i] = local_rgba;
	depth[i] = local_depth;
}

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;

inline __device__ Vector2f sample_cdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
	if (sample.x() < UNIFORM_SAMPLING_FRACTION) {
		sample.x() /= UNIFORM_SAMPLING_FRACTION;
		return sample;
	}

	sample.x() = (sample.x() - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

	cdf_y += img * res.y();

	// First select row according to cdf_y
	uint32_t y = binary_search(sample.y(), cdf_y, res.y());
	float prev = y > 0 ? cdf_y[y-1] : 0.0f;
	float pmf_y = cdf_y[y] - prev;
	sample.y() = (sample.y() - prev) / pmf_y;

	cdf_x_cond_y += img * res.y() * res.x() + y * res.x();

	// Then, select col according to x
	uint32_t x = binary_search(sample.x(), cdf_x_cond_y, res.x());
	prev = x > 0 ? cdf_x_cond_y[x-1] : 0.0f;
	float pmf_x = cdf_x_cond_y[x] - prev;
	sample.x() = (sample.x() - prev) / pmf_x;

	if (pdf) {
		*pdf = pmf_x * pmf_y * res.prod();
	}

	return {((float)x + sample.x()) / (float)res.x(), ((float)y + sample.y()) / (float)res.y()};
}

inline __device__ float pdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
	Vector2i p = (sample.cwiseProduct(res.cast<float>())).cast<int>().cwiseMax(0).cwiseMin(res - Vector2i::Ones());

	cdf_y += img * res.y();
	cdf_x_cond_y += img * res.y() * res.x() + p.y() * res.x();

	float pmf_y = cdf_y[p.y()];
	if (p.y() > 0) {
		pmf_y -= cdf_y[p.y()-1];
	}

	float pmf_x = cdf_x_cond_y[p.x()];
	if (p.x() > 0) {
		pmf_x -= cdf_x_cond_y[p.x()-1];
	}

	// Probability mass of picking the pixel
	float pmf = pmf_x * pmf_y;

	// To convert to probability density, divide by area of pixel
	return UNIFORM_SAMPLING_FRACTION + pmf * res.prod() * (1.0f - UNIFORM_SAMPLING_FRACTION);
}

inline __device__ Vector2f nerf_random_image_pos_training(default_rng_t& rng, const Vector2i& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const Vector2i& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
	Vector2f xy = random_val_2d(rng);

	if (cdf_x_cond_y) {
		xy = sample_cdf_2d(xy, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else if (pdf) {
		*pdf = 1.0f;
	}

	if (snap_to_pixel_centers) {
		Vector2f xy_before = xy;
		Vector2f xy_in_pixels = xy.cwiseProduct(resolution.cast<float>()).cwiseMax(Vector2f::Constant(0.0f)).cwiseMin((resolution - Vector2i::Ones()).cast<float>());
		Vector2f xy_in_pixels_offset = xy_in_pixels + Vector2f::Constant(0.5f);
		xy = (xy_in_pixels_offset.cwiseQuotient(resolution.cast<float>()));
		// printf("xy_before: %f %f, xy_in_pixels: %f %f, xy_in_pixels_offset: %f %f, xy_after: %f %f, resolution: %d %d\n", xy_before.x(), xy_before.y(), xy_in_pixels.x(), xy_in_pixels.y(), xy_in_pixels_offset.x(), xy_in_pixels_offset.y(), xy.x(), xy.y(), resolution.x(), resolution.y());

		// xy = (xy.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(resolution - Vector2i::Ones()).cast<float>() + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
		// printf("xy_before: %f %f, xy_after: %f %f, resolution: %f %f\n", xy_before.x(), xy_before.y(), xy.x(), xy.y(), resolution.x(), resolution.y());
	}

	return xy;
}

inline __device__ uint32_t image_idx(uint32_t base_idx, uint32_t n_rays, uint32_t n_rays_total, uint32_t n_training_images, const float* __restrict__ cdf = nullptr, float* __restrict__ pdf = nullptr) {
	if (cdf) {
		float sample = ld_random_val(base_idx + n_rays_total, 0xdeadbeef);
		uint32_t img = binary_search(sample, cdf, n_training_images);

		if (pdf) {
			float prev = img > 0 ? cdf[img-1] : 0.0f;
			*pdf = (cdf[img] - prev) * n_training_images;
		}

		return img;
	}

	// return ((base_idx + n_rays_total) * 56924617 + 96925573) % n_training_images;

	// Neighboring threads in the warp process the same image. Increases locality.
	if (pdf) {
		*pdf = 1.0f;
	}
	return (((base_idx + n_rays_total) * n_training_images) / n_rays) % n_training_images;
}

__global__ void generate_training_samples_nerf_with_global_movement(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	const float* __restrict__ distortion_data,
	const Vector2i distortion_resolution,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i cdf_res,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims,
	uint32_t cur_frame,
	uint32_t training_step,
	const network_precision_t* rotation,
	const network_precision_t* transition,
	const Eigen::Vector3f first_frame_offset
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
	Eigen::Vector2i resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(xy, resolution, 0);

	float transparency = read_rgba(xy, resolution, metadata[img].pixels, metadata[img].image_data_type).w();
	if (read_rgba(xy, resolution, metadata[img].pixels, metadata[img].image_data_type).x() <= 0.0f &&  random_val(rng) >= 0.9) {
		return;
	}

	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	float motionblur_time = random_val(rng);

	const Vector2f focal_length = metadata[img].focal_length;
	const Vector2f principal_point = metadata[img].principal_point;
	const float s0 = metadata[img].s0;

	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const CameraDistortion camera_distortion = metadata[img].camera_distortion;

	const Matrix<float, 3, 4> xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, xy, motionblur_time);

	Ray ray_unnormalized;
	// Rays need to be inferred from the camera matrix
	ray_unnormalized.o = xform.col(3);
	if (camera_distortion.mode == ECameraDistortionMode::FTheta) {
		ray_unnormalized.d = f_theta_undistortion(xy - principal_point, camera_distortion.params, {0.f, 0.f, 1.f});
	} else {
		ray_unnormalized.d = {
			(xy.x()-principal_point.x())*resolution.x() / focal_length.x(),
			(xy.y()-principal_point.y())*resolution.y() / focal_length.y(),
			1.0f,
		};
		//if (i == 0){
		//	printf("%f %f %d %f\n",xy.x(),principal_point.x(),resolution.x(),focal_length.x());
		//	printf("%f %f %d %f\n\n",xy.y(),principal_point.y(),resolution.y(),focal_length.y());
		//}
		

		if (camera_distortion.mode == ECameraDistortionMode::Iterative) {
			iterative_camera_undistortion(camera_distortion.params, &ray_unnormalized.d.x(), &ray_unnormalized.d.y());
		}
	}

	if (distortion_data) {
		ray_unnormalized.d.head<2>() += read_image<2>(distortion_data, distortion_resolution, xy);
	}

	ray_unnormalized.d = (xform.block<3, 3>(0, 0) * ray_unnormalized.d); // NOT normalized

	Eigen::Vector3f ray_o = ray_unnormalized.o;
	Eigen::Vector3f dir = ray_unnormalized.d.normalized();

	ray_o -= first_frame_offset;
	if (rotation != nullptr && transition != nullptr) {
		// apply global rotation and transition
		#if rotation_reprensentation
			global_movement_with_rotation_6d<network_precision_t>(rotation, transition, ray_o, dir);
		#else
			global_movement_with_rotation_quaternion<network_precision_t>(rotation, transition, ray_o, dir);
		#endif
	}

	Vector2f tminmax = aabb.ray_intersect(ray_o, dir); // ray depth min and ray depth max
	float cone_angle = calc_cone_angle(dir.dot(xform.col(2)), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), 0.0f);

	float startt = tminmax.x();
	startt += calc_dt(startt, cone_angle) * random_val(rng);
	Vector3f idir = dir.cwiseInverse();

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t=startt;
	Vector3f pos;

	while (aabb.contains(pos = ray_o + t * dir) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
		}
	}
	if (j == 0 && !train_envmap) {
		return;
	}
	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	ray_indices_out[ray_idx] = i;
	rays_out_unnormalized[ray_idx] = ray_unnormalized; // 
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	Vector3f warped_dir = warp_direction(dir);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray_o + t * dir) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, dir, idir, res);
		}
	}
	if (max_level_rand_training) {
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}

__device__ LossAndGradient loss_and_gradient(const Vector4f& target, const Vector4f& prediction, ELossType loss_type) {
	switch (loss_type) {
		case ELossType::L1:          return l1_loss(target, prediction); break;
		default: case ELossType::L2: return mse_loss(target, prediction); break;
	}
}

__global__ void compute_loss_kernel_train_nerf_with_global_movement(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	float original_loss_scale,
	int padded_output_width,
	const float* __restrict__ envmap_data,
	float* __restrict__ envmap_gradient,
	const Vector2i envmap_resolution,
	ELossType envmap_loss_type,
	Array3f background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata_normal,
	const TrainingImageMetadata* __restrict__ metadata_albedo,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	float* __restrict__ loss_output,
	float* __restrict__ ek_loss_output,
	float* __restrict__ mask_loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i error_map_res,
	const Vector2i error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	Eigen::Vector2i sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	const Eigen::Array3f* __restrict__ exposure,
	Eigen::Array3f* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance,
	float variance,
	uint32_t training_step,
	const network_precision_t* rotation,
	const network_precision_t* transition,
	Eigen::Vector3f first_frame_offset,
	const float mask_loss_weight,
	const float ek_loss_weight,
	const float cos_anneal_ratio,
	const bool apply_L2,
	const bool apply_supernormal,
	const bool apply_rgbplus,
	const bool apply_relu,
	const bool apply_bce,
	const bool apply_light_opti,
	const bool apply_no_albedo,
	const TrainingXForm* training_xforms
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;
	float racine_3 = sqrtf(3.0f);

	float EPSILON = 1e-4f;

	// const float cos_anneal_ratio = 1.0f;

	Array4f rgb_ray = Array4f::Zero(); // modification
	Vector3f hitpoint = Vector3f::Zero();

	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float img_pdf = 1.0f;
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);

	float depth_ray = 0.f;
	float weight_sum = 0.f;
	uint32_t compacted_numsteps = 0;
	Eigen::Vector3f ray_o = rays_in_unnormalized[i].o;
	Eigen::Vector3f dir = rays_in_unnormalized[i].d.normalized();

	Eigen::Vector2i resolution = metadata_normal[img].resolution;
	float xy_pdf = 1.0f;
	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &xy_pdf);
		
	const Matrix<float, 3, 4>& xform = training_xforms[img].start;
	Eigen::Matrix3f Rt = xform.block<3, 3>(0, 0);

	Array3f exposure_scale = (0.6931471805599453f * exposure[img]).exp();
	Array4f texsamp_albedo = read_rgba(xy, resolution, metadata_albedo[img].pixels, metadata_albedo[img].image_data_type);
	Array4f texsamp_normal = read_rgba(xy, resolution, metadata_normal[img].pixels, metadata_normal[img].image_data_type);

	Array3f normal_value = linear_to_srgb(exposure_scale * texsamp_normal.head<3>())*2.0f - 1.0f;
	normal_value[1] *= -1;
	normal_value[2] *= -1;
	normal_value /= normal_value.matrix().norm();

	Array4f albedo_value;
	Array3f albedo_value3f;
	if (apply_no_albedo){
		albedo_value = Array4f(1.0f,1.0f,1.0f,0.0f);
	}
	else {
		albedo_value3f = linear_to_srgb(exposure_scale * texsamp_albedo.head<3>());
		// Calculate the 1-norm of albedo3f
        float norm_value = albedo_value3f.matrix().norm();

        // Set albedo with the three values of albedo3f and the fourth as 1 - norm_value
		if (apply_rgbplus){
			if (apply_L2){
				albedo_value << albedo_value3f[0], albedo_value3f[1], albedo_value3f[2], sqrtf(max(0.0f,3 -  albedo_value3f[0]*albedo_value3f[0] - albedo_value3f[1]*albedo_value3f[1] - albedo_value3f[2]*albedo_value3f[2]));
			}
			else{
				albedo_value << albedo_value3f[0], albedo_value3f[1], albedo_value3f[2], 3 -  abs(albedo_value3f[0]) - abs(albedo_value3f[1]) - abs(albedo_value3f[2]);
			}
			
		}
		else{
			albedo_value << albedo_value3f[0], albedo_value3f[1], albedo_value3f[2], 0.0f;
		}
	}

	auto radians = [](float deg) { return deg * M_PI / 180.0f; };
	Eigen::Vector3f tilt = {radians(0.0f),radians(120.0f),radians(240.0f)};
	Eigen::Vector3f slant = {radians(54.74f), radians(54.74f), radians(54.74f)};

	Eigen::Vector3f sin_slant = slant.array().sin();
	Eigen::Vector3f cos_slant = slant.array().cos();
	Eigen::Vector3f cos_tilt = tilt.array().cos();
	Eigen::Vector3f sin_tilt = tilt.array().sin();

	Eigen::Matrix3f light_directions; // lights directions in the camera frame
	light_directions.row(0) = -sin_slant.cwiseProduct(cos_tilt);
	light_directions.row(1) = -sin_slant.cwiseProduct(sin_tilt);
	light_directions.row(2) = -cos_slant;

	if (apply_supernormal) {
		// Eigen::Matrix3f light_directions = [1, 0 , 0], [0, 1, 0], [0, 0, 1]
		light_directions = Eigen::Matrix3f::Identity(); 
	}

	// Initialize random number generator
    curandState state;
    curand_init(clock64(), i, 0, &state);
    
    // Generate random integer between 0 and 2
    int random_light = curand(&state) % 3;

	if (apply_light_opti){

		Eigen::Vector3f normalGT = normal_value.matrix();
		Eigen::Vector3f k;
		k << -normalGT[1], normalGT[0], 0;
		k /= k.norm();

		float cos_theta = normalGT[2];
		float sin_theta = std::sqrt(1 - cos_theta * cos_theta);

		Eigen::Matrix3f K;
		K << 0, -k[2], k[1],
			k[2], 0, -k[0],
			-k[1], k[0], 0;

		Eigen::Matrix3f KK = k * k.transpose();
		Eigen::Matrix3f R = cos_theta * Eigen::Matrix3f::Identity() + sin_theta * K + (1 - cos_theta) * KK;
		light_directions = -R * light_directions;
	}
	Eigen::Vector3f light_cam = light_directions.col(random_light); // in the camera frame
	Eigen::Vector3f light = Rt * light_cam; // in the world frame

	float shading_target; // shading target (GT) computed in the camera frame 
	if (apply_relu){ // Apply ReLU to shading in the 2.5 target shading
		shading_target = activation_function(normal_value.matrix().dot(light_cam), ENerfActivation::ReLU);
	}
	else{
		shading_target = normal_value.matrix().dot(light_cam);
	}
	 
	Array4f rgbtarget = albedo_value * shading_target;

	ray_o -= first_frame_offset;
	if (rotation != nullptr && transition != nullptr) {
		// apply global rotation and transition
		#if rotation_reprensentation
			global_movement_with_rotation_6d<network_precision_t>(rotation, transition, ray_o, dir);
		#else
			global_movement_with_rotation_quaternion<network_precision_t>(rotation, transition, ray_o, dir);
		#endif
	}

	Vector3f prev_deformed_pos = ray_o;
	const Vector3f const_offset = Vector3f{1e-8f,1e-8f,1e-8f};
	
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 16> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)network_output;
		Array4f albedo;
		Array3f albedo3f;
		if (apply_no_albedo){
			albedo = Array4f(1.0f,1.0f,1.0f,0.0f);
		}
		else {
			albedo3f = network_to_rgb(local_network_output, rgb_activation);
			// Calculate the 1-norm of albedo3f
			float norm_value = albedo3f.matrix().norm();

			// Set albedo with the three values of albedo3f and the fourth as 1 - norm_value
			if (apply_rgbplus){
				if (apply_L2){
					albedo << albedo3f[0], albedo3f[1], albedo3f[2], sqrtf(max(0.0f,3 -  albedo3f[0]*albedo3f[0] - albedo3f[1]*albedo3f[1] - albedo3f[2]*albedo3f[2]));
				}
				else{
					albedo << albedo3f[0], albedo3f[1], albedo3f[2], 3 -  abs(albedo3f[0]) - abs(albedo3f[1]) - abs(albedo3f[2]);
				}
			}
			else{
				albedo << albedo3f[0], albedo3f[1], albedo3f[2], 0.0f;
			}

			
			
		}
		 
		const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = (pos - ray_o).norm();

		#if BENT_DIR
			if (compacted_numsteps == 0){
				Vector3f deformed_viewdir{float(local_network_output[8]),float(local_network_output[9]),float(local_network_output[10])};
				dir = unwarp_direction(deformed_viewdir).normalized();
			}
		#endif

		float inv_s = __expf((tcnn::network_precision_t)10 * local_network_output[7]);

		// Neus rendering
		float sdf_value = float(local_network_output[3]);
		Array3f pos_gradient = network_to_pos_gradient(local_network_output, ENerfActivation::None); // dsdf_dpos => normal

		#if NORMAL_VECTORS_NORMALIZED
			float gradient_norm_for_dir = std::sqrt(pos_gradient[0]*pos_gradient[0] + pos_gradient[1]*pos_gradient[1] + pos_gradient[2]*pos_gradient[2] + 1e-5);
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2])/ gradient_norm_for_dir ;
		#else
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2]);
		#endif
		
		float iter_cos = -(activation_function(-true_cos * 0.5 + 0.5, ENerfActivation::ReLU) * (1.0 - cos_anneal_ratio) + \
			activation_function(-true_cos, ENerfActivation::ReLU) * cos_anneal_ratio);

		float estimated_next_sdf = sdf_value + iter_cos * dt * 0.5;
		float estimated_prev_sdf = sdf_value - iter_cos * dt * 0.5;

		float next_cdf = activation_function(estimated_next_sdf * inv_s, ENerfActivation::Logistic);
		float prev_cdf = activation_function(estimated_prev_sdf * inv_s, ENerfActivation::Logistic);

		float p = prev_cdf - next_cdf;
		float c = prev_cdf;
		float p_div_c = (p + 1e-5f) / (c + 1e-5f);
		const float alpha = tcnn::clamp(p_div_c, 0.0f, 1.0f); // alpha = 1.f - sigmoid(next_sdf / s) / sigmoid(prev_sdf / s);

		const float weight = alpha * T;
		Vector3f normal = pos_gradient.matrix();

		float shading; // shading computed in the world frame
		if (apply_relu){ // Apply ReLU to shading 
 			shading = activation_function(normal.dot(light), ENerfActivation::ReLU);
		}
		else {
			shading = normal.dot(light);
		}
		rgb_ray += weight * albedo * shading ; // modification TODO
		hitpoint += weight * pos;
		depth_ray += weight * cur_depth;
		weight_sum += weight;
		T *= (1.f - alpha);

		network_output += padded_output_width;
		coords_in += 1;
	}
	hitpoint /= (1.0f - T);

	// Must be same seed as above to obtain the same
	// background color.
	
	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	Array3f pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	Array4f envmap_value;
	if (envmap_data) {
		envmap_value = read_envmap(envmap_data, envmap_resolution, dir);
		background_color = envmap_value.head<3>() + background_color * (1.0f - envmap_value.w());
	}

	
	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;


	
	float mask_certainty = (float) (texsamp_albedo.w() > 0.99); // 1 should be with color

	loss_type = ELossType::L1 ; // Define a default L1 loss
	if (apply_L2){ // Change to a L2 loss
		loss_type = ELossType::L2;
	}

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type); // modification

	if (apply_rgbplus){
		lg.loss /= 2;
		lg.gradient /= 2;
	}

	lg.loss *= mask_certainty;
	lg.gradient *= mask_certainty;

	lg.loss /= img_pdf * xy_pdf;

	float mask_gt =(float) (texsamp_normal.w() > 0.99); // 1 should be with color
	
	float gradient_weight_sum;

	if (weight_sum >= 1.0 - 1e-4){
		weight_sum = 1.0 - 1e-4;
		gradient_weight_sum = 0.0f;
	}
	else if (weight_sum <= 1e-4){
		weight_sum = 1e-4;
		gradient_weight_sum = 0.0f;
	}
	else{
		float sigmoid_weight_sum = 1.0f / ( 1.0f + exp(-weight_sum));

		if (apply_bce){ // Classic BCE loss
			gradient_weight_sum = ((1 - mask_gt)/(1-weight_sum) - mask_gt/weight_sum) * mask_loss_weight;
		}
		else { // Sigmoid BCE loss
			gradient_weight_sum = (sigmoid_weight_sum - mask_gt) * mask_loss_weight;
		}
	}
	
	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * xy_pdf;

	float mean_loss = lg.loss;
	if (loss_output) {
		loss_output[i] = mean_loss / (float)n_rays;
	}

	if (mask_loss_output) { 
		float sigmoid_weight_sum = 1.0f / ( 1.0f + exp(-weight_sum));
		if (apply_bce){  // Classic BCE loss
			mask_loss_output[i] = - (mask_gt * log(weight_sum) + (1 - mask_gt) * log(1 - weight_sum));
		}
		else { // Sigmoid BCE loss
			mask_loss_output[i] = - (mask_gt * log(sigmoid_weight_sum) + (1 - mask_gt) * log(1 - sigmoid_weight_sum));
		}
	}

	if (ek_loss_output) {
		ek_loss_output[i] = 0.f;
	}

	if (error_map) {
		const Vector2f pos = (xy.cwiseProduct(error_map_res.cast<float>()) - Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(error_map_res.cast<float>() - Vector2f::Constant(1.0f + 1e-4f));
		const Vector2i pos_int = pos.cast<int>();
		const Vector2f weight = pos - pos_int.cast<float>();

		Vector2i idx = pos_int.cwiseMin(resolution - Vector2i::Constant(2)).cwiseMax(0);

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[img * error_map_res.prod() + y * error_map_res.x() + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			Vector2i sharpness_pos = xy.cwiseProduct(sharpness_resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(sharpness_resolution - Vector2i::Constant(1));
			float sharp = sharpness_data[img * sharpness_resolution.prod() + sharpness_pos.y() * sharpness_resolution.x() + sharpness_pos.x()] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x(),   idx.y(),   (1 - weight.x()) * (1 - weight.y()) * mean_loss);
		deposit_val(idx.x()+1, idx.y(),        weight.x()  * (1 - weight.y()) * mean_loss);
		deposit_val(idx.x(),   idx.y()+1, (1 - weight.x()) *      weight.y()  * mean_loss);
		deposit_val(idx.x()+1, idx.y()+1,      weight.x()  *      weight.y()  * mean_loss);
	}

	float loss_scale = original_loss_scale / n_rays;
	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	Array4f rgb_ray2 = { 0.f,0.f,0.f,0.0f }; // modification
	float weight_sum2 = 0.f;
	float depth_ray2 = 0.f;
	T = 1.f;

	prev_deformed_pos = ray_o;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const Vector3f pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = (pos - ray_o).norm();

		float dt = unwarp_dt(coord_in->dt);
		const tcnn::vector_t<tcnn::network_precision_t, 16> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)network_output;

		Array4f albedo;
		Array3f albedo3f;
		if (apply_no_albedo){
			albedo = Array4f(1.0f,1.0f,1.0f,0.0f);
		}
		else {
			albedo3f = network_to_rgb(local_network_output, rgb_activation);

			if (apply_rgbplus){
				if (apply_L2){
					albedo << albedo3f[0], albedo3f[1], albedo3f[2], sqrtf(max(0.0f,3 -  albedo3f[0]*albedo3f[0] - albedo3f[1]*albedo3f[1] - albedo3f[2]*albedo3f[2]));
				}
				else{
					albedo << albedo3f[0], albedo3f[1], albedo3f[2], 3 -  abs(albedo3f[0]) - abs(albedo3f[1]) - abs(albedo3f[2]);
				}
			}
			else{
				albedo << albedo3f[0], albedo3f[1], albedo3f[2], 0.0f;
			}
			
		}
		
		float inv_s = __expf((tcnn::network_precision_t)10 * local_network_output[7]);

		// Neus rendering
		float sdf_value = float(local_network_output[3]);
		Array3f pos_gradient = network_to_pos_gradient(local_network_output, ENerfActivation::None);
		#if NORMAL_VECTORS_NORMALIZED
			float gradient_norm_for_dir = std::sqrt(pos_gradient[0]*pos_gradient[0] + pos_gradient[1]*pos_gradient[1] + pos_gradient[2]*pos_gradient[2] + 1e-5);
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2])/ gradient_norm_for_dir ;
		#else
			float true_cos = (dir[0] * pos_gradient[0] + dir[1] * pos_gradient[1] + dir[2] * pos_gradient[2]);
		#endif
		float iter_cos = -(activation_function(-true_cos * 0.5 + 0.5, ENerfActivation::ReLU) * (1.0 - cos_anneal_ratio) + \
			activation_function(-true_cos, ENerfActivation::ReLU) * cos_anneal_ratio);

		float estimated_next_sdf = sdf_value + iter_cos * dt * 0.5;
		float estimated_prev_sdf = sdf_value - iter_cos * dt * 0.5;

		float next_cdf = activation_function(estimated_next_sdf * inv_s, ENerfActivation::Logistic);
		float prev_cdf = activation_function(estimated_prev_sdf * inv_s, ENerfActivation::Logistic);

		float p = prev_cdf - next_cdf;
		float c = prev_cdf;
		float p_div_c = (p + 1e-5f) / (c + 1e-5f);
		const float alpha = tcnn::clamp(p_div_c, 0.0f, 1.0f); // alpha = 1.f - sigmoid(next_sdf / s) / sigmoid(prev_sdf / s);

		const float weight = alpha * T;
		Vector3f normal = pos_gradient.matrix();

		float shading;
		if (apply_relu){
 			shading = activation_function(normal.dot(light), ENerfActivation::ReLU);
		}
		else {
			shading = normal.dot(light);
		}
		rgb_ray2 += weight * albedo * shading; // modification
		depth_ray2 += weight * depth;
		weight_sum2 += weight;
		T *= (1.f - alpha);

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const Array4f suffix = rgb_ray - rgb_ray2; // modification
		Array3f dloss_dn ;

		Matrix<float,1,4> albedo_transpose = albedo.matrix().transpose();
		Matrix<float,3,4> light_albedo = light * albedo_transpose;
		dloss_dn = weight * light_albedo * lg.gradient.matrix(); // modification : pas sur du tout ! produit matriciel ?
		
		Matrix<float,3,4> jac_rgb;
		jac_rgb.setZero();
			jac_rgb(0, 0) = 1.0f;
			jac_rgb(1, 1) = 1.0f;
			jac_rgb(2, 2) = 1.0f;

		if(apply_rgbplus){
			if (apply_L2){
				jac_rgb(0, 3) = -2*albedo[0]/(albedo[3]+1e-5);
				jac_rgb(1, 3) = -2*albedo[1]/(albedo[3]+1e-5);
				jac_rgb(2, 3) = -2*albedo[2]/(albedo[3]+1e-5);
			}
			else{
				jac_rgb(0, 3) = -sign(albedo[0]);
				jac_rgb(1, 3) = -sign(albedo[1]);
				jac_rgb(2, 3) = -sign(albedo[2]);
			}
			
		}


		const Array3f dloss_by_drgb = weight * shading * jac_rgb * lg.gradient.matrix();
		

		tcnn::vector_t<tcnn::network_precision_t, 16> local_dL_doutput;

		float opti_rgb = 1.0f;
		if (apply_no_albedo){
			opti_rgb = 0.0f;
		}

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = opti_rgb  * loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation));//+ fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = opti_rgb  * loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation)) ;//+ fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = opti_rgb  * loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation)) ;//+ fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		const float sum_weight_suffix = weight_sum - weight_sum2;

		float dloss_dalpha;

		dloss_dalpha = (
		lg.gradient.matrix().dot((T * albedo * shading - suffix).matrix())  // modification
		+ (gradient_weight_sum * (T - sum_weight_suffix))  // loss mask
		)/ (1.0f - alpha + 1e-5);
		
			
		float dalpha_d_e_minus_sigmoidx;
		float d_e_minus_sigmoidx_dsdf;
		float d_e_minus_sigmoidx_dinvs;
		float dloss_dvariance;
		float d_alpha_d_plus_e;
		float d_plus_e_dinvs;
		float d_plus_e_iter_cos;
		float d_e_minus_sigmoidx_diter_cos;
		if (p_div_c <= 0.0f || p_div_c >= 1.0f){
			dalpha_d_e_minus_sigmoidx = 0.0f;
			d_e_minus_sigmoidx_dsdf = 0.0f;
			d_e_minus_sigmoidx_dinvs = 0.0f;
			d_alpha_d_plus_e = 0.0f;
			d_plus_e_dinvs = 0.0f;
			d_plus_e_iter_cos = 0.0f;
			d_e_minus_sigmoidx_diter_cos = 0.0f;
		} 
		else{
			float plus_sigmoid_x = inv_s * iter_cos * dt;
			float plus_e = __expf(plus_sigmoid_x);
			float e_minus_sigmoid = __expf(-estimated_next_sdf * inv_s);
			dalpha_d_e_minus_sigmoidx = - (plus_e - 1 - 1e-5) / ((1 + 1e-5 + e_minus_sigmoid) * (1 + 1e-5 + e_minus_sigmoid));
			d_e_minus_sigmoidx_dsdf = - inv_s * e_minus_sigmoid;
			d_e_minus_sigmoidx_dinvs = - estimated_next_sdf * e_minus_sigmoid;


			float a = 1 + e_minus_sigmoid;
			float b = 1+plus_e*e_minus_sigmoid;
			float c = 1e-5 + 1/(1 + plus_e*e_minus_sigmoid);
			float delta = a * (b * b) * (c * c);

			dalpha_d_e_minus_sigmoidx = - (plus_e / (delta) - 1 / (a * a * c));

			d_alpha_d_plus_e = - e_minus_sigmoid / (delta) ;
			
			d_plus_e_dinvs = plus_e * iter_cos * dt;
			d_plus_e_iter_cos = plus_e * inv_s * dt;

			d_e_minus_sigmoidx_diter_cos = - inv_s * e_minus_sigmoid * dt * 0.5;

		}

		float dloss_dinvs = dloss_dalpha * (dalpha_d_e_minus_sigmoidx * d_e_minus_sigmoidx_dinvs + d_alpha_d_plus_e * d_plus_e_dinvs);
		dloss_dvariance = dloss_dinvs * inv_s * 10;

		// dalpha_dnormal
		float d_iter_cos_true_cos; 
		if (true_cos >= 0){
			d_iter_cos_true_cos = 0.0f;
		}
		else{
			d_iter_cos_true_cos = 1.0f;
		}
		float gradient_norm = std::sqrt(pos_gradient[0]*pos_gradient[0] + pos_gradient[1]*pos_gradient[1] + pos_gradient[2]*pos_gradient[2] + 1e-6);
		float pos_gradient_norm_inv = 1 - 1 / gradient_norm;
		float dloss_dnormal_norm = dloss_dalpha * (dalpha_d_e_minus_sigmoidx * d_e_minus_sigmoidx_diter_cos + d_plus_e_iter_cos * d_alpha_d_plus_e) * d_iter_cos_true_cos;

		float dloss_dsdf = dloss_dalpha * dalpha_d_e_minus_sigmoidx * d_e_minus_sigmoidx_dsdf ;

		local_dL_doutput[3] = loss_scale * dloss_dsdf;

		// add ek_loss to log
		if (ek_loss_output) {
			ek_loss_output[i] += (gradient_norm - 1.0f) * (gradient_norm - 1.0f);
		}
		
		// for large scene, we only cal ek_loss for points in the aabb_scale 1.0 box
		BoundingBox ek_aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
		ek_aabb.inflate(0.5f);

		bool inside_ek_aabb = ek_aabb.contains(pos);
	
		// ek_loss

		local_dL_doutput[4] = (tcnn::network_precision_t)(ek_loss_weight * 2 * original_loss_scale * pos_gradient_norm_inv * pos_gradient[0]);
		local_dL_doutput[5] = (tcnn::network_precision_t)(ek_loss_weight * 2 * original_loss_scale * pos_gradient_norm_inv * pos_gradient[1]);
		local_dL_doutput[6] = (tcnn::network_precision_t)(ek_loss_weight * 2 * original_loss_scale * pos_gradient_norm_inv * pos_gradient[2]);
		
		local_dL_doutput[7] = (tcnn::network_precision_t)(loss_scale*dloss_dvariance);

		#if NORMAL_VECTORS_NORMALIZED
			float gradient_norm_3 = gradient_norm * gradient_norm * gradient_norm;

			float dnormal_x = loss_scale * dloss_dnormal_norm * dir[0];
			float dnormal_y = loss_scale * dloss_dnormal_norm * dir[1];
			float dnormal_z = loss_scale * dloss_dnormal_norm * dir[2];
			
			float jacobian_x_x = (pos_gradient[1] * pos_gradient[1] + pos_gradient[2] * pos_gradient[2] )/ (gradient_norm_3);
			float jacobian_y_x = - pos_gradient[0] * pos_gradient[1] / gradient_norm_3;
			float jacobian_z_x = - pos_gradient[0] * pos_gradient[2] / gradient_norm_3;

			local_dL_doutput[8] = (tcnn::network_precision_t)(dnormal_x * jacobian_x_x + dnormal_y * jacobian_y_x + dnormal_z * jacobian_z_x);
			
			float jacobian_y_y = (pos_gradient[0] * pos_gradient[0] + pos_gradient[2] * pos_gradient[2] )/ (gradient_norm_3);
			float jacobian_x_y = - pos_gradient[0] * pos_gradient[1] / gradient_norm_3;
			float jacobian_z_y = - pos_gradient[1] * pos_gradient[2] / gradient_norm_3;

			local_dL_doutput[9] = (tcnn::network_precision_t)(dnormal_x * jacobian_x_y + dnormal_y * jacobian_y_y + dnormal_z * jacobian_z_y);

			float jacobian_z_z = (pos_gradient[0] * pos_gradient[0] + pos_gradient[1] * pos_gradient[1] )/ (gradient_norm_3);
			float jacobian_x_z = - pos_gradient[0] * pos_gradient[2] / gradient_norm_3;
			float jacobian_y_z = - pos_gradient[1] * pos_gradient[2] / gradient_norm_3;

			local_dL_doutput[10] = (tcnn::network_precision_t)(dnormal_x * jacobian_x_z + dnormal_y * jacobian_y_z + dnormal_z * jacobian_z_z);


		#else
			local_dL_doutput[8] = (tcnn::network_precision_t)(loss_scale * (dloss_dn.x() + dloss_dnormal_norm * dir[0]));
			local_dL_doutput[9] = (tcnn::network_precision_t)(loss_scale * (dloss_dn.y() + dloss_dnormal_norm * dir[1]));
			local_dL_doutput[10] = (tcnn::network_precision_t)(loss_scale * (dloss_dn.z() + dloss_dnormal_norm * dir[2]));

		#endif

		*(tcnn::vector_t<tcnn::network_precision_t, 16>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;
	}

	if (ek_loss_output) {
		ek_loss_output[i] /= (float)compacted_numsteps * (float) n_rays;
	}

}


__global__ void compute_cam_gradient_train_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	bool snap_to_pixel_centers,
	Vector3f* cam_pos_gradient,
	Vector3f* cam_rot_gradient,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const Vector2i distortion_resolution,
	Vector2f* cam_focal_length_gradient,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const Vector2i error_map_res
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
	Eigen::Vector2i resolution = metadata[img].resolution;

	const Matrix<float, 3, 4>& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = ray.d.normalized();
	Ray ray_gradient = { Vector3f::Zero(), Vector3f::Zero() };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		// pos = ray.o + t * ray.d;

		const Vector3f warped_pos = coords(j)->pos.p;
		const Vector3f pos_gradient = coords_gradient(j)->pos.p.cwiseProduct(warp_position_derivative(warped_pos, aabb));
		ray_gradient.o += pos_gradient;
		const Vector3f pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = (pos - ray.o).norm();
		const Vector3f dir_gradient = coords_gradient(j)->dir.d.cwiseProduct(warp_direction_derivative(coords(j)->dir.d));
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	// Projection of the raydir gradient onto the plane normal to raydir,
	// because that's the only degree of motion that the raydir has.
	ray_gradient.d -= ray.d * ray_gradient.d.dot(ray.d);

	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float xy_pdf = 1.0f;

	Vector2f xy = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &xy_pdf);

	if (distortion_gradient) {
		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		Vector3f image_plane_gradient = xform.block<3,3>(0,0).inverse() * ray_gradient.d;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient<2>(image_plane_gradient.head<2>() / xy_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, xy);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		#pragma unroll
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / xy_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		Vector3f angle_axis = ray.d.cross(ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		#pragma unroll
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / xy_pdf);
		}
	}
}

__global__ void compute_extra_dims_gradient_train_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	const uint32_t* __restrict__ rays_counter,
	float* extra_dims_gradient,
	uint32_t n_extra_dims,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ ray_indices_in,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords_gradient,
	const float* __restrict__ cdf_img
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}
	uint32_t base = numsteps_in[i*2+1];
	coords_gradient += base;
	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

	extra_dims_gradient += n_extra_dims * img;

	for (uint32_t j = 0; j < numsteps; ++j) {
		const float *src = coords_gradient(j)->get_extra_dims();
		for (uint32_t k = 0; k < n_extra_dims; ++k) {
			atomicAdd(&extra_dims_gradient[k], src[k]);
		}
	}
}

__global__ void shade_kernel_nerf(
	const uint32_t n_elements,
	Array4f* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	Array4f* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	NerfPayload& payload = payloads[i];

	Array4f tmp = rgba[i];

	if (render_mode == ERenderMode::Normals) {
		Array3f n = tmp.head<3>().matrix().normalized().array();
		tmp.head<3>() = (0.5f * n + Array3f::Constant(0.5f)) * tmp.w();
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.head<3>() = srgb_to_linear(tmp.head<3>());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.w());
	if (render_mode != ERenderMode::Slice && tmp.w() > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

__global__ void compact_kernel_nerf(
	const uint32_t n_elements,
	Array4f* src_rgba, float* src_depth, NerfPayload* src_payloads,
	Array4f* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
	Array4f* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_rgba[idx] = src_rgba[i];
		dst_depth[idx] = src_depth[i];
	} else if (src_rgba[i].w() > 0.001f) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_rgba[idx] = src_rgba[i];
		dst_final_depth[idx] = src_depth[i];
	}
}

__global__ void init_rays_with_payload_kernel_nerf(
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	Vector2i resolution,
	Vector2f focal_length,
	Matrix<float, 3, 4> camera_matrix0,
	Matrix<float, 3, 4> camera_matrix1,
	Vector4f rolling_shutter,
	Vector2f screen_center,
	Vector3f parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float plane_z,
	float dof,
	CameraDistortion camera_distortion,
	const float* __restrict__ envmap_data,
	const Vector2i envmap_resolution,
	Array4f* __restrict__ framebuffer,
	float* __restrict__ depthbuffer,
	const float* __restrict__ distortion_data,
	const Vector2i distortion_resolution,
	ERenderMode render_mode,
	const network_precision_t* rotation,
	const network_precision_t* transition,
	const Eigen::Vector3f first_frame_offset
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	if (plane_z < 0) {
		dof = 0.0;
	}


	// TODO: pixel_to_ray also immediately computes u,v for the pixel, so this is somewhat redundant
	float u = (x + 0.5f) * (1.f / resolution.x());
	float v = (y + 0.5f) * (1.f / resolution.y());
	float ray_time = rolling_shutter.x() + rolling_shutter.y() * u + rolling_shutter.z() * v + rolling_shutter.w() * ld_random_val(sample_index, idx * 72239731);
	Ray ray = pixel_to_ray(
		sample_index,
		{x, y},
		resolution,
		focal_length,
		camera_matrix0 * ray_time + camera_matrix1 * (1.f - ray_time),
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		plane_z,
		dof,
		camera_distortion,
		distortion_data,
		distortion_resolution
	);

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	if (plane_z < 0) {
		float n = ray.d.norm();
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		depthbuffer[idx] = -plane_z;
		return;
	}

	depthbuffer[idx] = 1e10f;

	ray.d = ray.d.normalized();

	ray.o -= first_frame_offset;
	if (rotation != nullptr && transition != nullptr) {
		// global_movement_with_rotation<network_precision_t>(rotation, transition, ray.o, ray.d);
		// apply global rotation and transition
		#if rotation_reprensentation
			global_movement_with_rotation_6d<network_precision_t>(rotation, transition, ray.o, ray.d);
		#else
			global_movement_with_rotation_quaternion<network_precision_t>(rotation, transition, ray.o, ray.d);
		#endif
		// ray.d = ray.d.normalized();
	}


	if (envmap_data) {
		framebuffer[idx] = read_envmap(envmap_data, envmap_resolution, ray.d);
	}

	float t = fmaxf(aabb.ray_intersect(ray.o, ray.d).x(), NERF_RENDERING_NEAR_DISTANCE()) + 1e-6f;

	if (!aabb.contains(ray.o + ray.d * t)) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		Vector2f offset = Vector2f::Zero();
		if (distortion_data) {
			offset += read_image<2>(distortion_data, distortion_resolution, Vector2f((float)x + 0.5f, (float)y + 0.5f).cwiseQuotient(resolution.cast<float>()));
		}
		framebuffer[idx].head<3>() = to_rgb(offset * 50.0f);
		framebuffer[idx].w() = 1.0f;
		depthbuffer[idx] = 1.0f;
		payload.origin = ray.o + ray.d * 10000.0f;
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;
}

static constexpr float MIN_PDF = 0.01f;

__global__ void construct_cdf_2d(
	uint32_t n_images,
	uint32_t height,
	uint32_t width,
	const float* __restrict__ data,
	float* __restrict__ cdf_x_cond_y,
	float* __restrict__ cdf_y
) {
	const uint32_t y = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
	if (y >= height || img >= n_images) return;

	const uint32_t offset_xy = img * height * width + y * width;
	data += offset_xy;
	cdf_x_cond_y += offset_xy;

	float cum = 0;
	for (uint32_t x = 0; x < width; ++x) {
		cum += data[x] + 1e-10f;
		cdf_x_cond_y[x] = cum;
	}

	cdf_y[img * height + y] = cum;
	float norm = __frcp_rn(cum);

	for (uint32_t x = 0; x < width; ++x) {
		cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float)(x+1) / (float)width;
	}
}

__global__ void construct_cdf_1d(
	uint32_t n_images,
	uint32_t height,
	float* __restrict__ cdf_y,
	float* __restrict__ cdf_img
) {
	const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
	if (img >= n_images) return;

	cdf_y += img * height;

	float cum = 0;
	for (uint32_t y = 0; y < height; ++y) {
		cum += cdf_y[y];
		cdf_y[y] = cum;
	}

	cdf_img[img] = cum;

	float norm = __frcp_rn(cum);
	for (uint32_t y = 0; y < height; ++y) {
		cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float)(y+1) / (float)height;
	}
}

__global__ void safe_divide(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	float local_divisor = divisor[i];
	inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
}


void Testbed::NerfTracer::init_rays_from_camera(
	uint32_t sample_index,
	uint32_t padded_output_width,
	uint32_t n_extra_dims,
	const Vector2i& resolution,
	const Vector2f& focal_length,
	const Matrix<float, 3, 4>& camera_matrix0,
	const Matrix<float, 3, 4>& camera_matrix1,
	const Vector4f& rolling_shutter,
	Vector2f screen_center,
	Vector3f parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	float plane_z,
	float dof,
	const CameraDistortion& camera_distortion,
	const float* envmap_data,
	const Vector2i& envmap_resolution,
	const float* distortion_data,
	const Vector2i& distortion_resolution,
	Eigen::Array4f* frame_buffer,
	float* depth_buffer,
	uint8_t *grid,
	int show_accel,
	float cone_angle_constant,
	ERenderMode render_mode,
	cudaStream_t stream,
	const network_precision_t* rotation,
	const network_precision_t* transition,
	const Eigen::Vector3f first_frame_offset
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x() * resolution.y();
	enlarge(n_pixels, padded_output_width, n_extra_dims, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x(), threads.x), div_round_up((uint32_t)resolution.y(), threads.y), 1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		render_aabb,
		plane_z,
		dof,
		camera_distortion,
		envmap_data,
		envmap_resolution,
		frame_buffer,
		depth_buffer,
		distortion_data,
		distortion_resolution,
		render_mode,
		rotation,
		transition,
		first_frame_offset
	);

	m_n_rays_initialized = resolution.x() * resolution.y();

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(Array4f), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

	linear_kernel(advance_pos_nerf, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		camera_matrix1.col(2),
		focal_length,
		sample_index,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		cone_angle_constant
	);
}

uint32_t Testbed::NerfTracer::trace(
	NerfNetwork<network_precision_t>& network,
	const BoundingBox& render_aabb,
	const BoundingBox& train_aabb,
	const uint32_t n_training_images,
	const TrainingXForm* training_xforms,
	const Vector2f& focal_length,
	float cone_angle_constant,
	const uint8_t* grid,
	ERenderMode render_mode,
	const Eigen::Matrix<float, 3, 4> &camera_matrix,
	float depth_scale,
	int visualized_layer,
	int visualized_dim,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance,
	float glow_y_cutoff,
	int glow_mode,
	const float* extra_dims_gpu,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}
	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter.data(), 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter.data(), 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
				m_alive_counter.data(), m_hit_counter.data()
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		uint32_t n_steps_between_compaction = tcnn::clamp(m_n_rays_initialized / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		linear_kernel(generate_next_nerf_network_inputs, 0, stream,
			n_alive,
			render_aabb,
			train_aabb,
			focal_length,
			camera_matrix.col(2),
			rays_current.payload,
			input_data,
			n_steps_between_compaction,
			grid,
			(show_accel>=0) ? show_accel : 0,
			cone_angle_constant,
			extra_dims_gpu
		);
		uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
		GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);


		network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			network.input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network.visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}
		linear_kernel(composite_kernel_nerf, 0, stream,
			n_alive,
			n_elements,
			i,
			train_aabb,
			glow_y_cutoff,
			glow_mode,
			n_training_images,
			training_xforms,
			camera_matrix,
			focal_length,
			depth_scale,
			rays_current.rgba,
			rays_current.depth,
			rays_current.payload,
			input_data,
			m_network_output,
			network.padded_output_width(),
			n_steps_between_compaction,
			render_mode,
			grid,
			rgb_activation,
			density_activation,
			show_accel,
			min_transmittance,
			network.variance(),
			network.training_step(),
			network.cos_anneal_ratio()
		);

		i += n_steps_between_compaction;
	}
	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;
	auto scratch = allocate_workspace_and_distribute<
		Array4f, float, NerfPayload, // m_rays[0]
		Array4f, float, NerfPayload, // m_rays[1]
		Array4f, float, NerfPayload, // m_rays_hit

		network_precision_t,
		float
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_network_output = std::get<9>(scratch);
	m_network_input = std::get<10>(scratch);
}

void Testbed::Nerf::Training::reset_extra_dims(default_rng_t &rng) {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(n_extra_dims * (dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
	float *dst = extra_dims_cpu.data();
	ArrayXf zero(n_extra_dims);
	extra_dims_opt.resize(dataset.n_images, AdamOptimizer<ArrayXf>(1e-4f, zero));
	for (uint32_t i = 0; i < dataset.n_images; ++i) {
		Eigen::Vector3f light_dir = warp_direction(dataset.metadata_normal[i].light_dir.normalized());
		extra_dims_opt[i].reset_state(Eigen::VectorXf(n_extra_dims));
		Eigen::ArrayXf &optimzer_value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			if (dataset.has_light_dirs && j < 3)
				dst[j] = light_dir[j];
			else
				dst[j] = random_val(rng) * 2.f - 1.f;
			optimzer_value[j] = dst[j];
		}
		dst += n_extra_dims;
	}
	extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);
}

const float* Testbed::get_inference_extra_dims(cudaStream_t stream) const {
	if (m_nerf_network->n_extra_dims() == 0) {
		return nullptr;
	}
	const float* extra_dims_src = m_nerf.training.extra_dims_gpu.data() + m_nerf.extra_dim_idx_for_inference * m_nerf.training.dataset.n_extra_dims();
	if (!m_nerf.training.dataset.has_light_dirs) {
		return extra_dims_src;
	}

	// the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
	// we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
	size_t size = m_nerf_network->n_extra_dims() * sizeof(float);
	float* dims_gpu = m_nerf.training.extra_dims_gpu.data() + m_nerf.training.dataset.n_images * m_nerf.training.dataset.n_extra_dims();
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
	Eigen::Vector3f light_dir = warp_direction(m_nerf.light_dir.normalized());
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(Eigen::Vector3f)), cudaMemcpyHostToDevice, stream));
	return dims_gpu;
}

void Testbed::render_nerf(CudaRenderBuffer& render_buffer, const Vector2i& max_res, const Vector2f& focal_length, const Matrix<float, 3, 4>& camera_matrix0, const Matrix<float, 3, 4>& camera_matrix1, const Vector4f& rolling_shutter, const Vector2f& screen_center, cudaStream_t stream) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = m_visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	const float* extra_dims_gpu = get_inference_extra_dims(stream);


	ScopeGuard tmp_memory_guard{[&]() {
		m_nerf.tracer.clear();
	}};

	// Our motion vector code can't undo f-theta and grid distortions -- so don't render these if DLSS is enabled.
	bool render_opencv_camera_distortion = m_nerf.render_with_camera_distortion && (!render_buffer.dlss() || m_nerf.render_distortion.mode == ECameraDistortionMode::Iterative);
	bool render_grid_camera_distortion = m_nerf.render_with_camera_distortion && !render_buffer.dlss();

	CameraDistortion camera_distortion = render_opencv_camera_distortion ? m_nerf.render_distortion : CameraDistortion{};

	m_nerf.tracer.init_rays_from_camera(
		render_buffer.spp(),
		m_network->padded_output_width(),
		m_nerf_network->n_extra_dims(),
		render_buffer.in_resolution(),
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		get_scaled_parallax_shift(),
		m_snap_to_pixel_centers,
		m_render_aabb,
		plane_z,
		m_dof,
		camera_distortion,
		m_envmap.envmap->params_inference(),
		m_envmap.resolution,
		render_grid_camera_distortion ? m_distortion.map->params_inference() : nullptr,
		m_distortion.resolution,
		render_buffer.frame_buffer(),
		render_buffer.depth_buffer(),
		m_nerf.density_grid_bitfield.data(),
		m_nerf.show_accel,
		m_nerf.cone_angle_constant,
		render_mode,
		stream,
		m_predict_global_movement ? m_nerf_network->rotation()->params(): nullptr,
		m_predict_global_movement ? m_nerf_network->transition()->params(): nullptr,
		m_first_frame_offset
	);
	


	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = m_nerf.tracer.n_rays_initialized();
	} else {
		float depth_scale = 1.0f / m_nerf.training.dataset.scale;
		n_hit = m_nerf.tracer.trace(
			*m_nerf_network,
			m_render_aabb,
			m_aabb,
			m_nerf.training.n_images_for_training,
			m_nerf.training.transforms.data(),
			focal_length,
			m_nerf.cone_angle_constant,
			m_nerf.density_grid_bitfield.data(),
			render_mode,
			camera_matrix1,
			depth_scale,
			m_visualized_layer,
			m_visualized_dimension,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.show_accel,
			m_nerf.rendering_min_transmittance,
			m_nerf.m_glow_y_cutoff,
			m_nerf.m_glow_mode,
			extra_dims_gpu,
			stream
		);
	}
	RaysNerfSoa& rays_hit = m_render_mode == ERenderMode::Slice ? m_nerf.tracer.rays_init() : m_nerf.tracer.rays_hit();

	if (m_render_mode == ERenderMode::Slice) {
		// Store colors in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, tcnn::batch_size_granularity);
		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

		m_nerf.vis_input.enlarge(n_elements * floats_per_coord);
		m_nerf.vis_rgba.enlarge(n_elements);
		linear_kernel(generate_nerf_network_inputs_at_current_position, 0, stream, n_hit, m_aabb, rays_hit.payload, PitchedPtr<NerfCoordinate>((NerfCoordinate*)m_nerf.vis_input.data(), 1, 0, extra_stride), extra_dims_gpu );

		GPUMatrix<float> positions_matrix((float*)m_nerf.vis_input.data(), floats_per_coord, n_elements);
		GPUMatrix<float> rgbsigma_matrix((float*)m_nerf.vis_rgba.data(), 4, n_elements);

		if (m_visualized_dimension == -1) {
			m_network->inference(stream, positions_matrix, rgbsigma_matrix);
			linear_kernel(compute_nerf_density, 0, stream, n_hit, m_nerf.vis_rgba.data(), m_nerf.rgb_activation, m_nerf.density_activation);
		} else {
			m_network->visualize_activation(stream, m_visualized_layer, m_visualized_dimension, positions_matrix, rgbsigma_matrix);
		}

		linear_kernel(shade_kernel_nerf, 0, stream,
			n_hit,
			m_nerf.vis_rgba.data(),
			nullptr,
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer(),
			render_buffer.depth_buffer()
		);
		return;
	}

	linear_kernel(shade_kernel_nerf, 0, stream,
		n_hit,
		rays_hit.rgba,
		rays_hit.depth,
		rays_hit.payload,
		m_render_mode,
		m_nerf.training.linear_colors,
		render_buffer.frame_buffer(),
		render_buffer.depth_buffer()
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<NerfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}
		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

void Testbed::Nerf::Training::set_camera_intrinsics(int frame_idx, float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return;
	}
	if (fx <= 0.f) fx = fy;
	if (fy <= 0.f) fy = fx;
	auto &m = dataset.metadata_normal[frame_idx];
	if (cx < 0.f) cx = -cx; else cx = cx / m.resolution.x();
	if (cy < 0.f) cy = -cy; else cy = cy / m.resolution.y();
	ECameraDistortionMode mode = (k1 || k2 || p1 || p2) ? ECameraDistortionMode::Iterative : ECameraDistortionMode::None;
	m.camera_distortion = { mode, k1, k2, p1, p2 };
	m.principal_point = { cx, cy };
	m.focal_length = { fx, fy };
	update_metadata_normal(frame_idx, frame_idx + 1);
	update_metadata_albedo(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::set_camera_extrinsics(int frame_idx, const Eigen::Matrix<float, 3, 4> &camera_to_world) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return;
	}

	dataset.xforms[frame_idx].start = dataset.xforms[frame_idx].end = dataset.nerf_matrix_to_ngp(camera_to_world);
	cam_rot_offset[frame_idx].reset_state();
	cam_pos_offset[frame_idx].reset_state();
	cam_exposure[frame_idx].reset_state();
	update_transforms(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::reset_camera_extrinsics() {
	for (auto&& opt : cam_rot_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_pos_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_exposure) {
		opt.reset_state();
	}
}

void Testbed::Nerf::Training::export_camera_extrinsics(const std::string& filename, bool export_extrinsics_in_quat_format) {
	tlog::info() << "Saving a total of " << n_images_for_training << " poses to " << filename;
	nlohmann::json trajectory;
	for(int i = 0; i < n_images_for_training; ++i) {
		nlohmann::json frame {{"id", i}};

		const Eigen::Matrix<float, 3, 4> p_nerf = get_camera_extrinsics(i);
		if (export_extrinsics_in_quat_format) {
			// Assume 30 fps
			frame["time"] =  i*0.033f;
			// Convert the pose from NeRF to Quaternion format.
			const Eigen::Matrix<float, 3, 3> conv_coords_l {{ 0.f,  1.f,  0.f},
															{ 0.f,  0.f, -1.f},
															{-1.f,  0.f,  0.f}};
			const Eigen::Matrix<float, 4, 4> conv_coords_r {{ 1.f,  0.f,  0.f,  0.f},
															{ 0.f, -1.f,  0.f,  0.f},
															{ 0.f,  0.f, -1.f,  0.f},
															{ 0.f,  0.f,  0.f,  1.f}};
			const Eigen::Matrix<float, 3, 4> p_quat = conv_coords_l * p_nerf * conv_coords_r;

			const Eigen::Quaternionf rot_q {p_quat.block<3, 3>(0, 0)};
			frame["q"] = {rot_q.w(), rot_q.x(), rot_q.y(), rot_q.z()};
			frame["t"] = {p_quat(0, 3), p_quat(1, 3), p_quat(2, 3)};
		} else {
			frame["transform_matrix"] = {p_nerf.row(0), p_nerf.row(1), p_nerf.row(2)};
		}

		trajectory.emplace_back(frame);
	}
	std::ofstream file(filename);
    file << std::setw(2) << trajectory << std::endl;
}

Eigen::Matrix<float, 3, 4> Testbed::Nerf::Training::get_camera_extrinsics(int frame_idx) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return Eigen::Matrix<float, 3, 4>::Identity();
	}
	return dataset.ngp_matrix_to_nerf(transforms[frame_idx].start);
}

void Testbed::Nerf::Training::update_metadata_normal(int first, int last) {
	if (last < 0) {
		last = dataset.n_images;
	}

	if (last > dataset.n_images) {
		last = dataset.n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}
	metadata_normal_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(metadata_normal_gpu.data() + first, dataset.metadata_normal.data() + first, n * sizeof(TrainingImageMetadata), cudaMemcpyHostToDevice));
}

void Testbed::Nerf::Training::update_metadata_albedo(int first, int last) {
	if (last < 0) {
		last = dataset.n_images;
	}

	if (last > dataset.n_images) {
		last = dataset.n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}
	metadata_albedo_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(metadata_albedo_gpu.data() + first, dataset.metadata_albedo.data() + first, n * sizeof(TrainingImageMetadata), cudaMemcpyHostToDevice));
}


void Testbed::Nerf::Training::update_transforms(int first, int last) {
	if (last < 0) {
		last=dataset.n_images;
	}

	if (last > dataset.n_images) {
		last = dataset.n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}

	if (transforms.size() < last) {
		transforms.resize(last);
	}

	for (uint32_t i = 0; i < n; ++i) {
		auto xform = dataset.xforms[i + first];
		Vector3f rot = cam_rot_offset[i + first].variable();
		float angle = rot.norm();
		rot /= angle;

		if (angle > 0) {
			xform.start.block<3, 3>(0, 0) = AngleAxisf(angle, rot) * xform.start.block<3, 3>(0, 0);
			xform.end.block<3, 3>(0, 0) = AngleAxisf(angle, rot) * xform.end.block<3, 3>(0, 0);
		}

		xform.start.col(3) += cam_pos_offset[i + first].variable();
		xform.end.col(3) += cam_pos_offset[i + first].variable();
		transforms[i + first] = xform;
	}

	transforms_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
}

void Testbed::create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr) {
	m_nerf.training.dataset = ngp::create_empty_nerf_dataset(n_images, aabb_scale, is_hdr);
	load_nerf();
	m_nerf.training.n_images_for_training = 0;
	m_training_data_available = true;
}

void Testbed::load_nerf() {
	if (!m_data_path.empty()) {

		if (m_data_path.is_directory()) {
			for (const auto& path : fs::directory{m_data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					auto idx = path.basename().find("transform");
					if (idx != std::string::npos){
						tlog::info() << path.str();
						all_json_paths.emplace_back(path);
					}
				}
			}
		} else if (equals_case_insensitive(m_data_path.extension(), "msgpack")) {
			load_snapshot(m_data_path.str());
			set_train(false);
			return;
		} else if (equals_case_insensitive(m_data_path.extension(), "json")) {
			tlog::info() << "Json here 2 ";
			all_json_paths.emplace_back(m_data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}
		
		// sorted all_json_paths
		std::sort(all_json_paths.begin(), all_json_paths.end(), [](const fs::path& a, const fs::path& b) {
			return a.basename() < b.basename();
		});
		// printf all_json_path
		for (const auto& path : all_json_paths) {
			printf("founded json file: %s\n", path.str().c_str());
		}

		all_training_time_frame = all_json_paths.size();
		printf("total frame: %d\n",all_training_time_frame);

		std::vector<fs::path> tmp_json_paths;
		tmp_json_paths.emplace_back(all_json_paths[0]);
		printf("tmp_json_paths size: %d\n", (int)tmp_json_paths.size());

		m_nerf.training.dataset = ngp::load_nerf(tmp_json_paths, m_nerf.sharpen);
	}

	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.update_metadata_normal();
	m_nerf.training.update_metadata_albedo();


	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f));
	m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, Array3f::Zero());
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = Vector2f::Zero();
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	m_nerf.training.reset_extra_dims(m_rng);

	if (m_nerf.training.dataset.has_rays) {
		m_nerf.training.near_distance = 0.0f;
		// m_nerf.training.optimize_exposure = true;
	}

	// Uncomment the following line to see how the network learns distortion from scratch rather than
	// starting from the distortion that's described by the training data.
	// m_nerf.training.dataset.camera_distortion = {};

	// Perturbation of the training cameras -- for debugging the online extrinsics learning code
	float perturb_amount = 0.0f;
	if (perturb_amount > 0.f) {
		for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
			Vector3f rot = random_val_3d(m_rng) * perturb_amount;
			float angle = rot.norm();
			rot /= angle;
			auto trans = random_val_3d(m_rng);
			m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0);
			m_nerf.training.dataset.xforms[i].start.col(3) += trans * perturb_amount;
			m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0);
			m_nerf.training.dataset.xforms[i].end.col(3) += trans * perturb_amount;
		}
	}

	m_nerf.training.update_transforms();

	if (!m_nerf.training.dataset.metadata_normal.empty()) {
		m_nerf.render_distortion = m_nerf.training.dataset.metadata_normal[0].camera_distortion;
		m_screen_center = Eigen::Vector2f::Constant(1.f) - m_nerf.training.dataset.metadata_normal[0].principal_point;
	}

	if (!m_nerf.training.dataset.metadata_albedo.empty()) {
		m_nerf.render_distortion = m_nerf.training.dataset.metadata_albedo[0].camera_distortion;
		m_screen_center = Eigen::Vector2f::Constant(1.f) - m_nerf.training.dataset.metadata_albedo[0].principal_point;
	}

	if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{std::string{"NeRF dataset's `aabb_scale` must be a power of two, but is "} + std::to_string(m_nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{
			std::string{"NeRF dataset must have `aabb_scale <= "} + std::to_string(max_aabb_scale) +
			"`, but is " + std::to_string(m_nerf.training.dataset.aabb_scale) +
			". You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling."
		};
	}

	m_aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);
	// m_nerf.cone_angle_constant = (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::load_nerf(uint32_t frame_time_idx, bool is_downsample) {
	if (!m_data_path.empty()) {
		auto transfer_to_downsample_json = [] (const auto& path, const bool is_downsample) {
			if (!is_downsample) {
				return path.str();
			}
			std::string downsample_path = path.stem().str() + std::string{"_downsample.json"};
			printf("load downsample json: %s\n", downsample_path.c_str());
			return downsample_path;
		};

		std::vector<fs::path> tmp_json_paths;
		tmp_json_paths.emplace_back(transfer_to_downsample_json(all_json_paths[frame_time_idx], is_downsample));

		m_nerf.training.dataset = ngp::load_nerf(tmp_json_paths, m_nerf.sharpen);

		tlog::info() << "Load data of frame: " << frame_time_idx;
	}

	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.update_metadata_normal();
	m_nerf.training.update_metadata_albedo();


	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f));
	m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, Vector3f::Zero());
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, Array3f::Zero());
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = Vector2f::Zero();
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	m_nerf.training.reset_extra_dims(m_rng);

	if (m_nerf.training.dataset.has_rays) {
		m_nerf.training.near_distance = 0.0f;
		// m_nerf.training.optimize_exposure = true;
	}

	// Uncomment the following line to see how the network learns distortion from scratch rather than
	// starting from the distortion that's described by the training data.
	// m_nerf.training.dataset.camera_distortion = {};

	// Perturbation of the training cameras -- for debugging the online extrinsics learning code
	float perturb_amount = 0.0f;
	if (perturb_amount > 0.f) {
		for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
			Vector3f rot = random_val_3d(m_rng) * perturb_amount;
			float angle = rot.norm();
			rot /= angle;
			auto trans = random_val_3d(m_rng);
			m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0);
			m_nerf.training.dataset.xforms[i].start.col(3) += trans * perturb_amount;
			m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0);
			m_nerf.training.dataset.xforms[i].end.col(3) += trans * perturb_amount;
		}
	}

	m_nerf.training.update_transforms();

	if (!m_nerf.training.dataset.metadata_normal.empty()) {
		m_nerf.render_distortion = m_nerf.training.dataset.metadata_normal[0].camera_distortion;
		m_screen_center = Eigen::Vector2f::Constant(1.f) - m_nerf.training.dataset.metadata_normal[0].principal_point;
	}

	if (!m_nerf.training.dataset.metadata_albedo.empty()) {
		m_nerf.render_distortion = m_nerf.training.dataset.metadata_albedo[0].camera_distortion;
		m_screen_center = Eigen::Vector2f::Constant(1.f) - m_nerf.training.dataset.metadata_albedo[0].principal_point;
	}

	if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{std::string{"NeRF dataset's `aabb_scale` must be a power of two, but is "} + std::to_string(m_nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{
			std::string{"NeRF dataset must have `aabb_scale <= "} + std::to_string(max_aabb_scale) +
			"`, but is " + std::to_string(m_nerf.training.dataset.aabb_scale) +
			". You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling."
		};
	}

	m_aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::reset_density_grid_nerf(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * (m_nerf.max_cascade + 1);

	printf("***********reset_density_grid_nerf!********\n");

	uint32_t n_uniform_density_grid_samples = n_elements;
	
	uint32_t n_nonuniform_density_grid_samples = 0;

	float decay = m_nerf.training.density_grid_decay;


	m_nerf.density_grid.resize(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

	// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
	// if (!m_nerf.training.dataset.has_rays) {
	// 	linear_kernel(mark_untrained_density_grid, 0, stream, n_elements, m_nerf.density_grid.data(),
	// 		m_nerf.training.n_images_for_training,
	// 		m_nerf.training.metadata_gpu.data(),
	// 		m_nerf.training.transforms_gpu.data(),
	// 		true
	// 	);
	// } else {
	// 	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
	// }

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));

	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_nerf.training.density_grid_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_nerf.training.density_grid_rng.advance();

		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, n_density_grid_samples);
		GPUMatrix<float> density_grid_position_matrix((float*)density_grid_positions, sizeof(NerfPosition)/sizeof(float), n_density_grid_samples);
		m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	update_density_grid_mean_and_bitfield(stream);
}

void Testbed::update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * (m_nerf.max_cascade + 1);

	m_nerf.density_grid.resize(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

	if (m_training_step == 0 || m_nerf.training.n_images_for_training != m_nerf.training.n_images_for_training_prev) {
		m_nerf.training.n_images_for_training_prev = m_nerf.training.n_images_for_training;
		if (m_training_step == 0) {
			m_nerf.density_grid_ema_step = 0;
		}
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
	}

	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_nerf.training.density_grid_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_nerf.training.density_grid_rng.advance();

		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, n_density_grid_samples);
		GPUMatrix<float> density_grid_position_matrix((float*)density_grid_positions, sizeof(NerfPosition)/sizeof(float), n_density_grid_samples);
		m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	update_density_grid_mean_and_bitfield(stream);
}

void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	// reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements * NERF_CASCADES()); }, m_nerf.density_grid_mean.data(), n_elements * NERF_CASCADES(), stream);
	// linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());


	reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	}

}

void Testbed::Nerf::Training::Counters::prepare_for_training_steps(cudaStream_t stream) {
	numsteps_counter.enlarge(1);
	numsteps_counter_compacted.enlarge(1);
	loss.enlarge(rays_per_batch);
	ek_loss.enlarge(rays_per_batch);
	mask_loss.enlarge(rays_per_batch);
	CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter_compacted.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(loss.data(), 0, sizeof(float)*rays_per_batch, stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ek_loss.data(), 0, sizeof(float)*rays_per_batch, stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(mask_loss.data(), 0, sizeof(float)*rays_per_batch, stream));
}

float Testbed::Nerf::Training::Counters::update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	std::vector<uint32_t> counter_cpu(1);
	std::vector<uint32_t> compacted_counter_cpu(1);
	numsteps_counter.copy_to_host(counter_cpu);
	numsteps_counter_compacted.copy_to_host(compacted_counter_cpu);
	measured_batch_size = 0;
	measured_batch_size_before_compaction = 0;

	if (counter_cpu[0] == 0 || compacted_counter_cpu[0] == 0) {
		return 0.f;
	}

	measured_batch_size_before_compaction = counter_cpu[0];
	measured_batch_size = compacted_counter_cpu[0];

	float loss_scalar = 0.0;
	if (get_loss_scalar) {
		loss_scalar = reduce_sum(loss.data(), rays_per_batch, stream) * (float)measured_batch_size / (float)target_batch_size;
		ek_loss_scalar = reduce_sum(ek_loss.data(), rays_per_batch, stream) * (float)measured_batch_size / (float)target_batch_size;
		mask_loss_scalar = reduce_sum(mask_loss.data(), rays_per_batch, stream) * (float)measured_batch_size / (float)target_batch_size;
	}

	rays_per_batch = (uint32_t)((float)rays_per_batch * (float)target_batch_size / (float)measured_batch_size);
	rays_per_batch = std::min(next_multiple(rays_per_batch, tcnn::batch_size_granularity), 1u << 18);

	return loss_scalar;
}

void Testbed::train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	m_nerf_network->m_training_step = m_training_step;

	if (m_nerf.training.include_sharpness_in_error) {
		size_t n_cells = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();
		if (m_nerf.training.sharpness_grid.size() < n_cells) {
			m_nerf.training.sharpness_grid.enlarge(NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES());
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		}

		if (m_training_step == 0) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		} else {
			linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
		}
	}
	m_nerf.training.counters_rgb.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
	}

	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
	if (train_extra_dims) {
		uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
		m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(), 0, m_nerf.training.extra_dims_gradient_gpu.get_bytes(), stream));
	}

	if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata_normal.empty()) {
		uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb.rays_per_batch) / m_nerf.training.dataset.n_images;
		Eigen::Vector2i res = m_nerf.training.dataset.metadata_normal[0].resolution;
		m_nerf.training.error_map.resolution = Vector2i::Constant((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)).cwiseMin(res);
		m_nerf.training.error_map.data.resize(m_nerf.training.error_map.resolution.prod() * m_nerf.training.dataset.n_images);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
	}

	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	if (envmap_gradient) {
		CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float)*m_envmap.envmap->n_params(), stream));
	}

	train_nerf_step(
		target_batch_size,
		m_nerf.training.counters_rgb.rays_per_batch,
		m_nerf.training.counters_rgb.numsteps_counter.data(),
		m_nerf.training.counters_rgb.numsteps_counter_compacted.data(),
		m_nerf.training.counters_rgb.loss.data(),
		m_nerf.training.counters_rgb.ek_loss.data(),
		m_nerf.training.counters_rgb.mask_loss.data(),
		m_training_stream
	);

	if (m_train_canonical) {
		m_trainer->optimizer_step(stream, LOSS_SCALE);
	}
	if (m_train_delta) {
		m_global_move.trainer->optimizer_step(stream, LOSS_SCALE);
	}

	++m_training_step;

	if (m_predict_global_movement){ // we need predict global movement for the next frames
		if (current_training_time_frame == 0){
			m_canonical_training_step = m_training_step;	
		}
		else{
			if (m_training_step < m_predict_global_movement_training_step) { // we are still training the global movement prediction
				m_canonical_training_step == min(first_frame_max_training_step, next_frame_max_training_step); // the canonical training status is still in the next frame
			}
			else{
				m_canonical_training_step = m_training_step - m_predict_global_movement_training_step; // start training the canonical 
			}
		}
	}
	else{ // we are not going to predict the global movement
		m_canonical_training_step = m_training_step;
	}


	if (envmap_gradient) {
		m_envmap.trainer->optimizer_step(stream, LOSS_SCALE);
	}

	float loss_scalar = m_nerf.training.counters_rgb.update_after_training(target_batch_size, get_loss_scalar, stream);
	bool zero_records = m_nerf.training.counters_rgb.measured_batch_size == 0;
	if (get_loss_scalar) {
		m_loss_scalar.update(loss_scalar);
		m_ek_loss_scalar.update(m_nerf.training.counters_rgb.ek_loss_scalar);
		m_mask_loss_scalar.update(m_nerf.training.counters_rgb.mask_loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar.set(0.f);
		m_ek_loss_scalar.set(0.f);
		m_mask_loss_scalar.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
		m_train = false;
	}

	// Compute CDFs from the error map
	m_nerf.training.n_steps_since_error_map_update += 1;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;
	if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
		m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
		m_nerf.training.error_map.cdf_x_cond_y.resize(m_nerf.training.error_map.cdf_resolution.prod() * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y() * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_img.resize(m_nerf.training.dataset.n_images);

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y(), threads.x), div_round_up((uint32_t)m_nerf.training.dataset.n_images, threads.y), 1 };
		construct_cdf_2d<<<blocks, threads, 0, stream>>>(
			m_nerf.training.dataset.n_images, m_nerf.training.error_map.cdf_resolution.y(), m_nerf.training.error_map.cdf_resolution.x(),
			m_nerf.training.error_map.data.data(),
			m_nerf.training.error_map.cdf_x_cond_y.data(),
			m_nerf.training.error_map.cdf_y.data()
		);
		linear_kernel(construct_cdf_1d, 0, stream,
			m_nerf.training.dataset.n_images,
			m_nerf.training.error_map.cdf_resolution.y(),
			m_nerf.training.error_map.cdf_y.data(),
			m_nerf.training.error_map.cdf_img.data()
		);

		// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
		m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
		m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
		std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
		float cum = 0;
		for (float& f : cdf_img_cpu) {
			cum += f;
			f = cum;
		}
		float norm = 1.0f / cum;
		for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
			constexpr float MIN_PMF = 0.1f;
			m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.dataset.n_images;
			cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i+1) / (float)m_nerf.training.dataset.n_images;
		}
		m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

		// Reset counters and decrease update rate.
		m_nerf.training.n_steps_since_error_map_update = 0;
		m_nerf.training.n_rays_since_error_map_update = 0;
		m_nerf.training.error_map.is_cdf_valid = true;

		m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
	}

	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += 1;


	if (train_extra_dims) {
		std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
		std::vector<float> &extra_dims_new_values = extra_dims_gradient; // just create an alias to make the code clearer.
		m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);
		// Optimization step
		for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
			ArrayXf gradient(n_extra_dims);
			gradient.setZero();
			for (uint32_t j = 0; j<n_extra_dims; ++j) {
				gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE;
				if (isnan(gradient[j])) {
					printf("OH NO %d %d %0.3f\n", i,j, gradient[j]);
				}
			}

			float l2_reg = 1e-4f;
			gradient = m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

			//m_nerf.training.extra_dims_opt[i].set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.extra_dims_opt[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));

			m_nerf.training.extra_dims_opt[i].step(gradient);

			const ArrayXf &value = m_nerf.training.extra_dims_opt[i].variable();
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				extra_dims_new_values[i * n_extra_dims + j] = value[j];
			}
		}

		//m_nerf.training.extra_dims_gpu.copy_from_host(extra_dims_new_values);
		CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.extra_dims_gpu.data(), extra_dims_new_values.data(), m_nerf.training.n_images_for_training * n_extra_dims * sizeof(float) , cudaMemcpyHostToDevice, stream));
	}

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
	if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		float per_camera_loss_scale = (float)m_nerf.training.n_images_for_training / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

		if (m_nerf.training.optimize_extrinsics) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				Vector3f pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
				Vector3f rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.extrinsic_l2_reg;
				pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
				rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

				m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_pos_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
				m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_rot_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));

				m_nerf.training.cam_pos_offset[i].step(pos_gradient);
				m_nerf.training.cam_rot_offset[i].step(rot_gradient);
			}

			m_nerf.training.update_transforms();
		}

		if (m_nerf.training.optimize_distortion) {
			linear_kernel(safe_divide, 0, stream,
				m_distortion.map->n_params(),
				m_distortion.map->gradients(),
				m_distortion.map->gradient_weights()
			);
			m_distortion.trainer->optimizer_step(stream, LOSS_SCALE*(float)m_nerf.training.n_steps_between_cam_updates);
		}

		if (m_nerf.training.optimize_focal_length) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_focal_length_gradient.data(),m_nerf.training.cam_focal_length_gradient_gpu.data(),m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(),cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			Vector2f focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
			float l2_reg = m_nerf.training.intrinsic_l2_reg;
			focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
			m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_focal_length_offset.step() / 128)),m_optimizer->learning_rate() / 1000.0f));
			m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
			m_nerf.training.update_metadata_normal();
			m_nerf.training.update_metadata_albedo();
		}

		if (m_nerf.training.optimize_exposure) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			Array3f mean_exposure = Array3f::Constant(0.0f);

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				Array3f gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.exposure_l2_reg;
				gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

				m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.cam_exposure[i].step(gradient);

				mean_exposure += m_nerf.training.cam_exposure[i].variable();
			}

			mean_exposure /= m_nerf.training.n_images_for_training;

			// Renormalize
			std::vector<Array3f> cam_exposures(m_nerf.training.n_images_for_training);
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
			}

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.cam_exposure_gpu.get_bytes(), cudaMemcpyHostToDevice, stream));
		}

		m_nerf.training.n_steps_since_cam_update = 0;
	}
}

void Testbed::train_nerf_step(uint32_t target_batch_size, uint32_t n_rays_per_batch, uint32_t* counter, uint32_t* compacted_counter, float* loss, float* ek_loss, float* mask_loss, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		uint32_t, // ray_indices
		Ray, // rays
		uint32_t, // numsteps
		float, // coords
		float, // max_level
		network_precision_t, // mlp_out
		network_precision_t, // dloss_dmlp_out
		float, // coords_compacted
		float, // coords_gradient
		float, // max_level_compacted
		uint32_t // ray_counter
	>(
		stream, &alloc, 
		n_rays_per_batch, // ray_indices
		n_rays_per_batch, // rays
		n_rays_per_batch * 2, // numsteps
		max_samples * floats_per_coord, // coords
		max_samples, // max_level
		std::max(target_batch_size, max_samples) * padded_output_width, // mlp_out
		target_batch_size * padded_output_width, // dloss_dmlp_out
		target_batch_size * floats_per_coord, // coords_compacted
		target_batch_size * floats_per_coord, // coords_gradient
		target_batch_size, // max_level_compacted
		1 // ray_counter
	);
	
	// TODO: C++17 structured binding
	uint32_t* ray_indices = std::get<0>(scratch);
	Ray* rays_unnormalized = std::get<1>(scratch);
	uint32_t* numsteps = std::get<2>(scratch);
	float* coords = std::get<3>(scratch);
	float* max_level = std::get<4>(scratch);
	network_precision_t* mlp_out = std::get<5>(scratch);
	network_precision_t* dloss_dmlp_out = std::get<6>(scratch);
	float* coords_compacted = std::get<7>(scratch);
	float* coords_gradient = std::get<8>(scratch);
	float* max_level_compacted = std::get<9>(scratch);
	uint32_t* ray_counter = std::get<10>(scratch);

	uint32_t max_inference;
	if (m_nerf.training.counters_rgb.measured_batch_size_before_compaction == 0) {
		m_nerf.training.counters_rgb.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(m_nerf.training.counters_rgb.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	}

	GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step == 0 || m_canonical_training_step == 0) {
		m_nerf.training.counters_rgb.n_rays_total = 0;
	}

	uint32_t n_rays_total = m_nerf.training.counters_rgb.n_rays_total;
	m_nerf.training.counters_rgb.n_rays_total += n_rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += n_rays_per_batch;

	// If we have an envmap, prepare its gradient buffer
	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

	bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
	bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));

	linear_kernel(generate_training_samples_nerf_with_global_movement, 0, stream,
		n_rays_per_batch,
		m_aabb,
		max_inference,
		n_rays_total,
		m_rng,
		ray_counter,
		counter,
		ray_indices,
		rays_unnormalized,
		numsteps,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_nerf.training.n_images_for_training,
		m_nerf.training.metadata_normal_gpu.data(),
		m_nerf.training.transforms_gpu.data(),
		m_nerf.density_grid_bitfield.data(),
		m_max_level_rand_training,
		max_level,
		m_nerf.training.snap_to_pixel_centers,
		m_nerf.training.train_envmap,
		m_nerf.cone_angle_constant,
		m_distortion.map->params(),
		m_distortion.resolution,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		m_nerf.training.error_map.cdf_resolution,
		m_nerf.training.extra_dims_gpu.data(),
		m_nerf_network->n_extra_dims(),
		current_training_time_frame,
		m_training_step,
		m_predict_global_movement ? m_nerf_network->rotation()->params(): nullptr,
		m_predict_global_movement ? m_nerf_network->transition()->params(): nullptr,
		m_first_frame_offset
	);

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
	}

	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

	if (hg_enc) {
		hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
	}

	linear_kernel(compute_loss_kernel_train_nerf_with_global_movement, 0, stream,
		n_rays_per_batch,
		m_aabb,
		n_rays_total,
		m_rng,
		target_batch_size,
		ray_counter,
		LOSS_SCALE,
		padded_output_width,
		m_envmap.envmap->params(),
		envmap_gradient,
		m_envmap.resolution,
		m_envmap.loss_type,
		m_background_color.head<3>(),
		m_color_space,
		m_nerf.training.random_bg_color,
		m_nerf.training.linear_colors,
		m_nerf.training.n_images_for_training,
		m_nerf.training.metadata_normal_gpu.data(),
		m_nerf.training.metadata_albedo_gpu.data(),
		mlp_out,
		compacted_counter,
		ray_indices,
		rays_unnormalized,
		numsteps,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		dloss_dmlp_out,
		m_nerf.training.loss_type,
		loss,
		ek_loss,
		mask_loss,
		m_max_level_rand_training,
		max_level_compacted,
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.training.snap_to_pixel_centers,
		accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		m_nerf.training.error_map.resolution,
		m_nerf.training.error_map.cdf_resolution,
		include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
		m_nerf.training.dataset.sharpness_resolution,
		m_nerf.training.sharpness_grid.data(),
		m_nerf.density_grid.data(),
		m_nerf.density_grid_mean.data(),
		m_nerf.training.cam_exposure_gpu.data(),
		m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
		m_nerf.training.depth_supervision_lambda,
		m_nerf.training.near_distance,
		m_nerf_network->variance(),
		m_training_step,
		m_predict_global_movement ? m_nerf_network->rotation()->params(): nullptr,
		m_predict_global_movement ? m_nerf_network->transition()->params(): nullptr,
		m_first_frame_offset,
		m_mask_loss_weight,
		m_ek_loss_weight,
		m_nerf_network->cos_anneal_ratio(),
		m_apply_L2,
		m_apply_supernormal,
		m_apply_rgbplus,
		m_apply_relu,
		m_apply_bce,
		m_light_opti,
		m_no_albedo,
		m_nerf.training.transforms_gpu.data()
	);


	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, compacted_counter, dloss_dmlp_out
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), n_threads_linear, 0, stream>>>(
		target_batch_size, floats_per_coord, compacted_counter, (float*)coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, compacted_counter, max_level_compacted
	);

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length;
	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	bool prepare_input_gradients = train_camera || train_extra_dims;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);
	std::vector<uint32_t> compacted_counter_cpu(1);
	CUDA_CHECK_THROW(cudaMemcpyAsync(compacted_counter_cpu.data(), compacted_counter, sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost, stream));
	std::vector<uint32_t> counter_cpu(1);
	CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu.data(), counter, sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost, stream));

	m_nerf_network->indeed_batch_size = target_batch_size;

	{

	auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
	m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);

	}

	if (train_extra_dims) {
		// Compute extra-dim gradients
		linear_kernel(compute_extra_dims_gradient_train_nerf, 0, stream,
		n_rays_per_batch,
			n_rays_total,
			ray_counter,
			m_nerf.training.extra_dims_gradient_gpu.data(),
			m_nerf.training.dataset.n_extra_dims(),
			m_nerf.training.n_images_for_training,
			ray_indices,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr
		);
	}

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf, 0, stream,
			n_rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
			m_nerf.training.n_images_for_training,
			m_nerf.training.metadata_normal_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}
}

void Testbed::training_prep_nerf(uint32_t batch_size, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	float alpha = m_nerf.training.density_grid_decay;
	uint32_t n_cascades = m_nerf.max_cascade+1;

	if (m_canonical_training_step < 256) {
		update_density_grid_nerf(alpha, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()*n_cascades, 0, stream);
	} else {
		update_density_grid_nerf(alpha, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()/4*n_cascades, NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE()/4*n_cascades, stream);
	}
}

void Testbed::optimise_mesh_step(uint32_t n_steps) {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
	GPUMemory<float> coords(n_verts * floats_per_coord);
	GPUMemory<network_precision_t> mlp_out(n_verts * padded_output_width);

	GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
	GPUMatrix<network_precision_t, RM> density_matrix(mlp_out.data(), padded_output_width, n_verts);

	const float* extra_dims_gpu = get_inference_extra_dims(m_inference_stream);

	for (uint32_t i = 0; i < n_steps; ++i) {
		linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_inference_stream,
			n_verts,
			m_aabb,
			m_mesh.verts.data(),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride),
			extra_dims_gpu
		);

		// For each optimizer step, we need the density at the given pos...
		m_nerf_network->sdf(m_inference_stream, positions_matrix, density_matrix);
		// ...as well as the input gradient w.r.t. density, which we will store in the nerf coords.
		m_nerf_network->input_gradient(m_inference_stream, 3, positions_matrix, positions_matrix);
		// and the 1ring centroid for laplacian smoothing
		compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);

		// With these, we can compute a gradient that points towards the threshold-crossing of density...
		compute_mesh_opt_gradients(
			m_mesh.thresh,
			m_mesh.verts,
			m_mesh.vert_normals,
			m_mesh.verts_smoothed,
			mlp_out.data(),
			floats_per_coord,
			(const float*)coords.data(),
			m_mesh.verts_gradient,
			m_mesh.smooth_amount,
			m_mesh.density_amount,
			m_mesh.inflate_amount
		);

		// ...that we can pass to the optimizer.
		m_mesh.verts_optimizer->step(m_inference_stream, 1.0f, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	}
}

void Testbed::compute_mesh_vertex_colors() {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	m_mesh.vert_colors.resize(n_verts);
	m_mesh.vert_colors.memset(0);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		const float* extra_dims_gpu = get_inference_extra_dims(m_inference_stream);

		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
		GPUMemory<float> coords(n_verts * floats_per_coord);
		GPUMemory<float> mlp_out(n_verts * 7);

		GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
		GPUMatrix<float> color_matrix(mlp_out.data(), 7, n_verts);
		linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_inference_stream, n_verts, m_aabb, m_mesh.verts.data(), PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride), extra_dims_gpu);
		m_network->inference(m_inference_stream, positions_matrix, color_matrix);
		linear_kernel(extract_srgb_with_activation, 0, m_inference_stream, n_verts * 3, 3, mlp_out.data(), (float*)m_mesh.vert_colors.data(), m_nerf.rgb_activation, m_nerf.training.linear_colors);
	}
}

GPUMemory<float> Testbed::get_density_on_grid(Vector3i res3d, const BoundingBox& aabb) {
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	GPUMemory<float> density(n_elements);

	const uint32_t batch_size = std::min(n_elements, 1u<<20);
	bool nerf_mode = m_testbed_mode == ETestbedMode::Nerf;

	const uint32_t padded_output_width = nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,
		network_precision_t
	>(m_inference_stream, &alloc, n_elements, batch_size * padded_output_width);

	NerfPosition* positions = std::get<0>(scratch);
	network_precision_t* mlp_out = std::get<1>(scratch);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x(), threads.x), div_round_up((uint32_t)res3d.y(), threads.y), div_round_up((uint32_t)res3d.z(), threads.z) };

	// 这里可以设置mc的范围
	BoundingBox unit_cube = BoundingBox{Vector3f::Zero(), Vector3f::Ones()};
	generate_grid_samples_nerf_uniform<<<blocks, threads, 0, m_inference_stream>>>(res3d, m_nerf.density_grid_ema_step, aabb, nerf_mode ? m_aabb : unit_cube , positions);

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, local_batch_size);

		GPUMatrix<float> positions_matrix((float*)(positions + offset), sizeof(NerfPosition)/sizeof(float), local_batch_size);

		if (nerf_mode) {
			m_nerf_network->sdf(m_inference_stream, positions_matrix, density_matrix);
		} else {
			m_network->inference_mixed_precision(m_inference_stream, positions_matrix, density_matrix);
		}
		linear_kernel(grid_samples_half_to_float, 0, m_inference_stream,
			local_batch_size,
			m_aabb,
			density.data() + offset , //+ axis_step * n_elements,
			mlp_out,
			m_nerf.density_activation,
			positions + offset,
			nerf_mode ? m_nerf.density_grid.data() : nullptr,
			m_nerf.max_cascade
		);
	}

	return density;
}

GPUMemory<Eigen::Array4f> Testbed::get_rgba_on_grid(Vector3i res3d, Eigen::Vector3f ray_dir) {
	const uint32_t n_elements = (res3d.x()*res3d.y()*res3d.z());
	GPUMemory<Eigen::Array4f> rgba(n_elements);
	GPUMemory<NerfCoordinate> positions(n_elements);
	const uint32_t batch_size = std::min(n_elements, 1u<<20);

	// generate inputs
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x(), threads.x), div_round_up((uint32_t)res3d.y(), threads.y), div_round_up((uint32_t)res3d.z(), threads.z) };
	generate_grid_samples_nerf_uniform_dir<<<blocks, threads, 0, m_inference_stream>>>(res3d, m_nerf.density_grid_ema_step, m_render_aabb, m_aabb, ray_dir, positions.data());

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		// run network
		GPUMatrix<float> positions_matrix((float*) (positions.data() + offset), sizeof(NerfCoordinate)/sizeof(float), local_batch_size);
		GPUMatrix<float> rgbsigma_matrix((float*) (rgba.data() + offset), 4, local_batch_size);
		m_network->inference(m_inference_stream, positions_matrix, rgbsigma_matrix);

		// convert network output to RGBA (in place)
		linear_kernel(compute_nerf_density, 0, m_inference_stream, local_batch_size, rgba.data() + offset, m_nerf.rgb_activation, m_nerf.density_activation);
	}
	return rgba;
}

int Testbed::marching_cubes(Vector3i res3d, const BoundingBox& aabb, float thresh) {
	res3d.x() = next_multiple((unsigned int)res3d.x(), 16u);
	res3d.y() = next_multiple((unsigned int)res3d.y(), 16u);
	res3d.z() = next_multiple((unsigned int)res3d.z(), 16u);

	tlog::info() << res3d.x() << res3d.y() << res3d.z() ;

	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}

	GPUMemory<float> density = get_density_on_grid(res3d, aabb);
	// marching_cubes_gpu(m_inference_stream, m_render_aabb, res3d, thresh, density, m_mesh.verts, m_mesh.indices);
	marching_cubes_gpu(m_inference_stream, aabb, res3d, thresh, density, m_mesh.verts, m_mesh.indices);

	uint32_t n_verts = (uint32_t)m_mesh.verts.size();

	// transform mesh

	#if rotation_reprensentation
		tcnn::linear_kernel(transform_mesh_with_6d<tcnn::network_precision_t>, 0, m_training_stream, n_verts,
			m_nerf_network->rotation()->params(),
			m_nerf_network->transition()->params(),
			(float*)m_mesh.verts.data());
	#else
		tcnn::linear_kernel(transform_mesh_with_quaternion<tcnn::network_precision_t>, 0, m_training_stream, n_verts,
			m_nerf_network->rotation()->params(),
			m_nerf_network->transition()->params(),
			(float*)m_mesh.verts.data());
	#endif

	m_mesh.verts_gradient.resize(n_verts);

	m_mesh.trainable_verts = std::make_shared<TrainableBuffer<3, 1, float>>(Matrix<int, 1, 1>{(int)n_verts});
	m_mesh.verts_gradient.copy_from_device(m_mesh.verts); // Make sure the vertices don't get destroyed in the initialization

	pcg32 rnd{m_seed};
	m_mesh.trainable_verts->initialize_params(rnd, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	m_mesh.verts.copy_from_device(m_mesh.verts_gradient);

	m_mesh.verts_optimizer.reset(create_optimizer<float>({
		{"otype", "Adam"},
		{"learning_rate", 1e-4},
		{"beta1", 0.9f},
		{"beta2", 0.99f},
	}));

	m_mesh.verts_optimizer->allocate(m_mesh.trainable_verts);

	compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);
	compute_mesh_vertex_colors();

	return (int)(m_mesh.indices.size()/3);
}

uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
	return density_grid_bitfield.data() + grid_mip_offset(mip)/8;
}

int Testbed::find_best_training_view(int default_view) {
	int bestimage = default_view;
	float bestscore = 1000.f;
	for (int i = 0; i < m_nerf.training.n_images_for_training; ++i) {
		float score = (m_nerf.training.transforms[i].start.col(3) - m_camera.col(3)).norm();
		score += 0.25f * (m_nerf.training.transforms[i].start.col(2) - m_camera.col(2)).norm();
		if (score < bestscore) {
			bestscore = score;
			bestimage = i;
		}
	}
	return bestimage;
}

NGP_NAMESPACE_END
