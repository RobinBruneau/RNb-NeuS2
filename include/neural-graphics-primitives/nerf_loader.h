/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_loader.h
 *  @author Alex Evans, NVIDIA
 *  @brief  Ability to load nerf datasets.
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <filesystem/path.h>


#include <json/json.hpp>

#include <vector>

NGP_NAMESPACE_BEGIN

// how much to scale the scene by vs the original nerf dataset; we want to fit the thing in the unit cube
static constexpr float NERF_SCALE = 0.33f;

struct TrainingImageMetadata {
	// Camera intrinsics and additional data associated with a NeRF training image
	// the memory to back the pixels and rays is held by GPUMemory objects in the NerfDataset and copied here.
	const void* pixels = nullptr;
	EImageDataType image_data_type = EImageDataType::Half;

	const float* depth = nullptr;
	const Ray* rays = nullptr;

	CameraDistortion camera_distortion = {};
	Eigen::Vector2i resolution = Eigen::Vector2i::Zero();
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);
	Eigen::Vector4f rolling_shutter = Eigen::Vector4f::Zero();
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f); // TODO: replace this with more generic float[] of task-specific metadata.
	float s0 = 0.0f;
};

struct InterfaceMetadata {
	Eigen::Vector3f* vertices = nullptr;
	Eigen::Vector3i* faces = nullptr;
	Eigen::Vector3f* normals = nullptr;
	float* d;
	float ior = 1.0;
	int n_faces = 0;
	int n_vertices = 0;
};

inline size_t image_type_size(EImageDataType type) {
	switch (type) {
		case EImageDataType::None: return 0;
		case EImageDataType::Byte: return 1;
		case EImageDataType::Half: return 2;
		case EImageDataType::Float: return 4;
		default: return 0;
	}
}

inline size_t depth_type_size(EDepthDataType type) {
	switch (type) {
		case EDepthDataType::UShort: return 2;
		case EDepthDataType::Float: return 4;
		default: return 0;
	}
}

struct NerfDataset {
	std::vector<tcnn::GPUMemory<Ray>> raymemory_normal;
	std::vector<tcnn::GPUMemory<uint16_t>> pixelmemory_normal;
	std::vector<tcnn::GPUMemory<float>> depthmemory_normal;

	std::vector<tcnn::GPUMemory<Ray>> raymemory_albedo;
	std::vector<tcnn::GPUMemory<uint16_t>> pixelmemory_albedo;
	std::vector<tcnn::GPUMemory<float>> depthmemory_albedo;

	std::vector<TrainingImageMetadata> metadata_normal;
	std::vector<TrainingImageMetadata> metadata_albedo;

	std::vector<TrainingXForm> xforms;
	tcnn::GPUMemory<float> sharpness_data;
	Eigen::Vector2i sharpness_resolution = {0, 0};
	tcnn::GPUMemory<float> envmap_data;

	BoundingBox render_aabb = {};
	Eigen::Vector3f up = {0.0f, 1.0f, 0.0f};
	Eigen::Vector3f offset = {0.0f, 0.0f, 0.0f};

	float n2w_s = 1;
	Eigen::Vector3f n2w_t = {0.0f, 0.0f, 0.0f};

	size_t n_images = 0;
	size_t n_lights = 0;
	size_t n_views = 0;
	size_t height = 0;
	size_t width = 0;
	Eigen::Vector2i envmap_resolution = {0, 0};
	float scale = 1.0f;
	int aabb_scale = 1;
	bool from_mitsuba = false;
	bool from_na = false;
	bool is_hdr = false;
	bool wants_importance_sampling = true;
	bool has_rays = false;

	uint32_t n_extra_learnable_dims = 0;
	bool has_light_dirs = false;


	uint32_t n_extra_dims() const {
		return (has_light_dirs ? 3u : 0u) + n_extra_learnable_dims;
	}

	void set_training_image_normal(int frame_idx, const Eigen::Vector2i& image_resolution, const void* pixels, const void* depth_pixels, float depth_scale, bool image_data_on_gpu, EImageDataType image_type, EDepthDataType depth_type, float sharpen_amount = 0.f, bool white_transparent = false, bool black_transparent = false, uint32_t mask_color = 0, const Ray *rays = nullptr);

	void set_training_image_albedo(int frame_idx, const Eigen::Vector2i& image_resolution, const void* pixels, const void* depth_pixels, float depth_scale, bool image_data_on_gpu, EImageDataType image_type, EDepthDataType depth_type, float sharpen_amount = 0.f, bool white_transparent = false, bool black_transparent = false, uint32_t mask_color = 0, const Ray *rays = nullptr);

	void free_memory(){
		// Free GPU memory for normal dataset
		for (auto& ray_memory : raymemory_normal) {
			cudaFree(ray_memory.data());
		}
		for (auto& pixel_memory : pixelmemory_normal) {
			cudaFree(pixel_memory.data());
		}
		for (auto& depth_memory : depthmemory_normal) {
			cudaFree(depth_memory.data());
		}
	
		// Free GPU memory for albedo dataset
		for (auto& ray_memory : raymemory_albedo) {
			cudaFree(ray_memory.data());
		}
		for (auto& pixel_memory : pixelmemory_albedo) {
			cudaFree(pixel_memory.data());
		}
		for (auto& depth_memory : depthmemory_albedo) {
			cudaFree(depth_memory.data());
		}
	
		// Free GPU memory for other resources
		cudaFree(sharpness_data.data());
		cudaFree(envmap_data.data());
	
		// Optionally, you can clear vectors
		raymemory_normal.clear();
		pixelmemory_normal.clear();
		depthmemory_normal.clear();
	
		raymemory_albedo.clear();
		pixelmemory_albedo.clear();
		depthmemory_albedo.clear();
	
		metadata_normal.clear();
		metadata_albedo.clear();
		xforms.clear();
	}

	Eigen::Vector3f nerf_direction_to_ngp(const Eigen::Vector3f& nerf_dir) {
		Eigen::Vector3f result = nerf_dir;
		if (from_mitsuba) {
			result *= -1;
		} else {
			result=Eigen::Vector3f(result.y(), result.z(), result.x());
		}
		return result;
	}

	Eigen::Matrix<float, 3, 4> nerf_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix) {
		Eigen::Matrix<float, 3, 4> result = nerf_matrix;
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = result.col(3) * scale + offset;
		if (from_na){
			result.col(1) *= -1;
			result.col(2) *= -1;
		}
		else if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz<-yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(1);
			result.row(1) = (Eigen::Vector4f)result.row(2);
			result.row(2) = tmp;
		}

		return result;
	}

	Eigen::Matrix<float, 3, 4> ngp_matrix_to_nerf(const Eigen::Matrix<float, 3, 4>& ngp_matrix) {
		Eigen::Matrix<float, 3, 4> result = ngp_matrix;
		if (from_na){
			result.col(1) *= -1;
			result.col(2) *= -1;			
		}
		else if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz->yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(2);
			result.row(2) = (Eigen::Vector4f)result.row(1);
			result.row(1) = tmp;
		}
		result.col(1) *= -1;
		result.col(2) *= -1;
		result.col(3) = (result.col(3) - offset) / scale;
		return result;
	}

	void nerf_ray_to_ngp(Ray& ray, bool scale_direction = false) {
		ray.o = ray.o * scale + offset;
		if (scale_direction)
			ray.d *= scale;

		float tmp = ray.o[0];
		ray.o[0] = ray.o[1];
		ray.o[1] = ray.o[2];
		ray.o[2] = tmp;

		tmp = ray.d[0];
		ray.d[0] = ray.d[1];
		ray.d[1] = ray.d[2];
		ray.d[2] = tmp;
	}
};

NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount = 0.f);
NerfDataset create_empty_nerf_dataset(size_t n_images, int aabb_scale = 1, bool is_hdr = false);

NGP_NAMESPACE_END
