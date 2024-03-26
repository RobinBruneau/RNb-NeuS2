/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface for input encodings
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/random.h>

#include <stdint.h>

#define RESAMPLE_BY_SCALE 1 // else resample by base_resolution


TCNN_NAMESPACE_BEGIN

enum class InterpolationType {
	Nearest,
	Linear,
	Smoothstep,
};

InterpolationType string_to_interpolation_type(const std::string& interpolation_type);

std::string to_string(InterpolationType interpolation_type);

template <typename T>
class Encoding : public DifferentiableObject<float, T, T> {
public:
	virtual ~Encoding() { }

	void inference_mixed_precision_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>& output,
		bool use_inference_params = true
	) override {
		this->forward(stream, input, &output, use_inference_params, false);
	}

	virtual void set_alignment(uint32_t alignment) = 0;
	virtual uint32_t min_alignment() const = 0;

	virtual MatrixLayout preferred_output_layout() const = 0;

	virtual std::pair<uint32_t, uint32_t> downsample(cudaStream_t stream, const uint8_t* density_grid, const uint32_t max_cascade, const uint32_t nerf_gridsize, 
														uint32_t downsample_scale=2, uint32_t downsample_start_level=8) { return std::pair<uint32_t, uint32_t>(0, 0); }
	virtual std::pair<uint32_t, uint32_t> upsample(cudaStream_t stream, const uint8_t* density_grid, const uint32_t max_cascade, const uint32_t nerf_gridsize, 
														uint32_t upsample_scale=2, uint32_t upsample_start_level=8) { return std::pair<uint32_t, uint32_t>(0, 0); }
	virtual std::pair<uint32_t, uint32_t> reset_gridlevel(cudaStream_t stream, const uint8_t* density_grid, const uint32_t max_cascade, const uint32_t nerf_gridsize) 
														{ return std::pair<uint32_t, uint32_t>(0, 0); }

	// By default, an encoding has no parameters
	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override { }
	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override { }
	size_t n_params() const override { return 0; }

	virtual T* params() const { return nullptr; }

	virtual T* params_inference() const { return nullptr; }

	virtual T* params_gradients() const { return nullptr; }

	virtual T* params_backward() const { return nullptr; }

	virtual void tv_backward(cudaStream_t stream, const float loss_scale, uint8_t * density_grid, uint32_t max_cascade, uint32_t nerf_cascade, default_rng_t& rng, const uint32_t max_num_elements = 1e7, bool use_inference_params = false) { }

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return {}; }

	// uint32_t m_training_step;
	
	virtual void set_training_step(int training_step) { 
		// m_training_step = training_step;
	}
	virtual void set_base_grid(std::shared_ptr<tcnn::Encoding<T>> base_grid) { 
		throw std::runtime_error{"Encoding: set_base_grid not implemented"};
	}
	virtual const uint32_t* hashmap_offsets_table() const {
		throw std::runtime_error{"Encoding: hashmap_offsets_table not implemented"};
	}
	virtual const T* grid(bool use_inference_params=false) const { 
		throw std::runtime_error{"Encoding: grid not implemented"};
	}

};

template <typename T>
Encoding<T>* create_encoding(uint32_t n_dims_to_encode, const json& params, uint32_t alignment = 8);

TCNN_NAMESPACE_END
