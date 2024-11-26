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

/** @file   adam.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the adam optimizer with support for
 *          the AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <json/json.hpp>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

#define COMPONENT_OPTIMIZE_STEP_VERBOSE 0
#define LEARNING_RATE_VERBOSE 0

template <typename T>
__global__ void adam_step(
	const uint32_t n_elements,
	// const uint32_t n_matrix_weights,
	const uint32_t n_components,
	const uint32_t n_weights_canonical,
	const uint32_t n_weights_canonical_covered_by_matrices,
	const uint32_t n_weights_delta,
	const uint32_t n_weights_delta_covered_by_matrices,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float loss_scale,
	float learning_rate,
	const float non_matrix_learning_rate_factor,
	const bool optimize_matrix_params,
	const bool optimize_non_matrix_params,
	const bool optimize_canonical_params,
	const bool optimize_delta_params,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	float* __restrict__ second_moments,
	uint32_t* __restrict__ param_steps,
	std::pair<uint32_t, bool>* __restrict__ n_weights_optimize,
	const bool only_sdf_training,
	const bool only_reflectance_training) 
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float gradient = (float)gradients[i] / loss_scale;
	
	bool is_canonical = (i >= n_weights_delta);
	// bool is_canonical = (i < n_weights_canonical);
	if (!is_canonical && !optimize_delta_params) {
		return;
	}
	if (is_canonical && !optimize_canonical_params) {
		return;
	}

	bool is_matrices_param = true;
	// if (is_canonical) {
	// 	if (i >= n_weights_canonical_covered_by_matrices) is_matrices_param = false;
	// } else {
	// 	if (i >= n_weights_canonical + n_weights_delta_covered_by_matrices) is_matrices_param = false;
	// }
	if (is_canonical) {
		if (i >= n_weights_delta + n_weights_canonical_covered_by_matrices) is_matrices_param = false;
	} else {
		if (i >= n_weights_delta_covered_by_matrices) is_matrices_param = false;
	}

	if (!is_matrices_param) {
		if (!optimize_non_matrix_params || gradient == 0) {
			return;
		}
	} else {
		if (!optimize_matrix_params) {
			return;
		}
	}

	uint32_t final_number;
	bool found_sdf = false;
	bool found_reflectance = false;
	for (int32_t  j=n_components-1; j>=0; j--) {
		//printf("%d\n",j);
		if (i < n_weights_optimize[j].first) {
			if (j==0){
				//printf("%d\n",i);
				found_sdf = true;
				found_reflectance = false;
			}
			if (j==2){
				//printf("%d\n",i);
				found_sdf = true;
				found_reflectance = false;
			}
			if (j==3){
				//printf("%d\n",i);
				found_sdf = true;
				found_reflectance = false;
			}
			if (j==1){
				found_sdf = false;
				found_reflectance = true;
			}
			else{
				found_sdf = false;
			}
		}
	}

	//printf("only_sdf_training: %s AND only_reflectance_training: %s\n", only_sdf_training ? "true" : "false",only_reflectance_training ? "true" : "false");

	bool continuer = true;

	if (found_reflectance){
		if (only_sdf_training){
			continuer = false;
		}
	}
	else{
		if (only_reflectance_training){
			continuer = false;
		}
	}


	if (continuer){
		const float weight_fp = weights_full_precision[i];

		// if (i < n_matrix_weights) {
		if (is_matrices_param) {
			// No L2 reg for non-matrix params
			gradient += l2_reg * weight_fp;
		}

		const float gradient_sq = gradient * gradient;

		float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
		const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;

		// if (i >= n_matrix_weights) {
		if (!is_matrices_param) {
			// Potentially different learning rate for non-matrix params
			learning_rate *= non_matrix_learning_rate_factor;
		}

		// Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
		const uint32_t current_step = ++param_steps[i];
		learning_rate *= sqrtf(1 - powf(beta2, (float)current_step)) / (1 - powf(beta1, (float)current_step));

		// Follow AdaBound paradigm
		const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

		const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
		const float new_weight = decayed_weight - effective_learning_rate * first_moment;

		weights_full_precision[i] = new_weight;
		weights[i] = (T)new_weight;
	}
	
}

template <typename T>
class AdamOptimizer : public Optimizer<T> {
public:
	AdamOptimizer(const json& params) {
		update_hyperparams(params);
	}

	void update_components_optimize() {
		uint32_t n_components = m_n_weights_components.size();
		uint32_t i = 0;
		for (tcnn::json::iterator it = m_n_weights_components.begin(); it != m_n_weights_components.end(); ++it) {
			tcnn::json j = it.value()[1];
			auto component_name = j[0].get<std::string>();
			std::string target_string = "density_network";
			if (m_optimize_params_components.contains(component_name)) {
				m_n_weights_optimize_cpu[i].second = m_optimize_params_components[component_name].get<bool>();
			}
			i++;
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_n_weights_optimize.data(), m_n_weights_optimize_cpu.data(), n_components * sizeof(std::pair<uint32_t, bool>), cudaMemcpyHostToDevice));
	}

	void allocate(std::shared_ptr<ParametricObject<T>> target) override {
		uint32_t size = (uint32_t)target->n_params();

		m_n_weights = size;
		if (m_n_weights <= m_first_moments.size()) {
			return;
		}

		m_first_moments.resize(size);
		m_first_moments.memset(0);

		m_second_moments.resize(size);
		m_second_moments.memset(0);

		m_param_steps.resize(size);
		m_param_steps.memset(0);

		m_n_weights_components = target->n_params_components();
		std::cout << m_n_weights_components << std::endl;
		
		uint32_t n_components = m_n_weights_components.size();
		m_n_weights_optimize.resize(n_components);
		m_n_weights_optimize.memset(0);

		m_n_weights_optimize_cpu.resize(n_components);
		uint32_t offset = 0;
		uint32_t i = 0;
		// for (uint32_t i=0; i<n_components; i++) {
		for (tcnn::json::iterator it = m_n_weights_components.begin(); it != m_n_weights_components.end(); ++it) {
			// tcnn::json j = m_n_weights_components.at(i);
			tcnn::json j = it.value()[1];
			uint32_t n_params = j[1].get<uint32_t>();
			offset += n_params;
			m_n_weights_optimize_cpu[i].first = offset;
			m_n_weights_optimize_cpu[i].second = true;
			auto component_name = j[0].get<std::string>();
			if (m_optimize_params_components.contains(component_name)) {
				if (m_optimize_params_components[component_name].get<bool>() == false) {
					m_n_weights_optimize_cpu[i].second = false;
				}
			}
			i++;
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_n_weights_optimize.data(), m_n_weights_optimize_cpu.data(), n_components * sizeof(std::pair<uint32_t, bool>), cudaMemcpyHostToDevice));

		m_n_weights_canonical = (uint32_t)target->n_params_canonical();
		m_n_weights_delta = (uint32_t)target->n_params_delta();
		printf("m_n_weights: %d\n", m_n_weights);
		printf("m_n_weights_canonical: %d\n", m_n_weights_canonical);
		printf("m_n_weights_delta: %d\n", m_n_weights_delta);

		if (m_n_weights_delta == 0) {
			m_n_weights_canonical_covered_by_matrices = 0;
			m_n_weights_delta_covered_by_matrices = 0;
			auto layer_sizes = target->layer_sizes();

			for (size_t i = 0; i < layer_sizes.size(); ++i) {
				m_n_weights_canonical_covered_by_matrices += layer_sizes[i].first * layer_sizes[i].second;
			}
		}
		else {
			m_n_weights_canonical_covered_by_matrices = 0;
			auto layer_sizes_canonical = target->layer_sizes_canonical();

			for (size_t i = 0; i < layer_sizes_canonical.size(); ++i) {
				m_n_weights_canonical_covered_by_matrices += layer_sizes_canonical[i].first * layer_sizes_canonical[i].second;
			}

			printf("m_n_weights_canonical_covered_by_matrices: %d\n", m_n_weights_canonical_covered_by_matrices);

			m_n_weights_delta_covered_by_matrices = 0;
			auto layer_sizes_delta = target->layer_sizes_delta();

			for (size_t i = 0; i < layer_sizes_delta.size(); ++i) {
				m_n_weights_delta_covered_by_matrices += layer_sizes_delta[i].first * layer_sizes_delta[i].second;
			}

			printf("m_n_weights_delta_covered_by_matrices: %d\n", m_n_weights_delta_covered_by_matrices);

		}
		
	}

	void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;
		update_components_optimize();
#if COMPONENT_OPTIMIZE_STEP_VERBOSE
		if (step() % 20 == 0) {
			printf("m_optimize_canonical_params: %d\n", m_optimize_canonical_params);
			printf("m_optimize_delta_params: %d\n", m_optimize_delta_params);
			std::cout << m_optimize_params_components << std::endl;
		}
#endif

#if LEARNING_RATE_VERBOSE
		printf("step: %d, learning rate: %f\n", m_current_step, m_base_learning_rate);
#endif

		float lower_lr_bound = 0;
		float upper_lr_bound = std::numeric_limits<float>::max();

		// AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
		if (m_adabound) {
			lower_lr_bound = 0.1f - 0.1f / ((1 - m_beta2) * (float)step() + 1);
			upper_lr_bound = 0.1f + 0.1f / ((1 - m_beta2) * (float)step());
		}

		uint32_t n_weights_to_optimize = n_weights();
		uint32_t n_components = m_n_weights_components.size();

		linear_kernel(adam_step<T>, 0, stream,
			n_weights_to_optimize,
			n_components,
			m_n_weights_canonical,
			m_n_weights_canonical_covered_by_matrices,
			m_n_weights_delta,
			m_n_weights_delta_covered_by_matrices,
			m_relative_weight_decay,
			m_absolute_weight_decay,
			loss_scale,
			m_base_learning_rate,
			m_non_matrix_learning_rate_factor,
			m_optimize_matrix_params,
			m_optimize_non_matrix_params,
			m_optimize_canonical_params,
			m_optimize_delta_params,
			m_beta1,
			m_beta2,
			m_epsilon,
			lower_lr_bound,
			upper_lr_bound,
			m_l2_reg,
			weights_full_precision,
			weights,
			gradients,
			m_first_moments.data(),
			m_second_moments.data(),
			m_param_steps.data(),
			m_n_weights_optimize.data(),
			m_only_sdf_training,
			m_only_reflectance_training
		);
	}

	void only_sdf_training(bool activate) override {
		m_only_sdf_training = activate;
	}
	void only_reflectance_training(bool activate) override {
		m_only_reflectance_training = activate;
	}

	float learning_rate() const override {
		return m_base_learning_rate;
	}

	void set_learning_rate(float val) override {
		m_base_learning_rate = val;
	}

	uint32_t step() const override {
		return m_current_step;
	}

	uint32_t n_weights() const override {
		return m_n_weights;
	}

	T* custom_weights() const override {
		return nullptr;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("beta1")) {
			m_beta1 = params["beta1"];
		}

		if (params.contains("beta2")) {
			m_beta2 = params["beta2"];
		}

		if (params.contains("epsilon")) {
			m_epsilon = params["epsilon"];
		}

		if (params.contains("learning_rate")) {
			m_base_learning_rate = params["learning_rate"];
		}

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}

		if (params.contains("adabound")) {
			m_adabound = params["adabound"];
		}

		if (params.contains("relative_decay")) {
			m_relative_weight_decay = params["relative_decay"];
		}

		if (params.contains("absolute_decay")) {
			m_absolute_weight_decay = params["absolute_decay"];
		}

		if (params.contains("non_matrix_learning_rate_factor")) {
			m_non_matrix_learning_rate_factor = params["non_matrix_learning_rate_factor"];
		}

		if (params.contains("optimize_matrix_params")) {
			m_optimize_matrix_params = params["optimize_matrix_params"];
		}

		if (params.contains("optimize_non_matrix_params")) {
			m_optimize_non_matrix_params = params["optimize_non_matrix_params"];
		}

		if (params.contains("optimize_canonical_params")) {
			m_optimize_canonical_params = params["optimize_canonical_params"];
		}
		
		if (params.contains("optimize_delta_params")) {
			m_optimize_delta_params = params["optimize_delta_params"];
		}


		if (params.contains("optimize_params_components")) {
			m_optimize_params_components = params["optimize_params_components"];
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Adam"},
			{"beta1", m_beta1},
			{"beta2", m_beta2},
			{"epsilon", m_epsilon},
			{"learning_rate", m_base_learning_rate},
			{"l2_reg", m_l2_reg},
			{"adabound", m_adabound},
			{"relative_decay", m_relative_weight_decay},
			{"absolute_decay", m_absolute_weight_decay},
			{"non_matrix_learning_rate_factor", m_non_matrix_learning_rate_factor},
			{"optimize_matrix_params", m_optimize_matrix_params},
			{"optimize_non_matrix_params", m_optimize_non_matrix_params},
			{"optimize_canonical_params", m_optimize_canonical_params},
			{"optimize_delta_params", m_optimize_delta_params},
			{"optimize_params_components", m_optimize_params_components},
		};
	}

	json serialize() const override {
		json data;
		data["current_step"] = m_current_step;
		data["base_learning_rate"] = m_base_learning_rate;
		data["first_moments_binary"] = m_first_moments;
		data["second_moments_binary"] = m_second_moments;
		data["param_steps_binary"] = m_param_steps;
		return data;
	}

	void deserialize(const json& data) override {
		m_first_moments = data["first_moments_binary"];
		m_second_moments = data["second_moments_binary"];
		if (data.contains("param_steps_binary")) {
			m_param_steps = data["param_steps_binary"];
		} else {
			m_param_steps.resize(m_second_moments.size());
			m_param_steps.memset(0);
		}
		m_current_step = data["current_step"];
		m_base_learning_rate = data["base_learning_rate"];
	}

private:
	uint32_t m_n_weights;
	tcnn::json m_n_weights_components;
	uint32_t m_n_weights_canonical;
	uint32_t m_n_weights_canonical_covered_by_matrices;
	uint32_t m_n_weights_delta;
	uint32_t m_n_weights_delta_covered_by_matrices;

	GPUMemory<float> m_first_moments;
	GPUMemory<float> m_second_moments;
	GPUMemory<uint32_t> m_param_steps;
	
	GPUMemory<std::pair<uint32_t, bool>> m_n_weights_optimize;
	std::vector<std::pair<uint32_t, bool>> m_n_weights_optimize_cpu;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_non_matrix_learning_rate_factor = 1.0f;
	float m_base_learning_rate = 1e-3f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.999f;
	float m_epsilon = 1e-8f;
	float m_l2_reg = 1e-8f;

	float m_relative_weight_decay = 0.0f;
	float m_absolute_weight_decay = 0.0f;

	bool m_adabound = false;

	bool m_only_sdf_training = false;
	bool m_only_reflectance_training = false;

	bool m_optimize_matrix_params = true;
	bool m_optimize_non_matrix_params = true;
	tcnn::json m_optimize_params_components = tcnn::json({});
	bool m_optimize_canonical_params = true; // canonical network
	bool m_optimize_delta_params = true; // delta network
};

TCNN_NAMESPACE_END
