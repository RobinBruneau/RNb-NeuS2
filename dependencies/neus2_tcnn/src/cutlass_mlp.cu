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

/** @file   cutlass_mlp.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void set_constant_value_view_vector(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const T value,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	output(dim_idx, elem_idx) = value;
}

template <typename T>
__global__ void matrix_multiple(
	const uint32_t n_elements,
	const uint32_t row,
	const uint32_t col,
	const uint32_t batch,
	tcnn::MatrixView<T> back,
	tcnn::MatrixView<T> front,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t row_id = i / col;
	const uint32_t col_id = i - row_id * col;
	for (uint32_t j = 0; j < batch; j++){
		output(row_id, col_id) += back(row_id,j) * front(col_id, j);
		printf("row:%d,col:%d,output,back,front:%f,%f,%f\n",row_id,col_id,(float)output(row_id, col_id),(float)back(row_id,j),(float)front(col_id, j));
	}

}

template <typename T>
__global__ void debug_log(
	const uint32_t  n_elements,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	printf("%d: %.10f\n",i, (float)output(i,0));
}

template <typename T>
CutlassMLP<T>::CutlassMLP(
	uint32_t input_width,
	uint32_t network_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
m_input_width{input_width},
m_network_width{network_width},
m_output_width{output_width},
m_n_hidden_layers{n_hidden_layers},
m_activation{activation},
m_output_activation{output_activation},
m_can_fuse_activation{activation != Activation::Sine}
{
	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	if (n_hidden_layers > 0) {
		m_n_hidden_matmuls = n_hidden_layers-1;
	} else {
		m_n_hidden_matmuls = 0;
	}

	// Create matrices related to weights
	if (n_hidden_layers == 0) {
		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);

		m_weight_matrices_stored.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_inference_stored.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_gradient_matrices_stored.emplace_back(nullptr, m_padded_output_width, m_input_width);
	} else {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

		m_weight_matrices_stored.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_inference_stored.emplace_back(nullptr, m_network_width, m_input_width);
		m_gradient_matrices_stored.emplace_back(nullptr, m_network_width, m_input_width);

		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_network_width);
			m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);

			m_weight_matrices_stored.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_inference_stored.emplace_back(nullptr, m_network_width, m_network_width);
			m_gradient_matrices_stored.emplace_back(nullptr, m_network_width, m_network_width);
		}

		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

		m_weight_matrices_stored.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_inference_stored.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_gradient_matrices_stored.emplace_back(nullptr, m_padded_output_width, m_network_width);
	}

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}

	// 1 stream per matrix.
	m_training_splitk_streams.resize(m_n_hidden_layers + 1);
	m_training_splitk_events.resize(m_n_hidden_layers + 1);

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T>
CutlassMLP<T>::~CutlassMLP() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		free_gpu_memory_arena(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename CutlassLayer, typename T>
bool compute_layer(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output,
	GPUMatrixDynamic<T>& activation_output
) {
	bool can_fuse_activation = true;
	if (!is_inference) {
		// Never disallow fusing if the caller passes the same output and activation_output buffers... in that case,
		// invertibility of the activation function may be ignored.
		can_fuse_activation &= activation != Activation::Sine || &output == &activation_output;
		// can_fuse_activation = false;
	}

	if (can_fuse_activation) {
		fc_multiply<CutlassLayer>(stream, weights, input, output, activation);
	} else {
		fc_multiply<CutlassLayer>(stream, weights, input, output);
		activation_gpu(stream, activation, output, activation_output);
	}

	return can_fuse_activation;
	// return true;
}

template <typename CutlassLayer, typename T>
bool compute_inference_layer(
	cudaStream_t stream,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output
) {
	return compute_layer<CutlassLayer>(stream, true, activation, weights, input, output, output);
}

template <typename T>
void CutlassMLP<T>::inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// If there are no hidden layers, the network is just a simple matmul.
	if (m_n_hidden_layers == 0) {
		compute_inference_layer<LastLayer>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, output);
		return;
	}

	uint32_t batch_size = input.n();
	GPUMatrix<T> inference_tmp[2] = {
		GPUMatrix<T>{m_network_width, batch_size, stream},
		GPUMatrix<T>{m_network_width, batch_size, stream},
	};

	m_inference_graph.capture_and_execute(stream, false, [&]() {
		// Run the actual network
		{
			uint32_t tmp_idx = 0;

			// Input layer
			compute_inference_layer<FullLayer>(stream, m_activation, input_weight_matrix(use_inference_params), input, inference_tmp[tmp_idx++ % 2]);

			// Hidden layers
			for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
				compute_inference_layer<FullLayer>(stream, m_activation, weight_matrix_at(use_inference_params, i), inference_tmp[(tmp_idx + 1) % 2], inference_tmp[tmp_idx % 2]);
				++tmp_idx;
			}

			// Output
			compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(use_inference_params), inference_tmp[(tmp_idx + 1) % 2], output);
		}
	});
}

template <typename T>
std::unique_ptr<Context> CutlassMLP<T>::forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
	// If there are no hidden layers, the network is just a simple matmul. No tmp buffers required
	if (m_n_hidden_layers == 0) {
		if (output) {
			compute_layer<LastLayer>(stream, false, m_output_activation, input_weight_matrix(use_inference_params), input, *output, *output);
			// compute_layer<LastLayer>(stream, true, m_output_activation, input_weight_matrix(use_inference_params), input, *output, *output);
		}
		return std::make_unique<ForwardContext>(); // Nothing to save -- empty context
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

	// Run the actual network
	uint32_t tmp_idx = 0;

	bool fused = compute_layer<FullLayer>(
		stream,
		false,
		m_activation,
		input_weight_matrix(use_inference_params),
		input,
		// forward->hidden_input.at(tmp_idx),
		forward->hidden.at(tmp_idx),
		m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
	);
	tmp_idx += fused ? 1 : 2;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		fused = compute_layer<FullLayer>(
			stream,
			false,
			m_activation,
			weight_matrix_at(use_inference_params, i),
			forward->hidden.at(tmp_idx-1),
			// forward->hidden_input.at(tmp_idx),
			// forward->hidden.at(tmp_idx)
			forward->hidden.at(tmp_idx),
			m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
		);
		tmp_idx += fused ? 1 : 2;
	}

	if (output) {
		compute_layer<LastLayer>(stream, false, m_output_activation, output_weight_matrix(use_inference_params), forward->hidden.at(tmp_idx-1), *output, *output);
		// compute_layer<LastLayer>(stream, true, m_output_activation, output_weight_matrix(use_inference_params), forward->hidden.at(tmp_idx-1), *output, *output);
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::backward_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	EGradientMode param_gradients_mode
) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		backward_tmp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	GPUMatrixDynamic<T> backward_output_tmp;
	if (m_output_activation != Activation::None) {
		backward_output_tmp = {m_padded_output_width, batch_size, stream, dL_doutput.layout()};
		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
		// activation_backward_output_gpu: stream, input_elements, activation, activation_value, input_value, output_value)
	}

	// Backprop
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;

	{
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

		// If there are no hidden layers, the network is just a simple matmul
		if (m_n_hidden_layers == 0) {
			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaEventRecord(m_training_splitk_events.at(0), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(0), m_training_splitk_events.at(0), 0);

				// Compute weight gradients
				fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(0), tmp_dL_doutput, input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);

				cudaEventRecord(m_training_splitk_events.at(0), m_training_splitk_streams.at(0));
			}

			if (dL_dinput) {
				fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, *dL_dinput);
			}

			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaStreamWaitEvent(stream, m_training_splitk_events.at(0), 0);
			}
			return;
		}

		uint32_t tmp_idx = (m_can_fuse_activation ? (m_n_hidden_matmuls+1) : ((m_n_hidden_matmuls+1) * 2)) - 1;
		uint32_t backward_tmp_idx = 0;

		if (param_gradients_mode != EGradientMode::Ignore) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);

			// Compute weight gradients
			fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);

			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		if (!m_can_fuse_activation) {
			fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx));
			activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
		} else {
			fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
		}

		tmp_idx -= m_can_fuse_activation ? 1 : 2;
		++backward_tmp_idx;

		// layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
			}

			if (!m_can_fuse_activation) {
				fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), backward_tmp.at(backward_tmp_idx));
				activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
			} else {
				fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
			}

			tmp_idx -= m_can_fuse_activation ? 1 : 2;
			++backward_tmp_idx;
		}

		if (param_gradients_mode != EGradientMode::Ignore) {
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
			fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If requested, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput) {
			// optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
		}
	}

	if (param_gradients_mode != EGradientMode::Ignore) {
		// All the per-layer split-k matrix multiplications summing over
		// the batch are computed in parallel streams to the actual
		// backpropagation. Here, we need to wait for all of these to complete.
		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
}

// Assume that derivative is a square matrix. n_row == n_col.
template <typename T, bool SET_ROW = true>
__global__ void apply_relu_to_derivative_by_forward(
	const uint32_t n_elements,
	const uint32_t n_row,
	const uint32_t n_col,
	const tcnn::MatrixView<T> forward,
	tcnn::MatrixView<T> derivative
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t i_row = i / n_row;
	if (forward(i_row, 0) < T(0)) {
		const uint32_t i_col = i - n_row * i_row;
		if (SET_ROW) {
			derivative(i_row, i_col) = T(0);
		}
		else {
			derivative(i_col, i_row) = T(0);
		}
	}
}

template <typename T, bool SET_ROW = true>
__global__ void apply_relu_to_derivative_by_forward_batch(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<T> forward,
	tcnn::MatrixView<T> derivative
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	// printf("apply relu:%f!\n",(float)forward(dim_idx, elem_idx));
	// printf("elem_idx:%d,dim_idx:%d,value: %f!\n",elem_idx,dim_idx,(float)forward(dim_idx, elem_idx));
	if (forward(dim_idx, elem_idx) < T(0)) {
	// if (forward(dim_idx, elem_idx) <= T(0)) {
		// printf("elem_idx:%d,dim_idx:%d\n",elem_idx,dim_idx);
		derivative(dim_idx, elem_idx) = T(0);
	}
}

template <typename T>
__global__ void kernel_compute_update_weight(
	const uint32_t n_elements,
	const uint32_t n_row,
	const uint32_t n_col,
	const tcnn::MatrixView<T> tmp_matrix_front,
	const tcnn::MatrixView<T> tmp_matrix_back,
	tcnn::MatrixView<T> update_weight,
	EGradientMode param_gradients_mode
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t i_row = i / n_row;
	const uint32_t i_col = i - n_row * i_row;

	T tmp_matrix_front_sum = 0.0f;
	T tmp_matrix_back_sum = 0.0f;
	for (uint32_t k = 0; k < n_row; ++k) {
		tmp_matrix_front_sum += tmp_matrix_front(k, i_col);
	}
	for (uint32_t l = 0; l < n_col; ++l) {
		tmp_matrix_back_sum += tmp_matrix_back(i_row, l);
	}
	// printf("tmp_matrix_back_sum %f\n",tmp_matrix_back_sum);
	// printf("tmp_matrix_front_sum %f\n",tmp_matrix_front_sum);
	if (param_gradients_mode == EGradientMode::Accumulate){
		update_weight(i_row, i_col) += tmp_matrix_front_sum * tmp_matrix_back_sum;
	}
	else if (param_gradients_mode == EGradientMode::Overwrite){
		update_weight(i_row, i_col) = tmp_matrix_front_sum * tmp_matrix_back_sum;
	}
	// printf("update_weight: %f", update_weight(i_row, i_col));
}

// Implemented by Yiming Wang <w752531540@gmail.com>
// According to Equation 8,9 in NeuS2 <https://arxiv.org/abs/2212.05231>
template <typename T>
void CutlassMLP<T>::backward_backward_input_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& dL_ddLdinput,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_ddLdoutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	EGradientMode param_gradients_mode
) {

	// there exists m_hidden_layers + 1 layers in total.
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	uint32_t num_tmp_matrix = m_n_hidden_layers + 3; // since that there exist one more default output layer
	std::vector<GPUMatrix<T>> tmp_front_multiply(num_tmp_matrix);

	for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i - 1).m();
		tmp_front_multiply[i].set_size_unsafe(matrix_size, batch_size);
	}
	auto tmp_front_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_front_multiply);
	tmp_front_multiply[0] = GPUMatrix<T>{dL_ddLdinput.data(),input_weight_matrix(use_inference_params).n(),batch_size};
	
	std::vector<GPUMatrix<T>> tmp_back_multiply(num_tmp_matrix);
	
	for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i-1).n();
		tmp_back_multiply[i].set_size_unsafe(matrix_size, batch_size);
	}
	auto tmp_back_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_back_multiply);
	tmp_back_multiply[num_tmp_matrix-1] = GPUMatrix<T>{dL_doutput.data(),output_weight_matrix(use_inference_params).m(),batch_size};
	
	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;

	{
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);
		
		for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
			auto& cur_tmp_front_multiply = tmp_front_multiply.at(i);
			fc_multiply<FullLayer>(stream, 
				full_weight_matrix_at(use_inference_params, i-1),
				tmp_front_multiply.at(i-1), 
				forward.hidden.at(i-1), 
				cur_tmp_front_multiply, 
				m_activation, true);

		}

		for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
			auto& cur_tmp_back_multiply = tmp_back_multiply.at(i);
			fc_multiply<FullLayer>(stream, 
				full_weight_matrix_at(use_inference_params, i-1).transposed(),
				tmp_back_multiply.at(i+1),
				forward.hidden.at(i-2),  
				cur_tmp_back_multiply, 
				m_activation, true);

		}
	
		for (uint32_t i = 1; i < num_tmp_matrix - 1; ++i) {
			auto& gradient_matrix = m_gradient_matrices.at(i-1);
			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaEventRecord(m_training_splitk_events.at(i-1), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(i-1), m_training_splitk_events.at(i-1), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(i-1), tmp_back_multiply.at(i+1), tmp_front_multiply.at(i-1).transposed(), gradient_matrix, split_k_factor, param_gradient_beta);
				cudaEventRecord(m_training_splitk_events.at(i-1), m_training_splitk_streams.at(i-1));
			}
		}

	}

	if (param_gradients_mode != EGradientMode::Ignore) {

		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}

}

template <typename T>
std::unique_ptr<typename CutlassMLP<T>::ForwardContext> CutlassMLP<T>::allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::set_params(T* params, T* inference_params, T* backward_params, T* gradients) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data_unsafe(params + current_pos);
		m_weight_matrices_inference[i].set_data_unsafe(inference_params + current_pos);
		m_gradient_matrices[i].set_data_unsafe(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}
}

template <typename T>
void CutlassMLP<T>::initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	set_params(params, inference_params, backward_params, gradients);

	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		m_weight_matrices_full_precision[i].set_data_unsafe(params_full_precision + current_pos);
		current_pos += m_weight_matrices_full_precision[i].n_elements();

		if (m_activation == Activation::Sine) {
			if (i == 0) {
				m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				m_weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

template <typename T>
void CutlassMLP<T>::read_params(std::vector<T*> params, std::vector<T*> inference_params, std::vector<T*> backward_params, std::vector<T*> gradients){

	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		
		m_weight_matrices_stored[i].set_data_unsafe(m_weight_matrices[i].data());
		m_weight_matrices_inference_stored[i].set_data_unsafe(m_weight_matrices_inference[i].data());
		m_gradient_matrices_stored[i].set_data_unsafe(m_gradient_matrices[i].data());
	}

	return;
}


template <typename T>
void CutlassMLP<T>::set_params_from_matrix(std::vector<T*> params, std::vector<T*> inference_params, std::vector<T*> backward_params, std::vector<T*> gradients){

	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {

		m_weight_matrices[i].set_data_unsafe(m_weight_matrices_stored[i].data());
		m_weight_matrices_inference[i].set_data_unsafe(m_weight_matrices_inference_stored[i].data());
		m_gradient_matrices[i].set_data_unsafe(m_gradient_matrices_stored[i].data());
	}

	return;
}

// Explicitly instantiate CutlassMLP classes.
template class CutlassMLP<network_precision_t>;

TCNN_NAMESPACE_END
