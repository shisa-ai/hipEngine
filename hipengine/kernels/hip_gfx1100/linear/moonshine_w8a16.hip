// Moonshine row-major per-output-channel symmetric W8A16 kernels.
// FP16 activations/boundaries, INT8 weights, FP32 accumulation and scaling.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <math.h>

namespace {

using half_t = _Float16;

__device__ inline float moonshine_w8a16_block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  if (blockDim.x == 32) return value;
  __shared__ float partial[8];
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int index = 0; index < waves; ++index) total += partial[index];
    return total;
  }
  return 0.0f;
}

__device__ inline float moonshine_w8a16_dot(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    int64_t input_offset,
    int64_t weight_offset,
    int64_t in_features) {
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[input_offset + feature]) *
        static_cast<float>(static_cast<int>(weight[weight_offset + feature]));
  }
  return moonshine_w8a16_block_sum(accumulator);
}

__global__ void moonshine_w8a16_lm_head_wave8_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 8 + wave;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  float accumulator = 0.0f;
  for (int64_t feature = lane; feature < in_features; feature += 32) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(static_cast<int>(weight[column * in_features + feature]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    accumulator += __shfl_down(accumulator, offset);
  }
  if (lane == 0) {
    output[row * out_features + column] =
        static_cast<half_t>(accumulator * scales[column]);
  }
}

__global__ void moonshine_w8a16_projection_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  const float total = moonshine_w8a16_dot(
      input, weight, row * in_features, column * in_features, in_features);
  if (threadIdx.x == 0) {
    output[row * out_features + column] =
        static_cast<half_t>(total * scales[column]);
  }
}

__global__ __launch_bounds__(32)
void moonshine_w8a16_mlp_fc1_gated_silu_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t intermediate_size) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= intermediate_size) return;
  const int64_t gate_column = column + intermediate_size;
  float value_accumulator = 0.0f;
  float gate_accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features; feature += 32) {
    const float input_value = static_cast<float>(input[row * in_features + feature]);
    value_accumulator += input_value * static_cast<float>(static_cast<int>(
        weight[column * in_features + feature]));
    gate_accumulator += input_value * static_cast<float>(static_cast<int>(
        weight[gate_column * in_features + feature]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    value_accumulator += __shfl_down(value_accumulator, offset);
    gate_accumulator += __shfl_down(gate_accumulator, offset);
  }
  if (threadIdx.x == 0) {
    const half_t value_boundary = static_cast<half_t>(
        value_accumulator * scales[column] + static_cast<float>(bias[column]));
    const half_t gate_boundary = static_cast<half_t>(
        gate_accumulator * scales[gate_column] +
        static_cast<float>(bias[gate_column]));
    const float value = static_cast<float>(value_boundary);
    const float gate = static_cast<float>(gate_boundary);
    output[row * intermediate_size + column] = static_cast<half_t>(
        value * (gate / (1.0f + expf(-gate))));
  }
}

__global__ void moonshine_w8a16_mlp_fc2_residual_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const half_t* __restrict__ bias,
    const half_t* __restrict__ residual,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  const float total = moonshine_w8a16_dot(
      input, weight, row * in_features, column * in_features, in_features) *
      scales[column] + static_cast<float>(bias[column]);
  if (threadIdx.x == 0) {
    const half_t projection_boundary = static_cast<half_t>(total);
    output[row * out_features + column] = static_cast<half_t>(
        static_cast<float>(residual[row * out_features + column]) +
        static_cast<float>(projection_boundary));
  }
}

__global__ void moonshine_w8a16_qkv_triple_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight_a,
    const float* __restrict__ scale_a,
    const int8_t* __restrict__ weight_b,
    const float* __restrict__ scale_b,
    const int8_t* __restrict__ weight_c,
    const float* __restrict__ scale_c,
    half_t* __restrict__ output_a,
    half_t* __restrict__ output_b,
    half_t* __restrict__ output_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t out_c_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int8_t* weight = weight_a;
  const float* scale = scale_a;
  half_t* output = output_a;
  int64_t local_column = column;
  int64_t width = out_a_features;
  if (column >= out_a_features) {
    if (column < out_a_features + out_b_features) {
      weight = weight_b;
      scale = scale_b;
      output = output_b;
      local_column = column - out_a_features;
      width = out_b_features;
    } else {
      weight = weight_c;
      scale = scale_c;
      output = output_c;
      local_column = column - out_a_features - out_b_features;
      width = out_c_features;
    }
  }
  if (row >= rows || local_column >= width) return;
  const float total = moonshine_w8a16_dot(
      input, weight, row * in_features, local_column * in_features, in_features);
  if (threadIdx.x == 0) {
    output[row * width + local_column] =
        static_cast<half_t>(total * scale[local_column]);
  }
}

__global__ void moonshine_w8a16_cross_kv_pair_head_major_kernel(
    const half_t* __restrict__ input,
    const int8_t* __restrict__ weight_a,
    const float* __restrict__ scale_a,
    const int8_t* __restrict__ weight_b,
    const float* __restrict__ scale_b,
    half_t* __restrict__ output_a,
    half_t* __restrict__ output_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t head_dim) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  const bool second = column >= out_a_features;
  const int64_t local_column = second ? column - out_a_features : column;
  const int64_t width = second ? out_b_features : out_a_features;
  if (row >= rows || local_column >= width) return;
  const int8_t* weight = second ? weight_b : weight_a;
  const float* scale = second ? scale_b : scale_a;
  half_t* output = second ? output_b : output_a;
  const float total = moonshine_w8a16_dot(
      input, weight, row * in_features, local_column * in_features, in_features);
  if (threadIdx.x == 0) {
    const int64_t head = local_column / head_dim;
    const int64_t dimension = local_column - head * head_dim;
    output[(head * rows + row) * head_dim + dimension] =
        static_cast<half_t>(total * scale[local_column]);
  }
}

bool valid_shape(int64_t rows, int64_t in_features, int64_t threads) {
  return rows > 0 && in_features > 0 && threads > 0 && threads <= 256 &&
      (threads % 32) == 0;
}

}  // namespace

extern "C" int hipengine_moonshine_w8a16_lm_head_wave8(
    const half_t* input,
    const int8_t* weight,
    const float* scales,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_lm_head_wave8_kernel<<<
      dim3((out_features + 7) / 8, rows), dim3(256), 0, stream>>>(
      input, weight, scales, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_w8a16_projection(
    const half_t* input,
    const int8_t* weight,
    const float* scales,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_projection_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      input, weight, scales, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_w8a16_mlp_fc1_gated_silu(
    const half_t* input,
    const int8_t* weight,
    const float* scales,
    const half_t* bias,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t intermediate_size,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || intermediate_size <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_mlp_fc1_gated_silu_kernel<<<
      dim3(intermediate_size, rows), dim3(32), 0, stream>>>(
      input, weight, scales, bias, output, rows, in_features, intermediate_size);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_w8a16_mlp_fc2_residual(
    const half_t* input,
    const int8_t* weight,
    const float* scales,
    const half_t* bias,
    const half_t* residual,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_mlp_fc2_residual_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      input, weight, scales, bias, residual, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_w8a16_qkv_triple(
    const half_t* input,
    const int8_t* weight_a,
    const float* scale_a,
    const int8_t* weight_b,
    const float* scale_b,
    const int8_t* weight_c,
    const float* scale_c,
    half_t* output_a,
    half_t* output_b,
    half_t* output_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t out_c_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_a_features <= 0 ||
      out_b_features <= 0 || out_c_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_qkv_triple_kernel<<<
      dim3(out_a_features + out_b_features + out_c_features, rows),
      dim3(threads), 0, stream>>>(
      input, weight_a, scale_a, weight_b, scale_b, weight_c, scale_c,
      output_a, output_b, output_c, rows, in_features, out_a_features,
      out_b_features, out_c_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_w8a16_cross_kv_pair_head_major(
    const half_t* input,
    const int8_t* weight_a,
    const float* scale_a,
    const int8_t* weight_b,
    const float* scale_b,
    half_t* output_a,
    half_t* output_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_a_features <= 0 ||
      out_b_features <= 0 || head_dim <= 0 ||
      (out_a_features % head_dim) != 0 || (out_b_features % head_dim) != 0) {
    return hipErrorInvalidValue;
  }
  moonshine_w8a16_cross_kv_pair_head_major_kernel<<<
      dim3(out_a_features + out_b_features, rows), dim3(threads), 0, stream>>>(
      input, weight_a, scale_a, weight_b, scale_b, output_a, output_b,
      rows, in_features, out_a_features, out_b_features, head_dim);
  return hipGetLastError();
}
