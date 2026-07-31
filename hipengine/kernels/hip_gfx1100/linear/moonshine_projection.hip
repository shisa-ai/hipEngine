// Moonshine source-F16 dense projection baselines with FP32 accumulation.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

using half_t = _Float16;

__device__ inline float moonshine_block_sum(float value) {
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

__global__ void moonshine_f16_projection_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator);
  if (threadIdx.x == 0) {
    output[row * out_features + column] = static_cast<half_t>(total);
  }
}

__global__ void moonshine_f16_lm_head_projection_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator);
  if (threadIdx.x == 0) {
    output[row * out_features + column] = static_cast<half_t>(total);
  }
}

__global__ void moonshine_f16_lm_head_projection_wave8_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
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
        static_cast<float>(weight[column * in_features + feature]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    accumulator += __shfl_down(accumulator, offset);
  }
  if (lane == 0) {
    output[row * out_features + column] = static_cast<half_t>(accumulator);
  }
}

__global__ void moonshine_f16_projection_bias_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_features) return;
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator) +
      static_cast<float>(bias[column]);
  if (threadIdx.x == 0) {
    output[row * out_features + column] = static_cast<half_t>(total);
  }
}

__global__ void moonshine_f16_projection_pair_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    half_t* __restrict__ output_a,
    half_t* __restrict__ output_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_a_features + out_b_features) return;
  const bool first = column < out_a_features;
  const int64_t local_column = first ? column : column - out_a_features;
  const int64_t width = first ? out_a_features : out_b_features;
  const half_t* weight = first ? weight_a : weight_b;
  half_t* output = first ? output_a : output_b;
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[local_column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator);
  if (threadIdx.x == 0) {
    output[row * width + local_column] = static_cast<half_t>(total);
  }
}

__global__ void moonshine_f16_projection_pair_head_major_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    half_t* __restrict__ output_a,
    half_t* __restrict__ output_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t head_dim) {
  const int64_t column = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || column >= out_a_features + out_b_features) return;
  const bool first = column < out_a_features;
  const int64_t local_column = first ? column : column - out_a_features;
  const half_t* weight = first ? weight_a : weight_b;
  half_t* output = first ? output_a : output_b;
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[local_column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator);
  if (threadIdx.x == 0) {
    const int64_t head = local_column / head_dim;
    const int64_t dimension = local_column - head * head_dim;
    output[(head * rows + row) * head_dim + dimension] =
        static_cast<half_t>(total);
  }
}

__global__ void moonshine_f16_projection_triple_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    const half_t* __restrict__ weight_c,
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
  if (row >= rows ||
      column >= out_a_features + out_b_features + out_c_features) return;
  const half_t* weight;
  half_t* output;
  int64_t local_column;
  int64_t width;
  if (column < out_a_features) {
    weight = weight_a;
    output = output_a;
    local_column = column;
    width = out_a_features;
  } else if (column < out_a_features + out_b_features) {
    weight = weight_b;
    output = output_b;
    local_column = column - out_a_features;
    width = out_b_features;
  } else {
    weight = weight_c;
    output = output_c;
    local_column = column - out_a_features - out_b_features;
    width = out_c_features;
  }
  float accumulator = 0.0f;
  for (int64_t feature = threadIdx.x; feature < in_features;
       feature += blockDim.x) {
    accumulator += static_cast<float>(input[row * in_features + feature]) *
        static_cast<float>(weight[local_column * in_features + feature]);
  }
  const float total = moonshine_block_sum(accumulator);
  if (threadIdx.x == 0) {
    output[row * width + local_column] = static_cast<half_t>(total);
  }
}

bool valid_shape(int64_t rows, int64_t in_features, int64_t threads) {
  return rows > 0 && in_features > 0 && threads > 0 && threads <= 256 &&
      (threads % 32) == 0;
}

}  // namespace

extern "C" int hipengine_moonshine_f16_projection(
    const half_t* input,
    const half_t* weight,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_f16_projection_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      input, weight, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_lm_head_projection(
    const half_t* input,
    const half_t* weight,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_f16_lm_head_projection_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      input, weight, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_lm_head_projection_wave8(
    const half_t* input,
    const half_t* weight,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_f16_lm_head_projection_wave8_kernel<<<
      dim3((out_features + 7) / 8, rows), dim3(256), 0, stream>>>(
      input, weight, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_projection_bias(
    const half_t* input,
    const half_t* weight,
    const half_t* bias,
    half_t* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_f16_projection_bias_kernel<<<
      dim3(out_features, rows), dim3(threads), 0, stream>>>(
      input, weight, bias, output, rows, in_features, out_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_projection_pair(
    const half_t* input,
    const half_t* weight_a,
    const half_t* weight_b,
    half_t* output_a,
    half_t* output_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, in_features, threads) || out_a_features <= 0 ||
      out_b_features <= 0) {
    return hipErrorInvalidValue;
  }
  moonshine_f16_projection_pair_kernel<<<
      dim3(out_a_features + out_b_features, rows), dim3(threads), 0, stream>>>(
      input, weight_a, weight_b, output_a, output_b, rows, in_features,
      out_a_features, out_b_features);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_projection_pair_head_major(
    const half_t* input,
    const half_t* weight_a,
    const half_t* weight_b,
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
  moonshine_f16_projection_pair_head_major_kernel<<<
      dim3(out_a_features + out_b_features, rows), dim3(threads), 0, stream>>>(
      input, weight_a, weight_b, output_a, output_b, rows, in_features,
      out_a_features, out_b_features, head_dim);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_f16_projection_triple(
    const half_t* input,
    const half_t* weight_a,
    const half_t* weight_b,
    const half_t* weight_c,
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
  moonshine_f16_projection_triple_kernel<<<
      dim3(out_a_features + out_b_features + out_c_features, rows),
      dim3(threads), 0, stream>>>(
      input, weight_a, weight_b, weight_c, output_a, output_b, output_c,
      rows, in_features, out_a_features, out_b_features, out_c_features);
  return hipGetLastError();
}
