// Moonshine batch-one FP16 encoder primitives for CUDA ``sm_120a``.
//
// Covers the pre-convolution front end (conv1+tanh, GroupNorm(1) over the full
// channel/length plane, conv2+gelu, conv3+gelu), the exact-erf GELU used by the
// encoder MLP, full-sequence partial RoPE, and the non-causal full-sequence
// encoder self-attention.  The remaining encoder ops (LayerNorm, head-major
// Q/K/V projection, o-projection, fc1/fc2) reuse the C1b/C1c/C1d primitives.
// All accumulation is FP32 with FP16 rounding at each stored boundary, matching
// the compiled PyTorch CUDA FP16 encoder used as the C4 bring-up oracle.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;
constexpr int64_t kMoonshineChannels = 416;
constexpr int64_t kMoonshineConv2Channels = 832;
constexpr int64_t kMoonshineHeads = 8;
constexpr int64_t kMoonshineHeadDim = 52;
constexpr float kLowest = -3.4028234663852886e38f;
constexpr float kGeluConstant = 0.70710678118654752440f;  // 1 / sqrt(2)

__device__ inline float moonshine_gelu_f32(float value) {
  return 0.5f * value * (1.0f + erff(value * kGeluConstant));
}

__device__ inline half_t moonshine_gelu_half(half_t value) {
  return static_cast<half_t>(moonshine_gelu_f32(static_cast<float>(value)));
}

// ---- conv1 (1 -> 416, kernel 127, stride 64) + tanh ------------------------
// One block per output channel; threads stride over output positions.
__global__ __launch_bounds__(256) void moonshine_conv1_tanh_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    half_t* __restrict__ output,
    int64_t length,
    int64_t out_length) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x);
  const half_t* w = weight + channel * 127;
  half_t* out = output + channel * out_length;
  for (int64_t position = threadIdx.x; position < out_length;
       position += blockDim.x) {
    float accumulator = 0.0f;
    const half_t* src = input + position * 64;
#pragma unroll 8
    for (int64_t k = 0; k < 127; ++k) {
      accumulator +=
          static_cast<float>(src[k]) * static_cast<float>(w[k]);
    }
    out[position] = static_cast<half_t>(tanhf(accumulator));
  }
}

// ---- conv (in_channels -> out_channels, kernel 7/3) + bias + gelu ---------
// One block per output position; each thread owns one output channel and reads
// the shared input window plus its contiguous weight row.  ``kRowMajorOutput``
// selects the row-major ``[position, channels]`` layout used for the final
// conv3 hidden (so the permute is fused) versus the channel-major
// ``[channels, position]`` layout consumed by the next conv stage.
template <int kInChannels, int kKernel, int kOutChannels, bool kRowMajorOutput = false>
__global__ __launch_bounds__(kOutChannels)
void moonshine_conv_gelu_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t in_length,
    int64_t out_length,
    int64_t stride) {
  constexpr int kWindow = kInChannels * kKernel;
  __shared__ half_t window[kWindow];
  const int64_t position = static_cast<int64_t>(blockIdx.x);
  for (int index = threadIdx.x; index < kWindow; index += blockDim.x) {
    const int kernel_index = index % kKernel;
    const int channel = index / kKernel;
    window[index] = input[channel * in_length + position * stride + kernel_index];
  }
  __syncthreads();
  const int64_t out_channel = static_cast<int64_t>(threadIdx.x);
  const half_t* w = weight + out_channel * kWindow;
  float accumulator = static_cast<float>(bias[out_channel]);
#pragma unroll 8
  for (int index = 0; index < kWindow; ++index) {
    accumulator +=
        static_cast<float>(window[index]) * static_cast<float>(w[index]);
  }
  const half_t result = moonshine_gelu_half(static_cast<half_t>(accumulator));
  if (kRowMajorOutput) {
    output[position * kOutChannels + out_channel] = result;
  } else {
    output[out_channel * out_length + position] = result;
  }
}

// ---- GroupNorm(1) over the full [channels, length] plane ------------------
// Kernel 1: one block per channel reduces the channel row to partial sum/sumsq.
__global__ __launch_bounds__(256) void moonshine_groupnorm_partial_fp16_kernel(
    const half_t* __restrict__ input,
    float* __restrict__ partial,
    int64_t channels,
    int64_t length) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x);
  float sum = 0.0f;
  float sum_sq = 0.0f;
  const half_t* row = input + channel * length;
  for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
    const float value = static_cast<float>(row[index]);
    sum += value;
    sum_sq += value * value;
  }
  __shared__ float block_sum[256];
  __shared__ float block_sum_sq[256];
  block_sum[threadIdx.x] = sum;
  block_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int offset = 256 / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
      block_sum_sq[threadIdx.x] += block_sum_sq[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[channel * 2] = block_sum[0];
    partial[channel * 2 + 1] = block_sum_sq[0];
  }
}

// Kernel 2: one block reduces the per-channel partials to mean and rstd.
__global__ void moonshine_groupnorm_finalize_fp16_kernel(
    const float* __restrict__ partial,
    float* __restrict__ mean_rstd,
    int64_t channels,
    int64_t length,
    float eps) {
  float total_sum = 0.0f;
  float total_sq = 0.0f;
  for (int64_t channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    total_sum += partial[channel * 2];
    total_sq += partial[channel * 2 + 1];
  }
  __shared__ float block_sum[256];
  __shared__ float block_sum_sq[256];
  block_sum[threadIdx.x] = total_sum;
  block_sum_sq[threadIdx.x] = total_sq;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
      block_sum_sq[threadIdx.x] += block_sum_sq[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int64_t count = channels * length;
    const float mean = block_sum[0] / static_cast<float>(count);
    const float variance =
        block_sum_sq[0] / static_cast<float>(count) - mean * mean;
    mean_rstd[0] = mean;
    mean_rstd[1] = rsqrtf(variance + eps);
  }
}

// Kernel 3: apply per-element affine over the whole plane.
__global__ void moonshine_groupnorm_apply_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    const half_t* __restrict__ bias,
    const float* __restrict__ mean_rstd,
    half_t* __restrict__ output,
    int64_t channels,
    int64_t length) {
  const int64_t count = channels * length;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t channel = index / length;
    const float value = static_cast<float>(input[index]);
    const float mean = mean_rstd[0];
    const float rstd = mean_rstd[1];
    const float scaled = (value - mean) * rstd * static_cast<float>(weight[channel]) +
        static_cast<float>(bias[channel]);
    output[index] = static_cast<half_t>(scaled);
  }
}

// ---- elementwise exact GELU -------------------------------------------------
__global__ void moonshine_gelu_fp16_kernel(
    const half_t* __restrict__ input,
    half_t* __restrict__ output,
    int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = moonshine_gelu_half(input[index]);
  }
}

// ---- full-sequence partial RoPE over [heads, sequence, head_dim] -----------
__device__ inline half_t moonshine_encoder_rotate_value(
    const half_t* source,
    int64_t head_offset,
    int64_t sequence,
    int64_t sequence_stride,
    int64_t dimension,
    int64_t rotary_dim,
    const half_t* cos,
    const half_t* sin,
    int64_t max_positions) {
  if (dimension >= rotary_dim) {
    return source[head_offset + sequence * sequence_stride + dimension];
  }
  const int64_t pair = dimension >> 1;
  const int64_t pair_offset = head_offset + sequence * sequence_stride + (pair << 1);
  const float first = static_cast<float>(source[pair_offset]);
  const float second = static_cast<float>(source[pair_offset + 1]);
  const int64_t table_offset = sequence * (rotary_dim >> 1) + pair;
  const float cosine = static_cast<float>(cos[table_offset]);
  const float sine = static_cast<float>(sin[table_offset]);
  const float value = (dimension & 1)
      ? second * cosine + first * sine
      : first * cosine - second * sine;
  return static_cast<half_t>(value);
}

__global__ __launch_bounds__(256) void moonshine_encoder_rope_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key,
    const half_t* __restrict__ cos,
    const half_t* __restrict__ sin,
    half_t* __restrict__ query_output,
    half_t* __restrict__ key_output,
    int64_t heads,
    int64_t sequence,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t elements = heads * sequence * head_dim;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements || sequence > max_positions) return;
  const int64_t dimension = index % head_dim;
  const int64_t position = (index / head_dim) % sequence;
  const int64_t head = index / (head_dim * sequence);
  const int64_t head_offset = head * sequence * head_dim;
  query_output[index] = moonshine_encoder_rotate_value(
      query, head_offset, position, head_dim, dimension, rotary_dim,
      cos, sin, max_positions);
  key_output[index] = moonshine_encoder_rotate_value(
      key, head_offset, position, head_dim, dimension, rotary_dim,
      cos, sin, max_positions);
}

// ---- non-causal full-sequence encoder self-attention ------------------------
// Grid (sequence, heads); one 32-lane wave per (query position, head) attends
// the query against every key with the online FP32 softmax used by the decoder
// attention kernels.  Head-major [heads, sequence, head_dim] Q/K/V in, row-major
// [sequence, hidden] output written directly so o_proj reads it without a
// transpose pass.
__device__ inline void moonshine_encoder_attention_position(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t head,
    int64_t query_position,
    int64_t sequence,
    float scale,
    bool use_mask) {
  const int64_t lane = static_cast<int64_t>(threadIdx.x) & 31;
  const int64_t head_offset = head * sequence * kMoonshineHeadDim;
  const int64_t query_offset = head_offset + query_position * kMoonshineHeadDim;
  const int64_t first_dim = lane;
  const int64_t second_dim = lane + 32;
  const float query_first = static_cast<float>(query[query_offset + first_dim]);
  const float query_second = second_dim < kMoonshineHeadDim
      ? static_cast<float>(query[query_offset + second_dim])
      : 0.0f;
  float output_first = 0.0f;
  float output_second = 0.0f;
  float running_max = kLowest;
  float denominator = 0.0f;

  for (int64_t token = 0; token < sequence; ++token) {
    const bool visible = !use_mask || mask[token] != 0;
    float dot = 0.0f;
    if (visible) {
      const int64_t cache_offset = head_offset + token * kMoonshineHeadDim;
      dot = query_first * static_cast<float>(key_cache[cache_offset + first_dim]);
      if (second_dim < kMoonshineHeadDim) {
        dot += query_second * static_cast<float>(key_cache[cache_offset + second_dim]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset, 32);
    }
    if (!visible) continue;
    const float score = __shfl_sync(0xffffffffu, dot, 0, 32) * scale;
    const float next_max = fmaxf(running_max, score);
    const float previous_weight = denominator > 0.0f
        ? expf(running_max - next_max)
        : 0.0f;
    const float current_weight = expf(score - next_max);
    denominator = denominator * previous_weight + current_weight;
    if (second_dim < kMoonshineHeadDim) {
      output_second = output_second * previous_weight + current_weight *
          static_cast<float>(value_cache[head_offset + token * kMoonshineHeadDim + second_dim]);
    }
    output_first = output_first * previous_weight + current_weight *
        static_cast<float>(value_cache[head_offset + token * kMoonshineHeadDim + first_dim]);
    running_max = next_max;
  }

  const float inverse_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  const int64_t row_output = query_position * (kMoonshineHeads * kMoonshineHeadDim) +
      head * kMoonshineHeadDim;
  output[row_output + first_dim] =
      static_cast<half_t>(output_first * inverse_denominator);
  if (second_dim < kMoonshineHeadDim) {
    output[row_output + second_dim] =
        static_cast<half_t>(output_second * inverse_denominator);
  }
}

__global__ __launch_bounds__(32) void moonshine_encoder_attention_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t sequence,
    float scale) {
  const int64_t head = static_cast<int64_t>(blockIdx.y);
  const int64_t query_position = static_cast<int64_t>(blockIdx.x);
  if (head >= kMoonshineHeads || query_position >= sequence) return;
  moonshine_encoder_attention_position(
      query, key_cache, value_cache, mask, output, head, query_position,
      sequence, scale, mask != nullptr);
}

// -------------------------------------------------------------------------
// Batch-plane variants (C8 phase 2): one grid dimension per batch plane so
// the row-generic layernorm/projection/gelu kernels compose at M=B*sequence
// while the conv front end, GroupNorm, head-major transpose, RoPE, and the
// non-causal self-attention process B fixed-length planes independently.
// -------------------------------------------------------------------------

// ---- batch conv1 (1 -> 416, kernel 127, stride 64) + tanh ------------------
// Grid (416, batch); one block per (output channel, plane).
__global__ __launch_bounds__(256) void moonshine_conv1_tanh_batch_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    half_t* __restrict__ output,
    int64_t batch,
    int64_t length,
    int64_t out_length) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x);
  const int64_t plane = static_cast<int64_t>(blockIdx.y);
  const half_t* w = weight + channel * 127;
  half_t* out = output + (plane * kMoonshineChannels + channel) * out_length;
  const half_t* src = input + plane * length;
  for (int64_t position = threadIdx.x; position < out_length;
       position += blockDim.x) {
    float accumulator = 0.0f;
    const half_t* plane_src = src + position * 64;
#pragma unroll 8
    for (int64_t k = 0; k < 127; ++k) {
      accumulator +=
          static_cast<float>(plane_src[k]) * static_cast<float>(w[k]);
    }
    out[position] = static_cast<half_t>(tanhf(accumulator));
  }
}

// ---- batch conv (in_channels -> out_channels, kernel 7/3) + bias + gelu ----
// Grid (out_length, batch); each block owns one output position of one plane.
template <int kInChannels, int kKernel, int kOutChannels, bool kRowMajorOutput = false>
__global__ __launch_bounds__(kOutChannels)
void moonshine_conv_gelu_batch_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t batch,
    int64_t in_length,
    int64_t out_length,
    int64_t stride) {
  constexpr int kWindow = kInChannels * kKernel;
  __shared__ half_t window[kWindow];
  const int64_t position = static_cast<int64_t>(blockIdx.x);
  const int64_t plane = static_cast<int64_t>(blockIdx.y);
  const half_t* plane_input = input + plane * kInChannels * in_length;
  for (int index = threadIdx.x; index < kWindow; index += blockDim.x) {
    const int kernel_index = index % kKernel;
    const int channel = index / kKernel;
    window[index] =
        plane_input[channel * in_length + position * stride + kernel_index];
  }
  __syncthreads();
  const int64_t out_channel = static_cast<int64_t>(threadIdx.x);
  const half_t* w = weight + out_channel * kWindow;
  float accumulator = static_cast<float>(bias[out_channel]);
#pragma unroll 8
  for (int index = 0; index < kWindow; ++index) {
    accumulator +=
        static_cast<float>(window[index]) * static_cast<float>(w[index]);
  }
  const half_t result = moonshine_gelu_half(static_cast<half_t>(accumulator));
  if (kRowMajorOutput) {
    output[(plane * out_length + position) * kOutChannels + out_channel] = result;
  } else {
    output[(plane * kOutChannels + out_channel) * out_length + position] = result;
  }
}

// ---- batch GroupNorm(1) over each [channels, length] plane -----------------
// Kernel 1: grid (channels, batch); per (channel, plane) partial sum/sumsq.
__global__ __launch_bounds__(256) void moonshine_groupnorm_partial_batch_fp16_kernel(
    const half_t* __restrict__ input,
    float* __restrict__ partial,
    int64_t batch,
    int64_t channels,
    int64_t length) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x);
  const int64_t plane = static_cast<int64_t>(blockIdx.y);
  float sum = 0.0f;
  float sum_sq = 0.0f;
  const half_t* row = input + (plane * channels + channel) * length;
  for (int64_t index = threadIdx.x; index < length; index += blockDim.x) {
    const float value = static_cast<float>(row[index]);
    sum += value;
    sum_sq += value * value;
  }
  __shared__ float block_sum[256];
  __shared__ float block_sum_sq[256];
  block_sum[threadIdx.x] = sum;
  block_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int offset = 256 / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
      block_sum_sq[threadIdx.x] += block_sum_sq[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[(plane * channels + channel) * 2] = block_sum[0];
    partial[(plane * channels + channel) * 2 + 1] = block_sum_sq[0];
  }
}

// Kernel 2: grid (batch); one block reduces one plane to mean and rstd.
__global__ void moonshine_groupnorm_finalize_batch_fp16_kernel(
    const float* __restrict__ partial,
    float* __restrict__ mean_rstd,
    int64_t batch,
    int64_t channels,
    int64_t length,
    float eps) {
  const int64_t plane = static_cast<int64_t>(blockIdx.x);
  float total_sum = 0.0f;
  float total_sq = 0.0f;
  for (int64_t channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    total_sum += partial[(plane * channels + channel) * 2];
    total_sq += partial[(plane * channels + channel) * 2 + 1];
  }
  __shared__ float block_sum[256];
  __shared__ float block_sum_sq[256];
  block_sum[threadIdx.x] = total_sum;
  block_sum_sq[threadIdx.x] = total_sq;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
      block_sum_sq[threadIdx.x] += block_sum_sq[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int64_t count = channels * length;
    const float mean = block_sum[0] / static_cast<float>(count);
    const float variance =
        block_sum_sq[0] / static_cast<float>(count) - mean * mean;
    mean_rstd[plane * 2] = mean;
    mean_rstd[plane * 2 + 1] = rsqrtf(variance + eps);
  }
}

// Kernel 3: grid over (batch * count); apply per-plane mean/rstd.
__global__ void moonshine_groupnorm_apply_batch_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    const half_t* __restrict__ bias,
    const float* __restrict__ mean_rstd,
    half_t* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t length) {
  const int64_t count = channels * length;
  const int64_t total = batch * count;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t plane = index / count;
    const int64_t local = index - plane * count;
    const int64_t channel = local / length;
    const float value = static_cast<float>(input[index]);
    const float mean = mean_rstd[plane * 2];
    const float rstd = mean_rstd[plane * 2 + 1];
    const float scaled = (value - mean) * rstd * static_cast<float>(weight[channel]) +
        static_cast<float>(bias[channel]);
    output[index] = static_cast<half_t>(scaled);
  }
}

// ---- batch transpose row-major [B, seq, hidden] -> head-major [B, heads, seq, dim] -
__global__ __launch_bounds__(256)
void moonshine_encoder_transpose_head_major_batch_fp16_kernel(
    const half_t* __restrict__ input,
    half_t* __restrict__ output,
    int64_t batch,
    int64_t sequence,
    int64_t heads,
    int64_t head_dim) {
  const int64_t hidden = heads * head_dim;
  const int64_t per_plane = sequence * hidden;
  const int64_t total = batch * per_plane;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int64_t plane = index / per_plane;
  const int64_t local = index - plane * per_plane;
  const int64_t dimension = local % head_dim;
  const int64_t position = (local / head_dim) % sequence;
  const int64_t head = local / (head_dim * sequence);
  const int64_t row_hidden = head * head_dim + dimension;
  const int64_t out_index =
      ((plane * heads + head) * sequence + position) * head_dim + dimension;
  output[out_index] = input[plane * per_plane + position * hidden + row_hidden];
}

// ---- batch full-sequence partial RoPE over [B, heads, sequence, head_dim] ---
__global__ __launch_bounds__(256) void moonshine_encoder_rope_batch_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key,
    const half_t* __restrict__ cos,
    const half_t* __restrict__ sin,
    half_t* __restrict__ query_output,
    half_t* __restrict__ key_output,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t elements = batch * heads * sequence * head_dim;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements || sequence > max_positions) return;
  const int64_t per_plane = heads * sequence * head_dim;
  const int64_t plane = index / per_plane;
  const int64_t local = index - plane * per_plane;
  const int64_t dimension = local % head_dim;
  const int64_t position = (local / head_dim) % sequence;
  const int64_t head = local / (head_dim * sequence);
  const int64_t head_offset = (plane * heads + head) * sequence * head_dim;
  query_output[index] = moonshine_encoder_rotate_value(
      query, head_offset, position, head_dim, dimension, rotary_dim,
      cos, sin, max_positions);
  key_output[index] = moonshine_encoder_rotate_value(
      key, head_offset, position, head_dim, dimension, rotary_dim,
      cos, sin, max_positions);
}

// ---- batch non-causal full-sequence encoder self-attention -----------------
// Grid (sequence, heads, batch); one 32-lane wave per (plane, query, head).
// Q/K/V are batch head-major [B, heads, sequence, head_dim]; the output is
// written row-major [B, sequence, hidden] directly for the o_proj read.
__device__ inline void moonshine_encoder_attention_position_batch(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t plane,
    int64_t head,
    int64_t query_position,
    int64_t sequence,
    float scale,
    bool use_mask) {
  const int64_t lane = static_cast<int64_t>(threadIdx.x) & 31;
  const int64_t plane_head_offset =
      (plane * kMoonshineHeads + head) * sequence * kMoonshineHeadDim;
  const int64_t query_offset =
      plane_head_offset + query_position * kMoonshineHeadDim;
  const int64_t first_dim = lane;
  const int64_t second_dim = lane + 32;
  const float query_first = static_cast<float>(query[query_offset + first_dim]);
  const float query_second = second_dim < kMoonshineHeadDim
      ? static_cast<float>(query[query_offset + second_dim])
      : 0.0f;
  float output_first = 0.0f;
  float output_second = 0.0f;
  float running_max = kLowest;
  float denominator = 0.0f;

  for (int64_t token = 0; token < sequence; ++token) {
    const bool visible = !use_mask || mask[plane * sequence + token] != 0;
    float dot = 0.0f;
    if (visible) {
      const int64_t cache_offset =
          plane_head_offset + token * kMoonshineHeadDim;
      dot = query_first * static_cast<float>(key_cache[cache_offset + first_dim]);
      if (second_dim < kMoonshineHeadDim) {
        dot += query_second *
            static_cast<float>(key_cache[cache_offset + second_dim]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset, 32);
    }
    if (!visible) continue;
    const float score = __shfl_sync(0xffffffffu, dot, 0, 32) * scale;
    const float next_max = fmaxf(running_max, score);
    const float previous_weight = denominator > 0.0f
        ? expf(running_max - next_max)
        : 0.0f;
    const float current_weight = expf(score - next_max);
    denominator = denominator * previous_weight + current_weight;
    if (second_dim < kMoonshineHeadDim) {
      output_second = output_second * previous_weight + current_weight *
          static_cast<float>(value_cache[plane_head_offset + token * kMoonshineHeadDim + second_dim]);
    }
    output_first = output_first * previous_weight + current_weight *
        static_cast<float>(value_cache[plane_head_offset + token * kMoonshineHeadDim + first_dim]);
    running_max = next_max;
  }

  const float inverse_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  const int64_t row_output =
      (plane * sequence + query_position) * (kMoonshineHeads * kMoonshineHeadDim) +
      head * kMoonshineHeadDim;
  output[row_output + first_dim] =
      static_cast<half_t>(output_first * inverse_denominator);
  if (second_dim < kMoonshineHeadDim) {
    output[row_output + second_dim] =
        static_cast<half_t>(output_second * inverse_denominator);
  }
}

__global__ __launch_bounds__(32) void moonshine_encoder_attention_batch_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t sequence,
    float scale) {
  const int64_t plane = static_cast<int64_t>(blockIdx.z);
  const int64_t head = static_cast<int64_t>(blockIdx.y);
  const int64_t query_position = static_cast<int64_t>(blockIdx.x);
  if (head >= kMoonshineHeads || query_position >= sequence) return;
  moonshine_encoder_attention_position_batch(
      query, key_cache, value_cache, mask, output, plane, head,
      query_position, sequence, scale, mask != nullptr);
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_conv1_tanh_fp16(
    const half_t* input, const half_t* weight, half_t* output,
    int64_t length, int64_t out_length, int64_t threads, cudaStream_t stream) {
  if (length <= 0 || out_length <= 0 || threads != 256) return 1;
  moonshine_conv1_tanh_fp16_kernel<<<kMoonshineChannels, 256, 0, stream>>>(
      input, weight, output, length, out_length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_conv2_gelu_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, int64_t in_length, int64_t out_length, int64_t threads,
    cudaStream_t stream) {
  if (in_length <= 0 || out_length <= 0 || threads != kMoonshineConv2Channels) {
    return 1;
  }
  moonshine_conv_gelu_fp16_kernel<kMoonshineChannels, 7, kMoonshineConv2Channels>
      <<<out_length, kMoonshineConv2Channels, 0, stream>>>(
          input, weight, bias, output, in_length, out_length, 3);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_conv3_gelu_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, int64_t in_length, int64_t out_length, int64_t threads,
    cudaStream_t stream) {
  if (in_length <= 0 || out_length <= 0 || threads != kMoonshineChannels) {
    return 1;
  }
  moonshine_conv_gelu_fp16_kernel<kMoonshineConv2Channels, 3, kMoonshineChannels, true>
      <<<out_length, kMoonshineChannels, 0, stream>>>(
          input, weight, bias, output, in_length, out_length, 2);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_groupnorm_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, float* partial, float* mean_rstd, int64_t channels,
    int64_t length, float eps, int64_t threads, cudaStream_t stream) {
  if (channels <= 0 || length <= 0 || threads != 256) return 1;
  moonshine_groupnorm_partial_fp16_kernel<<<channels, 256, 0, stream>>>(
      input, partial, channels, length);
  moonshine_groupnorm_finalize_fp16_kernel<<<1, 256, 0, stream>>>(
      partial, mean_rstd, channels, length, eps);
  const int64_t count = channels * length;
  moonshine_groupnorm_apply_fp16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      input, weight, bias, mean_rstd, output, channels, length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_gelu_fp16(
    const half_t* input, half_t* output, int64_t elements, int64_t threads,
    cudaStream_t stream) {
  if (elements <= 0 || threads != 256) return 1;
  moonshine_gelu_fp16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      input, output, elements);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_rope_fp16(
    const half_t* query, const half_t* key, const half_t* cos, const half_t* sin,
    half_t* query_output, half_t* key_output, int64_t heads, int64_t sequence,
    int64_t head_dim, int64_t rotary_dim, int64_t max_positions, int64_t threads,
    cudaStream_t stream) {
  if (heads <= 0 || sequence <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || sequence > max_positions || threads != 256) {
    return 1;
  }
  const int64_t elements = heads * sequence * head_dim;
  moonshine_encoder_rope_fp16_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      query, key, cos, sin, query_output, key_output, heads, sequence, head_dim,
      rotary_dim, max_positions);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

// ---- row-major [sequence, heads*head_dim] -> head-major [heads, seq, dim] ---
// The Q/K/V projections write the row-major hidden layout consumed by the
// o_proj and MLP kernels, but the rope and encoder self-attention kernels take
// head-major [heads, sequence, head_dim].  This single transpose pass bridges
// the two layouts so the validated C1c/C1e kernels compose unchanged.
__global__ __launch_bounds__(256) void moonshine_encoder_transpose_head_major_fp16_kernel(
    const half_t* __restrict__ input,
    half_t* __restrict__ output,
    int64_t sequence,
    int64_t heads,
    int64_t head_dim) {
  const int64_t hidden = heads * head_dim;
  const int64_t elements = sequence * hidden;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const int64_t dimension = index % head_dim;
  const int64_t position = (index / head_dim) % sequence;
  const int64_t head = index / (head_dim * sequence);
  const int64_t row_hidden = head * head_dim + dimension;
  output[index] = input[position * hidden + row_hidden];
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_fp16(
    const half_t* input, half_t* output, int64_t sequence, int64_t heads,
    int64_t head_dim, int64_t threads, cudaStream_t stream) {
  if (sequence <= 0 || heads <= 0 || head_dim <= 0 || threads != 256) return 1;
  const int64_t elements = sequence * heads * head_dim;
  moonshine_encoder_transpose_head_major_fp16_kernel<<<
      (elements + 255) / 256, 256, 0, stream>>>(
      input, output, sequence, heads, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_attention_fp16(
    const half_t* query, const half_t* key, const half_t* value,
    const int32_t* mask, half_t* output, int64_t heads, int64_t head_dim,
    int64_t sequence, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      sequence <= 0 || threads != 32) {
    return 1;
  }
  dim3 grid(sequence, kMoonshineHeads);
  moonshine_encoder_attention_fp16_kernel<<<grid, 32, 0, stream>>>(
      query, key, value, mask, output, sequence, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_conv1_tanh_batch_fp16(
    const half_t* input, const half_t* weight, half_t* output,
    int64_t batch, int64_t length, int64_t out_length, int64_t threads,
    cudaStream_t stream) {
  if (batch <= 0 || length <= 0 || out_length <= 0 || threads != 256) return 1;
  dim3 grid(kMoonshineChannels, batch);
  moonshine_conv1_tanh_batch_fp16_kernel<<<grid, 256, 0, stream>>>(
      input, weight, output, batch, length, out_length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_conv2_gelu_batch_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, int64_t batch, int64_t in_length, int64_t out_length,
    int64_t threads, cudaStream_t stream) {
  if (batch <= 0 || in_length <= 0 || out_length <= 0 ||
      threads != kMoonshineConv2Channels) {
    return 1;
  }
  dim3 grid(out_length, batch);
  moonshine_conv_gelu_batch_fp16_kernel<
      kMoonshineChannels, 7, kMoonshineConv2Channels>
      <<<grid, kMoonshineConv2Channels, 0, stream>>>(
          input, weight, bias, output, batch, in_length, out_length, 3);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_conv3_gelu_batch_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, int64_t batch, int64_t in_length, int64_t out_length,
    int64_t threads, cudaStream_t stream) {
  if (batch <= 0 || in_length <= 0 || out_length <= 0 ||
      threads != kMoonshineChannels) {
    return 1;
  }
  dim3 grid(out_length, batch);
  moonshine_conv_gelu_batch_fp16_kernel<
      kMoonshineConv2Channels, 3, kMoonshineChannels, true>
      <<<grid, kMoonshineChannels, 0, stream>>>(
          input, weight, bias, output, batch, in_length, out_length, 2);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_groupnorm_batch_fp16(
    const half_t* input, const half_t* weight, const half_t* bias,
    half_t* output, float* partial, float* mean_rstd, int64_t batch,
    int64_t channels, int64_t length, float eps, int64_t threads,
    cudaStream_t stream) {
  if (batch <= 0 || channels <= 0 || length <= 0 || threads != 256) return 1;
  dim3 partial_grid(channels, batch);
  moonshine_groupnorm_partial_batch_fp16_kernel<<<partial_grid, 256, 0, stream>>>(
      input, partial, batch, channels, length);
  moonshine_groupnorm_finalize_batch_fp16_kernel<<<batch, 256, 0, stream>>>(
      partial, mean_rstd, batch, channels, length, eps);
  const int64_t count = channels * length;
  moonshine_groupnorm_apply_batch_fp16_kernel<<<
      (count + 255) / 256, 256, 0, stream>>>(
      input, weight, bias, mean_rstd, output, batch, channels, length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_transpose_head_major_batch_fp16(
    const half_t* input, half_t* output, int64_t batch, int64_t sequence,
    int64_t heads, int64_t head_dim, int64_t threads, cudaStream_t stream) {
  if (batch <= 0 || sequence <= 0 || heads <= 0 || head_dim <= 0 ||
      threads != 256) {
    return 1;
  }
  const int64_t elements = batch * sequence * heads * head_dim;
  moonshine_encoder_transpose_head_major_batch_fp16_kernel<<<
      (elements + 255) / 256, 256, 0, stream>>>(
      input, output, batch, sequence, heads, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_rope_batch_fp16(
    const half_t* query, const half_t* key, const half_t* cos, const half_t* sin,
    half_t* query_output, half_t* key_output, int64_t batch, int64_t heads,
    int64_t sequence, int64_t head_dim, int64_t rotary_dim,
    int64_t max_positions, int64_t threads, cudaStream_t stream) {
  if (batch <= 0 || heads <= 0 || sequence <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || sequence > max_positions ||
      threads != 256) {
    return 1;
  }
  const int64_t elements = batch * heads * sequence * head_dim;
  moonshine_encoder_rope_batch_fp16_kernel<<<
      (elements + 255) / 256, 256, 0, stream>>>(
      query, key, cos, sin, query_output, key_output, batch, heads, sequence,
      head_dim, rotary_dim, max_positions);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_attention_batch_fp16(
    const half_t* query, const half_t* key, const half_t* value,
    const int32_t* mask, half_t* output, int64_t batch, int64_t heads,
    int64_t head_dim, int64_t sequence, float scale, int64_t threads,
    cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      batch <= 0 || sequence <= 0 || threads != 32) {
    return 1;
  }
  dim3 grid(sequence, kMoonshineHeads, batch);
  moonshine_encoder_attention_batch_fp16_kernel<<<grid, 32, 0, stream>>>(
      query, key, value, mask, output, sequence, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
