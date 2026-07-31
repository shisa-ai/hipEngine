// Moonshine batch-one FP16 self/cross attention over logical head dimension 52.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int64_t kMoonshineHeads = 8;
constexpr int64_t kMoonshineHeadDim = 52;
constexpr int64_t kWaveThreads = 32;

template <bool UseMask>
__device__ __forceinline__ void moonshine_attention_head_fp16(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t head,
    int64_t key_length,
    int64_t cache_stride,
    float scale) {
  const int64_t lane = static_cast<int64_t>(threadIdx.x) & (kWaveThreads - 1);
  const int64_t query_offset = head * kMoonshineHeadDim;
  const int64_t first_dim = lane;
  const int64_t second_dim = lane + kWaveThreads;
  const float query_first = static_cast<float>(query[query_offset + first_dim]);
  const float query_second = second_dim < kMoonshineHeadDim
      ? static_cast<float>(query[query_offset + second_dim])
      : 0.0f;
  float output_first = 0.0f;
  float output_second = 0.0f;
  float running_max = -3.4028234663852886e38f;
  float denominator = 0.0f;

  for (int64_t token = 0; token < key_length; ++token) {
    const bool visible = !UseMask || mask[token] != 0;
    float dot = 0.0f;
    int64_t cache_offset = 0;
    if (visible) {
      cache_offset = (head * cache_stride + token) * kMoonshineHeadDim;
      dot = query_first * static_cast<float>(key_cache[cache_offset + first_dim]);
      if (second_dim < kMoonshineHeadDim) {
        dot += query_second * static_cast<float>(key_cache[cache_offset + second_dim]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down(dot, offset, 32);
    }
    if (!visible) continue;

    const float score = __shfl(dot, 0, 32) * scale;
    const float next_max = fmaxf(running_max, score);
    const float previous_weight = denominator > 0.0f
        ? expf(running_max - next_max)
        : 0.0f;
    const float current_weight = expf(score - next_max);
    denominator = denominator * previous_weight + current_weight;
    output_first = output_first * previous_weight + current_weight *
        static_cast<float>(value_cache[cache_offset + first_dim]);
    if (second_dim < kMoonshineHeadDim) {
      output_second = output_second * previous_weight + current_weight *
          static_cast<float>(value_cache[cache_offset + second_dim]);
    }
    running_max = next_max;
  }

  const float inverse_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  output[query_offset + first_dim] =
      static_cast<half_t>(output_first * inverse_denominator);
  if (second_dim < kMoonshineHeadDim) {
    output[query_offset + second_dim] =
        static_cast<half_t>(output_second * inverse_denominator);
  }
}

__global__ __launch_bounds__(32) void moonshine_self_attention_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ output,
    int64_t capacity,
    float scale) {
  const int64_t position = position_ptr[0];
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (position < 0 || position >= capacity || head >= kMoonshineHeads) return;
  moonshine_attention_head_fp16<false>(
      query,
      key_cache,
      value_cache,
      nullptr,
      output,
      head,
      position + 1,
      capacity,
      scale);
}

__global__ __launch_bounds__(32) void moonshine_cross_attention_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t encoder_length,
    float scale) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= kMoonshineHeads) return;
  moonshine_attention_head_fp16<true>(
      query,
      key_cache,
      value_cache,
      mask,
      output,
      head,
      encoder_length,
      encoder_length,
      scale);
}

__global__ __launch_bounds__(256) void moonshine_cross_attention_grouped_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t encoder_length,
    float scale) {
  const int64_t head = static_cast<int64_t>(threadIdx.x) / kWaveThreads;
  moonshine_attention_head_fp16<true>(
      query,
      key_cache,
      value_cache,
      mask,
      output,
      head,
      encoder_length,
      encoder_length,
      scale);
}

template <int kWaves>
__global__ __launch_bounds__(kWaves * kWaveThreads)
void moonshine_cross_attention_parallel_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int32_t* __restrict__ mask,
    half_t* __restrict__ output,
    int64_t encoder_length,
    float scale) {
  constexpr float kLowest = -3.4028234663852886e38f;
  __shared__ float partial_max[kWaves];
  __shared__ float partial_denominator[kWaves];
  __shared__ float partial_output[kWaves][kMoonshineHeadDim];

  const int64_t lane = static_cast<int64_t>(threadIdx.x) & (kWaveThreads - 1);
  const int64_t wave = static_cast<int64_t>(threadIdx.x) / kWaveThreads;
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t query_offset = head * kMoonshineHeadDim;
  const int64_t first_dim = lane;
  const int64_t second_dim = lane + kWaveThreads;
  const float query_first = static_cast<float>(query[query_offset + first_dim]);
  const float query_second = second_dim < kMoonshineHeadDim
      ? static_cast<float>(query[query_offset + second_dim])
      : 0.0f;
  float output_first = 0.0f;
  float output_second = 0.0f;
  float running_max = kLowest;
  float denominator = 0.0f;

  for (int64_t token = wave; token < encoder_length; token += kWaves) {
    if (mask[token] == 0) continue;
    const int64_t cache_offset =
        (head * encoder_length + token) * kMoonshineHeadDim;
    float dot = query_first * static_cast<float>(key_cache[cache_offset + first_dim]);
    if (second_dim < kMoonshineHeadDim) {
      dot += query_second * static_cast<float>(key_cache[cache_offset + second_dim]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down(dot, offset, 32);
    }
    const float score = __shfl(dot, 0, 32) * scale;
    if (score > running_max) {
      const float previous_weight = denominator > 0.0f
          ? expf(running_max - score)
          : 0.0f;
      denominator = denominator * previous_weight + 1.0f;
      output_first = output_first * previous_weight +
          static_cast<float>(value_cache[cache_offset + first_dim]);
      if (second_dim < kMoonshineHeadDim) {
        output_second = output_second * previous_weight +
            static_cast<float>(value_cache[cache_offset + second_dim]);
      }
      running_max = score;
    } else {
      const float weight = expf(score - running_max);
      denominator += weight;
      output_first += weight * static_cast<float>(value_cache[cache_offset + first_dim]);
      if (second_dim < kMoonshineHeadDim) {
        output_second +=
            weight * static_cast<float>(value_cache[cache_offset + second_dim]);
      }
    }
  }

  if (lane == 0) {
    partial_max[wave] = running_max;
    partial_denominator[wave] = denominator;
  }
  partial_output[wave][first_dim] = output_first;
  if (second_dim < kMoonshineHeadDim) {
    partial_output[wave][second_dim] = output_second;
  }
  __syncthreads();
  if (wave != 0) return;

  float merged_max = kLowest;
#pragma unroll
  for (int part = 0; part < kWaves; ++part) {
    if (partial_denominator[part] > 0.0f) {
      merged_max = fmaxf(merged_max, partial_max[part]);
    }
  }
  float merged_denominator = 0.0f;
  float merged_first = 0.0f;
  float merged_second = 0.0f;
#pragma unroll
  for (int part = 0; part < kWaves; ++part) {
    if (partial_denominator[part] <= 0.0f) continue;
    const float part_scale = expf(partial_max[part] - merged_max);
    merged_denominator += partial_denominator[part] * part_scale;
    merged_first += partial_output[part][first_dim] * part_scale;
    if (second_dim < kMoonshineHeadDim) {
      merged_second += partial_output[part][second_dim] * part_scale;
    }
  }
  const float inverse_denominator =
      merged_denominator > 0.0f ? 1.0f / merged_denominator : 0.0f;
  output[query_offset + first_dim] =
      static_cast<half_t>(merged_first * inverse_denominator);
  if (second_dim < kMoonshineHeadDim) {
    output[query_offset + second_dim] =
        static_cast<half_t>(merged_second * inverse_denominator);
  }
}

bool valid_moonshine_attention_shape(
    int64_t heads,
    int64_t head_dim,
    int64_t length,
    float scale,
    int64_t threads) {
  return heads == kMoonshineHeads && head_dim == kMoonshineHeadDim &&
      length > 0 && isfinite(scale) && scale > 0.0f && threads == kWaveThreads;
}

}  // namespace

extern "C" int hipengine_moonshine_self_attention_fp16(
    const half_t* query,
    const half_t* key_cache,
    const half_t* value_cache,
    const int64_t* position,
    half_t* output,
    int64_t heads,
    int64_t head_dim,
    int64_t capacity,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_moonshine_attention_shape(
          heads, head_dim, capacity, scale, threads)) {
    return hipErrorInvalidValue;
  }
  moonshine_self_attention_fp16_kernel<<<
      dim3(heads), dim3(threads), 0, stream>>>(
      query, key_cache, value_cache, position, output, capacity, scale);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_cross_attention_fp16(
    const half_t* query,
    const half_t* key_cache,
    const half_t* value_cache,
    const int32_t* mask,
    half_t* output,
    int64_t heads,
    int64_t head_dim,
    int64_t encoder_length,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_moonshine_attention_shape(
          heads, head_dim, encoder_length, scale, threads)) {
    return hipErrorInvalidValue;
  }
  moonshine_cross_attention_fp16_kernel<<<
      dim3(heads), dim3(threads), 0, stream>>>(
      query, key_cache, value_cache, mask, output, encoder_length, scale);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_cross_attention_grouped_fp16(
    const half_t* query,
    const half_t* key_cache,
    const half_t* value_cache,
    const int32_t* mask,
    half_t* output,
    int64_t heads,
    int64_t head_dim,
    int64_t encoder_length,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_moonshine_attention_shape(
          heads, head_dim, encoder_length, scale, kWaveThreads) ||
      threads != kMoonshineHeads * kWaveThreads) {
    return hipErrorInvalidValue;
  }
  moonshine_cross_attention_grouped_fp16_kernel<<<
      dim3(1), dim3(threads), 0, stream>>>(
      query, key_cache, value_cache, mask, output, encoder_length, scale);
  return hipGetLastError();
}

extern "C" int hipengine_moonshine_cross_attention_parallel_fp16(
    const half_t* query,
    const half_t* key_cache,
    const half_t* value_cache,
    const int32_t* mask,
    half_t* output,
    int64_t heads,
    int64_t head_dim,
    int64_t encoder_length,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_moonshine_attention_shape(
          heads, head_dim, encoder_length, scale, kWaveThreads) ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return hipErrorInvalidValue;
  }
  if (threads == 64) {
    moonshine_cross_attention_parallel_fp16_kernel<2><<<
        dim3(heads), dim3(threads), 0, stream>>>(
        query, key_cache, value_cache, mask, output, encoder_length, scale);
  } else if (threads == 128) {
    moonshine_cross_attention_parallel_fp16_kernel<4><<<
        dim3(heads), dim3(threads), 0, stream>>>(
        query, key_cache, value_cache, mask, output, encoder_length, scale);
  } else {
    moonshine_cross_attention_parallel_fp16_kernel<8><<<
        dim3(heads), dim3(threads), 0, stream>>>(
        query, key_cache, value_cache, mask, output, encoder_length, scale);
  }
  return hipGetLastError();
}
