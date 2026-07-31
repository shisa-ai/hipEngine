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
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
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
