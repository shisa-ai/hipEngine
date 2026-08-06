// Moonshine batch-one FP16 self/cross attention over logical head dimension 52.
//
// CUDA ``sm_120a`` port of the correctness-qualified HIP reference in
// ``hip_gfx1100/attention/moonshine_attention.hip``. One wave (32 threads) per
// head computes an online FP32 softmax directly from the FP16 head-major cache
// without materializing a score plane; lanes 0..19 cover dims {d, d+32} and
// lanes 20..31 cover dim {d}. Multi-wave variants partition the visible
// sequence across 2/4/8 waves and merge partial (max, denominator, output)
// statistics in shared memory. Output boundaries are rounded FP16.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;
constexpr int64_t kMoonshineHeads = 8;
constexpr int64_t kMoonshineHeadDim = 52;
constexpr int64_t kWaveThreads = 32;
constexpr float kLowest = -3.4028234663852886e38f;

// Attend one head over ``key_length`` head-major keys (stride ``cache_stride``
// heads x tokens) with optional per-token mask. Each lane holds dims {d, d+32};
// the per-token dot is reduced across the wave and the online softmax state is
// updated in FP32.
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
  float running_max = kLowest;
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

template <int kWaves>
__global__ __launch_bounds__(kWaves * kWaveThreads)
void moonshine_self_attention_parallel_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key_cache,
    const half_t* __restrict__ value_cache,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ output,
    int64_t capacity,
    float scale) {
  __shared__ float partial_max[kWaves];
  __shared__ float partial_denominator[kWaves];
  __shared__ float partial_output[kWaves][kMoonshineHeadDim];

  const int64_t position = position_ptr[0];
  const int64_t lane = static_cast<int64_t>(threadIdx.x) & (kWaveThreads - 1);
  const int64_t wave = static_cast<int64_t>(threadIdx.x) / kWaveThreads;
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (position < 0 || position >= capacity || head >= kMoonshineHeads) return;
  const int64_t key_length = position + 1;
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

  for (int64_t token = wave; token < key_length; token += kWaves) {
    const int64_t cache_offset =
        (head * capacity + token) * kMoonshineHeadDim;
    float dot = query_first * static_cast<float>(key_cache[cache_offset + first_dim]);
    if (second_dim < kMoonshineHeadDim) {
      dot += query_second * static_cast<float>(key_cache[cache_offset + second_dim]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset, 32);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0, 32) * scale;
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
      dot += __shfl_down_sync(0xffffffffu, dot, offset, 32);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0, 32) * scale;
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

template <int kWaves>
void launch_self_parallel(const half_t* query, const half_t* key_cache,
                          const half_t* value_cache,
                          const int64_t* position, half_t* output,
                          int64_t capacity, float scale,
                          cudaStream_t stream) {
  moonshine_self_attention_parallel_fp16_kernel<kWaves><<<8, kWaves * 32, 0, stream>>>(
      query, key_cache, value_cache, position, output, capacity, scale);
}

template <int kWaves>
void launch_cross_parallel(const half_t* query, const half_t* key_cache,
                           const half_t* value_cache, const int32_t* mask,
                           half_t* output, int64_t encoder_length, float scale,
                           cudaStream_t stream) {
  moonshine_cross_attention_parallel_fp16_kernel<kWaves><<<8, kWaves * 32, 0, stream>>>(
      query, key_cache, value_cache, mask, output, encoder_length, scale);
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_self_attention_fp16(
    const half_t* query, const half_t* key_cache, const half_t* value_cache,
    const int64_t* position, half_t* output, int64_t heads, int64_t head_dim,
    int64_t capacity, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      capacity <= 0 || threads != kWaveThreads) {
    return 1;
  }
  moonshine_self_attention_fp16_kernel<<<8, 32, 0, stream>>>(
      query, key_cache, value_cache, position, output, capacity, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_self_attention_parallel_fp16(
    const half_t* query, const half_t* key_cache, const half_t* value_cache,
    const int64_t* position, half_t* output, int64_t heads, int64_t head_dim,
    int64_t capacity, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      capacity <= 0) {
    return 1;
  }
  if (threads == 64) {
    launch_self_parallel<2>(query, key_cache, value_cache, position, output,
                            capacity, scale, stream);
  } else if (threads == 128) {
    launch_self_parallel<4>(query, key_cache, value_cache, position, output,
                            capacity, scale, stream);
  } else if (threads == 256) {
    launch_self_parallel<8>(query, key_cache, value_cache, position, output,
                            capacity, scale, stream);
  } else {
    return 1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_cross_attention_fp16(
    const half_t* query, const half_t* key_cache, const half_t* value_cache,
    const int32_t* mask, half_t* output, int64_t heads, int64_t head_dim,
    int64_t encoder_length, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      encoder_length <= 0 || threads != kWaveThreads) {
    return 1;
  }
  moonshine_cross_attention_fp16_kernel<<<8, 32, 0, stream>>>(
      query, key_cache, value_cache, mask, output, encoder_length, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_cross_attention_grouped_fp16(
    const half_t* query, const half_t* key_cache, const half_t* value_cache,
    const int32_t* mask, half_t* output, int64_t heads, int64_t head_dim,
    int64_t encoder_length, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      encoder_length <= 0 || threads != 256) {
    return 1;
  }
  moonshine_cross_attention_grouped_fp16_kernel<<<1, 256, 0, stream>>>(
      query, key_cache, value_cache, mask, output, encoder_length, scale);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_cross_attention_parallel_fp16(
    const half_t* query, const half_t* key_cache, const half_t* value_cache,
    const int32_t* mask, half_t* output, int64_t heads, int64_t head_dim,
    int64_t encoder_length, float scale, int64_t threads, cudaStream_t stream) {
  if (heads != kMoonshineHeads || head_dim != kMoonshineHeadDim ||
      encoder_length <= 0) {
    return 1;
  }
  if (threads == 64) {
    launch_cross_parallel<2>(query, key_cache, value_cache, mask, output,
                             encoder_length, scale, stream);
  } else if (threads == 128) {
    launch_cross_parallel<4>(query, key_cache, value_cache, mask, output,
                             encoder_length, scale, stream);
  } else if (threads == 256) {
    launch_cross_parallel<8>(query, key_cache, value_cache, mask, output,
                             encoder_length, scale, stream);
  } else {
    return 1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
