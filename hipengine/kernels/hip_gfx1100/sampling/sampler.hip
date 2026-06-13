// Native row-wise sampler kernels for FP32 logits.
// Correctness-first S6 bring-up: bounded top-k + temperature softmax +
// counter-seeded sampling. Exact full-vocab top-p remains a separate path.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

namespace {

constexpr int kMaxSamplerTopK = 64;
constexpr float kNegInf = -3.4028234663852886e+38F;

__device__ inline bool better_pair_i32(float value, int32_t index, float best_value, int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__device__ inline void insert_topk_pair(
    float value,
    int32_t index,
    float* top_values,
    int32_t* top_indices,
    int top_k) {
  for (int pos = 0; pos < top_k; ++pos) {
    if (better_pair_i32(value, index, top_values[pos], top_indices[pos])) {
      for (int shift = top_k - 1; shift > pos; --shift) {
        top_values[shift] = top_values[shift - 1];
        top_indices[shift] = top_indices[shift - 1];
      }
      top_values[pos] = value;
      top_indices[pos] = index;
      return;
    }
  }
}

__device__ inline uint64_t splitmix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ inline float sampler_uniform01(uint64_t row_seed, uint64_t step_index, int64_t row) {
  const uint64_t row_component = (static_cast<uint64_t>(row) + 1ull) * 0xBF58476D1CE4E5B9ull;
  const uint64_t step_component = (step_index + 1ull) * 0x9E3779B97F4A7C15ull;
  const uint64_t bits = splitmix64(row_seed ^ row_component ^ step_component);
  const double unit = static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
  return static_cast<float>(unit);
}

__global__ void sampler_topk_temperature_rows_i32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ temperatures,
    const uint64_t* __restrict__ row_seeds,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_logprobs,
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t rows,
    int64_t vocab_size,
    int32_t top_k,
    uint64_t step_index) {
  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(shared_values + static_cast<int64_t>(blockDim.x) * top_k);

  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int top_count = min(top_k, static_cast<int32_t>(vocab_size));
  const int64_t logits_base = row * vocab_size;

  float local_values[kMaxSamplerTopK];
  int32_t local_indices[kMaxSamplerTopK];
  for (int pos = 0; pos < kMaxSamplerTopK; ++pos) {
    local_values[pos] = kNegInf;
    local_indices[pos] = INT32_MAX;
  }

  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[logits_base + i];
    if (isfinite(value)) {
      insert_topk_pair(value, static_cast<int32_t>(i), local_values, local_indices, top_count);
    }
  }

  for (int pos = 0; pos < top_count; ++pos) {
    const int64_t off = static_cast<int64_t>(tid) * top_k + pos;
    shared_values[off] = local_values[pos];
    shared_indices[off] = local_indices[pos];
  }
  __syncthreads();

  if (tid != 0) {
    return;
  }

  float final_values[kMaxSamplerTopK];
  int32_t final_indices[kMaxSamplerTopK];
  float candidate_logprobs[kMaxSamplerTopK];
  float candidate_weights[kMaxSamplerTopK];
  for (int pos = 0; pos < kMaxSamplerTopK; ++pos) {
    final_values[pos] = kNegInf;
    final_indices[pos] = INT32_MAX;
    candidate_logprobs[pos] = kNegInf;
    candidate_weights[pos] = 0.0f;
  }

  for (int thread = 0; thread < blockDim.x; ++thread) {
    for (int pos = 0; pos < top_count; ++pos) {
      const int64_t off = static_cast<int64_t>(thread) * top_k + pos;
      if (shared_indices[off] != INT32_MAX) {
        insert_topk_pair(shared_values[off], shared_indices[off], final_values, final_indices, top_count);
      }
    }
  }

  const int64_t top_out_base = row * static_cast<int64_t>(top_k);
  if (final_indices[0] == INT32_MAX) {
    out_indices[row] = -1;
    if (out_logprobs != nullptr) {
      out_logprobs[row] = kNegInf;
    }
    for (int pos = 0; pos < top_k; ++pos) {
      if (out_top_indices != nullptr) {
        out_top_indices[top_out_base + pos] = -1;
      }
      if (out_top_logprobs != nullptr) {
        out_top_logprobs[top_out_base + pos] = kNegInf;
      }
    }
    return;
  }

  const float temperature = temperatures[row];
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    out_indices[row] = final_indices[0];
    if (out_logprobs != nullptr) {
      out_logprobs[row] = 0.0f;
    }
    for (int pos = 0; pos < top_k; ++pos) {
      const bool valid = pos < top_count && final_indices[pos] != INT32_MAX;
      if (out_top_indices != nullptr) {
        out_top_indices[top_out_base + pos] = valid ? final_indices[pos] : -1;
      }
      if (out_top_logprobs != nullptr) {
        out_top_logprobs[top_out_base + pos] = pos == 0 ? 0.0f : kNegInf;
      }
    }
    return;
  }

  const float max_scaled = final_values[0] / temperature;
  float weight_sum = 0.0f;
  for (int pos = 0; pos < top_count; ++pos) {
    if (final_indices[pos] == INT32_MAX) {
      break;
    }
    const float scaled = final_values[pos] / temperature;
    const float weight = expf(scaled - max_scaled);
    candidate_weights[pos] = weight;
    weight_sum += weight;
  }

  int selected_pos = 0;
  if (weight_sum > 0.0f && isfinite(weight_sum)) {
    const float log_denom = logf(weight_sum) + max_scaled;
    for (int pos = 0; pos < top_count; ++pos) {
      if (final_indices[pos] == INT32_MAX) {
        break;
      }
      candidate_logprobs[pos] = final_values[pos] / temperature - log_denom;
    }
    const float threshold = sampler_uniform01(row_seeds[row], step_index, row) * weight_sum;
    float cumulative = 0.0f;
    selected_pos = top_count - 1;
    for (int pos = 0; pos < top_count; ++pos) {
      cumulative += candidate_weights[pos];
      if (threshold <= cumulative) {
        selected_pos = pos;
        break;
      }
    }
  } else {
    candidate_logprobs[0] = 0.0f;
    selected_pos = 0;
  }

  out_indices[row] = final_indices[selected_pos];
  if (out_logprobs != nullptr) {
    out_logprobs[row] = candidate_logprobs[selected_pos];
  }
  for (int pos = 0; pos < top_k; ++pos) {
    const bool valid = pos < top_count && final_indices[pos] != INT32_MAX;
    if (out_top_indices != nullptr) {
      out_top_indices[top_out_base + pos] = valid ? final_indices[pos] : -1;
    }
    if (out_top_logprobs != nullptr) {
      out_top_logprobs[top_out_base + pos] = valid ? candidate_logprobs[pos] : kNegInf;
    }
  }
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128;
}

}  // namespace

extern "C" int hipengine_sampler_topk_temperature_f32_rows_i32(
    const float* logits,
    const float* temperatures,
    const uint64_t* row_seeds,
    int32_t* out_indices,
    float* out_logprobs,
    int32_t* out_top_indices,
    float* out_top_logprobs,
    int64_t rows,
    int64_t vocab_size,
    int64_t top_k,
    uint64_t step_index,
    int64_t threads,
    hipStream_t stream) {
  if (logits == nullptr || temperatures == nullptr || row_seeds == nullptr || out_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || top_k <= 0 || top_k > kMaxSamplerTopK ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const size_t smem = static_cast<size_t>(threads) * static_cast<size_t>(top_k) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      sampler_topk_temperature_rows_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      smem,
      stream,
      logits,
      temperatures,
      row_seeds,
      out_indices,
      out_logprobs,
      out_top_indices,
      out_top_logprobs,
      rows,
      vocab_size,
      static_cast<int32_t>(top_k),
      step_index);
  return static_cast<int>(hipGetLastError());
}
