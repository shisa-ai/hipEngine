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

__device__ inline void write_sampler_commit(
    int64_t* __restrict__ out_indices_i64,
    float* __restrict__ out_values,
    int64_t row,
    int32_t index,
    float value) {
  if (out_indices_i64 != nullptr) {
    out_indices_i64[row] = static_cast<int64_t>(index);
  }
  if (out_values != nullptr) {
    out_values[row] = value;
  }
}

__device__ inline void write_sampler_top_logprob_empty(
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t top_out_base,
    int32_t top_logprobs) {
  if (out_top_indices == nullptr || out_top_logprobs == nullptr) {
    return;
  }
  for (int32_t pos = 0; pos < top_logprobs; ++pos) {
    out_top_indices[top_out_base + pos] = -1;
    out_top_logprobs[top_out_base + pos] = kNegInf;
  }
}

__device__ inline void write_sampler_top_logprob_argmax(
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t top_out_base,
    int32_t top_logprobs,
    int32_t index) {
  if (out_top_indices == nullptr || out_top_logprobs == nullptr) {
    return;
  }
  for (int32_t pos = 0; pos < top_logprobs; ++pos) {
    out_top_indices[top_out_base + pos] = pos == 0 ? index : -1;
    out_top_logprobs[top_out_base + pos] = pos == 0 ? 0.0f : kNegInf;
  }
}

__global__ void sampler_topk_temperature_rows_i32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ temperatures,
    const uint64_t* __restrict__ row_seeds,
    const float* __restrict__ top_ps,
    const float* __restrict__ min_ps,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_logprobs,
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t* __restrict__ out_indices_i64_commit,
    float* __restrict__ out_values_f32_commit,
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
    write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, -1, kNegInf);
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
    write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, final_indices[0], final_values[0]);
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
    int retained_count = top_count;
    const float top_p = top_ps != nullptr ? top_ps[row] : 1.0f;
    if (top_p < 1.0f) {
      if (top_p <= 0.0f) {
        retained_count = 1;
      } else {
        const float top_p_target = top_p * weight_sum;
        float cumulative_full = 0.0f;
        retained_count = top_count;
        for (int pos = 0; pos < top_count; ++pos) {
          if (final_indices[pos] == INT32_MAX) {
            retained_count = pos;
            break;
          }
          cumulative_full += candidate_weights[pos];
          if (cumulative_full >= top_p_target) {
            retained_count = pos + 1;
            break;
          }
        }
      }
    }
    if (retained_count <= 0) {
      retained_count = 1;
    }
    const float min_p = min_ps != nullptr ? min_ps[row] : 0.0f;
    if (min_p > 0.0f) {
      int min_p_count = 0;
      for (int pos = 0; pos < retained_count; ++pos) {
        if (final_indices[pos] == INT32_MAX) {
          break;
        }
        if (candidate_weights[pos] >= min_p) {
          ++min_p_count;
        }
      }
      retained_count = min_p_count > 0 ? min_p_count : 1;
    }

    float retained_weight_sum = 0.0f;
    for (int pos = 0; pos < retained_count; ++pos) {
      if (final_indices[pos] == INT32_MAX) {
        retained_count = pos;
        break;
      }
      retained_weight_sum += candidate_weights[pos];
    }
    if (retained_count <= 0 || !(retained_weight_sum > 0.0f) || !isfinite(retained_weight_sum)) {
      candidate_logprobs[0] = 0.0f;
      selected_pos = 0;
    } else {
      const float log_denom = logf(retained_weight_sum) + max_scaled;
      for (int pos = 0; pos < retained_count; ++pos) {
        candidate_logprobs[pos] = final_values[pos] / temperature - log_denom;
      }
      const float threshold = sampler_uniform01(row_seeds[row], step_index, row) * retained_weight_sum;
      float cumulative = 0.0f;
      selected_pos = retained_count - 1;
      for (int pos = 0; pos < retained_count; ++pos) {
        cumulative += candidate_weights[pos];
        if (threshold <= cumulative) {
          selected_pos = pos;
          break;
        }
      }
    }
    for (int pos = retained_count; pos < top_count; ++pos) {
      if (final_indices[pos] == INT32_MAX) {
        break;
      }
      candidate_logprobs[pos] = kNegInf;
    }
  } else {
    candidate_logprobs[0] = 0.0f;
    selected_pos = 0;
  }

  out_indices[row] = final_indices[selected_pos];
  if (out_logprobs != nullptr) {
    out_logprobs[row] = candidate_logprobs[selected_pos];
  }
  write_sampler_commit(
      out_indices_i64_commit, out_values_f32_commit, row, final_indices[selected_pos], final_values[selected_pos]);
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

__global__ void sampler_temperature_rows_i32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ temperatures,
    const uint64_t* __restrict__ row_seeds,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_logprobs,
    int64_t* __restrict__ out_indices_i64_commit,
    float* __restrict__ out_values_f32_commit,
    int64_t rows,
    int64_t vocab_size,
    uint64_t step_index) {
  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(shared_values + blockDim.x);

  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t logits_base = row * vocab_size;

  float local_max = kNegInf;
  int32_t local_index = INT32_MAX;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[logits_base + i];
    if (isfinite(value) && better_pair_i32(value, static_cast<int32_t>(i), local_max, local_index)) {
      local_max = value;
      local_index = static_cast<int32_t>(i);
    }
  }
  shared_values[tid] = local_max;
  shared_indices[tid] = local_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair_i32(
            shared_values[tid + stride], shared_indices[tid + stride], shared_values[tid], shared_indices[tid])) {
      shared_values[tid] = shared_values[tid + stride];
      shared_indices[tid] = shared_indices[tid + stride];
    }
    __syncthreads();
  }
  const float row_max = shared_values[0];
  const int32_t row_argmax = shared_indices[0];
  if (row_argmax == INT32_MAX) {
    if (tid == 0) {
      out_indices[row] = -1;
      if (out_logprobs != nullptr) {
        out_logprobs[row] = kNegInf;
      }
      write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, -1, kNegInf);
    }
    return;
  }

  const float temperature = temperatures[row];
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    if (tid == 0) {
      out_indices[row] = row_argmax;
      if (out_logprobs != nullptr) {
        out_logprobs[row] = 0.0f;
      }
      write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, row_argmax, row_max);
    }
    return;
  }

  float local_sum = 0.0f;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[logits_base + i];
    if (isfinite(value)) {
      local_sum += expf(value / temperature - row_max / temperature);
    }
  }
  shared_values[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_values[tid] += shared_values[tid + stride];
    }
    __syncthreads();
  }
  const float weight_sum = shared_values[0];
  if (tid != 0) {
    return;
  }
  if (!(weight_sum > 0.0f) || !isfinite(weight_sum)) {
    out_indices[row] = row_argmax;
    if (out_logprobs != nullptr) {
      out_logprobs[row] = 0.0f;
    }
    write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, row_argmax, row_max);
    return;
  }

  const float max_scaled = row_max / temperature;
  const float threshold = sampler_uniform01(row_seeds[row], step_index, row) * weight_sum;
  float cumulative = 0.0f;
  int32_t selected_index = row_argmax;
  float selected_logit = row_max;
  for (int64_t i = 0; i < vocab_size; ++i) {
    const float value = logits[logits_base + i];
    if (!isfinite(value)) {
      continue;
    }
    cumulative += expf(value / temperature - max_scaled);
    if (threshold <= cumulative) {
      selected_index = static_cast<int32_t>(i);
      selected_logit = value;
      break;
    }
  }
  out_indices[row] = selected_index;
  if (out_logprobs != nullptr) {
    out_logprobs[row] = selected_logit / temperature - (logf(weight_sum) + max_scaled);
  }
  write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, selected_index, selected_logit);
}

__global__ void sampler_temperature_top_logprobs_rows_i32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ temperatures,
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t rows,
    int64_t vocab_size,
    int32_t top_logprobs) {
  extern __shared__ unsigned char shared_raw[];
  const int top_count = min(top_logprobs, static_cast<int32_t>(vocab_size));
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices =
      reinterpret_cast<int32_t*>(shared_values + static_cast<int64_t>(blockDim.x) * top_count);
  float* shared_sums =
      reinterpret_cast<float*>(shared_indices + static_cast<int64_t>(blockDim.x) * top_count);

  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t base = row * vocab_size;
  const int64_t top_out_base = row * static_cast<int64_t>(top_logprobs);

  float local_values[kMaxSamplerTopK];
  int32_t local_indices[kMaxSamplerTopK];
  for (int pos = 0; pos < kMaxSamplerTopK; ++pos) {
    local_values[pos] = kNegInf;
    local_indices[pos] = INT32_MAX;
  }
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[base + i];
    if (isfinite(value)) {
      insert_topk_pair(value, static_cast<int32_t>(i), local_values, local_indices, top_count);
    }
  }
  for (int pos = 0; pos < top_count; ++pos) {
    const int64_t off = static_cast<int64_t>(tid) * top_count + pos;
    shared_values[off] = local_values[pos];
    shared_indices[off] = local_indices[pos];
  }
  __syncthreads();

  float final_values[kMaxSamplerTopK];
  int32_t final_indices[kMaxSamplerTopK];
  for (int pos = 0; pos < kMaxSamplerTopK; ++pos) {
    final_values[pos] = kNegInf;
    final_indices[pos] = INT32_MAX;
  }
  if (tid == 0) {
    for (int thread = 0; thread < blockDim.x; ++thread) {
      for (int pos = 0; pos < top_count; ++pos) {
        const int64_t off = static_cast<int64_t>(thread) * top_count + pos;
        if (shared_indices[off] != INT32_MAX) {
          insert_topk_pair(shared_values[off], shared_indices[off], final_values, final_indices, top_count);
        }
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int pos = 0; pos < top_count; ++pos) {
      shared_values[pos] = final_values[pos];
      shared_indices[pos] = final_indices[pos];
    }
  }
  __syncthreads();

  if (shared_indices[0] == INT32_MAX) {
    if (tid == 0) {
      write_sampler_top_logprob_empty(out_top_indices, out_top_logprobs, top_out_base, top_logprobs);
    }
    return;
  }

  const float temperature = temperatures[row];
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    if (tid == 0) {
      write_sampler_top_logprob_argmax(
          out_top_indices, out_top_logprobs, top_out_base, top_logprobs, shared_indices[0]);
    }
    return;
  }

  const float max_scaled = shared_values[0] / temperature;
  float local_sum = 0.0f;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[base + i];
    if (isfinite(value)) {
      local_sum += expf(value / temperature - max_scaled);
    }
  }
  shared_sums[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    __syncthreads();
  }
  const float weight_sum = shared_sums[0];
  if (tid != 0) {
    return;
  }
  if (!(weight_sum > 0.0f) || !isfinite(weight_sum)) {
    write_sampler_top_logprob_argmax(
        out_top_indices, out_top_logprobs, top_out_base, top_logprobs, shared_indices[0]);
    return;
  }
  const float log_denom = logf(weight_sum) + max_scaled;
  for (int pos = 0; pos < top_logprobs; ++pos) {
    const bool valid = pos < top_count && shared_indices[pos] != INT32_MAX;
    out_top_indices[top_out_base + pos] = valid ? shared_indices[pos] : -1;
    out_top_logprobs[top_out_base + pos] = valid ? shared_values[pos] / temperature - log_denom : kNegInf;
  }
}

__device__ inline bool sorted_after_previous(float value, int32_t index, bool has_previous, float prev_value, int32_t prev_index) {
  if (!has_previous) {
    return true;
  }
  return value < prev_value || (value == prev_value && index > prev_index);
}

__device__ inline bool find_next_sorted_token(
    const float* __restrict__ logits,
    int64_t base,
    int64_t vocab_size,
    bool has_previous,
    float prev_value,
    int32_t prev_index,
    float* out_value,
    int32_t* out_index) {
  float best_value = kNegInf;
  int32_t best_index = INT32_MAX;
  for (int64_t i = 0; i < vocab_size; ++i) {
    const float value = logits[base + i];
    const int32_t index = static_cast<int32_t>(i);
    if (isfinite(value) && sorted_after_previous(value, index, has_previous, prev_value, prev_index) &&
        better_pair_i32(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  if (best_index == INT32_MAX) {
    return false;
  }
  *out_value = best_value;
  *out_index = best_index;
  return true;
}

__global__ void sampler_top_p_temperature_rows_i32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ top_ps,
    const float* __restrict__ min_ps,
    const uint64_t* __restrict__ row_seeds,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_logprobs,
    int32_t* __restrict__ out_candidate_counts,
    int32_t* __restrict__ out_top_indices,
    float* __restrict__ out_top_logprobs,
    int64_t* __restrict__ out_indices_i64_commit,
    float* __restrict__ out_values_f32_commit,
    int64_t rows,
    int64_t vocab_size,
    int32_t top_logprobs,
    uint64_t step_index) {
  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(shared_values + blockDim.x);

  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t base = row * vocab_size;
  const int64_t top_out_base = row * static_cast<int64_t>(top_logprobs);

  float local_max = kNegInf;
  int32_t local_index = INT32_MAX;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[base + i];
    if (isfinite(value) && better_pair_i32(value, static_cast<int32_t>(i), local_max, local_index)) {
      local_max = value;
      local_index = static_cast<int32_t>(i);
    }
  }
  shared_values[tid] = local_max;
  shared_indices[tid] = local_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair_i32(
            shared_values[tid + stride], shared_indices[tid + stride], shared_values[tid], shared_indices[tid])) {
      shared_values[tid] = shared_values[tid + stride];
      shared_indices[tid] = shared_indices[tid + stride];
    }
    __syncthreads();
  }
  const float row_max = shared_values[0];
  const int32_t row_argmax = shared_indices[0];
  if (row_argmax == INT32_MAX) {
    if (tid == 0) {
      out_indices[row] = -1;
      if (out_logprobs != nullptr) {
        out_logprobs[row] = kNegInf;
      }
      if (out_candidate_counts != nullptr) {
        out_candidate_counts[row] = 0;
      }
      write_sampler_top_logprob_empty(out_top_indices, out_top_logprobs, top_out_base, top_logprobs);
      write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, -1, kNegInf);
    }
    return;
  }

  const float temperature = temperatures[row];
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    if (tid == 0) {
      out_indices[row] = row_argmax;
      if (out_logprobs != nullptr) {
        out_logprobs[row] = 0.0f;
      }
      if (out_candidate_counts != nullptr) {
        out_candidate_counts[row] = 1;
      }
      write_sampler_top_logprob_argmax(out_top_indices, out_top_logprobs, top_out_base, top_logprobs, row_argmax);
      write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, row_argmax, row_max);
    }
    return;
  }

  const float max_scaled = row_max / temperature;
  float local_sum = 0.0f;
  for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = logits[base + i];
    if (isfinite(value)) {
      local_sum += expf(value / temperature - max_scaled);
    }
  }
  shared_values[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_values[tid] += shared_values[tid + stride];
    }
    __syncthreads();
  }
  const float full_weight_sum = shared_values[0];
  if (tid != 0) {
    return;
  }
  if (!(full_weight_sum > 0.0f) || !isfinite(full_weight_sum)) {
    out_indices[row] = row_argmax;
    if (out_logprobs != nullptr) {
      out_logprobs[row] = 0.0f;
    }
    if (out_candidate_counts != nullptr) {
      out_candidate_counts[row] = 1;
    }
    write_sampler_top_logprob_argmax(out_top_indices, out_top_logprobs, top_out_base, top_logprobs, row_argmax);
    write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, row_argmax, row_max);
    return;
  }

  const float top_p = top_ps[row];
  const float min_p = min_ps[row];
  const float top_p_target = top_p * full_weight_sum;
  bool has_previous = false;
  float prev_value = 0.0f;
  int32_t prev_index = -1;
  float cumulative_full = 0.0f;
  int32_t top_p_count = 0;
  int32_t min_p_count = 0;
  while (true) {
    float value = 0.0f;
    int32_t index = -1;
    if (!find_next_sorted_token(logits, base, vocab_size, has_previous, prev_value, prev_index, &value, &index)) {
      break;
    }
    const float weight = expf(value / temperature - max_scaled);
    ++top_p_count;
    cumulative_full += weight;
    if (!(min_p > 0.0f) || weight >= min_p) {
      ++min_p_count;
    }
    prev_value = value;
    prev_index = index;
    has_previous = true;
    if (top_p <= 0.0f) {
      break;
    }
    if (top_p < 1.0f && cumulative_full >= top_p_target) {
      break;
    }
  }

  int32_t retained_count = (min_p > 0.0f) ? min_p_count : top_p_count;
  if (retained_count <= 0) {
    retained_count = 1;
  }

  has_previous = false;
  prev_value = 0.0f;
  prev_index = -1;
  float retained_weight_sum = 0.0f;
  for (int32_t rank = 0; rank < retained_count; ++rank) {
    float value = 0.0f;
    int32_t index = -1;
    if (!find_next_sorted_token(logits, base, vocab_size, has_previous, prev_value, prev_index, &value, &index)) {
      retained_count = rank;
      break;
    }
    retained_weight_sum += expf(value / temperature - max_scaled);
    prev_value = value;
    prev_index = index;
    has_previous = true;
  }
  if (retained_count <= 0 || !(retained_weight_sum > 0.0f) || !isfinite(retained_weight_sum)) {
    out_indices[row] = row_argmax;
    if (out_logprobs != nullptr) {
      out_logprobs[row] = 0.0f;
    }
    if (out_candidate_counts != nullptr) {
      out_candidate_counts[row] = 1;
    }
    write_sampler_top_logprob_argmax(out_top_indices, out_top_logprobs, top_out_base, top_logprobs, row_argmax);
    write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, row_argmax, row_max);
    return;
  }

  const float threshold = sampler_uniform01(row_seeds[row], step_index, row) * retained_weight_sum;
  has_previous = false;
  prev_value = 0.0f;
  prev_index = -1;
  float cumulative = 0.0f;
  int32_t selected_index = row_argmax;
  float selected_logit = row_max;
  float selected_weight = 1.0f;
  for (int32_t rank = 0; rank < retained_count; ++rank) {
    float value = 0.0f;
    int32_t index = -1;
    if (!find_next_sorted_token(logits, base, vocab_size, has_previous, prev_value, prev_index, &value, &index)) {
      break;
    }
    const float weight = expf(value / temperature - max_scaled);
    cumulative += weight;
    selected_index = index;
    selected_logit = value;
    selected_weight = weight;
    if (threshold <= cumulative) {
      break;
    }
    prev_value = value;
    prev_index = index;
    has_previous = true;
  }

  out_indices[row] = selected_index;
  if (out_logprobs != nullptr) {
    out_logprobs[row] = logf(selected_weight) - logf(retained_weight_sum);
  }
  if (out_candidate_counts != nullptr) {
    out_candidate_counts[row] = retained_count;
  }
  if (out_top_indices != nullptr && out_top_logprobs != nullptr) {
    const float log_denom = logf(retained_weight_sum) + max_scaled;
    has_previous = false;
    prev_value = 0.0f;
    prev_index = -1;
    for (int pos = 0; pos < top_logprobs; ++pos) {
      float value = 0.0f;
      int32_t index = -1;
      const bool valid =
          pos < retained_count && find_next_sorted_token(logits, base, vocab_size, has_previous, prev_value, prev_index, &value, &index);
      out_top_indices[top_out_base + pos] = valid ? index : -1;
      out_top_logprobs[top_out_base + pos] = valid ? value / temperature - log_denom : kNegInf;
      if (valid) {
        prev_value = value;
        prev_index = index;
        has_previous = true;
      }
    }
  }
  write_sampler_commit(out_indices_i64_commit, out_values_f32_commit, row, selected_index, selected_logit);
}

__global__ void sampler_apply_processors_rows_kernel(
    const float* __restrict__ logits,
    float* __restrict__ processed,
    const int32_t* __restrict__ bias_offsets,
    const int32_t* __restrict__ bias_token_ids,
    const float* __restrict__ bias_values,
    const int32_t* __restrict__ history_offsets,
    const int32_t* __restrict__ history_token_ids,
    const int32_t* __restrict__ history_counts,
    const float* __restrict__ repetition_penalties,
    const float* __restrict__ presence_penalties,
    const float* __restrict__ frequency_penalties,
    const int32_t* __restrict__ suppress_offsets,
    const int32_t* __restrict__ suppress_token_ids,
    const int32_t* __restrict__ min_tokens,
    const int32_t* __restrict__ eos_token_ids,
    const uint64_t* __restrict__ step_indices,
    int64_t rows,
    int64_t vocab_size) {
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t base = row * vocab_size;
  for (int64_t token = tid; token < vocab_size; token += blockDim.x) {
    const float value = logits[base + token];
    processed[base + token] = isfinite(value) ? value : -INFINITY;
  }
  __syncthreads();

  const int32_t bias_begin = bias_offsets[row];
  const int32_t bias_end = bias_offsets[row + 1];
  for (int32_t item = bias_begin + tid; item < bias_end; item += blockDim.x) {
    const int32_t token = bias_token_ids[item];
    if (token >= 0 && static_cast<int64_t>(token) < vocab_size) {
      processed[base + token] += bias_values[item];
    }
  }
  __syncthreads();

  const float repetition_penalty = repetition_penalties[row];
  const float presence_penalty = presence_penalties[row];
  const float frequency_penalty = frequency_penalties[row];
  const int32_t history_begin = history_offsets[row];
  const int32_t history_end = history_offsets[row + 1];
  for (int32_t item = history_begin + tid; item < history_end; item += blockDim.x) {
    const int32_t token = history_token_ids[item];
    if (token < 0 || static_cast<int64_t>(token) >= vocab_size) {
      continue;
    }
    float value = processed[base + token];
    if (repetition_penalty != 1.0f) {
      if (value < 0.0f) {
        value *= repetition_penalty;
      } else {
        value /= repetition_penalty;
      }
    }
    if (presence_penalty != 0.0f) {
      value -= presence_penalty;
    }
    if (frequency_penalty != 0.0f) {
      value -= frequency_penalty * static_cast<float>(history_counts[item]);
    }
    processed[base + token] = value;
  }
  __syncthreads();

  if (suppress_offsets != nullptr && suppress_token_ids != nullptr) {
    const int32_t suppress_begin = suppress_offsets[row];
    const int32_t suppress_end = suppress_offsets[row + 1];
    for (int32_t item = suppress_begin + tid; item < suppress_end; item += blockDim.x) {
      const int32_t token = suppress_token_ids[item];
      if (token >= 0 && static_cast<int64_t>(token) < vocab_size) {
        processed[base + token] = -INFINITY;
      }
    }
  }

  if (tid == 0 && min_tokens != nullptr && eos_token_ids != nullptr && step_indices != nullptr) {
    const int32_t min_token_count = min_tokens[row];
    const int32_t eos_token = eos_token_ids[row];
    if (min_token_count > 0 && step_indices[row] < static_cast<uint64_t>(min_token_count) && eos_token >= 0 &&
        static_cast<int64_t>(eos_token) < vocab_size) {
      processed[base + eos_token] = -INFINITY;
    }
  }
}

// Original in-tree full-vocabulary ordering. No bounded candidate approximation.
// 256-key tile bitonic sort followed by parallel binary-search merges (O(V log²V)).
// Scratch is caller-owned and every consumed element is overwritten per launch.
constexpr int kSamplerTile = 256;

__device__ inline uint64_t sampler_order_key(float value, uint32_t id) {
  if (!isfinite(value)) return UINT64_MAX;
  if (value == 0.0f) value = 0.0f;  // +/- zero have token-ID ties.
  uint32_t bits = __float_as_uint(value);
  uint32_t ascending = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
  return (static_cast<uint64_t>(~ascending) << 32) | id;
}

__global__ void sampler_sort_tiles_kernel(const float* logits, uint64_t* keys, int64_t vocab) {
  __shared__ uint64_t tile[kSamplerTile];
  const int t = threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kSamplerTile + t;
  const int64_t base = static_cast<int64_t>(blockIdx.y) * vocab;
  uint64_t key = i < vocab ? sampler_order_key(logits[base + i], i) : UINT64_MAX;
  tile[t] = key;
  __syncthreads();
  for (int size = 2; size <= kSamplerTile; size *= 2) {
    for (int stride = size / 2; stride; stride /= 2) {
      uint64_t other = tile[t ^ stride];
      bool take_min = ((t & size) == 0) == ((t & stride) == 0);
      key = take_min ? (key < other ? key : other) : (key > other ? key : other);
      __syncthreads();
      tile[t] = key;
      __syncthreads();
    }
  }
  if (i < vocab) keys[base + i] = key;
}

__global__ void sampler_merge_keys_kernel(const uint64_t* src, uint64_t* dst, int64_t vocab, int64_t width) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= vocab) return;
  int64_t base = static_cast<int64_t>(blockIdx.y) * vocab;
  int64_t pair = (i / (2 * width)) * (2 * width);
  bool right = i - pair >= width;
  int64_t own = right ? pair + width : pair;
  int64_t other = right ? pair : min(pair + width, vocab);
  int64_t end = min(other + width, vocab);
  int64_t lo = other, hi = end;
  uint64_t key = src[base + i];
  while (lo < hi) {
    int64_t mid = (lo + hi) / 2;
    uint64_t probe = src[base + mid];
    if (probe < key || (right && probe == key)) lo = mid + 1;
    else hi = mid;
  }
  dst[base + pair + (i - own) + (lo - other)] = key;
}

__device__ inline double sampler_centered_score(float value, float maximum, float temperature) {
  // Widen BEFORE subtraction and divide AFTER centering. Every finite FP32
  // logit difference / positive FP32 temperature fits FP64, including opposite
  // FLT_MAX endpoints and subnormal temperatures. Equal maxima stay exactly 0.
  return (static_cast<double>(value) - static_cast<double>(maximum)) /
      static_cast<double>(temperature);
}

__device__ inline float sampler_sorted_weight(const float* logits, const uint64_t* keys,
    int64_t base, int64_t rank, float temperature, float maximum) {
  uint64_t key = keys[base + rank];
  return key == UINT64_MAX ? 0.0f : expf(static_cast<float>(
      sampler_centered_score(logits[base + static_cast<uint32_t>(key)], maximum, temperature)));
}

__global__ void sampler_weight_scan_tiles_kernel(const float* logits, const uint64_t* keys,
    const float* temperatures, double* prefix, double* tile_totals, int64_t vocab, int64_t tiles) {
  __shared__ double scan[kSamplerTile];
  int t = threadIdx.x;
  int64_t row = blockIdx.y, base = row * vocab;
  int64_t i = static_cast<int64_t>(blockIdx.x) * kSamplerTile + t;
  float temp = temperatures[row];
  uint64_t first = keys[base];
  float maximum = first == UINT64_MAX ? 0.0f : logits[base + static_cast<uint32_t>(first)];
  double value = i < vocab && temp > 0.0f && isfinite(temp)
      ? sampler_sorted_weight(logits, keys, base, i, temp, maximum) : 0.0;
  scan[t] = value;
  __syncthreads();
  for (int stride = 1; stride < kSamplerTile; stride *= 2) {
    double add = t >= stride ? scan[t - stride] : 0.0;
    __syncthreads();
    value += add;
    scan[t] = value;
    __syncthreads();
  }
  if (i < vocab) prefix[base + i] = value;
  if (t == kSamplerTile - 1) tile_totals[row * tiles + blockIdx.x] = value;
}

__device__ inline double sampler_prefix_at(const double* prefix, const double* tiles, int64_t rank) {
  int64_t tile = rank / kSamplerTile;
  return prefix[rank] + (tile ? tiles[tile - 1] : 0.0);
}

__device__ inline int64_t sampler_prefix_lower_bound(const double* prefix, const double* tiles,
    int64_t count, double target) {
  int64_t lo = 0, hi = count - 1;
  while (lo < hi) {
    int64_t mid = (lo + hi) / 2;
    if (sampler_prefix_at(prefix, tiles, mid) < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__global__ void sampler_sorted_select_kernel(const float* logits, const uint64_t* keys,
    const float* temperatures, const float* top_ps, const float* min_ps, const uint64_t* seeds,
    const double* prefix, double* tile_totals, int32_t* out_indices, float* out_logprobs,
    int32_t* out_counts, int32_t* out_top_indices, float* out_top_logprobs,
    int64_t* out_commit_indices, float* out_commit_values, int64_t vocab, int64_t tiles,
    int top_logprobs, uint64_t step) {
  __shared__ double scan[kSamplerTile];
  __shared__ double carry;
  int t = threadIdx.x;
  int64_t row = blockIdx.x, base = row * vocab, top_base = row * top_logprobs;
  const uint64_t* row_keys = keys + base;
  const double* row_prefix = prefix + base;
  double* row_tiles = tile_totals + row * tiles;
  if (t == 0) carry = 0.0;
  __syncthreads();
  // Scan tile totals cooperatively too: a serial 970-load chain costs more
  // than the weight scan at model vocabulary sizes on gfx1151.
  for (int64_t start = 0; start < tiles; start += kSamplerTile) {
    int64_t index = start + t;
    double value = index < tiles ? row_tiles[index] : 0.0;
    scan[t] = value;
    __syncthreads();
    for (int stride = 1; stride < kSamplerTile; stride *= 2) {
      double add = t >= stride ? scan[t - stride] : 0.0;
      __syncthreads();
      value += add;
      scan[t] = value;
      __syncthreads();
    }
    if (index < tiles) row_tiles[index] = carry + value;
    __syncthreads();
    if (t == 0) carry += scan[kSamplerTile - 1];
    __syncthreads();
  }
  if (t != 0) return;
  double total = carry;
  if (row_keys[0] == UINT64_MAX) {
    out_indices[row] = -1;
    if (out_logprobs) out_logprobs[row] = kNegInf;
    if (out_counts) out_counts[row] = 0;
    write_sampler_top_logprob_empty(out_top_indices, out_top_logprobs, top_base, top_logprobs);
    write_sampler_commit(out_commit_indices, out_commit_values, row, -1, kNegInf);
    return;
  }
  int32_t argmax = static_cast<uint32_t>(row_keys[0]);
  float temperature = temperatures[row], max_value = logits[base + argmax];
  if (!(temperature > 0.0f) || !isfinite(temperature) || !(total > 0.0) || !isfinite(total)) {
    out_indices[row] = argmax;
    if (out_logprobs) out_logprobs[row] = 0.0f;
    if (out_counts) out_counts[row] = 1;
    write_sampler_top_logprob_argmax(out_top_indices, out_top_logprobs, top_base, top_logprobs, argmax);
    write_sampler_commit(out_commit_indices, out_commit_values, row, argmax, max_value);
    return;
  }
  // Find the finite extent, not a fixed-size candidate list.
  int64_t lo = 0, hi = vocab;
  while (lo < hi) { int64_t mid = (lo + hi) / 2;
    if (row_keys[mid] != UINT64_MAX) lo = mid + 1; else hi = mid; }
  int64_t count = lo;
  float top_p = top_ps[row], min_p = min_ps[row];
  if (top_p <= 0.0f) count = 1;
  else if (top_p < 1.0f) count = sampler_prefix_lower_bound(row_prefix, row_tiles, count, top_p * total) + 1;
  // Contract: nucleus on the full distribution, then min-p, retaining >=1.
  if (min_p > 0.0f) {
    lo = 0; hi = count;
    while (lo < hi) { int64_t mid = (lo + hi) / 2;
      if (sampler_sorted_weight(logits, keys, base, mid, temperature, max_value) >= min_p) lo = mid + 1;
      else hi = mid; }
    count = max(static_cast<int64_t>(1), lo);
  }
  double retained_sum = sampler_prefix_at(row_prefix, row_tiles, count - 1);
  double target = static_cast<double>(sampler_uniform01(seeds[row], step, row)) * retained_sum;
  int64_t rank = sampler_prefix_lower_bound(row_prefix, row_tiles, count, target);
  int32_t selected = static_cast<uint32_t>(row_keys[rank]);
  out_indices[row] = selected;
  if (out_logprobs) out_logprobs[row] =
      sampler_centered_score(logits[base + selected], max_value, temperature) - log(retained_sum);
  if (out_counts) out_counts[row] = count;
  if (out_top_indices && out_top_logprobs) {
    double denom = log(retained_sum);
    for (int pos = 0; pos < top_logprobs; ++pos) {
      bool valid = pos < count;
      int32_t id = valid ? static_cast<uint32_t>(row_keys[pos]) : -1;
      out_top_indices[top_base + pos] = id;
      out_top_logprobs[top_base + pos] = valid ?
          sampler_centered_score(logits[base + id], max_value, temperature) - denom : kNegInf;
    }
  }
  write_sampler_commit(out_commit_indices, out_commit_values, row, selected, logits[base + selected]);
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128;
}

}  // namespace

extern "C" int hipengine_sampler_top_p_temperature_f32_rows_i32(
    const float* logits,
    const float* temperatures,
    const float* top_ps,
    const float* min_ps,
    const uint64_t* row_seeds,
    int32_t* out_indices,
    float* out_logprobs,
    int32_t* out_candidate_counts,
    int32_t* out_top_indices,
    float* out_top_logprobs,
    int64_t* out_indices_i64_commit,
    float* out_values_f32_commit,
    int64_t rows,
    int64_t vocab_size,
    int64_t top_logprobs,
    uint64_t step_index,
    int64_t threads,
    hipStream_t stream) {
  if (logits == nullptr || temperatures == nullptr || top_ps == nullptr || min_ps == nullptr ||
      row_seeds == nullptr || out_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || top_logprobs < 0 ||
      top_logprobs > kMaxSamplerTopK || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      sampler_top_p_temperature_rows_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      smem,
      stream,
      logits,
      temperatures,
      top_ps,
      min_ps,
      row_seeds,
      out_indices,
      out_logprobs,
      out_candidate_counts,
      out_top_indices,
      out_top_logprobs,
      out_indices_i64_commit,
      out_values_f32_commit,
      rows,
      vocab_size,
      static_cast<int32_t>(top_logprobs),
      step_index);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampler_apply_processors_f32_rows(
    const float* logits,
    float* processed,
    const int32_t* bias_offsets,
    const int32_t* bias_token_ids,
    const float* bias_values,
    const int32_t* history_offsets,
    const int32_t* history_token_ids,
    const int32_t* history_counts,
    const float* repetition_penalties,
    const float* presence_penalties,
    const float* frequency_penalties,
    const int32_t* suppress_offsets,
    const int32_t* suppress_token_ids,
    const int32_t* min_tokens,
    const int32_t* eos_token_ids,
    const uint64_t* step_indices,
    int64_t rows,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (logits == nullptr || processed == nullptr || bias_offsets == nullptr || history_offsets == nullptr ||
      repetition_penalties == nullptr || presence_penalties == nullptr || frequency_penalties == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const bool has_min_token_processor = min_tokens != nullptr || eos_token_ids != nullptr || step_indices != nullptr;
  if (has_min_token_processor && (min_tokens == nullptr || eos_token_ids == nullptr || step_indices == nullptr)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      sampler_apply_processors_rows_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      logits,
      processed,
      bias_offsets,
      bias_token_ids,
      bias_values,
      history_offsets,
      history_token_ids,
      history_counts,
      repetition_penalties,
      presence_penalties,
      frequency_penalties,
      suppress_offsets,
      suppress_token_ids,
      min_tokens,
      eos_token_ids,
      step_indices,
      rows,
      vocab_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampler_sorted_f32_rows_i32(
    const float* logits, const float* temperatures, const float* top_ps, const float* min_ps,
    const uint64_t* seeds, int32_t* out_indices, float* out_logprobs, int32_t* out_counts,
    int32_t* out_top_indices, float* out_top_logprobs, int64_t* out_commit_indices,
    float* out_commit_values, void* scratch, uint64_t scratch_bytes,
    int64_t rows, int64_t vocab, int64_t top_logprobs, uint64_t step, hipStream_t stream) {
  if (!logits || !temperatures || !top_ps || !min_ps || !seeds || !out_indices || !scratch ||
      rows <= 0 || rows > 65535 || vocab <= 0 || vocab > INT32_MAX || top_logprobs < 0 ||
      top_logprobs > kMaxSamplerTopK || ((out_top_indices == nullptr) != (out_top_logprobs == nullptr)))
    return static_cast<int>(hipErrorInvalidValue);
  int64_t tiles = (vocab + kSamplerTile - 1) / kSamplerTile;
  uint64_t required = static_cast<uint64_t>(rows) * (24ull * vocab + 8ull * tiles);
  if (scratch_bytes < required || (reinterpret_cast<uintptr_t>(scratch) & 7))
    return static_cast<int>(hipErrorInvalidValue);
  auto* a = static_cast<uint64_t*>(scratch);
  auto* b = a + rows * vocab;
  auto* prefix = reinterpret_cast<double*>(b + rows * vocab);
  auto* totals = prefix + rows * vocab;
  dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(sampler_sort_tiles_kernel, grid, dim3(kSamplerTile), 0, stream, logits, a, vocab);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  for (int64_t width = kSamplerTile; width < vocab; width *= 2) {
    hipLaunchKernelGGL(sampler_merge_keys_kernel, grid, dim3(kSamplerTile), 0, stream, a, b, vocab, width);
    err = hipGetLastError();
    if (err != hipSuccess) return static_cast<int>(err);
    auto* tmp = a; a = b; b = tmp;
  }
  hipLaunchKernelGGL(sampler_weight_scan_tiles_kernel, grid, dim3(kSamplerTile), 0, stream,
      logits, a, temperatures, prefix, totals, vocab, tiles);
  err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  hipLaunchKernelGGL(sampler_sorted_select_kernel, dim3(rows), dim3(kSamplerTile), 0, stream,
      logits, a, temperatures, top_ps, min_ps, seeds, prefix, totals, out_indices, out_logprobs,
      out_counts, out_top_indices, out_top_logprobs, out_commit_indices, out_commit_values,
      vocab, tiles, static_cast<int>(top_logprobs), step);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampler_temperature_f32_rows_i32(
    const float* logits,
    const float* temperatures,
    const uint64_t* row_seeds,
    int32_t* out_indices,
    float* out_logprobs,
    int64_t* out_indices_i64_commit,
    float* out_values_f32_commit,
    int64_t rows,
    int64_t vocab_size,
    uint64_t step_index,
    int64_t threads,
    hipStream_t stream) {
  if (logits == nullptr || temperatures == nullptr || row_seeds == nullptr || out_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const size_t smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      sampler_temperature_rows_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      smem,
      stream,
      logits,
      temperatures,
      row_seeds,
      out_indices,
      out_logprobs,
      out_indices_i64_commit,
      out_values_f32_commit,
      rows,
      vocab_size,
      step_index);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampler_temperature_top_logprobs_f32_rows_i32(
    const float* logits,
    const float* temperatures,
    int32_t* out_top_indices,
    float* out_top_logprobs,
    int64_t rows,
    int64_t vocab_size,
    int64_t top_logprobs,
    int64_t threads,
    hipStream_t stream) {
  if (logits == nullptr || temperatures == nullptr || out_top_indices == nullptr || out_top_logprobs == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || vocab_size <= 0 || vocab_size > INT32_MAX || top_logprobs <= 0 ||
      top_logprobs > kMaxSamplerTopK || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  const size_t top_count = static_cast<size_t>(top_logprobs);
  const size_t smem =
      static_cast<size_t>(threads) * top_count * (sizeof(float) + sizeof(int32_t)) +
      static_cast<size_t>(threads) * sizeof(float);
  hipLaunchKernelGGL(
      sampler_temperature_top_logprobs_rows_i32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(t),
      smem,
      stream,
      logits,
      temperatures,
      out_top_indices,
      out_top_logprobs,
      rows,
      vocab_size,
      static_cast<int32_t>(top_logprobs));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampler_topk_temperature_f32_rows_i32(
    const float* logits,
    const float* temperatures,
    const uint64_t* row_seeds,
    const float* top_ps,
    const float* min_ps,
    int32_t* out_indices,
    float* out_logprobs,
    int32_t* out_top_indices,
    float* out_top_logprobs,
    int64_t* out_indices_i64_commit,
    float* out_values_f32_commit,
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
      top_ps,
      min_ps,
      out_indices,
      out_logprobs,
      out_top_indices,
      out_top_logprobs,
      out_indices_i64_commit,
      out_values_f32_commit,
      rows,
      vocab_size,
      static_cast<int32_t>(top_k),
      step_index);
  return static_cast<int>(hipGetLastError());
}
