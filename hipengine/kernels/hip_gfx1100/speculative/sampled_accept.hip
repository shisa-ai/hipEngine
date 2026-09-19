// Device-side sampled MTP acceptance for hipEngine.
//
// The argmax accept chain commits on device by comparing each drafted token
// against the target's top-1 id. A sampled row cannot do that: its decision is
// the coupled acceptance `min(1, p(x)/q(x))` followed, on rejection, by a draw
// from `normalize(max(0, p - q))`. Computing that on the host costs one full
// vocabulary logits row per verified prefix, which is what made a sampled cycle
// several times more expensive than the identical greedy cycle.
//
// These kernels do the same decision from device logits:
//
//   * `sampled_accept_row_stats_kernel` reduces each verified row's processed
//     logits to the two scalars a softmax needs (row max and inverse sum);
//   * `sampled_accept_chain_i32_kernel` walks one request's drafted chain with
//     the request's own sampler stream and emits the same bounded accept/commit
//     payload `dflash_accept_chain_i32_native_cycle` emits, so the resident
//     commit path is unchanged.
//
// Only the unfiltered support is handled here (`top_k == 0`, `top_p == 1.0`,
// `min_p == 0.0`, no active token constraint), which is the serving default and
// the case whose law the host oracle in `hipengine/generation/mtp_sampled_accept.py`
// defines. A filtered or constrained row keeps the host route; the caller
// decides that, not this file.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

namespace {

constexpr float kNegInf = -3.4028234663852886e+38F;

__device__ inline uint64_t sampled_splitmix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// One uniform per (seed, step, row, slot) from the same counter-based stream the
// native row sampler uses, so a row's sampled accept draws are reproducible from
// its seed and step index. Slot 0 is the acceptance test, slot 1 the residual or
// bonus sample, in the host oracle's draw order.
__device__ inline float sampled_uniform01(
    uint64_t row_seed,
    uint64_t step_index,
    int64_t row,
    int64_t slot) {
  const uint64_t row_component =
      (static_cast<uint64_t>(row) + 1ull) * 0xBF58476D1CE4E5B9ull;
  const uint64_t step_component =
      (step_index + 1ull) * 0x9E3779B97F4A7C15ull;
  const uint64_t slot_component =
      (static_cast<uint64_t>(slot) + 1ull) * 0x94D049BB133111EBull;
  const uint64_t bits = sampled_splitmix64(
      row_seed ^ row_component ^ step_component ^ slot_component);
  const double unit =
      static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
  return static_cast<float>(unit);
}

__device__ inline bool sampled_row_belongs_to_root(
    const int32_t* __restrict__ parent_rows,
    int64_t rows,
    int64_t row,
    int64_t root) {
  int64_t current = row;
  for (int64_t guard = 0; guard <= rows; ++guard) {
    if (current == root) {
      return true;
    }
    if (current < 0 || current >= rows) {
      return false;
    }
    current = static_cast<int64_t>(parent_rows[current]);
  }
  return false;
}

__global__ void sampled_accept_row_stats_kernel(
    const float* __restrict__ processed_logits,
    const float* __restrict__ temperatures,
    float* __restrict__ row_max,
    float* __restrict__ row_inv_sum,
    int64_t rows,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float temperature = temperatures[row];
  if (!(temperature > 0.0f) || !isfinite(temperature)) {
    // A greedy row has no sampled accept; the caller must not route it here.
    row_max[row] = 0.0f;
    row_inv_sum[row] = 0.0f;
    return;
  }
  const float inv_temperature = 1.0f / temperature;
  const int64_t base = row * vocab_size;

  float local_max = kNegInf;
  for (int64_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = processed_logits[base + index];
    if (isfinite(value)) {
      const float scaled = value * inv_temperature;
      if (scaled > local_max) {
        local_max = scaled;
      }
    }
  }
  __shared__ float shared_max[256];
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      const float other = shared_max[threadIdx.x + stride];
      if (other > shared_max[threadIdx.x]) {
        shared_max[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  const float maximum = shared_max[0];

  float local_sum = 0.0f;
  for (int64_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = processed_logits[base + index];
    if (isfinite(value)) {
      local_sum += expf(value * inv_temperature - maximum);
    }
  }
  __shared__ float shared_sum[256];
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_max[row] = maximum;
    const float total = shared_sum[0];
    row_inv_sum[row] = total > 0.0f && isfinite(total) ? 1.0f / total : 0.0f;
  }
}

// One probability under a row's processed softmax. Non-finite logits carry no
// mass, matching the host candidate filter.
__device__ inline float sampled_row_probability(
    const float* __restrict__ processed_logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum,
    int64_t row,
    int64_t token_id,
    int64_t vocab_size) {
  const float value = processed_logits[row * vocab_size + token_id];
  if (!isfinite(value)) {
    return 0.0f;
  }
  const float inv_temperature = 1.0f / temperatures[row];
  const float probability =
      expf(value * inv_temperature - row_max[row]) * row_inv_sum[row];
  if (!(probability > 0.0f) || !isfinite(probability)) {
    return 0.0f;
  }
  return probability > 1.0f ? 1.0f : probability;
}

// Draw one token from a row's distribution with one uniform, in ascending token
// id order exactly like the host `_sample` cumulative search. ``mass_scale`` is
// the total mass the caller's distribution carries over this scan: 1 for the
// target law itself, and ``1 - p(x)`` for the residual of a rejected draft at
// token ``x``, which the host renormalizes before drawing.
__device__ inline int32_t sampled_draw_from_row(
    const float* __restrict__ processed_logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum,
    int64_t row,
    int64_t vocab_size,
    int32_t excluded_token_id,
    float uniform,
    float mass_scale) {
  const float inv_temperature = 1.0f / temperatures[row];
  const float maximum = row_max[row];
  const float inv_sum = row_inv_sum[row];
  const float threshold = uniform * mass_scale;
  const int64_t base = row * vocab_size;
  float cumulative = 0.0f;
  int32_t last = -1;
  for (int64_t index = 0; index < vocab_size; ++index) {
    if (index == static_cast<int64_t>(excluded_token_id)) {
      continue;
    }
    const float value = processed_logits[base + index];
    if (!isfinite(value)) {
      continue;
    }
    const float probability = expf(value * inv_temperature - maximum) * inv_sum;
    if (!(probability > 0.0f) || !isfinite(probability)) {
      continue;
    }
    last = static_cast<int32_t>(index);
    cumulative += probability;
    if (cumulative > threshold) {
      return static_cast<int32_t>(index);
    }
  }
  return last;
}

__global__ void sampled_accept_chain_i32_kernel(
    const float* __restrict__ processed_logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ row_max,
    const float* __restrict__ row_inv_sum,
    const uint64_t* __restrict__ row_seeds,
    const uint64_t* __restrict__ step_indices,
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ parent_rows,
    const int32_t* __restrict__ draft_depths,
    const uint8_t* __restrict__ active_mask,
    const int32_t* __restrict__ remaining_decode,
    int32_t* __restrict__ accepted_counts,
    int32_t* __restrict__ commit_rows,
    int32_t* __restrict__ commit_tokens,
    int32_t* __restrict__ commit_positions,
    int32_t* __restrict__ next_tokens,
    uint8_t* __restrict__ full_accept,
    int32_t* __restrict__ committed_output_ids,
    int32_t* __restrict__ committed_output_lengths,
    int32_t* __restrict__ packed_payload,
    int32_t* __restrict__ visible_output_ids,
    int32_t* __restrict__ visible_output_lengths,
    int64_t* __restrict__ resident_positions,
    int64_t* __restrict__ resident_contexts,
    int32_t cursor_offset,
    int64_t rows,
    int64_t request_count,
    int64_t output_stride,
    int64_t vocab_size) {
  const int64_t request_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (request_index >= request_count) {
    return;
  }

  const int64_t root_row = request_index;
  const uint64_t row_seed = row_seeds[request_index];
  const uint64_t step_index = step_indices[request_index];
  int64_t row = root_row;
  int32_t accepted = 0;
  int32_t max_depth = 0;
  int32_t emitted = -1;
  const int32_t budget =
      remaining_decode == nullptr ? INT32_MAX : remaining_decode[request_index];
  const bool has_budget_room = budget > 0;
  const int64_t output_base = request_index * output_stride;

  for (int64_t slot = 0; slot < output_stride; ++slot) {
    committed_output_ids[output_base + slot] = -1;
    if (visible_output_ids != nullptr) {
      visible_output_ids[output_base + slot] = -1;
    }
  }
  committed_output_ids[output_base] = token_ids[root_row];

  for (int64_t candidate = request_count; candidate < rows; ++candidate) {
    if (active_mask[candidate] == 0) {
      continue;
    }
    if (!sampled_row_belongs_to_root(parent_rows, rows, candidate, root_row)) {
      continue;
    }
    const int32_t depth = draft_depths[candidate];
    if (depth > max_depth) {
      max_depth = depth;
    }
  }

  while (has_budget_room && accepted + 1 < budget) {
    int64_t matched_child = -1;
    for (int64_t candidate = request_count; candidate < rows; ++candidate) {
      if (active_mask[candidate] == 0) {
        continue;
      }
      if (static_cast<int64_t>(parent_rows[candidate]) != row) {
        continue;
      }
      if (!sampled_row_belongs_to_root(parent_rows, rows, candidate, root_row)) {
        continue;
      }
      matched_child = candidate;
      break;
    }
    if (matched_child < 0) {
      break;
    }
    const int32_t token_id = token_ids[matched_child];
    const float alpha = sampled_row_probability(
        processed_logits,
        temperatures,
        row_max,
        row_inv_sum,
        row,
        token_id,
        vocab_size);
    const float accept_uniform =
        sampled_uniform01(row_seed, step_index, row, 0);
    if (accept_uniform < alpha) {
      row = matched_child;
      ++accepted;
      if (accepted < output_stride) {
        committed_output_ids[output_base + accepted] = token_ids[row];
      }
      if (visible_output_ids != nullptr && accepted <= output_stride) {
        visible_output_ids[output_base + accepted - 1] = token_ids[row];
      }
      continue;
    }
    // Rejection: the residual is this row's law with the drafted token removed,
    // renormalized, drawn with the second uniform of the same step.
    emitted = sampled_draw_from_row(
        processed_logits,
        temperatures,
        row_max,
        row_inv_sum,
        row,
        vocab_size,
        token_id,
        sampled_uniform01(row_seed, step_index, row, 1),
        1.0f - alpha);
    break;
  }

  int32_t next_token = -1;
  if (has_budget_room) {
    if (emitted >= 0) {
      next_token = emitted;
    } else {
      next_token = sampled_draw_from_row(
          processed_logits,
          temperatures,
          row_max,
          row_inv_sum,
          row,
          vocab_size,
          -1,
          sampled_uniform01(row_seed, step_index, row, 1),
          1.0f);
    }
  }

  accepted_counts[request_index] = accepted;
  commit_rows[request_index] = static_cast<int32_t>(row);
  commit_tokens[request_index] = token_ids[row];
  const int32_t commit_position = positions[row];
  commit_positions[request_index] = commit_position;
  next_tokens[request_index] = next_token;
  full_accept[request_index] = accepted == max_depth ? 1 : 0;
  committed_output_lengths[request_index] = accepted + 1;
  const bool append_next = next_token >= 0 && accepted < output_stride;
  if (visible_output_ids != nullptr && append_next) {
    visible_output_ids[output_base + accepted] = next_token;
  }
  if (visible_output_lengths != nullptr) {
    visible_output_lengths[request_index] = accepted + (append_next ? 1 : 0);
  }
  if (resident_positions != nullptr && resident_contexts != nullptr) {
    const int64_t position =
        static_cast<int64_t>(commit_position) + static_cast<int64_t>(cursor_offset);
    resident_positions[request_index] = position;
    resident_contexts[request_index] = position + 1;
  }
  if (packed_payload != nullptr) {
    const int64_t payload_base = request_index * 7;
    packed_payload[payload_base + 0] = accepted;
    packed_payload[payload_base + 1] = static_cast<int32_t>(row);
    packed_payload[payload_base + 2] = token_ids[row];
    packed_payload[payload_base + 3] = commit_position;
    packed_payload[payload_base + 4] = next_token;
    packed_payload[payload_base + 5] = static_cast<int32_t>(full_accept[request_index]);
    packed_payload[payload_base + 6] = accepted + 1;
  }
}

bool valid_sampled_accept_shape(int64_t rows, int64_t request_count, int64_t output_stride) {
  return rows > 0 && request_count > 0 && request_count <= rows && output_stride > 0;
}

}  // namespace

extern "C" int hipengine_sampled_accept_row_stats_f32(
    const float* processed_logits,
    const float* temperatures,
    float* row_max,
    float* row_inv_sum,
    int64_t rows,
    int64_t vocab_size,
    hipStream_t stream) {
  if (processed_logits == nullptr || temperatures == nullptr || row_max == nullptr ||
      row_inv_sum == nullptr) {
    return hipErrorInvalidValue;
  }
  if (rows <= 0 || vocab_size <= 0) {
    return hipErrorInvalidValue;
  }
  const int threads = 256;
  sampled_accept_row_stats_kernel<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
      processed_logits, temperatures, row_max, row_inv_sum, rows, vocab_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_sampled_accept_chain_i32(
    const float* processed_logits,
    const float* temperatures,
    const float* row_max,
    const float* row_inv_sum,
    const uint64_t* row_seeds,
    const uint64_t* step_indices,
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* parent_rows,
    const int32_t* draft_depths,
    const uint8_t* active_mask,
    const int32_t* remaining_decode,
    int32_t* accepted_counts,
    int32_t* commit_rows,
    int32_t* commit_tokens,
    int32_t* commit_positions,
    int32_t* next_tokens,
    uint8_t* full_accept,
    int32_t* committed_output_ids,
    int32_t* committed_output_lengths,
    int32_t* packed_payload,
    int32_t* visible_output_ids,
    int32_t* visible_output_lengths,
    int64_t* resident_positions,
    int64_t* resident_contexts,
    int32_t cursor_offset,
    int64_t rows,
    int64_t request_count,
    int64_t output_stride,
    int64_t vocab_size,
    hipStream_t stream) {
  if (processed_logits == nullptr || temperatures == nullptr || row_max == nullptr ||
      row_inv_sum == nullptr || row_seeds == nullptr || step_indices == nullptr ||
      token_ids == nullptr || positions == nullptr || parent_rows == nullptr ||
      draft_depths == nullptr || active_mask == nullptr || accepted_counts == nullptr ||
      commit_rows == nullptr || commit_tokens == nullptr || commit_positions == nullptr ||
      next_tokens == nullptr || full_accept == nullptr ||
      committed_output_ids == nullptr || committed_output_lengths == nullptr) {
    return hipErrorInvalidValue;
  }
  if (!valid_sampled_accept_shape(rows, request_count, output_stride)) {
    return hipErrorInvalidValue;
  }
  if (vocab_size <= 0 || cursor_offset < 0) {
    return hipErrorInvalidValue;
  }
  const int threads = 1;
  sampled_accept_chain_i32_kernel<<<static_cast<unsigned int>(request_count), threads, 0, stream>>>(
      processed_logits,
      temperatures,
      row_max,
      row_inv_sum,
      row_seeds,
      step_indices,
      token_ids,
      positions,
      parent_rows,
      draft_depths,
      active_mask,
      remaining_decode,
      accepted_counts,
      commit_rows,
      commit_tokens,
      commit_positions,
      next_tokens,
      full_accept,
      committed_output_ids,
      committed_output_lengths,
      packed_payload,
      visible_output_ids,
      visible_output_lengths,
      resident_positions,
      resident_contexts,
      cursor_offset,
      rows,
      request_count,
      output_stride,
      vocab_size);
  return static_cast<int>(hipGetLastError());
}
