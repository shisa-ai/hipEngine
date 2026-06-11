// Device-side DFlash chain accept summary for hipEngine.
// Inputs are target-verifier row metadata plus target top-1 ids.  The kernel
// writes compact per-request summaries and committed output-id prefixes without
// copying full logits or doing token-by-token acceptance on the host.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ bool row_belongs_to_root(
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

__global__ void dflash_accept_chain_i32_kernel(
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ parent_rows,
    const int32_t* __restrict__ draft_depths,
    const uint8_t* __restrict__ active_mask,
    const int32_t* __restrict__ target_top1,
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
    int64_t* __restrict__ resident_positions,
    int64_t* __restrict__ resident_contexts,
    int64_t rows,
    int64_t request_count,
    int64_t output_stride) {
  const int64_t request_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (request_index >= request_count) {
    return;
  }

  const int64_t root_row = request_index;
  int64_t row = root_row;
  int32_t accepted = 0;
  int32_t max_depth = 0;
  const int32_t budget = remaining_decode == nullptr ? INT32_MAX : remaining_decode[request_index];
  const bool has_budget_room = budget > 0;
  const int64_t output_base = request_index * output_stride;

  for (int64_t slot = 0; slot < output_stride; ++slot) {
    committed_output_ids[output_base + slot] = -1;
  }
  committed_output_ids[output_base] = token_ids[root_row];

  for (int64_t candidate = request_count; candidate < rows; ++candidate) {
    if (active_mask[candidate] == 0) {
      continue;
    }
    if (!row_belongs_to_root(parent_rows, rows, candidate, root_row)) {
      continue;
    }
    const int32_t depth = draft_depths[candidate];
    if (depth > max_depth) {
      max_depth = depth;
    }
  }

  while (has_budget_room && accepted < budget) {
    const int32_t expected = target_top1[row];
    int64_t matched_child = -1;
    for (int64_t candidate = request_count; candidate < rows; ++candidate) {
      if (active_mask[candidate] == 0 || static_cast<int64_t>(parent_rows[candidate]) != row) {
        continue;
      }
      if (token_ids[candidate] != expected) {
        continue;
      }
      if (!row_belongs_to_root(parent_rows, rows, candidate, root_row)) {
        continue;
      }
      matched_child = candidate;
      break;
    }
    if (matched_child < 0) {
      break;
    }
    row = matched_child;
    ++accepted;
    if (accepted < output_stride) {
      committed_output_ids[output_base + accepted] = token_ids[row];
    }
  }

  accepted_counts[request_index] = accepted;
  commit_rows[request_index] = static_cast<int32_t>(row);
  commit_tokens[request_index] = token_ids[row];
  const int32_t commit_position = positions[row];
  commit_positions[request_index] = commit_position;
  next_tokens[request_index] = (budget >= 0 && accepted >= budget) ? -1 : target_top1[row];
  full_accept[request_index] = accepted == max_depth ? 1 : 0;
  committed_output_lengths[request_index] = accepted + 1;
  if (resident_positions != nullptr && resident_contexts != nullptr) {
    const int64_t position = static_cast<int64_t>(commit_position);
    resident_positions[request_index] = position;
    resident_contexts[request_index] = position + 1;
  }
  if (packed_payload != nullptr) {
    const int64_t payload_base = request_index * 7;
    packed_payload[payload_base + 0] = accepted;
    packed_payload[payload_base + 1] = static_cast<int32_t>(row);
    packed_payload[payload_base + 2] = token_ids[row];
    packed_payload[payload_base + 3] = commit_position;
    packed_payload[payload_base + 4] = next_tokens[request_index];
    packed_payload[payload_base + 5] = static_cast<int32_t>(full_accept[request_index]);
    packed_payload[payload_base + 6] = accepted + 1;
  }
}

bool valid_accept_shape(int64_t rows, int64_t request_count, int64_t output_stride) {
  return rows > 0 && request_count > 0 && request_count <= rows && output_stride > 0;
}

}  // namespace

extern "C" int hipengine_dflash_accept_chain_i32(
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* parent_rows,
    const int32_t* draft_depths,
    const uint8_t* active_mask,
    const int32_t* target_top1,
    const int32_t* remaining_decode,
    int32_t* accepted_counts,
    int32_t* commit_rows,
    int32_t* commit_tokens,
    int32_t* commit_positions,
    int32_t* next_tokens,
    uint8_t* full_accept,
    int32_t* committed_output_ids,
    int32_t* committed_output_lengths,
    int64_t rows,
    int64_t request_count,
    int64_t output_stride,
    hipStream_t stream) {
  if (!valid_accept_shape(rows, request_count, output_stride)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const int blocks = static_cast<int>((request_count + threads - 1) / threads);
  hipLaunchKernelGGL(
      dflash_accept_chain_i32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      token_ids,
      positions,
      parent_rows,
      draft_depths,
      active_mask,
      target_top1,
      remaining_decode,
      accepted_counts,
      commit_rows,
      commit_tokens,
      commit_positions,
      next_tokens,
      full_accept,
      committed_output_ids,
      committed_output_lengths,
      nullptr,
      nullptr,
      nullptr,
      rows,
      request_count,
      output_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_accept_chain_i32_packed(
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* parent_rows,
    const int32_t* draft_depths,
    const uint8_t* active_mask,
    const int32_t* target_top1,
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
    int64_t rows,
    int64_t request_count,
    int64_t output_stride,
    hipStream_t stream) {
  if (!valid_accept_shape(rows, request_count, output_stride) || packed_payload == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const int blocks = static_cast<int>((request_count + threads - 1) / threads);
  hipLaunchKernelGGL(
      dflash_accept_chain_i32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      token_ids,
      positions,
      parent_rows,
      draft_depths,
      active_mask,
      target_top1,
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
      nullptr,
      nullptr,
      rows,
      request_count,
      output_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_accept_chain_i32_packed_update_state(
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* parent_rows,
    const int32_t* draft_depths,
    const uint8_t* active_mask,
    const int32_t* target_top1,
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
    int64_t* resident_positions,
    int64_t* resident_contexts,
    int64_t rows,
    int64_t request_count,
    int64_t output_stride,
    hipStream_t stream) {
  if (!valid_accept_shape(rows, request_count, output_stride) || packed_payload == nullptr ||
      resident_positions == nullptr || resident_contexts == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const int blocks = static_cast<int>((request_count + threads - 1) / threads);
  hipLaunchKernelGGL(
      dflash_accept_chain_i32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      token_ids,
      positions,
      parent_rows,
      draft_depths,
      active_mask,
      target_top1,
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
      resident_positions,
      resident_contexts,
      rows,
      request_count,
      output_stride);
  return static_cast<int>(hipGetLastError());
}
