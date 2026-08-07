// Moonshine fused FP16 LM-head projection + stable argmax for ``sm_120a``.
//
// C1f candidate: a single bounded pass over the tied ``[vocab, hidden]`` FP16
// weight stream that computes each row's FP16 logit with exactly the ordered
// FP32 accumulation of the C1c plain 256-thread ``moonshine_f16_lm_head_projection_kernel``
// (so the fused result is byte-identical to the two-step projection+argmax
// baseline), then tracks a stable running best with the lowest-index tie break
// of ``moonshine_argmax_fp16``.  Stage 1 emits only per-block partial maxima;
// stage 2 reduces them.  No full logit plane is materialized.
//
// Scratch is bounded and caller-owned: ``num_blocks`` (value, index) partials
// plus the final (index, value) pair.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace {

using half_t = __half;

// Byte-identical block reduction to the C1c projection baseline.  The warp
// butterfly order and the cross-warp ``partial[8]`` serial sum must not change.
__device__ inline float moonshine_block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
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

// Lowest index wins on FP16-visible ties, matching ``moonshine_stable_argmax``
// and the ``better_pair`` tie break used by ``moonshine_argmax_fp16``.
__device__ inline bool better_pair(float value, int64_t index, float best_value,
                                   int64_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__global__ void moonshine_lm_head_argmax_stage1_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t in_features,
    int64_t vocab_size,
    int64_t rows_per_block) {
  const int64_t block_begin = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  const int64_t block_end = min(block_begin + rows_per_block, vocab_size);
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  for (int64_t column = block_begin; column < block_end; ++column) {
    float accumulator = 0.0f;
    // Same ordered FP32 accumulation as moonshine_f16_lm_head_projection_kernel
    // (256 threads, ascending feature stride).  in_features == 416 fits one
    // stride, so this reproduces the baseline byte-for-byte.
    for (int64_t feature = threadIdx.x; feature < in_features;
         feature += blockDim.x) {
      accumulator += static_cast<float>(input[feature]) *
          static_cast<float>(weight[column * in_features + feature]);
    }
    const float total = moonshine_block_sum(accumulator);
    // Release the shared partial[] before the next row reuses it.
    __syncthreads();
    if (threadIdx.x == 0) {
      const float logit = __half2float(__float2half_rn(total));
      if (better_pair(logit, column, best_value, best_index)) {
        best_value = logit;
        best_index = column;
      }
    }
  }
  if (threadIdx.x == 0) {
    block_values[blockIdx.x] = best_value;
    block_indices[blockIdx.x] = best_index;
  }
}

__global__ void moonshine_lm_head_argmax_final_kernel(
    const float* __restrict__ block_values,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t* __restrict__ out_index,
    float* __restrict__ out_value) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  for (int64_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    const float value = block_values[i];
    const int64_t index = block_indices[i];
    if (better_pair(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best_value;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = values[threadIdx.x + stride];
      const int64_t other_index = indices[threadIdx.x + stride];
      if (better_pair(other_value, other_index, values[threadIdx.x],
                      indices[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out_index = indices[0];
    *out_value = values[0];
  }
}

// ---------------------------------------------------------------------------
// C8 phase-1 static-batch variants (grid-Y == batch row).  Stage 1 processes
// ``rows_per_block`` vocab rows per block for one batch row; stage 2 reduces
// that row's ``num_blocks`` partials.  Every accumulation, block sum, and
// tie-break is byte-identical to the single-row kernel, so the batch result is
// bit-exact vs B sequential single-row calls.
// ---------------------------------------------------------------------------

__global__ void moonshine_lm_head_argmax_batch_stage1_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t in_features,
    int64_t vocab_size,
    int64_t rows_per_block,
    int64_t num_blocks) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t input_offset = row * in_features;
  const int64_t scratch_offset = row * num_blocks;
  const int64_t block_begin = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  const int64_t block_end = min(block_begin + rows_per_block, vocab_size);
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  for (int64_t column = block_begin; column < block_end; ++column) {
    float accumulator = 0.0f;
    for (int64_t feature = threadIdx.x; feature < in_features;
         feature += blockDim.x) {
      accumulator += static_cast<float>(input[input_offset + feature]) *
          static_cast<float>(weight[column * in_features + feature]);
    }
    const float total = moonshine_block_sum(accumulator);
    __syncthreads();
    if (threadIdx.x == 0) {
      const float logit = __half2float(__float2half_rn(total));
      if (better_pair(logit, column, best_value, best_index)) {
        best_value = logit;
        best_index = column;
      }
    }
  }
  if (threadIdx.x == 0) {
    block_values[scratch_offset + blockIdx.x] = best_value;
    block_indices[scratch_offset + blockIdx.x] = best_index;
  }
}

__global__ void moonshine_lm_head_argmax_batch_final_kernel(
    const float* __restrict__ block_values,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t* __restrict__ out_index,
    float* __restrict__ out_value) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t scratch_offset = row * num_blocks;
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  for (int64_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    const float value = block_values[scratch_offset + i];
    const int64_t index = block_indices[scratch_offset + i];
    if (better_pair(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best_value;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = values[threadIdx.x + stride];
      const int64_t other_index = indices[threadIdx.x + stride];
      if (better_pair(other_value, other_index, values[threadIdx.x],
                      indices[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_index[row] = indices[0];
    out_value[row] = values[0];
  }
}

// ---------------------------------------------------------------------------
// C6/RR-8 fused wave8 + stable top-1 candidate.  Each 256-thread block is 8
// warps; warp ``wave`` computes exactly ONE vocab column (``blockIdx.x*8 +
// wave``) with the ``moonshine_f16_lm_head_projection_wave8_kernel`` arithmetic
// (lane-stride 32 FP32 accumulation + warp butterfly, NO cross-warp serial
// ``partial[8]`` sum and NO per-column ``__syncthreads``), then the block does a
// single cross-warp reduction over its 8 columns.  This is the fused form of
// the C6-screened wave8 projection: 8 columns progress in parallel per block
// instead of serially, at the cost of FP32-reassociation-level logit deltas vs
// the exact C1f fused stage (the review requires complete-token + logit-margin
// gates plus a measured per-token win before admission).  The stable
// lowest-index tie break and bounded ``num_blocks`` partial scratch are
// identical to the exact fused path.
// ---------------------------------------------------------------------------

__global__ void moonshine_lm_head_argmax_wave8_stage1_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t in_features,
    int64_t vocab_size) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 8 + wave;
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  if (column < vocab_size) {
    float accumulator = 0.0f;
    // Same lane-stride 32 FP32 accumulation as the wave8 projection kernel.
    for (int64_t feature = lane; feature < in_features; feature += 32) {
      accumulator += static_cast<float>(input[feature]) *
          static_cast<float>(weight[column * in_features + feature]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
      const float logit = __half2float(__float2half_rn(accumulator));
      best_value = logit;
      best_index = column;
    }
  }
  __shared__ float s_values[8];
  __shared__ int64_t s_indices[8];
  if (lane == 0) {
    s_values[wave] = best_value;
    s_indices[wave] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_best_value = -FLT_MAX;
    int64_t block_best_index = INT64_MAX;
    for (int w = 0; w < 8; ++w) {
      if (better_pair(s_values[w], s_indices[w], block_best_value,
                      block_best_index)) {
        block_best_value = s_values[w];
        block_best_index = s_indices[w];
      }
    }
    block_values[blockIdx.x] = block_best_value;
    block_indices[blockIdx.x] = block_best_index;
  }
}

__global__ void moonshine_lm_head_argmax_wave8_batch_stage1_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ weight,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t in_features,
    int64_t vocab_size,
    int64_t num_blocks) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t input_offset = row * in_features;
  const int64_t scratch_offset = row * num_blocks;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * 8 + wave;
  float best_value = -FLT_MAX;
  int64_t best_index = INT64_MAX;
  if (column < vocab_size) {
    float accumulator = 0.0f;
    for (int64_t feature = lane; feature < in_features; feature += 32) {
      accumulator += static_cast<float>(input[input_offset + feature]) *
          static_cast<float>(weight[column * in_features + feature]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
      const float logit = __half2float(__float2half_rn(accumulator));
      best_value = logit;
      best_index = column;
    }
  }
  __shared__ float s_values[8];
  __shared__ int64_t s_indices[8];
  if (lane == 0) {
    s_values[wave] = best_value;
    s_indices[wave] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_best_value = -FLT_MAX;
    int64_t block_best_index = INT64_MAX;
    for (int w = 0; w < 8; ++w) {
      if (better_pair(s_values[w], s_indices[w], block_best_value,
                      block_best_index)) {
        block_best_value = s_values[w];
        block_best_index = s_indices[w];
      }
    }
    block_values[scratch_offset + blockIdx.x] = block_best_value;
    block_indices[scratch_offset + blockIdx.x] = block_best_index;
  }
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16(
    const half_t* input,
    const half_t* weight,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t in_features,
    int64_t vocab_size,
    int64_t rows_per_block,
    cudaStream_t stream) {
  if (in_features <= 0 || vocab_size <= 0 || rows_per_block <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t num_blocks = (vocab_size + rows_per_block - 1) / rows_per_block;
  moonshine_lm_head_argmax_stage1_kernel<<<
      static_cast<unsigned int>(num_blocks), 256u, 0, stream>>>(
      input, weight, block_values, block_indices, in_features, vocab_size,
      rows_per_block);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  moonshine_lm_head_argmax_final_kernel<<<1u, 256u, 0, stream>>>(
      block_values, block_indices, num_blocks, out_index, out_value);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int hipengine_cuda_sm120a_moonshine_lm_head_argmax_batch_fp16(
    const half_t* input,
    const half_t* weight,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t in_features,
    int64_t vocab_size,
    int64_t rows_per_block,
    int64_t batch,
    cudaStream_t stream) {
  if (in_features <= 0 || vocab_size <= 0 || rows_per_block <= 0 ||
      batch <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t num_blocks = (vocab_size + rows_per_block - 1) / rows_per_block;
  moonshine_lm_head_argmax_batch_stage1_kernel<<<
      dim3(static_cast<unsigned int>(num_blocks),
           static_cast<unsigned int>(batch)),
      256u, 0, stream>>>(
      input, weight, block_values, block_indices, in_features, vocab_size,
      rows_per_block, num_blocks);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  moonshine_lm_head_argmax_batch_final_kernel<<<
      dim3(1u, static_cast<unsigned int>(batch)), 256u, 0, stream>>>(
      block_values, block_indices, num_blocks, out_index, out_value);
  return static_cast<int>(cudaGetLastError());
}

// Fused wave8 + stable top-1 (C6/RR-8 candidate).  ``num_blocks`` is fixed at
// ``ceil(vocab_size / 8)`` because each block's 8 warps each own one column
// (no rows-per-block knob).  The caller supplies that many (value, index)
// scratch pairs, i.e. ``lm_head_argmax_wave8_scratch_elements``.
extern "C" int hipengine_cuda_sm120a_moonshine_lm_head_argmax_wave8_fp16(
    const half_t* input,
    const half_t* weight,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t in_features,
    int64_t vocab_size,
    cudaStream_t stream) {
  if (in_features <= 0 || vocab_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t num_blocks = (vocab_size + 7) / 8;
  moonshine_lm_head_argmax_wave8_stage1_kernel<<<
      static_cast<unsigned int>(num_blocks), 256u, 0, stream>>>(
      input, weight, block_values, block_indices, in_features, vocab_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  moonshine_lm_head_argmax_final_kernel<<<1u, 256u, 0, stream>>>(
      block_values, block_indices, num_blocks, out_index, out_value);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int hipengine_cuda_sm120a_moonshine_lm_head_argmax_wave8_batch_fp16(
    const half_t* input,
    const half_t* weight,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t in_features,
    int64_t vocab_size,
    int64_t batch,
    cudaStream_t stream) {
  if (in_features <= 0 || vocab_size <= 0 || batch <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t num_blocks = (vocab_size + 7) / 8;
  moonshine_lm_head_argmax_wave8_batch_stage1_kernel<<<
      dim3(static_cast<unsigned int>(num_blocks),
           static_cast<unsigned int>(batch)),
      256u, 0, stream>>>(
      input, weight, block_values, block_indices, in_features, vocab_size,
      num_blocks);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  moonshine_lm_head_argmax_batch_final_kernel<<<
      dim3(1u, static_cast<unsigned int>(batch)), 256u, 0, stream>>>(
      block_values, block_indices, num_blocks, out_index, out_value);
  return static_cast<int>(cudaGetLastError());
}
