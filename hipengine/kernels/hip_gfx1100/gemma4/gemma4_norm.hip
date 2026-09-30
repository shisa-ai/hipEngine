// Gemma 4 norm, router pre-scale, and layer-output family.
//
// Gemma 4's RMSNorm is *not* the Gemma-2 form. HuggingFace's Gemma4RMSNorm
// computes `normed * weight` where the existing qwen35 kernels compute
// `normed * (1 + weight)`. Reusing those would add one to every weight and
// corrupt every norm in the model, so these kernels are separate.
//
// Gemma 4 also builds two norms with `with_scale=False`: the value norm on
// global (full attention) layers, where V = K and no v_proj tensor exists, and
// the router's input norm. Those have no weight tensor at all.
//
// Every weight tensor these kernels read is F32 in the artifact. The norm
// weights, the router scale, the router projection, and the layer scalar are
// all F32; only the projections and the expert stacks are quantized.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGemma4NormThreads = 256;

__device__ inline float gemma4_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

template <typename scalar_t>
__device__ inline scalar_t gemma4_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_to_scalar<uint16_t>(float value) {
  return gemma4_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_to_float<uint16_t>(uint16_t value) {
  return gemma4_bf16_bits_to_float(value);
}

__device__ inline void gemma4_block_sum(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
}

// Plain-weight RMSNorm over a whole row. `weight` is F32 and applied as-is.
template <typename scalar_t>
__global__ void gemma4_rmsnorm_kernel(
    const scalar_t* hidden_states,
    const float* weight,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] =
        gemma4_to_scalar<scalar_t>(value * inv_rms * weight[idx]);
  }
}

// Weightless RMSNorm. Used for the value norm on global layers and for the
// router's input norm, neither of which has a weight tensor.
template <typename scalar_t>
__global__ void gemma4_rmsnorm_weightless_kernel(
    const scalar_t* hidden_states,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>(value * inv_rms);
  }
}

// Multi-output RMSNorm: up to three outputs in one launch (D6 fusion).
//
// The layer runs three normalizations over the SAME residual row (pre-FFN,
// pre-FFN-2, the weightless router norm) and two over different rows (the
// dense and expert branch outputs). At decode, rows == 1, so each of those
// launches is a single 256-thread block whose real work is sub-microsecond —
// the family cost is dispatch, not arithmetic. One block computing the outputs
// sequentially pays one dispatch.
//
// Bit-exactness contract: each output performs the identical reduction tree
// (`gemma4_block_sum` over the same strided loop) and the identical write
// expression as its standalone kernel, so every output equals the chain
// bitwise. The trailing __syncthreads() fences partial[] so the next output's
// block_sum cannot overwrite inv_rms before every thread has consumed it.
// A null weight selects the weightless expression, matching
// gemma4_rmsnorm_weightless_kernel exactly.
template <typename scalar_t>
__global__ void gemma4_multi_rmsnorm_kernel(
    const scalar_t* in0,
    const scalar_t* in1,
    const scalar_t* in2,
    const float* w0,
    const float* w1,
    const float* w2,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    int count,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const scalar_t* inputs[3] = {in0, in1, in2};
  const float* weights[3] = {w0, w1, w2};
  scalar_t* outputs[3] = {out0, out1, out2};
  for (int k = 0; k < count; ++k) {
    const scalar_t* input = inputs[k];
    const float* weight = weights[k];
    scalar_t* out = outputs[k];
    float sumsq = 0.0f;
    for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
      const float value = gemma4_to_float(input[row_offset + idx]);
      sumsq += value * value;
    }
    gemma4_block_sum(partial, sumsq);
    const float inv_rms =
        rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
    for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
      const float value = gemma4_to_float(input[row_offset + idx]);
      if (weight == nullptr) {
        out[row_offset + idx] = gemma4_to_scalar<scalar_t>(value * inv_rms);
      } else {
        out[row_offset + idx] =
            gemma4_to_scalar<scalar_t>(value * inv_rms * weight[idx]);
      }
    }
    __syncthreads();
  }
}

// Per-head plain-weight RMSNorm over `head_dim`, one block per head.
// Gemma 4 applies this to queries and keys, and to keys again for the value on
// global layers. `weight` may be null, which is the weightless case.
template <typename scalar_t>
__global__ void gemma4_head_rmsnorm_kernel(
    const scalar_t* hidden_states,
    const float* weight,
    scalar_t* out,
    float eps,
    int64_t head_dim) {
  extern __shared__ float partial[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_offset = row * head_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(head_dim)) + eps);
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    const float scale = weight == nullptr ? 1.0f : weight[idx];
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>(value * inv_rms * scale);
  }
}

// The router's input transform: weightless RMSNorm followed by
// `scale[i] * hidden_size**-0.5`.
//
// Gemma4TextRouter does exactly this before its projection. Folding the two
// scalars into one multiply keeps the logits kernel reading an already-scaled
// row, so the router needs no kernel of its own.
template <typename scalar_t>
__global__ void gemma4_router_prescale_kernel(
    const scalar_t* hidden_states,
    const float* scale,
    scalar_t* out,
    float eps,
    float root_size,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] =
        gemma4_to_scalar<scalar_t>(value * inv_rms * scale[idx] * root_size);
  }
}

// Top-k selection over one token's logits: iterative block-parallel argmax,
// then softmax over the selected values only. This is the same math as the
// `qwen35_router_select` chain stage -- the body copies
// `qwen35_router_select_from_logits` from `moe/router.hip` (including its
// warp-shuffle fix history) because device helpers are per translation unit
// and the norm family does not include the router family's source.
//
// Shared memory (dynamic): `num_experts * sizeof(float)` at `shared_logits`.
// Shared memory (static): warp-reduction scratch plus `top_vals[16]`.
__device__ void gemma4_router_select_from_logits(
    const float* __restrict__ token_logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    float* __restrict__ shared_logits,
    int64_t token,
    int64_t logits_stride,
    int64_t num_experts,
    int64_t top_k) {
  // Warp-reduction scratch: blockDim.x / warpSize <= 16 for allowed router workgroups.
  __shared__ float warp_max_vals[16];
  __shared__ int32_t warp_max_idxs[16];
  // Top-k buffer for softmax. top_k is asserted <= 16 at the wrapper.
  __shared__ float top_vals[16];

  const int lane_id = static_cast<int>(threadIdx.x) & (warpSize - 1);
  const int warp_id = static_cast<int>(threadIdx.x) / warpSize;
  const int num_warps = static_cast<int>(blockDim.x) / warpSize;

  // Parallel load of this token's logits into shared memory.
  for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    shared_logits[expert] = token_logits[expert];
  }
  __syncthreads();

  // Iterative top-k: each iteration runs a block-parallel argmax, records
  // the winner, and marks that expert's score as -inf so the next
  // iteration finds the next-best.
  for (int64_t pos = 0; pos < top_k; ++pos) {
    // 1) Thread-local argmax across strided experts.
    float my_val = -3.4028234663852886e38f;
    int32_t my_idx = -1;
    for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
      const float v = shared_logits[expert];
      if (v > my_val) {
        my_val = v;
        my_idx = static_cast<int32_t>(expert);
      }
    }
    // 2) Warp-level argmax via shuffle. On gfx1100 CU mode, warpSize==64.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_val = __shfl_down(my_val, offset);
      const int32_t other_idx = __shfl_down(my_idx, offset);
      if (other_val > my_val) {
        my_val = other_val;
        my_idx = other_idx;
      }
    }
    if (lane_id == 0) {
      warp_max_vals[warp_id] = my_val;
      warp_max_idxs[warp_id] = my_idx;
    }
    __syncthreads();
    // 3) First warp reduces the per-warp maxes.
    if (warp_id == 0) {
      float v = (lane_id < num_warps) ? warp_max_vals[lane_id] : -3.4028234663852886e38f;
      int32_t i = (lane_id < num_warps) ? warp_max_idxs[lane_id] : -1;
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        const float other_v = __shfl_down(v, offset);
        const int32_t other_i = __shfl_down(i, offset);
        if (other_v > v) {
          v = other_v;
          i = other_i;
        }
      }
      if (lane_id == 0) {
        top_vals[pos] = v;
        selected[token * top_k + pos] = static_cast<int64_t>(i);
        shared_logits[i] = -3.4028234663852886e38f;
      }
    }
    __syncthreads();
  }

  // Softmax over top_k values. top_k <= 16 so this is trivial; leave it
  // in thread 0 to avoid extra syncs for a cost that no longer matters.
  if (threadIdx.x == 0) {
    const float max_value = top_vals[0];
    float denom = 0.0f;
    for (int64_t pos = 0; pos < top_k; ++pos) {
      denom += expf(top_vals[pos] - max_value);
    }
    denom = fmaxf(denom, 1.0e-20f);
    for (int64_t pos = 0; pos < top_k; ++pos) {
      routing[token * top_k + pos] = expf(top_vals[pos] - max_value) / denom;
    }
  }
}

// Whole Gemma 4 router for one token: weightless prescale, projection,
// top-k softmax, and per-expert scale in a single launch.
//
// The grid is one block per expert row of the projection. Every block
// recomputes the prescaled row into shared memory -- the same arithmetic and
// the same bf16 rounding as `gemma4_router_prescale_kernel`, so the products
// below see exactly the bits the unfused chain's buffer would hold -- reduces
// its own logit, and fences into the global logits buffer. The last block to
// arrive runs the selection (the cooperative-output pattern from
// `moe/router.hip`'s topk kernels) and applies `per_expert_scale` in place.
//
// `selected` doubles as the arrival counter, so the host wrapper zeroes it on
// the stream before the launch; the selection overwrites every int64 it
// touched, and top_k >= 1 guarantees the counter bytes do not survive.
//
// Not bit-identical to the unfused chain: the projection's f32 reduction
// order differs from `qwen35_router_logits_bf16_f32w`'s. The teacher-forced
// campaign gate covers that class of change, exactly as it covers the
// token-tile logits variant in the chain.
template <typename scalar_t>
__global__ void gemma4_router_topk_fused_kernel(
    const scalar_t* __restrict__ hidden,
    const float* __restrict__ scale,
    const float* __restrict__ proj,
    const float* __restrict__ per_expert,
    float* __restrict__ logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    float eps,
    float root_size,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ float shmem[];
  float* prescaled = shmem;                              // [hidden_size]
  float* partial = shmem + hidden_size;                  // [blockDim.x]
  float* shared_logits = partial + blockDim.x;           // [num_experts]
  __shared__ int is_last_block;

  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);

  // Phase 1: weightless norm, learned scale, and hidden_size**-0.5 -- the
  // `gemma4_router_prescale_kernel` arithmetic, rounded to bf16 so the
  // projection consumes the same values the chain's output buffer holds.
  float sumsq = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden[idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden[idx]);
    prescaled[idx] = gemma4_to_float(
        gemma4_to_scalar<uint16_t>(value * inv_rms * scale[idx] * root_size));
  }
  __syncthreads();

  // Phase 2: this expert's logit over the bf16-prescaled row, f32 weight
  // row, 8-wide vectors with a scalar tail, tree-reduced into `partial[0]`.
  const float* weight_row = proj + row * hidden_size;
  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  for (int64_t k = tid * 8; k + 7 < hidden_size; k += vec_stride) {
    acc += prescaled[k + 0] * weight_row[k + 0];
    acc += prescaled[k + 1] * weight_row[k + 1];
    acc += prescaled[k + 2] * weight_row[k + 2];
    acc += prescaled[k + 3] * weight_row[k + 3];
    acc += prescaled[k + 4] * weight_row[k + 4];
    acc += prescaled[k + 5] * weight_row[k + 5];
    acc += prescaled[k + 6] * weight_row[k + 6];
    acc += prescaled[k + 7] * weight_row[k + 7];
  }
  for (int64_t k = (hidden_size & ~7) + tid; k < hidden_size; k += blockDim.x) {
    acc += prescaled[k] * weight_row[k];
  }
  partial[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    logits[row] = partial[0];
    __threadfence();
    int32_t* counter = reinterpret_cast<int32_t*>(selected);
    const int prior = atomicAdd(counter, 1);
    is_last_block =
        (prior + 1 == static_cast<int32_t>(num_experts)) ? 1 : 0;
  }
  __syncthreads();
  if (is_last_block == 0) {
    return;
  }

  // Phase 3 (last block): select, renormalize, and scale by the selected
  // experts' per-expert scale -- the router_select and expert_weight_scale
  // chain stages over token 0.
  gemma4_router_select_from_logits(
      logits, selected, routing, shared_logits, 0, num_experts, num_experts, top_k);
  if (tid == 0) {
    for (int64_t pos = 0; pos < top_k; ++pos) {
      routing[pos] *= per_expert[selected[pos]];
    }
  }
}

// `out = (residual + rmsnorm(x, weight)) * layer_scalar`.
//
// This is the tail of a Gemma 4 decoder layer: the post-feedforward norm, the
// residual add, and the trained per-layer output scale, which is a real scalar
// (0.0703125 on layer 0, 0.1953125 on the last) rather than 1.
template <typename scalar_t>
__global__ void gemma4_add_rmsnorm_scale_kernel(
    const scalar_t* hidden_states,
    const scalar_t* residual,
    const float* weight,
    const float* layer_scalar,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const float scalar = layer_scalar == nullptr ? 1.0f : layer_scalar[0];
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    const float normed = value * inv_rms * weight[idx];
    const float base = gemma4_to_float(residual[row_offset + idx]);
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>((base + normed) * scalar);
  }
}

// D6 tail fold: post_ffw_norm_1 -> branch_add -> add_rmsnorm_scale in ONE
// launch.
//
// The three standalone kernels are strictly sequential on the main stream and
// each intermediate has exactly one consumer (dense -> branch_add -> the tail
// norm), so one block keeps them in registers at decode. The stage arithmetic
// mirrors the standalone kernels exactly, including the bf16 rounding of the
// intermediate `dense_norm` and `branch_sum` that the chain's buffers would
// apply: stage 2 computes `to_scalar(to_float(dense_norm) + to_float(expert))`
// where dense_norm is already the rounded scalar_t the norm kernel would have
// stored, and stage 3 accumulates sumsq over those rounded sums in the row
// stride add_rmsnorm_scale uses. Every output bit therefore equals the chain.
//
// The __syncthreads() after inv_rms1 fences partial[]: every thread must have
// consumed the stage-1 reduction before any thread's stage-2 block_sum
// overwrites it.
template <typename scalar_t>
__global__ void gemma4_dense_combine_rmsnorm_scale_kernel(
    const scalar_t* dense,
    const scalar_t* experts,
    const scalar_t* residual,
    const float* dense_weight,
    const float* tail_weight,
    const float* layer_scalar,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const float scalar = layer_scalar == nullptr ? 1.0f : layer_scalar[0];

  // Stage 1: rmsnorm of the dense branch output (post_feedforward_layernorm_1).
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(dense[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms1 =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  __syncthreads();

  // Stage 2: branch add with the chain's bf16 rounding, accumulating the tail
  // norm's sumsq over the rounded sums (the same values add_rmsnorm_scale
  // would read back from the branch_sum buffer).
  float sumsq2 = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(dense[row_offset + idx]);
    const scalar_t dense_norm =
        gemma4_to_scalar<scalar_t>(value * inv_rms1 * dense_weight[idx]);
    const scalar_t branch_sum_bits = gemma4_to_scalar<scalar_t>(
        gemma4_to_float(dense_norm) + gemma4_to_float(experts[row_offset + idx]));
    const float branch_sum = gemma4_to_float(branch_sum_bits);
    sumsq2 += branch_sum * branch_sum;
  }
  gemma4_block_sum(partial, sumsq2);
  const float inv_rms2 =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);

  // Stage 3: (residual + rmsnorm(branch_sum, tail_weight)) * layer_scalar,
  // written to the residual buffer in place.
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(dense[row_offset + idx]);
    const scalar_t dense_norm =
        gemma4_to_scalar<scalar_t>(value * inv_rms1 * dense_weight[idx]);
    const scalar_t branch_sum_bits = gemma4_to_scalar<scalar_t>(
        gemma4_to_float(dense_norm) + gemma4_to_float(experts[row_offset + idx]));
    const float branch_sum = gemma4_to_float(branch_sum_bits);
    const float normed = branch_sum * inv_rms2 * tail_weight[idx];
    const float base = gemma4_to_float(residual[row_offset + idx]);
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>((base + normed) * scalar);
  }
}

// `weights[token, slot] *= per_expert_scale[expert[token, slot]]`.
//
// Gemma4TextRouter applies the per-expert scale after renormalizing the top-k
// weights, so it lands on the selected slots only.
__global__ void gemma4_expert_weight_scale_kernel(
    float* weights,
    const int64_t* selected,
    const float* per_expert_scale,
    int64_t rows,
    int64_t top_k) {
  const int64_t total = rows * top_k;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    weights[idx] *= per_expert_scale[selected[idx]];
  }
}

// `out = a + b` over a flat buffer, used to combine the dense and expert
// branches before their shared post-norm.
template <typename scalar_t>
__global__ void gemma4_branch_add_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = gemma4_to_scalar<scalar_t>(
        gemma4_to_float(a[idx]) + gemma4_to_float(b[idx]));
  }
}

// Elementwise `out = x * scale` over a whole buffer.
//
// Gemma 4 scales its embedding by sqrt(hidden_size). That scale cannot be folded
// into the following RMSNorm even though a norm would not see it, because the
// scaled embedding *is* the residual stream: the attention and MLP outputs are
// added back to it, so dropping the scale leaves the residual wrong by exactly
// that factor.
template <typename scalar_t>
__global__ void gemma4_scale_kernel(
    const scalar_t* x,
    scalar_t* out,
    int64_t total,
    float scale) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = gemma4_to_scalar<scalar_t>(gemma4_to_float(x[idx]) * scale);
  }
}

// Splits one fused q/k/v projection output -- a row-major
// [rows, q_width + (parts - 1) * kv_width] buffer -- into the separate q, k
// and v buffers the attention path already reads. P6 fuses the three projections
// into one resident weight and one gemma4_project launch; this is the
// device-side split that keeps every consumer on its existing ABI, which is
// exactly what route (b) of #41 chose over widening the attention, norm,
// rotary and cache-append signatures.
//
// parts == 2 covers k_eq_v layers, whose artifact carries no attn_v: the
// third region is simply never addressed and ``v`` stays untouched. The
// launcher rejects any stride that disagrees with (parts, q_width, kv_width)
// so a mis-sized fused buffer cannot silently land in the wrong array.
__global__ void gemma4_qkv_split_bf16_kernel(
    const uint16_t* fused,
    int64_t stride,
    int64_t rows,
    int64_t q_width,
    int64_t kv_width,
    int parts,
    uint16_t* q,
    uint16_t* k,
    uint16_t* v) {
  const int64_t total = rows * stride;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = idx / stride;
    const int64_t col = idx - row * stride;
    const uint16_t value = fused[idx];
    if (col < q_width) {
      q[row * q_width + col] = value;
    } else {
      const int64_t kv_col = col - q_width;
      if (kv_col < kv_width) {
        k[row * kv_width + kv_col] = value;
      } else if (parts == 3) {
        v[row * kv_width + kv_col - kv_width] = value;
      }
    }
  }
}

}  // namespace

extern "C" int hipengine_gemma4_qkv_split_bf16(
    const uint16_t* fused,
    int64_t stride,
    int64_t rows,
    int64_t q_width,
    int64_t kv_width,
    int parts,
    uint16_t* q,
    uint16_t* k,
    uint16_t* v,
    hipStream_t stream) {
  if (rows <= 0 || q_width <= 0 || kv_width <= 0 ||
      (parts != 2 && parts != 3) ||
      stride != q_width + static_cast<int64_t>(parts - 1) * kv_width) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * stride;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_qkv_split_bf16_kernel,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      fused,
      stride,
      rows,
      q_width,
      kv_width,
      parts,
      q,
      k,
      v);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_rmsnorm_f32w_bf16(
    const uint16_t* hidden_states,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_rmsnorm_f32w_f32(
    const float* hidden_states,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_kernel<float>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_rmsnorm_weightless_bf16(
    const uint16_t* hidden_states,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_weightless_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_multi_rmsnorm_bf16(
    const uint16_t* in0,
    const uint16_t* in1,
    const uint16_t* in2,
    const float* w0,
    const float* w1,
    const float* w2,
    uint16_t* out0,
    uint16_t* out1,
    uint16_t* out2,
    int count,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || count < 1 || count > 3 ||
      in0 == nullptr || out0 == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (count >= 2 && (in1 == nullptr || out1 == nullptr)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (count == 3 && (in2 == nullptr || out2 == nullptr)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_multi_rmsnorm_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      in0,
      in1,
      in2,
      w0,
      w1,
      w2,
      out0,
      out1,
      out2,
      count,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_head_rmsnorm_f32w_bf16(
    const uint16_t* hidden_states,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_head_rmsnorm_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_router_prescale_bf16(
    const uint16_t* hidden_states,
    const float* scale,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    float root_size,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_router_prescale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      scale,
      out,
      eps,
      root_size,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_router_topk_fused_bf16(
    const uint16_t* hidden,
    const float* scale,
    const float* proj,
    const float* per_expert,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    float eps,
    float root_size,
    hipStream_t stream) {
  if (hidden_size <= 0 || num_experts <= 0 || top_k <= 0 || top_k > 16) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shmem =
      (static_cast<size_t>(hidden_size) + kGemma4NormThreads +
       static_cast<size_t>(num_experts)) * sizeof(float);
  if (shmem > 49152) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_router_topk_fused_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(kGemma4NormThreads),
      shmem,
      stream,
      hidden,
      scale,
      proj,
      per_expert,
      logits,
      selected,
      routing,
      eps,
      root_size,
      hidden_size,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_add_rmsnorm_scale_bf16(
    const uint16_t* hidden_states,
    const uint16_t* residual,
    const float* weight,
    const float* layer_scalar,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_add_rmsnorm_scale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      residual,
      weight,
      layer_scalar,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_dense_combine_rmsnorm_scale_bf16(
    const uint16_t* dense,
    const uint16_t* experts,
    const uint16_t* residual,
    const float* dense_weight,
    const float* tail_weight,
    const float* layer_scalar,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || dense == nullptr || experts == nullptr ||
      residual == nullptr || dense_weight == nullptr || tail_weight == nullptr ||
      out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_dense_combine_rmsnorm_scale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      dense,
      experts,
      residual,
      dense_weight,
      tail_weight,
      layer_scalar,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_expert_weight_scale_f32(
    float* weights,
    const int64_t* selected,
    const float* per_expert_scale,
    int64_t rows,
    int64_t top_k,
    hipStream_t stream) {
  if (rows <= 0 || top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * top_k;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(
      gemma4_expert_weight_scale_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      weights,
      selected,
      per_expert_scale,
      rows,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_branch_add_bf16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_branch_add_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      a,
      b,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_scale_bf16(
    const uint16_t* x,
    uint16_t* out,
    int64_t total,
    float scale,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_scale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      x,
      out,
      total,
      scale);
  return static_cast<int>(hipGetLastError());
}

// Final-logit softapping: x = tanh(x / cap) * cap, elementwise and in place.
// The host expression this replaces (numpy's np.tanh over the full vocab,
// ``Gemma4Runner._collect_logits``) measured 0.528 ms of decode host time per
// step — about a third of the measured host wall gap (X7 row). Applying it
// here, before the D2H copy of the logits, moves that work onto the device.
// tanhf rounds independently of numpy's libm, so the Python-side gate pins
// the contract: two f32 ulps of the capped range elementwise, greedy argmax
// position exact on unique tops, tie sets preserved (tests/
// test_gpu_gemma4_softcap_kernel.py).
__global__ void gemma4_logit_softcap_f32_kernel(
    float* x,
    int64_t total,
    float cap) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    x[idx] = tanhf(x[idx] / cap) * cap;
  }
}

extern "C" int hipengine_gemma4_logit_softcap_f32(
    float* x,
    int64_t total,
    float cap,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_logit_softcap_f32_kernel,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      x,
      total,
      cap);
  return static_cast<int>(hipGetLastError());
}

// Greedy argmax over the (optionally capped) logits: emit the index and the
// winning value so the greedy route transfers 12 bytes instead of the full
// 1 MB vocab copy (D10 first half). Semantics are np.argmax's exactly, since
// that is what Gemma4Runner.next_token runs on the host path today: the FIRST
// maximum wins, and a NaN beats every finite value with the first NaN winning
// among NaNs (np.argmax returns the first NaN's index). The ordering below is
// piecewise-total -- {NaN group, lowest index} above {value desc, index asc}
// -- so any reduction tree yields the same winner as a left-to-right scan.
__device__ inline bool gemma4_argmax_beats(
    float candidate_value,
    int64_t candidate_index,
    float incumbent_value,
    int64_t incumbent_index) {
  const bool candidate_nan = isnan(candidate_value);
  const bool incumbent_nan = isnan(incumbent_value);
  if (candidate_nan != incumbent_nan) {
    return candidate_nan;
  }
  if (candidate_nan) {
    return candidate_index < incumbent_index;
  }
  if (candidate_value > incumbent_value) {
    return true;
  }
  if (candidate_value < incumbent_value) {
    return false;
  }
  // Equal (covers +0/-0): the first index wins, like np.argmax.
  return candidate_index < incumbent_index;
}

__global__ void gemma4_logit_argmax_stage1_kernel(
    const float* x,
    int64_t total,
    float* partial_values,
    int64_t* partial_indices) {
  extern __shared__ float shared_values[];  // [threads] values + [threads] idx
  int64_t* shared_indices =
      reinterpret_cast<int64_t*>(shared_values + blockDim.x);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool have = false;
  float best_value = 0.0f;
  int64_t best_index = -1;
  for (; i < total; i += stride) {
    const float value = x[i];
    if (!have) {
      best_value = value;
      best_index = i;
      have = true;
    } else if (gemma4_argmax_beats(value, i, best_value, best_index)) {
      best_value = value;
      best_index = i;
    }
  }

  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;  // -1 marks "no candidate"
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const int64_t other_index = shared_indices[threadIdx.x + offset];
      const int64_t my_index = shared_indices[threadIdx.x];
      if (other_index >= 0 &&
          (my_index < 0 ||
           gemma4_argmax_beats(
               shared_values[threadIdx.x + offset],
               other_index,
               shared_values[threadIdx.x],
               my_index))) {
        shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_values[blockIdx.x] = shared_values[0];
    partial_indices[blockIdx.x] = shared_indices[0];
  }
}

__global__ void gemma4_logit_argmax_stage2_kernel(
    const float* partial_values,
    const int64_t* partial_indices,
    int64_t partial_count,
    void* out) {
  extern __shared__ float shared_values[];
  int64_t* shared_indices =
      reinterpret_cast<int64_t*>(shared_values + blockDim.x);

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool have = false;
  float best_value = 0.0f;
  int64_t best_index = -1;
  for (; i < partial_count; i += stride) {
    const int64_t candidate_index = partial_indices[i];
    if (candidate_index < 0) {
      continue;  // empty stage-1 lane
    }
    const float candidate_value = partial_values[i];
    if (!have) {
      best_value = candidate_value;
      best_index = candidate_index;
      have = true;
    } else if (gemma4_argmax_beats(
                   candidate_value, candidate_index, best_value, best_index)) {
      best_value = candidate_value;
      best_index = candidate_index;
    }
  }

  shared_values[threadIdx.x] = best_value;
  shared_indices[threadIdx.x] = best_index;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const int64_t other_index = shared_indices[threadIdx.x + offset];
      const int64_t my_index = shared_indices[threadIdx.x];
      if (other_index >= 0 &&
          (my_index < 0 ||
           gemma4_argmax_beats(
               shared_values[threadIdx.x + offset],
               other_index,
               shared_values[threadIdx.x],
               my_index))) {
        shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // out layout: bytes 0..7 int64 index, bytes 8..11 f32 value.
    reinterpret_cast<int64_t*>(out)[0] = shared_indices[0];
    reinterpret_cast<float*>(reinterpret_cast<int64_t*>(out) + 1)[0] =
        shared_values[0];
  }
}

// (const float* x, int64 total, void* scratch, int64 scratch_blocks,
//  void* out, stream): out must hold 16 bytes; scratch holds
// scratch_blocks floats followed by scratch_blocks int64s (size it with
// gemma4_logit_argmax_scratch_bytes in Python). Enqueue order makes this
// safe to run right after the softcap on the same logits buffer.
extern "C" int hipengine_gemma4_logit_argmax_f32(
    const float* x,
    int64_t total,
    void* scratch,
    int64_t scratch_blocks,
    void* out,
    hipStream_t stream) {
  if (total <= 0 || out == nullptr || scratch == nullptr ||
      scratch_blocks <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (blocks > scratch_blocks) {
    // Fewer blocks just means each thread strides further; never write past
    // the scratch the caller declared.
    blocks = scratch_blocks;
  }
  float* partial_values = static_cast<float*>(scratch);
  int64_t* partial_indices =
      reinterpret_cast<int64_t*>(static_cast<float*>(scratch) + blocks);
  const unsigned int shared_bytes =
      static_cast<unsigned int>(threads) * (sizeof(float) + sizeof(int64_t));
  hipLaunchKernelGGL(
      gemma4_logit_argmax_stage1_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      shared_bytes,
      stream,
      x,
      total,
      partial_values,
      partial_indices);
  const hipError_t first = hipGetLastError();
  if (first != hipSuccess) {
    return static_cast<int>(first);
  }
  // One block reduces the partials; shared memory holds them all.
  hipLaunchKernelGGL(
      gemma4_logit_argmax_stage2_kernel,
      dim3(1),
      dim3(threads),
      shared_bytes,
      stream,
      partial_values,
      partial_indices,
      blocks,
      out);
  return static_cast<int>(hipGetLastError());
}
