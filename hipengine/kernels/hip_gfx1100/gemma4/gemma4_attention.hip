// Gemma 4 prefill attention on gfx1100.
//
// Every Qwen3.5 prefill attention variant is *gate-shaped*: it reads an
// attention gate and multiplies the result by `sigmoid(gate)`. Gemma 4 has no
// such gate. Laguna's ungated prefill kernel hard-codes Laguna's head geometry,
// so neither can be reused and this family exists.
//
// Three details of the Gemma 4 attention contract are easy to get wrong:
//
//   * `scale` is always 1.0, never `head_dim**-0.5`. Gemma 4 folds the softmax
//     scaling into the query norm weight, so the kernel multiplies by whatever
//     scale it is handed rather than assuming the usual reciprocal square root.
//   * The mask is a *keep* mask supplied by the caller, not an implied causal
//     test. Sliding-window layers need `key > query - window` in addition to
//     `key <= query`, and which layers those are is a config decision, so the
//     kernel applies the mask and does not re-derive it.
//   * GQA maps query head `h` to key/value head `h / (num_heads / num_kv_heads)`.
//
// This kernel is a correctness-first block kernel: one block per (token, head)
// pair, three passes over the key block (max, denominator, weighted sum). It is
// not the fastest possible shape — a flash-style single-pass kernel with an
// online softmax would cut the key traffic threefold — but it is the one that is
// obviously right, and prefill speed is not the current constraint.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGemma4AttentionMaxThreads = 256;

__device__ inline float gemma4_attn_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_attn_float_to_bf16_bits(float value) {
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
__device__ inline scalar_t gemma4_attn_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_attn_to_scalar<uint16_t>(float value) {
  return gemma4_attn_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_attn_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_attn_to_float<uint16_t>(uint16_t value) {
  return gemma4_attn_bf16_bits_to_float(value);
}

__device__ inline float gemma4_attn_block_max(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return partial[0];
}

__device__ inline float gemma4_attn_block_sum(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return partial[0];
}

// One block per (token, head). Shared layout:
//   q_s      [head_dim]     the query row, loaded once
//   logits_s [tokens]       per-key logits, written in pass 1 and reused
//   partial  [blockDim]     reduction scratch
template <typename scalar_t>
__global__ void gemma4_attention_prefill_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* logits_s = q_s + head_dim;
  float* partial = logits_s + keys;

  const int64_t token = static_cast<int64_t>(blockIdx.x) / num_heads;
  const int64_t head = static_cast<int64_t>(blockIdx.x) - token * num_heads;
  const int64_t kv_groups = num_heads / num_kv_heads;
  const int64_t kv_head = head / kv_groups;

  const scalar_t* q_row = query + (token * num_heads + head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = gemma4_attn_to_float(q_row[d]);
  }
  __syncthreads();

  // The key count is separate from the query count so this same kernel serves
  // both a dense prefill block (keys == tokens) and a decode step reading a
  // KV cache (tokens == 1, keys == live context).
  const uint8_t* mask_row = keep_mask + token * keys;

  // Pass 1: logits and their running maximum. A masked-out key is -inf so it
  // contributes nothing to either the denominator or the numerator.
  float running_max = -INFINITY;
  for (int64_t j = 0; j < keys; ++j) {
    float logit = -INFINITY;
    if (mask_row[j] != 0) {
      const scalar_t* k_row = key + (j * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dot += q_s[d] * gemma4_attn_to_float(k_row[d]);
      }
      logit = gemma4_attn_block_sum(partial, dot) * scale;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      logits_s[j] = logit;
    }
    running_max = fmaxf(running_max, logit);
    __syncthreads();
  }
  const float row_max = gemma4_attn_block_max(partial, running_max);
  __syncthreads();

  // Pass 2: denominator.
  float local_sum = 0.0f;
  for (int64_t j = threadIdx.x; j < keys; j += blockDim.x) {
    local_sum += expf(logits_s[j] - row_max);
  }
  const float denominator = gemma4_attn_block_sum(partial, local_sum);
  __syncthreads();

  // Pass 3: weighted sum of V, then the normalise.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < keys; ++j) {
      const float weight = expf(logits_s[j] - row_max);
      if (weight == 0.0f) {
        continue;
      }
      const scalar_t* v_row = value + (j * num_kv_heads + kv_head) * head_dim;
      acc += weight * gemma4_attn_to_float(v_row[d]);
    }
    out[(token * num_heads + head) * head_dim + d] =
        gemma4_attn_to_scalar<scalar_t>(acc / denominator);
  }
}

template <typename scalar_t>
int launch_gemma4_attention_prefill(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = static_cast<int>(
      head_dim < kGemma4AttentionMaxThreads ? head_dim : kGemma4AttentionMaxThreads);
  const size_t shared_bytes =
      static_cast<size_t>(head_dim + keys + threads) * sizeof(float);
  hipLaunchKernelGGL(
      gemma4_attention_prefill_kernel<scalar_t>,
      dim3(static_cast<unsigned int>(tokens * num_heads)),
      dim3(static_cast<unsigned int>(threads)),
      shared_bytes,
      stream,
      query,
      key,
      value,
      keep_mask,
      out,
      num_heads,
      num_kv_heads,
      head_dim,
      scale,
      tokens,
      keys);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gemma4_attention_prefill_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* value,
    const uint8_t* keep_mask,
    uint16_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_prefill<uint16_t>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}

// F32 entry point for validation against the CPU reference: the reference works
// in f32, so comparing against it through a bf16 round trip would measure the
// rounding rather than the kernel.
extern "C" int hipengine_gemma4_attention_prefill_f32(
    const float* query,
    const float* key,
    const float* value,
    const uint8_t* keep_mask,
    float* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_prefill<float>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}
