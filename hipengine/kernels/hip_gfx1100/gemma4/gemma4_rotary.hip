// Gemma 4 partial rotary.
//
// Gemma 4 rotates with rotate_half over the full head width: element `i` pairs
// with element `i + head_dim // 2`, and HuggingFace builds its table as
// `emb = cat(freqs, freqs)` so both halves of a pair read the same angle. On
// global layers only the first `rope_angles` pairs carry a real inverse
// frequency; every remaining pair gets exactly zero, which makes its cosine one
// and its sine zero and leaves those elements untouched.
//
// This is the same pairing the Qwen3.5 partial rotary kernel uses, and calling
// that one with `rotary_dim = head_dim` would work, but it is FP32-only while
// the attention and KV-write kernels Gemma 4 shares are BF16, and the fused
// head-norm-plus-rotary variant applies the Gemma-2 `1 + weight` norm. Gemma 4
// therefore needs its own entry points.
//
// The rotated elements are NOT a contiguous prefix: they are `[0, rope_angles)`
// together with `[head_dim // 2, head_dim // 2 + rope_angles)`. Passing a
// rotary_dim smaller than head_dim would rotate the wrong pairs.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float gemma4_rotary_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_rotary_float_to_bf16_bits(float value) {
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
__device__ inline scalar_t gemma4_rotary_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_rotary_to_scalar<uint16_t>(float value) {
  return gemma4_rotary_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_rotary_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_rotary_to_float<uint16_t>(uint16_t value) {
  return gemma4_rotary_bf16_bits_to_float(value);
}

// One block per (token, head). `cos`/`sin` carry one row of head_dim angles per
// token, which is the doubled layout described above.
//
// `query` and `key` are separate buffers with their own head counts, so each is
// indexed by its own row stride. A shared `num_q_heads + num_kv_heads` stride
// would read and write past the end of whichever buffer is narrower.
template <typename scalar_t>
__global__ void gemma4_partial_rotary_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    scalar_t* __restrict__ query_out,
    scalar_t* __restrict__ key_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim) {
  const int64_t total_heads = num_q_heads + num_kv_heads;
  const int64_t token = blockIdx.y;
  const int64_t head = blockIdx.x;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t heads_in_buffer = is_query ? num_q_heads : num_kv_heads;
  const int64_t row = (token * heads_in_buffer + local_head) * head_dim;
  const float* table_cos = cos + token * head_dim;
  const float* table_sin = sin + token * head_dim;
  const int64_t half_rotary = rotary_dim / 2;

  const scalar_t* src = is_query ? query : key;
  scalar_t* dst = is_query ? query_out : key_out;
  if (src == nullptr) {
    return;
  }

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = gemma4_rotary_to_float(src[row + dim]);
    if (dim < rotary_dim) {
      const int64_t pair_dim =
          dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = gemma4_rotary_to_float(src[row + pair_dim]);
      const float rotated = dim < half_rotary ? -paired : paired;
      dst[row + dim] = gemma4_rotary_to_scalar<scalar_t>(
          value * table_cos[dim] + rotated * table_sin[dim]);
    } else {
      dst[row + dim] = gemma4_rotary_to_scalar<scalar_t>(value);
    }
  }
  (void)total_heads;
}

// Copy the raw K projection into the V slot for `attention_k_eq_v` layers.
// Gemma 4 global layers have no `attn_v.weight`; V is K before the value norm,
// which is why the copy happens on the unrotated, unnormalized projection.
template <typename scalar_t>
__global__ void gemma4_k_to_v_kernel(
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ value,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    value[idx] = key[idx];
  }
}

}  // namespace

extern "C" int hipengine_gemma4_partial_rotary_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const float* cos,
    const float* sin,
    uint16_t* query_out,
    uint16_t* key_out,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_heads = num_q_heads + num_kv_heads;
  const int threads = head_dim < 256 ? static_cast<int>(head_dim) : 256;
  hipLaunchKernelGGL(
      gemma4_partial_rotary_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(total_heads),
           static_cast<unsigned int>(tokens)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      cos,
      sin,
      query_out,
      key_out,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_partial_rotary_f32(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_out,
    float* key_out,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_heads = num_q_heads + num_kv_heads;
  const int threads = head_dim < 256 ? static_cast<int>(head_dim) : 256;
  hipLaunchKernelGGL(
      gemma4_partial_rotary_kernel<float>,
      dim3(static_cast<unsigned int>(total_heads),
           static_cast<unsigned int>(tokens)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      cos,
      sin,
      query_out,
      key_out,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_k_to_v_bf16(
    const uint16_t* key,
    uint16_t* value,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_k_to_v_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      total);
  return static_cast<int>(hipGetLastError());
}
