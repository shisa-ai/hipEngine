// Gemma 4 routed-expert FFN primitives.
//
// Gemma 4's expert block is a fused gate_up projection, a tanh-approximation
// GeGLU, a down projection, and a routing-weighted accumulate:
//
//   projection = gate_up[expert] @ row
//   gate, up   = split(projection, 2)
//   activated  = gelu_tanh(gate) * up
//   out[token] += down[expert] @ activated * weight
//
// The gather/compact machinery, the per-expert GEMV, and the paged attention
// path are shared with other architectures and reused as-is. Only the
// activation and the accumulate are Gemma 4-specific here: the existing
// activation kernels in the tree are SwiGLU (maple) or vision/vibevoice
// one-offs, and `gelu_tanh` is not `silu`, so reusing one would silently change
// the arithmetic.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float gemma4_moe_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_moe_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

// gelu_pytorch_tanh, matching the reference element for element:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gemma4_moe_gelu_tanh(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// Fused GeGLU over a `(rows, 2 * intermediate)` gate_up buffer. The first
// `intermediate` columns are the gate, the second half is up.
__global__ void gemma4_gelu_tanh_mul_kernel(
    const uint16_t* __restrict__ gate_up,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t intermediate) {
  const int64_t total = rows * intermediate;
  const int64_t fused = 2 * intermediate;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = idx / intermediate;
    const int64_t col = idx - row * intermediate;
    const int64_t base = row * fused + col;
    const float gate = gemma4_moe_bf16_bits_to_float(gate_up[base]);
    const float up = gemma4_moe_bf16_bits_to_float(gate_up[base + intermediate]);
    out[idx] = gemma4_moe_float_to_bf16_bits(gemma4_moe_gelu_tanh(gate) * up);
  }
}

// Invert the compact permutation: `lane_to_row[sorted_lanes[row]] = row`.
// Every lane appears exactly once because every selected expert has at least
// one lane, so the compact list is a full permutation of `total_lanes`.
//
// `sorted_lanes` is int64 because that is the width the group-compact kernel
// emits; `lane_to_row` is int32 because it holds row indices, which the
// accumulate kernel consumes.
__global__ void gemma4_moe_lane_to_row_kernel(
    const int64_t* __restrict__ sorted_lanes,
    int32_t* __restrict__ lane_to_row,
    int64_t total_lanes) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < total_lanes;
       row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    lane_to_row[sorted_lanes[row]] = static_cast<int32_t>(row);
  }
}

// Weighted accumulate, one block per token.
//
// Compaction orders rows by expert, so a token's `top_k` lanes land in `top_k`
// different compact rows, usually in different experts. An earlier version of
// this kernel gave each compact row its own block and did a read-modify-write
// on the token row; two lanes of one token then raced on the same output
// elements. Owning a whole token per block removes the race without atomics:
// each column is accumulated in a register by exactly one thread and written
// once.
//
// The accumulation is FP32 across lanes, matching the reference. Each expert's
// down-projection result is BF16 before it is accumulated, which the reference
// keeps in FP32; that is a per-lane rounding difference, not a structural one.
__global__ void gemma4_moe_weighted_accumulate_kernel(
    const uint16_t* __restrict__ expert_out,
    const int32_t* __restrict__ lane_to_row,
    const float* __restrict__ sorted_weights,
    uint16_t* __restrict__ out,
    int64_t tokens,
    int64_t hidden,
    int64_t top_k) {
  const int64_t token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  const int64_t lane_base = token * top_k;
  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    float accumulated = 0.0f;
    for (int64_t slot = 0; slot < top_k; ++slot) {
      const int64_t lane = lane_base + slot;
      const int64_t row = lane_to_row[lane];
      if (row < 0) {
        continue;
      }
      accumulated += gemma4_moe_bf16_bits_to_float(
                         expert_out[row * hidden + col]) *
                     sorted_weights[row];
    }
    out[token * hidden + col] = gemma4_moe_float_to_bf16_bits(accumulated);
  }
}

// Zero the output before the accumulate pass.
__global__ void gemma4_moe_zero_kernel(uint16_t* __restrict__ out, int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = static_cast<uint16_t>(0);
  }
}

}  // namespace

extern "C" int hipengine_gemma4_gelu_tanh_mul_bf16(
    const uint16_t* gate_up,
    uint16_t* out,
    int64_t rows,
    int64_t intermediate,
    hipStream_t stream) {
  if (gate_up == nullptr || out == nullptr || rows <= 0 || intermediate <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * intermediate;
  const int64_t blocks = (total + 255) / 256;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_gelu_tanh_mul_kernel,
      dim3(static_cast<unsigned int>(capped)),
      dim3(256),
      0,
      stream,
      gate_up,
      out,
      rows,
      intermediate);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_moe_zero_bf16(
    uint16_t* out,
    int64_t total,
    hipStream_t stream) {
  if (out == nullptr || total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total + 255) / 256;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_moe_zero_kernel,
      dim3(static_cast<unsigned int>(capped)),
      dim3(256),
      0,
      stream,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_moe_lane_to_row_i32(
    const int64_t* sorted_lanes,
    int32_t* lane_to_row,
    int64_t total_lanes,
    hipStream_t stream) {
  if (sorted_lanes == nullptr || lane_to_row == nullptr || total_lanes <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total_lanes + 255) / 256;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_moe_lane_to_row_kernel,
      dim3(static_cast<unsigned int>(capped)),
      dim3(256),
      0,
      stream,
      sorted_lanes,
      lane_to_row,
      total_lanes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_moe_weighted_accumulate_bf16(
    const uint16_t* expert_out,
    const int32_t* lane_to_row,
    const float* sorted_weights,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden,
    int64_t top_k,
    hipStream_t stream) {
  if (expert_out == nullptr || lane_to_row == nullptr ||
      sorted_weights == nullptr || out == nullptr || tokens <= 0 ||
      hidden <= 0 || top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = hidden < 256 ? static_cast<int>(hidden) : 256;
  hipLaunchKernelGGL(
      gemma4_moe_weighted_accumulate_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(threads),
      0,
      stream,
      expert_out,
      lane_to_row,
      sorted_weights,
      out,
      tokens,
      hidden,
      top_k);
  return static_cast<int>(hipGetLastError());
}
