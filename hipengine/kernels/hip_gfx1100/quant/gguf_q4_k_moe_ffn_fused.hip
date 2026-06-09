// B1: fused selected-expert GGUF Q4_K MoE FFN megakernel (M16.3).
//
// One block per selected (token, expert) row computes the *whole* expert FFN:
//   gate = Wg_e @ x ; up = Wu_e @ x ; inter = silu(gate) * up ; down = Wd_e @ inter
// keeping the ffn_len-wide intermediate on-chip (LDS) so the gate_up-output HBM
// write and down-input HBM read vanish, and the three big-grid GEMV launches
// collapse to one. Output is the per-selected-row down projection
// ``out[rows, hidden]``; the routing-weighted combine stays a separate kernel.
//
// Structure: thread-owns-output (no cross-thread reduction). Each thread loops
// over the columns it owns and accumulates the full K dot product, so the kernel
// is row-invariant by construction (a row's result is identical alone or batched).
// Q4_K dequant + selected-expert addressing forked from gguf_q4_k_gemv.hip.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
}

// Round an FFN intermediate to the engine's working precision. The bf16 output
// variant rounds gate/up and silu(gate)*up to bf16 so the fused kernel is
// numerically equivalent to the unfused dual_gemv -> silu_mul -> down chain it
// replaces (which materializes those intermediates as bf16). The f32 variant
// keeps full fp32 as the reference path.
template <typename out_t>
__device__ inline float round_intermediate(float value) {
  return value;
}

template <>
__device__ inline float round_intermediate<uint16_t>(float value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
}

__device__ inline uint8_t gguf_q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

// Dequantize a single Q4_K weight element at column k of a weight row.
__device__ inline float gguf_q4_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * 32 + lane];
  const uint8_t q = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const float scale = static_cast<float>(gguf_q4_k_scale(scales, subblock));
  const float minv = static_cast<float>(gguf_q4_k_min(scales, subblock));
  return d * scale * static_cast<float>(q) - dmin * minv;
}

// Partial dot product over one 256-element Q4_K superblock against staged
// activations ``x_block[0..255]``. The fp16 d/dmin header and per-subblock
// scale/min are decoded ONCE per block (vs per element in gguf_q4_k_weight),
// which is the dominant inner-loop cost. Numerically identical to summing
// ``x_block[i] * gguf_q4_k_weight(row, block_idx*256 + i)``.
__device__ inline float gguf_q4_k_block_dot(
    const uint8_t* __restrict__ row, int block_idx, const float* __restrict__ x_block) {
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  float acc = 0.0f;
  for (int sb = 0; sb < 8; ++sb) {
    const float scale = d * static_cast<float>(gguf_q4_k_scale(scales, sb));
    const float minv = dmin * static_cast<float>(gguf_q4_k_min(scales, sb));
    const uint8_t* qsb = qs + (sb >> 1) * 32;
    const bool hi = sb & 1;
    const float* xsb = x_block + sb * 32;
    for (int lane = 0; lane < 32; ++lane) {
      const uint8_t packed = qsb[lane];
      const uint8_t q = hi ? (packed >> 4) : (packed & 0x0Fu);
      acc += xsb[lane] * (scale * static_cast<float>(q) - minv);
    }
  }
  return acc;
}

template <typename scalar_t, typename out_t>
__launch_bounds__(256, 2)
__global__ void gguf_q4_k_selected_ffn_fused_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ up_w,
    const uint8_t* __restrict__ down_w,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int hidden,
    int ffn_len) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * hidden;
  // Inactive / out-of-range expert lanes emit zeros so the downstream combine
  // (routing weight * down) is well defined.
  if (expert_id < 0 || expert_id >= num_experts) {
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
      store_output<out_t>(out_row, c, 0.0f);
    }
    return;
  }

  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const scalar_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * hidden;

  // Dynamic LDS: hidden f32 (staged x) + ffn_len f32 (silu(gate)*up intermediate).
  extern __shared__ float smem[];
  float* x_s = smem;
  float* inter_s = smem + hidden;

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    x_s[i] = scalar_to_float<scalar_t>(x_row_ptr[i]);
  }
  __syncthreads();

  const int gate_row_bytes = (hidden / QK_K) * Q4_K_BLOCK_BYTES;
  const int64_t gate_expert_bytes = static_cast<int64_t>(ffn_len) * gate_row_bytes;
  const uint8_t* __restrict__ gate_e = gate_w + expert_id * gate_expert_bytes;
  const uint8_t* __restrict__ up_e = up_w + expert_id * gate_expert_bytes;
  const int hidden_blocks = hidden / QK_K;
  for (int c = threadIdx.x; c < ffn_len; c += blockDim.x) {
    const uint8_t* __restrict__ wg = gate_e + static_cast<int64_t>(c) * gate_row_bytes;
    const uint8_t* __restrict__ wu = up_e + static_cast<int64_t>(c) * gate_row_bytes;
    float g = 0.0f;
    float u = 0.0f;
    for (int b = 0; b < hidden_blocks; ++b) {
      const float* __restrict__ x_block = x_s + b * QK_K;
      g += gguf_q4_k_block_dot(wg, b, x_block);
      u += gguf_q4_k_block_dot(wu, b, x_block);
    }
    // Match the unfused chain's bf16 working precision (dual_gemv outputs bf16
    // gate/up; silu_mul materializes a bf16 intermediate with expf).
    const float gb = round_intermediate<out_t>(g);
    const float ub = round_intermediate<out_t>(u);
    const float silu = gb / (1.0f + expf(-gb));
    inter_s[c] = round_intermediate<out_t>(silu * ub);
  }
  __syncthreads();

  const int down_row_bytes = (ffn_len / QK_K) * Q4_K_BLOCK_BYTES;
  const int64_t down_expert_bytes = static_cast<int64_t>(hidden) * down_row_bytes;
  const uint8_t* __restrict__ down_e = down_w + expert_id * down_expert_bytes;
  const int ffn_blocks = ffn_len / QK_K;
  for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
    const uint8_t* __restrict__ wd = down_e + static_cast<int64_t>(c) * down_row_bytes;
    float acc = 0.0f;
    for (int b = 0; b < ffn_blocks; ++b) {
      acc += gguf_q4_k_block_dot(wd, b, inter_s + b * QK_K);
    }
    store_output<out_t>(out_row, c, acc);
  }
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_selected_ffn_fused_out(
    const scalar_t* x,
    const int64_t* selected,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    const uint8_t* down_w,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t hidden,
    int64_t ffn_len,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_w == nullptr || up_w == nullptr ||
      down_w == nullptr || out == nullptr || x_rows <= 0 || rows <= 0 ||
      (rows % x_rows) != 0 || num_experts <= 0 || hidden <= 0 || ffn_len <= 0 ||
      (hidden % QK_K) != 0 || (ffn_len % QK_K) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(rows));
  const size_t shared_bytes =
      static_cast<size_t>(hidden + ffn_len) * sizeof(float);
  gguf_q4_k_selected_ffn_fused_kernel<scalar_t, out_t><<<grid, block, shared_bytes, stream>>>(
      x,
      selected,
      gate_w,
      up_w,
      down_w,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(hidden),
      static_cast<int>(ffn_len));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
    const void* x,
    const void* selected,
    const void* gate_w,
    const void* up_w,
    const void* down_w,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t hidden,
    int64_t ffn_len,
    int64_t threads,
    void* stream) {
  return launch_gguf_q4_k_selected_ffn_fused_out<uint16_t, uint16_t>(
      static_cast<const uint16_t*>(x),
      static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(gate_w),
      static_cast<const uint8_t*>(up_w),
      static_cast<const uint8_t*>(down_w),
      static_cast<uint16_t*>(out),
      x_rows,
      rows,
      num_experts,
      hidden,
      ffn_len,
      threads,
      static_cast<hipStream_t>(stream));
}

extern "C" int hipengine_gguf_q4_k_selected_ffn_fused_f32_f32_out(
    const void* x,
    const void* selected,
    const void* gate_w,
    const void* up_w,
    const void* down_w,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t hidden,
    int64_t ffn_len,
    int64_t threads,
    void* stream) {
  return launch_gguf_q4_k_selected_ffn_fused_out<float, float>(
      static_cast<const float*>(x),
      static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(gate_w),
      static_cast<const uint8_t*>(up_w),
      static_cast<const uint8_t*>(down_w),
      static_cast<float*>(out),
      x_rows,
      rows,
      num_experts,
      hidden,
      ffn_len,
      threads,
      static_cast<hipStream_t>(stream));
}
