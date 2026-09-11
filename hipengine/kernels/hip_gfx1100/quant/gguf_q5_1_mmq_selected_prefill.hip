// Raw Q5_1 selected-MoE MMQ prefill (Q8_1 activations, DP4A dot4) for gfx11.
//
// Consumes compact expert-sorted activation rows prequantized to the
// block_q8_1_mmq_ds4 layout (128-element blocks, four fp16 (scale, sum)
// pairs; the producer is gguf_q8_1_mmq_ds4_pack_bf16) and raw GGUF Q5_1
// expert weights. Per 32-value Q5_1 block with x quantized per 32 lanes:
//   y += d_b * xd_s * dot4(lo_nibble, xq)
//     + 16*d_b * xd_s * dot4(hi_bit, xq)
//     + m_b * xsum_s
// which equals sum w*x for w = (nibble + 16*hi_bit)*d + m with the Q8_1
// scale xd_s applied to the integer dot and the raw block sum xsum_s
// carrying the m offset. Output is BF16. Arithmetic differs from the
// strict float-dequant grouped owner; promotion runs under the production
// gate with the strict grouped chain as fallback.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK5_1 = 32;
constexpr int Q5_1_BLOCK_BYTES = 24;  // 2*fp16 + u32 qh + 16B qs (+ 2B pad)
constexpr int Q8_1_MMQ_BLOCK = 128;

struct block_q8_1_mmq_ds4 {
  uint16_t ds4[8];  // four fp16 (scale, sum) pairs, one per 32 lanes
  int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq_ds4) == 144, "unexpected ds4 block size");

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
  return static_cast<float>(*reinterpret_cast<const half_t*>(&bits));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  uint32_t bits;
  *reinterpret_cast<float*>(&bits) = value;
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding = 0x7FFFu + lsb;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

__device__ __forceinline__ int dot4_u8_s8(uint32_t a, uint32_t b, int acc) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), acc, false);
#else
  int8_t b0 = static_cast<int8_t>(b & 0xFF);
  int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFF);
  int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFF);
  int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFF);
  return acc + static_cast<int>(a & 0xFF) * b0 +
      static_cast<int>((a >> 8) & 0xFF) * b1 +
      static_cast<int>((a >> 16) & 0xFF) * b2 +
      static_cast<int>((a >> 24) & 0xFF) * b3;
#endif
}

__device__ __forceinline__ uint32_t lo_plane_nibbles(uint32_t bytes4) {
  // Low nibble of byte i stays in lane i of the little-endian u32 load.
  return bytes4 & 0x0F0F0F0Fu;
}

__device__ __forceinline__ uint32_t load_qs_bytes(const uint8_t* qs, int w) {
  return static_cast<uint32_t>(qs[4 * w]) |
      (static_cast<uint32_t>(qs[4 * w + 1]) << 8) |
      (static_cast<uint32_t>(qs[4 * w + 2]) << 16) |
      (static_cast<uint32_t>(qs[4 * w + 3]) << 24);
}

__device__ __forceinline__ uint32_t expand_hi_bits(uint32_t bits4) {
  // Four consecutive qh bits -> four 0/1 byte lanes.
  return (bits4 & 1u) | ((bits4 & 2u) << 7) | ((bits4 & 4u) << 14) |
      ((bits4 & 8u) << 21);
}

constexpr int ROWS_PER_PASS = 16;

__global__ __launch_bounds__(128, 4)
void q5_1_mmq_ds4_selected_prefill_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_ds4,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int planes) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) * 128 + threadIdx.x;
  if (out_col >= out_features) return;
  const int64_t ds4_blocks = in_features / Q8_1_MMQ_BLOCK;
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) {
      continue;
    }
    const uint8_t* weight_row = qweight +
        (expert * out_features + out_col) *
        (in_features / QK5_1) * Q5_1_BLOCK_BYTES;
    for (int64_t row_base = start; row_base < end; row_base += ROWS_PER_PASS) {
      const int rcount = static_cast<int>(
          end - row_base < ROWS_PER_PASS ? end - row_base : ROWS_PER_PASS);
      float acc[ROWS_PER_PASS];
#pragma unroll
      for (int r = 0; r < ROWS_PER_PASS; ++r) acc[r] = 0.0f;
      for (int64_t kb = 0; kb < ds4_blocks; ++kb) {
        const uint8_t* w_base =
            weight_row + kb * (Q8_1_MMQ_BLOCK / QK5_1) * Q5_1_BLOCK_BYTES;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
          const uint8_t* w_sub = w_base + sub * Q5_1_BLOCK_BYTES;
          const float d =
              fp16_bits_to_float(*reinterpret_cast<const uint16_t*>(w_sub));
          const float m =
              fp16_bits_to_float(*reinterpret_cast<const uint16_t*>(w_sub + 2));
          const uint32_t qh =
              *reinterpret_cast<const uint32_t*>(w_sub + 4);
          for (int plane = 0; plane < planes; ++plane) {
          const block_q8_1_mmq_ds4* x_base =
              x_ds4 + static_cast<int64_t>(plane) * compact_rows * ds4_blocks +
              row_base * ds4_blocks + kb;
          for (int r = 0; r < rcount; ++r) {
            const block_q8_1_mmq_ds4* x_block = x_base + r * ds4_blocks;
            const int32_t* xw =
                reinterpret_cast<const int32_t*>(x_block->qs + sub * 32);
            int dot_q = 0;
            int dot_h = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              const uint32_t bytes4 = load_qs_bytes(w_sub + 8, w);
              const uint32_t lo = lo_plane_nibbles(bytes4);
              const uint32_t hi = lo_plane_nibbles(bytes4 >> 4);
              const uint32_t x_lo = static_cast<uint32_t>(xw[w]);
              const uint32_t x_hi = static_cast<uint32_t>(xw[4 + w]);
              dot_q = dot4_u8_s8(lo, x_lo, dot_q);
              dot_q = dot4_u8_s8(hi, x_hi, dot_q);
              dot_h = dot4_u8_s8(expand_hi_bits(qh >> (4 * w)), x_lo, dot_h);
              dot_h = dot4_u8_s8(
                  expand_hi_bits(qh >> (16 + 4 * w)), x_hi, dot_h);
            }
            const float xd =
                fp16_bits_to_float(x_block->ds4[2 * sub]);
            const float xs =
                fp16_bits_to_float(x_block->ds4[2 * sub + 1]);
            acc[r] += d * xd * static_cast<float>(dot_q);
            acc[r] += 16.0f * d * xd * static_cast<float>(dot_h);
            // Only plane 0 carries the full block sum; residual planes hold
            // residual sums and must not enter the m offset.
            if (plane == 0) {
              acc[r] += m * xs;
            }
          }
          }
        }
      }
      for (int r = 0; r < rcount; ++r) {
        out[(row_base + r) * out_features + out_col] =
            float_to_bf16_bits(acc[r]);
      }
    }
  }
}

}  // namespace

extern "C" int hipengine_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
    const void* x_ds4,
    const int64_t* expert_start,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t planes,
    hipStream_t stream) {
  if (x_ds4 == nullptr || expert_start == nullptr || qweight == nullptr ||
      out == nullptr || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % Q8_1_MMQ_BLOCK) != 0 ||
      out_features <= 0 || planes <= 0 || planes > 3) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int out_tiles =
      static_cast<unsigned int>((out_features + 127) / 128);
  const unsigned int expert_grid = num_experts > 512u ? 512u
      : static_cast<unsigned int>(num_experts);
  hipLaunchKernelGGL(
      (q5_1_mmq_ds4_selected_prefill_kernel),
      dim3(out_tiles, expert_grid), dim3(128), 0, stream,
      static_cast<const block_q8_1_mmq_ds4*>(x_ds4), expert_start, qweight,
      out, compact_rows, num_experts, in_features, out_features,
      static_cast<int>(planes));
  return static_cast<int>(hipGetLastError());
}
