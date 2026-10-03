// Raw Q8_0 selected-MoE MMQ prefill (Q8_1 activations, DP4A dot4) for gfx11.
//
// The Q8_0 twin of gguf_q5_1_mmq_selected_prefill.hip, for the same reason that
// file exists: an expert down projection whose weight is not the quant the fast
// route accepts has no MMQ path and runs the fp32 grouped family instead. In
// Gemma 4 26B-A4B UD-Q4_K_XL that is exactly one layer of thirty -- blk.29's
// ffn_down_exps is Q8_0 where the other 29 are Q5_1 -- and that one layer's down
// projection measured 43.85 ms of a 873 ms 512-token prefill, 5.0 percent of the
// step, against 5.30 ms for a sibling at the same shape.
//
// Consumes the same compact expert-sorted activation rows prequantized to the
// block_q8_1_mmq_ds4 layout (128-element blocks, four fp16 (scale, sum) pairs;
// the producer is gguf_q8_1_mmq_ds4_pack_bf16) as the Q5_1 route, so the two
// share an activation pack and differ only in how the weight is decoded. Per
// 32-value Q8_0 block with x quantized per 32 lanes:
//
//   y += d_b * xd_s * dot4(qs, xq)
//
// Q8_0 is the simpler of the two: its block carries a single fp16 scale and 32
// signed bytes, with no min term and no fifth bit, so the Q5_1 route's
// ``m_b * xsum_s`` correction and its two-plane nibble/bit decode both drop out.
// The bytes are genuine int8 rather than a 0..15 nibble, so the dot must be
// signed on both operands; ``__builtin_amdgcn_sudot4`` with both negate flags
// set is signed-by-signed (probed: 0xFF * 0xFF returns 1, not 65025). The Q5_1
// route passes the same flags and is unaffected, because its nibbles and its
// expanded fifth bits are non-negative.
//
// Arithmetic differs from the strict float-dequant grouped owner in the same way
// the Q5_1 route's does -- the error is bounded by the activation quantization
// step rather than by reassociation -- so it is held to the same envelope in
// tests/test_unit_gemma4_expert_route.py.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK8_0 = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;  // fp16 d + 32 signed bytes
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

__device__ __forceinline__ int dot4_s8_s8(uint32_t a, uint32_t b, int acc) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), acc, false);
#else
  int8_t a0 = static_cast<int8_t>(a & 0xFF);
  int8_t a1 = static_cast<int8_t>((a >> 8) & 0xFF);
  int8_t a2 = static_cast<int8_t>((a >> 16) & 0xFF);
  int8_t a3 = static_cast<int8_t>((a >> 24) & 0xFF);
  int8_t b0 = static_cast<int8_t>(b & 0xFF);
  int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFF);
  int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFF);
  int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFF);
  return acc + static_cast<int>(a0) * b0 + static_cast<int>(a1) * b1 +
      static_cast<int>(a2) * b2 + static_cast<int>(a3) * b3;
#endif
}

constexpr int ROWS_PER_PASS = 16;

// The DS4 activation block comes in two layouts that differ only in how the
// four (scale, sum) pairs are stored: fp16 for a hidden-state input, and fp32
// for a post-SiLU one. Both put the eight pairs first and the 128 int8 quants
// after them, so the offsets are shared and only the stride and the scale read
// change. The fp16 layout cannot hold a scale above 65504, and a down
// projection's input is a GeGLU output whose magnitude is a property of the
// model rather than of this route, so the fp32 layout is the correct one there.
constexpr int DS4_BLOCK_BYTES = 16 + Q8_1_MMQ_BLOCK;
constexpr int DS4_F32_BLOCK_BYTES = 32 + Q8_1_MMQ_BLOCK;

template <bool F32_SCALES>
__device__ __forceinline__ float x_block_scale(const uint8_t* x_block,
                                               int index) {
  if (F32_SCALES) {
    return reinterpret_cast<const float*>(x_block)[index];
  }
  return fp16_bits_to_float(
      reinterpret_cast<const uint16_t*>(x_block)[index]);
}

template <bool F32_SCALES>
__global__ __launch_bounds__(128, 4)
void q8_0_mmq_ds4_selected_prefill_kernel(
    const uint8_t* __restrict__ x_ds4,
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
  // Round up, exactly as the Q5_1 consumer does: a width that is not a multiple
  // of the 128-wide DS4 block still needs its trailing block read, and flooring
  // here is what silently dropped the last 64 of a 704-wide expert projection.
  const int64_t ds4_blocks = (in_features + Q8_1_MMQ_BLOCK - 1) / Q8_1_MMQ_BLOCK;
  const int64_t x_block_bytes =
      F32_SCALES ? DS4_F32_BLOCK_BYTES : DS4_BLOCK_BYTES;
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) {
      continue;
    }
    const uint8_t* weight_row = qweight +
        (expert * out_features + out_col) * (in_features / QK8_0) *
        Q8_0_BLOCK_BYTES;
    for (int64_t row_base = start; row_base < end; row_base += ROWS_PER_PASS) {
      const int rcount = static_cast<int>(
          end - row_base < ROWS_PER_PASS ? end - row_base : ROWS_PER_PASS);
      float acc[ROWS_PER_PASS];
#pragma unroll
      for (int r = 0; r < ROWS_PER_PASS; ++r) acc[r] = 0.0f;
      for (int64_t kb = 0; kb < ds4_blocks; ++kb) {
        // Q8_0 blocks of real weight data in this 128-wide group. It is 4 for
        // every group but a trailing partial one, and the row's byte layout
        // holds exactly (in_features/32) of them, so reading a fifth would run
        // past the row. Skipping is exact: the dropped sub-blocks multiply
        // activation elements that do not exist, and the pack writes zeros
        // there, so their contribution is zero either way.
        const bool full_group = (kb + 1) * Q8_1_MMQ_BLOCK <= in_features;
        const int64_t sub_blocks =
            (in_features - kb * Q8_1_MMQ_BLOCK + QK8_0 - 1) / QK8_0;
        const uint8_t* w_base =
            weight_row + kb * (Q8_1_MMQ_BLOCK / QK8_0) * Q8_0_BLOCK_BYTES;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
          if (!full_group && sub >= sub_blocks) {
            continue;
          }
          const uint8_t* w_sub = w_base + sub * Q8_0_BLOCK_BYTES;
          const float d =
              fp16_bits_to_float(*reinterpret_cast<const uint16_t*>(w_sub));
          for (int plane = 0; plane < planes; ++plane) {
            const uint8_t* x_base = x_ds4 +
                (static_cast<int64_t>(plane) * compact_rows * ds4_blocks +
                 row_base * ds4_blocks + kb) * x_block_bytes;
            for (int r = 0; r < rcount; ++r) {
              const uint8_t* x_block = x_base + r * ds4_blocks * x_block_bytes;
              const int32_t* xw = reinterpret_cast<const int32_t*>(
                  x_block + (F32_SCALES ? 32 : 16) + sub * 32);
              const int32_t* ww =
                  reinterpret_cast<const int32_t*>(w_sub + 2);
              int dot = 0;
#pragma unroll
              for (int w = 0; w < 8; ++w) {
                dot = dot4_s8_s8(
                    static_cast<uint32_t>(ww[w]),
                    static_cast<uint32_t>(xw[w]), dot);
              }
              const float xd = x_block_scale<F32_SCALES>(x_block, 2 * sub);
              acc[r] += d * xd * static_cast<float>(dot);
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

extern "C" int hipengine_q8_0_mmq_ds4_selected_prefill_bf16_bf16_out(
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
      in_features <= 0 || (in_features % QK8_0) != 0 ||
      out_features <= 0 || planes <= 0 || planes > 3) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int out_tiles =
      static_cast<unsigned int>((out_features + 127) / 128);
  const unsigned int expert_grid = num_experts > 512u ? 512u
      : static_cast<unsigned int>(num_experts);
  hipLaunchKernelGGL(
      (q8_0_mmq_ds4_selected_prefill_kernel<false>),
      dim3(out_tiles, expert_grid), dim3(128), 0, stream,
      static_cast<const uint8_t*>(x_ds4), expert_start, qweight,
      out, compact_rows, num_experts, in_features, out_features,
      static_cast<int>(planes));
  return static_cast<int>(hipGetLastError());
}

// The fp32-activation-scale twin. A down projection's input is a GeGLU output,
// so it is the one that belongs on the expert down route; see DS4_BLOCK_BYTES.
extern "C" int hipengine_q8_0_mmq_ds4_f32_selected_prefill_bf16_bf16_out(
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
      in_features <= 0 || (in_features % QK8_0) != 0 ||
      out_features <= 0 || planes <= 0 || planes > 3) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int out_tiles =
      static_cast<unsigned int>((out_features + 127) / 128);
  const unsigned int expert_grid = num_experts > 512u ? 512u
      : static_cast<unsigned int>(num_experts);
  hipLaunchKernelGGL(
      (q8_0_mmq_ds4_selected_prefill_kernel<true>),
      dim3(out_tiles, expert_grid), dim3(128), 0, stream,
      static_cast<const uint8_t*>(x_ds4), expert_start, qweight,
      out, compact_rows, num_experts, in_features, out_features,
      static_cast<int>(planes));
  return static_cast<int>(hipGetLastError());
}
