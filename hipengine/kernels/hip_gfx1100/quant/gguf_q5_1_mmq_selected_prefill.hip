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
  // Ceil: K need not be a multiple of 128 (the down projection is K=704,
  // i.e. 22 Q5_1 blocks = 5.5 DS4 blocks). The pack zero-fills the trailing
  // lanes of the partial block, so those lanes contribute exactly 0 to dot4,
  // to the block sum and therefore to the m offset.
  const int64_t q51_blocks = in_features / QK5_1;
  const int64_t ds4_blocks =
      (in_features + Q8_1_MMQ_BLOCK - 1) / Q8_1_MMQ_BLOCK;
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
        // A partial final DS4 block has fewer than 4 Q5_1 blocks left in the
        // weight row (704 = 22, so the 6th DS4 block holds only 2); reading
        // past row 22 would be a 48-byte out-of-bounds read.
        const int64_t remaining =
            q51_blocks - kb * (Q8_1_MMQ_BLOCK / QK5_1);
        const int sub_count =
            remaining < 4 ? static_cast<int>(remaining) : 4;
#pragma unroll
        for (int sub = 0; sub < sub_count; ++sub) {
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

__device__ inline int pack_i8x4(const int8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return static_cast<int>(out);
}

// Pack four Q5_1 values starting at k0 into one i32, one value per byte,
// for the int8 matrix core. Q5_1 keeps values 0..15 in the low nibble of
// qs[k] for k < 16 and the high nibble of qs[k-16] for k >= 16, with the
// 5th bit in qh; the value is nibble + 16*hi, unsigned in 0..31, and the
// minimum travels separately as the block's m. The wmma operand therefore
// matches the scalar leaf's split `dot_q + 16*dot_h` exactly.
__device__ inline int q5_1_pack4(const uint8_t* qs, uint32_t qh, int k0) {
  uint32_t out = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k = k0 + j;
    const uint32_t nib = (k < 16)
        ? static_cast<uint32_t>(qs[k] & 0x0F)
        : static_cast<uint32_t>(qs[k - 16] >> 4);
    const uint32_t v = nib | (((qh >> k) & 1u) << 4);
    out |= (v & 0xFFu) << (8 * j);
  }
  return static_cast<int>(out);
}

// Int8 matrix-core down-projection consumer (P10). Same DS4 activation
// layout and same 32-column x 16-row tile, 64-thread workgroup shape as
// gguf_q4_k_q8_1_mmq_prefill.hip's q4_k_q8_1_wmma32_dense_kernel; only the
// weight-side pack differs, Q5_1 blocks of 32 K values stepping the same
// kt in {0,1} halves the sibling uses. Two waves own 32 columns between
// them, each lane owning one column and eight output rows.
__global__ __launch_bounds__(64, 1)
void q5_1_q8_1_wmma_prefill_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_ds4,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int planes) {
  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_tile = static_cast<int64_t>(blockIdx.x);
  const int expert = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  const int64_t q51_blocks = in_features / QK5_1;
  const int64_t ds4_blocks =
      (in_features + Q8_1_MMQ_BLOCK - 1) / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = q51_blocks * Q5_1_BLOCK_BYTES;

  const int64_t col_base = col_tile * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const bool valid_col = global_col < out_features;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (qweight + (static_cast<int64_t>(expert) * out_features + global_col) *
             row_bytes)
      : qweight;

  const int64_t start = expert_start[expert];
  const int64_t end = expert_start[expert + 1];
  if (start < 0 || end < start || end > compact_rows || start == end) return;

  for (int64_t row0 = start; row0 < end; row0 += 16) {
    float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int64_t a_row = row0 + lane16;
    const bool valid_a = a_row < end;

    for (int64_t b = 0; b < q51_blocks; ++b) {
      const uint8_t* __restrict__ w_sub = weight_row + b * Q5_1_BLOCK_BYTES;
      float d = 0.0f;
      float m = 0.0f;
      uint32_t qh = 0;
      if (valid_col) {
        d = fp16_bits_to_float(*reinterpret_cast<const uint16_t*>(w_sub));
        m = fp16_bits_to_float(*reinterpret_cast<const uint16_t*>(w_sub + 2));
        qh = *reinterpret_cast<const uint32_t*>(w_sub + 4);
      }
      // Four 32-value Q5_1 blocks per 128-value DS4 block; a partial final
      // DS4 block simply is not addressed, because b stops at q51_blocks.
      const int64_t xb_blk = b >> 2;
      const int ds4_idx = static_cast<int>(b & 3);

      for (int plane = 0; plane < planes; ++plane) {
        const block_q8_1_mmq_ds4* __restrict__ x_plane =
            x_ds4 + static_cast<int64_t>(plane) * compact_rows * ds4_blocks;

        int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
          int32x4_t a_vec = {0, 0, 0, 0};
          if (valid_a) {
            const block_q8_1_mmq_ds4* __restrict__ xb =
                x_plane + a_row * ds4_blocks + xb_blk;
            const int8_t* __restrict__ xq =
                xb->qs + ds4_idx * QK5_1 + kt * 16;
            a_vec = int32x4_t{pack_i8x4(xq + 0), pack_i8x4(xq + 4),
                              pack_i8x4(xq + 8), pack_i8x4(xq + 12)};
          }
          int32x4_t b_vec = {0, 0, 0, 0};
          if (valid_col) {
            b_vec = int32x4_t{q5_1_pack4(w_sub + 8, qh, kt * 16 + 0),
                              q5_1_pack4(w_sub + 8, qh, kt * 16 + 4),
                              q5_1_pack4(w_sub + 8, qh, kt * 16 + 8),
                              q5_1_pack4(w_sub + 8, qh, kt * 16 + 12)};
          }
          dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, a_vec, true, b_vec, dot_acc, false);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int64_t orow = row0 + half_id + j * 2;
          if (valid_col && orow < end) {
            const block_q8_1_mmq_ds4* __restrict__ xb =
                x_plane + orow * ds4_blocks + xb_blk;
            const float xd =
                fp16_bits_to_float(xb->ds4[2 * ds4_idx]);
            const float xs =
                fp16_bits_to_float(xb->ds4[2 * ds4_idx + 1]);
            acc_f[j] += d * xd * static_cast<float>(dot_acc[j]);
            // Only plane 0 carries the full block sum, as in the scalar leaf.
            if (plane == 0) acc_f[j] += m * xs;
          }
        }
      }
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t orow = row0 + half_id + j * 2;
      if (valid_col && orow < end) {
        out[orow * out_features + global_col] = float_to_bf16_bits(acc_f[j]);
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
      in_features <= 0 ||
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

// P10: int8 matrix-core twin of the scalar leaf above. Same arguments, so
// an A/B needs only a different symbol. 32-column x 16-row tiles; the grid's
// y axis carries the expert, whose row range the block walks itself.
extern "C" int hipengine_q5_1_mmq_ds4_wmma_prefill_bf16_bf16_out(
    const void* x_ds4,
    const int64_t* expert_start,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int planes,
    hipStream_t stream) {
  if (x_ds4 == nullptr || expert_start == nullptr || qweight == nullptr ||
      out == nullptr || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || in_features % QK5_1 != 0 || out_features <= 0 ||
      planes <= 0 || planes > 3 || num_experts > 65535u) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int col_tiles =
      static_cast<unsigned int>((out_features + 31) / 32);
  hipLaunchKernelGGL(
      (q5_1_q8_1_wmma_prefill_kernel),
      dim3(col_tiles, static_cast<unsigned int>(num_experts)), dim3(64), 0,
      stream, static_cast<const block_q8_1_mmq_ds4*>(x_ds4), expert_start,
      qweight, out, compact_rows, num_experts, in_features, out_features,
      static_cast<int>(planes));
  return static_cast<int>(hipGetLastError());
}
