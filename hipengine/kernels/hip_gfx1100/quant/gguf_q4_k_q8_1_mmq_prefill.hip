// Dense bulk-prefill raw-Q4_K x DS4-Q8_1 integer MMQ: leaf A/B screen.
//
// Screen for the 2026-09-08 engine-comparison PP8192 attribution
// (worklog pp8192-gap-attribution-8bb338): the timed prefill is ~63%
// projection T16 WMMA, and the candidate arithmetic class is the nasone32
// bulk-prefill class — raw Q4_K weights with Q8_1 activations in an integer
// MMQ consumer with efa4e8641-family load reuse. The 2026-08-12 rejections
// bar repeating the T16-payload I128/J128 or I64/J64 dataflow (Q4 ~2.1x
// slower; Q5 +3.0-4.4%), so both kernels here consume RAW GGUF Q4_K bytes —
// a materially different weight-consumer dataflow — in two classes:
//
//   * mmq32: local128 workgroup, 32-column x 32-row output tile, per-256-
//     block LDS staging of the raw weight columns and the DS4 activation
//     rows, sudot4 (dp4a) accumulation (dense port of the retained raw Q5
//     MMQ32 C8-owner idiom from gguf_k_mmq_prefill.hip).
//   * wmma32: direct-global iu8 WMMA 16x16x16, 32-column x 16-row tiles
//     (dense port of the 2026-06-16 selected ds4-wmma32 family winner from
//     gguf_q4_k_q8_1_selected_prefill.hip).
//
// Each class has ctl/vdr siblings with the SAME thread mapping, the SAME
// integer dot terms and the SAME per-subblock f32 evaluation order. vdr only
// hoists loads: the Q4_K block header d/dmin is read once per 256-block
// instead of once per 32-element subblock, and mmq32_vdr additionally
// pre-decodes the eight subblock scale/min products per block into
// registers. The RED contract in tests/test_gguf_q4_k_q8_1_mmq_prefill.py is
// BIT-EXACT ctl/vdr equality per class: the only admissible difference is
// load scheduling, never values.
//
// Activations arrive as llama.cpp-style DS4 block_q8_1_mmq from the existing
// gguf_q8_1_mmq_ds4_pack_bf16 (one residual pass: per-32 fp16 scale =
// maxabs/127, fp16 sum = raw f32 sum). The -dmin*min correction consumes the
// raw sum because x = scale * q8 exactly up to quantization error.
//
// This is a leaf screen — the production dense Q4 projection prefill owners
// (float Q4T16 WMMA) are untouched and remain the registered exact paths.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + 12;  // d + dmin + 12 scale bytes
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int Q8_1_BLOCK = 32;
constexpr int Q8_1_MMQ_BLOCK = 4 * Q8_1_BLOCK;  // == 128
constexpr int PACKS_PER_SUBBLOCK = Q4_K_SUBBLOCK / 4;  // == 8

// llama.cpp DS4 block_q8_1_mmq: four (fp16 scale, fp16 sum) pairs for the
// four 32-element groups, followed by 128 int8 quants.
struct gguf_q8_1_ds4_block {
  uint16_t ds4[8];
  int8_t qs[Q8_1_MMQ_BLOCK];
};
static_assert(sizeof(gguf_q8_1_ds4_block) == 144, "DS4 block layout must match GGML");

constexpr int MMQ_TILE_COLS = 32;
constexpr int MMQ_TILE_ROWS = 32;
constexpr int MMQ_THREADS = 128;  // 4 row-groups of 8 rows x 32 columns

__device__ inline float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  return fp16_bits_to_float(bits);
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

__device__ inline int q4_k_dp4a(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  const int8_t b0 = static_cast<int8_t>(b & 0xFFu);
  const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFFu);
  const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFFu);
  const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFFu);
  return c + static_cast<int>(a & 0xFFu) * static_cast<int>(b0) +
         static_cast<int>((a >> 8) & 0xFFu) * static_cast<int>(b1) +
         static_cast<int>((a >> 16) & 0xFFu) * static_cast<int>(b2) +
         static_cast<int>((a >> 24) & 0xFFu) * static_cast<int>(b3);
#endif
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

// One u32 holds four consecutive nibble values of one subblock.
__device__ inline uint32_t q4_k_pack4(
    const uint8_t* qs, int subblock, int lane4) {
  const int pair = subblock >> 1;
  const uint32_t packed =
      *reinterpret_cast<const uint32_t*>(qs + pair * Q4_K_SUBBLOCK + lane4);
  return (subblock & 1) ? ((packed >> 4) & 0x0F0F0F0Fu)
                        : (packed & 0x0F0F0F0Fu);
}

__device__ inline int pack_i8x4(const int8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return static_cast<int>(out);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t index, float value) {
  out[index] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(
    uint16_t* out, int64_t index, float value) {
  out[index] = float_to_bf16_bits(value);
}

// ---------------------------------------------------------------------------
// mmq32: staged dp4a consumer, 32-column x 32-row output tile per local128
// workgroup. Per 256-K block the workgroup stages the 32 raw weight blocks
// and the 32 activation rows into LDS, then every thread accumulates eight
// rows of one column with eight dp4a packs per subblock.
// ---------------------------------------------------------------------------
template <typename out_t, bool VDR>
__launch_bounds__(MMQ_THREADS, 2) __global__ void q4_k_q8_1_mmq32_dense_kernel(
    const gguf_q8_1_ds4_block* __restrict__ x_q8,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * MMQ_TILE_COLS;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * MMQ_TILE_ROWS;
  const int tid = threadIdx.x;
  const int col = tid & 31;    // column within the tile
  const int row_grp = tid >> 5;  // 0..3, eight rows each
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

  __shared__ alignas(16) uint8_t w_tile[MMQ_TILE_COLS * Q4_K_BLOCK_BYTES];
  __shared__ alignas(16) int8_t a_tile[MMQ_TILE_ROWS * QK_K];
  __shared__ alignas(16) float a_ds[MMQ_TILE_ROWS * Q4_K_SUBBLOCKS * 2];

  float acc[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    // --- cooperative staging: 32 raw weight blocks (u32 words) ---
    constexpr int W_WORDS = Q4_K_BLOCK_BYTES / 4;  // 36
    for (int i = tid; i < MMQ_TILE_COLS * W_WORDS; i += MMQ_THREADS) {
      const int c = i / W_WORDS;
      const int w = i % W_WORDS;
      const int64_t gcol = col0 + c;
      uint32_t word = 0;
      if (gcol < out_features) {
        word = *reinterpret_cast<const uint32_t*>(
            qweight + gcol * row_bytes + blk * Q4_K_BLOCK_BYTES + w * 4);
      }
      *reinterpret_cast<uint32_t*>(w_tile + i * 4) = word;
    }
    // --- cooperative staging: 32 activation rows, 72 u32 words each ---
    constexpr int A_WORDS =
        2 * (static_cast<int>(sizeof(gguf_q8_1_ds4_block)) / 4);  // 72
    constexpr int DS4_WORDS =
        static_cast<int>(sizeof(gguf_q8_1_ds4_block)) / 4;  // 36
    for (int i = tid; i < MMQ_TILE_ROWS * A_WORDS; i += MMQ_THREADS) {
      const int r = i / A_WORDS;
      const int w = i % A_WORDS;
      const int64_t grow = row0 + r;
      const int half = w / DS4_WORDS;        // DS4 block within the 256-K slice
      const int ww = w % DS4_WORDS;         // u32 within that DS4 block
      uint32_t word = 0;
      if (grow < rows) {
        const gguf_q8_1_ds4_block* __restrict__ xb =
            x_q8 + grow * q8_blocks_per_row + blk * 2 + half;
        word = *reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(xb) + ww * 4);
      }
      if (ww < 4) {
        // ds4 pairs: u32 0..3 = groups 0..3 of (fp16 scale, fp16 sum)
        const int g = ww;
        const int sb = half * 4 + g;
        a_ds[r * 16 + sb * 2] = fp16_bits_to_float(
            static_cast<uint16_t>(word & 0xFFFFu));
        a_ds[r * 16 + sb * 2 + 1] = fp16_bits_to_float(
            static_cast<uint16_t>(word >> 16));
      } else {
        *reinterpret_cast<uint32_t*>(
            a_tile + r * QK_K + half * Q8_1_MMQ_BLOCK + (ww - 4) * 4) = word;
      }
    }
    __syncthreads();

    const uint8_t* __restrict__ wcol = w_tile + col * Q4_K_BLOCK_BYTES;
    // vdr: hoist the block header and all eight subblock scale/min
    // products once per 256-K block (adjacent-chunk amortization).
    float d_scale_h[Q4_K_SUBBLOCKS];
    float dmin_min_h[Q4_K_SUBBLOCKS];
    if constexpr (VDR) {
      const float d = fp16_bytes_to_float(wcol);
      const float dmin = fp16_bytes_to_float(wcol + 2);
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        d_scale_h[sb] = d * static_cast<float>(gguf_q4_k_scale(wcol + 4, sb));
        dmin_min_h[sb] =
            dmin * static_cast<float>(gguf_q4_k_min(wcol + 4, sb));
      }
    }

    const uint8_t* __restrict__ qs = wcol + Q4_K_QS_OFFSET;
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      float scale_f;
      float min_f;
      if constexpr (VDR) {
        scale_f = d_scale_h[sb];
        min_f = dmin_min_h[sb];
      } else {
        const float d = fp16_bytes_to_float(wcol);
        const float dmin = fp16_bytes_to_float(wcol + 2);
        scale_f = d * static_cast<float>(gguf_q4_k_scale(wcol + 4, sb));
        min_f = dmin * static_cast<float>(gguf_q4_k_min(wcol + 4, sb));
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int trow = row_grp * 8 + r;
        const int64_t grow = row0 + trow;
        if (grow >= rows) break;  // r increases, so the rest are invalid
        int dot = 0;
#pragma unroll
        for (int p = 0; p < PACKS_PER_SUBBLOCK; ++p) {
          const uint32_t q4 = q4_k_pack4(qs, sb, p * 4);
          const uint32_t x8 = *reinterpret_cast<const uint32_t*>(
              a_tile + trow * QK_K + sb * Q4_K_SUBBLOCK + p * 4);
          dot = q4_k_dp4a(q4, x8, dot);
        }
        const float xd = a_ds[trow * 16 + sb * 2];
        const float xsum = a_ds[trow * 16 + sb * 2 + 1];
        acc[r] += scale_f * xd * static_cast<float>(dot) - min_f * xsum;
      }
    }
    __syncthreads();  // staging of the next block overwrites the tiles
  }

#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int trow = row_grp * 8 + r;
    const int64_t grow = row0 + trow;
    if (grow < rows) {
      store_output<out_t>(out, grow * out_features + col0 + col, acc[r]);
    }
  }
}

// ---------------------------------------------------------------------------
// wmma32: direct-global iu8 WMMA consumer, 32-column x 16-row output tiles,
// 64 threads (dense port of the 2026-06-16 selected ds4-wmma32 family
// winner). No LDS staging; every thread walks the raw weight row and the
// DS4 activation rows directly.
// ---------------------------------------------------------------------------
template <typename out_t, bool VDR>
__launch_bounds__(64, 1) __global__ void q4_k_q8_1_wmma32_dense_kernel(
    const gguf_q8_1_ds4_block* __restrict__ x_q8,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col_tile = blockIdx.x;
  const int64_t row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = col_tile * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const bool valid_col = global_col < out_features;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (qweight + global_col * row_bytes) : qweight;

  // A-fragment row for this thread; D-fragment rows are half_id + j*2.
  const int64_t a_row = row_tile * 16 + lane16;
  const bool valid_a = a_row < rows;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr =
        weight_row + blk * Q4_K_BLOCK_BYTES;
    // vdr: read the block header once per 256-K block.
    float d_h = 0.0f;
    float dmin_h = 0.0f;
    if constexpr (VDR) {
      if (valid_col) {
        d_h = fp16_bytes_to_float(block_ptr);
        dmin_h = fp16_bytes_to_float(block_ptr + 2);
      }
    }
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        float d;
        float dmin;
        if constexpr (VDR) {
          d = d_h;
          dmin = dmin_h;
        } else {
          d = fp16_bytes_to_float(block_ptr);
          dmin = fp16_bytes_to_float(block_ptr + 2);
        }
        scale_f =
            d * static_cast<float>(gguf_q4_k_scale(block_ptr + 4, sb));
        min_f = dmin * static_cast<float>(gguf_q4_k_min(block_ptr + 4, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a) {
          const gguf_q8_1_ds4_block* __restrict__ xb =
              x_q8 + a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const int8_t* __restrict__ xq =
              xb->qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
          b_vec = int32x4_t{
              static_cast<int>(q4_k_pack4(qs, sb, kt * 16 + 0)),
              static_cast<int>(q4_k_pack4(qs, sb, kt * 16 + 4)),
              static_cast<int>(q4_k_pack4(qs, sb, kt * 16 + 8)),
              static_cast<int>(q4_k_pack4(qs, sb, kt * 16 + 12)),
          };
        }
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t orow = row_tile * 16 + half_id + j * 2;
        if (valid_col && orow < rows) {
          const gguf_q8_1_ds4_block* __restrict__ xb =
              x_q8 + orow * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const float xd = fp16_bits_to_float(xb->ds4[2 * ds4_idx + 0]);
          const float xsum = fp16_bits_to_float(xb->ds4[2 * ds4_idx + 1]);
          acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) -
                      min_f * xsum;
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t orow = row_tile * 16 + half_id + j * 2;
    if (valid_col && orow < rows) {
      store_output<out_t>(out, orow * out_features + global_col, acc_f[j]);
    }
  }
}

}  // namespace

// mmq32 launchers: 32-column x 32-row tiles, local128 workgroups.
#define DEFINE_MMQ32_LAUNCH(SYMBOL, OUT_T, VDR)                              \
  extern "C" int SYMBOL(                                                     \
      const void* xq,                                                        \
      const uint8_t* qweight,                                                \
      void* out,                                                             \
      int64_t rows,                                                          \
      int64_t in_features,                                                   \
      int64_t out_features,                                                  \
      hipStream_t stream) {                                                  \
    if (xq == nullptr || qweight == nullptr || out == nullptr ||             \
        rows <= 0 || rows > (1 << 20) || in_features <= 0 ||                  \
        (in_features % QK_K) != 0 || out_features <= 0 ||                    \
        (out_features % 32) != 0) {                                          \
      return static_cast<int>(hipErrorInvalidValue);                         \
    }                                                                        \
    const dim3 block(MMQ_THREADS);                                           \
    const dim3 grid(                                                         \
        static_cast<unsigned int>(out_features / 32),                        \
        static_cast<unsigned int>((rows + 31) / 32));                        \
    hipLaunchKernelGGL(                                                     \
        (q4_k_q8_1_mmq32_dense_kernel<OUT_T, VDR>), grid, block, 0, stream,  \
        reinterpret_cast<const gguf_q8_1_ds4_block*>(xq),                   \
        qweight,                                                            \
        reinterpret_cast<OUT_T*>(out),                                      \
        rows,                                                                \
        in_features,                                                         \
        out_features);                                                       \
    return static_cast<int>(hipGetLastError());                              \
  }

DEFINE_MMQ32_LAUNCH(hipengine_gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out, float, false)
DEFINE_MMQ32_LAUNCH(hipengine_gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out, float, true)
DEFINE_MMQ32_LAUNCH(hipengine_gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out, uint16_t, false)
DEFINE_MMQ32_LAUNCH(hipengine_gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out, uint16_t, true)

// wmma32 launchers: 32-column x 16-row tiles, 64-thread workgroups.
#define DEFINE_WMMA32_LAUNCH(SYMBOL, OUT_T, VDR)                             \
  extern "C" int SYMBOL(                                                     \
      const void* xq,                                                        \
      const uint8_t* qweight,                                                \
      void* out,                                                             \
      int64_t rows,                                                          \
      int64_t in_features,                                                   \
      int64_t out_features,                                                  \
      hipStream_t stream) {                                                  \
    if (xq == nullptr || qweight == nullptr || out == nullptr ||             \
        rows <= 0 || rows > (1 << 20) || in_features <= 0 ||                  \
        (in_features % QK_K) != 0 || out_features <= 0 ||                    \
        (out_features % 32) != 0) {                                          \
      return static_cast<int>(hipErrorInvalidValue);                         \
    }                                                                        \
    const dim3 block(64);                                                    \
    const dim3 grid(                                                         \
        static_cast<unsigned int>(out_features / 32),                        \
        static_cast<unsigned int>((rows + 15) / 16));                        \
    hipLaunchKernelGGL(                                                      \
        (q4_k_q8_1_wmma32_dense_kernel<OUT_T, VDR>), grid, block, 0, stream, \
        reinterpret_cast<const gguf_q8_1_ds4_block*>(xq),                    \
        qweight,                                                            \
        reinterpret_cast<OUT_T*>(out),                                       \
        rows,                                                                \
        in_features,                                                         \
        out_features);                                                       \
    return static_cast<int>(hipGetLastError());                              \
  }

DEFINE_WMMA32_LAUNCH(hipengine_gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out, float, false)
DEFINE_WMMA32_LAUNCH(hipengine_gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out, float, true)
DEFINE_WMMA32_LAUNCH(hipengine_gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out, uint16_t, false)
DEFINE_WMMA32_LAUNCH(hipengine_gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out, uint16_t, true)
