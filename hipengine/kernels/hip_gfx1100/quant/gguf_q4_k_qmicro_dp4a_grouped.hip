// Grouped q8_1-dp4a Q4_K qmicro-T16 kernel (C8-P2 Q4 pool candidate).
//
// Consumes the qmicro T16 tile layout produced by
// repack_gguf_q4_k_tile16_qmicro (d/dmin fp16 planes + 24-bit scale/min
// metadata records + the col-major nibble-pair q section) and the PLAIN
// gguf_q8_1 activation producer (gguf_q4_k_quantize_bf16_q8_1) shared
// with the retained grouped rowtile16-w2 owner, so a production decode
// pass needs no second quantize launch. Thread structure, load
// coalescing, and reduction mirror the proven Q5/Q6 grouped dp4a
// kernels: one thread per 4-k group across an 8-column tile, u32
// record/q loads straight from global, no LDS staging, no per-k
// barriers. Unlike the Q5 planar sibling this kernel reads the qmicro
// layout DIRECTLY: the [subblock][k][col-pair] q section yields two
// contiguous u32 loads per (subblock, quartet, 4-k pack), and each
// column's dp4a operand is assembled with nibble shifts — no separate
// planar conversion and no extra sidecar bytes.
//
// Arithmetic class: integer dp4a decode with f32 scale application and a
// changed f32 accumulation order versus the retained grouped rowtile16-w2
// owner (T2). The Q4_K minimum correction is DISTRIBUTED per 4-k pack
// (term -= wmin * d8 * sum(qs of the pack), sum via dp4a(0x01010101,...)
// exactly like the registered single-row Q4 dp4a GEMV and the Q6 grouped
// kernel's -32 correction); the pack sums add up to the exact integer
// group sum, so no f32 group-sum sidecar is needed.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int QK_K = 256;
constexpr int Q8_BLOCK = 128;
constexpr int Q8_GROUP = 32;
constexpr int T16_COLS = 16;
constexpr int SUBBLOCKS = 8;
constexpr int SUBBLOCK = 32;
constexpr int ROW_TILE = 8;

constexpr int Q4_QMICRO_T16_D_OFFSET = 0;
constexpr int Q4_QMICRO_T16_DMIN_OFFSET =
    Q4_QMICRO_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_META_OFFSET =
    Q4_QMICRO_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_META_RECORDS_BYTES =
    2 * SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q4_QMICRO_T16_Q_OFFSET =
    Q4_QMICRO_T16_META_OFFSET + Q4_QMICRO_T16_META_RECORDS_BYTES;
constexpr int Q4_QMICRO_T16_BLOCK_BYTES =
    Q4_QMICRO_T16_Q_OFFSET + SUBBLOCKS * SUBBLOCK * (T16_COLS / 2);

using half_t = _Float16;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_GROUP];
};
static_assert(sizeof(gguf_q8_1_block) == 36,
              "GGUF q8_1 block layout must match GGML");

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = static_cast<uint16_t>(ptr[0]) |
           (static_cast<uint16_t>(ptr[1]) << 8);
  return static_cast<float>(in.f16);
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

__device__ inline int dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  const int8_t b0 = static_cast<int8_t>(b & 0xFFu);
  const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFFu);
  const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFFu);
  const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFFu);
  return static_cast<int>(a & 0xFFu) * b0 +
         static_cast<int>((a >> 8) & 0xFFu) * b1 +
         static_cast<int>((a >> 16) & 0xFFu) * b2 +
         static_cast<int>((a >> 24) & 0xFFu) * b3;
#endif
}

__device__ inline uint32_t qmicro_unpack_record_contiguous(
    const uint8_t* __restrict__ record) {
  // Q4_K/Q5_K qmicro meta records pack four 6-bit coefficients
  // contiguously (c0 | c1<<6 | c2<<12 | c3<<18).
  const uint32_t word = static_cast<uint32_t>(record[0]) |
                        (static_cast<uint32_t>(record[1]) << 8) |
                        (static_cast<uint32_t>(record[2]) << 16);
  return word & 0x00FFFFFFu;
}

template <typename output_t>
__device__ inline void store_output(output_t* out, int64_t index, float value) {
  out[index] = static_cast<output_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(
    uint16_t* out, int64_t index, float value) {
  out[index] = float_to_bf16_bits(value);
}

template <typename output_t>
__global__ __launch_bounds__(128, 4) void
q4_k_qmicro_q8_1_dp4a_grouped_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles,
    output_t* __restrict__ out,
    int in_features,
    int out_features,
    int q8_blocks_per_row) {
  constexpr int kCols = 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int out_group = static_cast<int>(blockIdx.x);
  const int row_base = static_cast<int>(blockIdx.y) * ROW_TILE;
  const int out_tile = out_group >> 1;
  const int col_base = (out_group & 1) * kCols;
  const int quartet_base = col_base >> 2;
  const int64_t blocks_per_row = in_features / QK_K;
  const int groups4 = in_features >> 2;

  float acc[ROW_TILE][kCols];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int c = 0; c < kCols; ++c) acc[r][c] = 0.0f;

  for (int group = tid; group < groups4; group += blockDim.x) {
    const int pack = group & 7;
    const int sb = (group >> 3) & 7;
    const int64_t block_idx = group >> 6;
    const uint8_t* __restrict__ tile =
        tiles +
        (static_cast<int64_t>(out_tile) * blocks_per_row + block_idx) *
            Q4_QMICRO_T16_BLOCK_BYTES;

    // Scale/min metadata: 24-bit records, 4 x 6-bit coefficients per
    // (half, subblock, quartet); half 0 = scale, half 1 = min — the SAME
    // half-major record order as the Q5_K qmicro/planar metadata
    // (verified byte-level: min(sb=0, col0) sits at meta byte 96).
    // The coefficients are PER-COLUMN: T16 columns are distinct output
    // rows, each with its own d/dmin fp16 in the 16-entry planes.
    float wscale[kCols];
    float wmin[kCols];
#pragma unroll
    for (int qi = 0; qi < 2; ++qi) {
      const int quartet = quartet_base + qi;
      const uint32_t scale_word = qmicro_unpack_record_contiguous(
          tile + Q4_QMICRO_T16_META_OFFSET + (sb * 4 + quartet) * 3);
      const uint32_t min_word = qmicro_unpack_record_contiguous(
          tile + Q4_QMICRO_T16_META_OFFSET +
          (SUBBLOCKS * 4 + sb * 4 + quartet) * 3);
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        const int col = quartet * 4 + col4;
        const float d = fp16_bytes_to_float(
            tile + Q4_QMICRO_T16_D_OFFSET + col * 2);
        const float dmin = fp16_bytes_to_float(
            tile + Q4_QMICRO_T16_DMIN_OFFSET + col * 2);
        wscale[qi * 4 + col4] =
            d * static_cast<float>((scale_word >> (6 * col4)) & 0x3Fu);
        wmin[qi * 4 + col4] =
            dmin * static_cast<float>((min_word >> (6 * col4)) & 0x3Fu);
      }
    }

    // Weight quants: the q section is [subblock][k][col-pair] with an
    // 8-byte stride per k row, so each k row contributes ONE 2-byte load
    // (the quartet's two col-pair bytes: low nibble = even col, high
    // nibble = odd col) and the dp4a operand accumulates one nibble per
    // k across the pack.
    uint32_t wpack[kCols];
#pragma unroll
    for (int qi = 0; qi < 2; ++qi) {
      const int quartet = quartet_base + qi;
      uint32_t qcols[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const uint8_t* __restrict__ qptr =
            tile + Q4_QMICRO_T16_Q_OFFSET +
            (static_cast<int64_t>(sb) * SUBBLOCK + pack * 4 + kk) * 8 +
            quartet * 2;
        const uint16_t pair_bytes =
            *reinterpret_cast<const uint16_t*>(qptr);
        qcols[0] |=
            static_cast<uint32_t>((pair_bytes >> 0) & 0xFu) << (8 * kk);
        qcols[1] |=
            static_cast<uint32_t>((pair_bytes >> 4) & 0xFu) << (8 * kk);
        qcols[2] |=
            static_cast<uint32_t>((pair_bytes >> 8) & 0xFu) << (8 * kk);
        qcols[3] |=
            static_cast<uint32_t>((pair_bytes >> 12) & 0xFu) << (8 * kk);
      }
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        wpack[qi * 4 + col4] = qcols[col4];
      }
    }

#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const int row = row_base + r;
      // Plain q8_1 blocks are 32-k each: the group's block index is
      // block_idx * SUBBLOCKS + sb (8 blocks per 256-k QK_K block).
      const gguf_q8_1_block* __restrict__ xb =
          xq + static_cast<int64_t>(row) * q8_blocks_per_row +
          block_idx * SUBBLOCKS + sb;
      const int8_t* __restrict__ group_qs = xb->qs;
      const uint32_t x_pack =
          *reinterpret_cast<const uint32_t*>(group_qs + pack * 4);
      const float xd = static_cast<float>(xb->d);
      // Distributed Q4_K minimum correction: sum the pack's four int8
      // activations with an all-ones dp4a (exact); the per-pack
      // corrections add up to the exact 32-wide group sum.
      const int x_sum4 = dp4a_u8_s8(0x01010101u, x_pack, 0);
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int dot = dp4a_u8_s8(wpack[c], x_pack, 0);
        acc[r][c] +=
            xd * (wscale[c] * static_cast<float>(dot) -
                  wmin[c] * static_cast<float>(x_sum4));
      }
    }
  }

  // Wave-level reduce then the cross-wave xchg roundtrip (the proven Q5/Q6
  // grouped structure). Lane 0 of each wave writes its reduced partials;
  // the first kCols threads sum the four wave slots and store.
  __shared__ float xchg[4 * ROW_TILE * kCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      float v = acc[r][c];
      v += __shfl_down(v, 16);
      v += __shfl_down(v, 8);
      v += __shfl_down(v, 4);
      v += __shfl_down(v, 2);
      v += __shfl_down(v, 1);
      if (sub_lane == 0) xchg[(sub_id * ROW_TILE + r) * kCols + c] = v;
    }
  __syncthreads();
  if (tid < kCols) {
    const int c = tid;
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      float v = 0.0f;
#pragma unroll
      for (int s = 0; s < 4; ++s)
        v += xchg[(s * ROW_TILE + r) * kCols + c];
      const int64_t row = row_base + r;
      const int64_t col = static_cast<int64_t>(out_group) * kCols + c;
      store_output<output_t>(
          out, row * static_cast<int64_t>(out_features) + col, v);
    }
  }
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_qmicro_dp4a_grouped_bf16_bf16_out(
    const void* xq,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || tiles == nullptr || out == nullptr || rows < 8 ||
      (rows % ROW_TILE) != 0 || rows > 64 || in_features <= 0 ||
      (in_features % QK_K) != 0 || (in_features % Q8_BLOCK) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / 8),
      static_cast<unsigned int>(rows / ROW_TILE));
  hipLaunchKernelGGL(
      (q4_k_qmicro_q8_1_dp4a_grouped_kernel<uint16_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      tiles,
      out,
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(in_features / Q8_GROUP));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_qmicro_dp4a_grouped_bf16_f32_out(
    const void* xq,
    const uint8_t* tiles,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || tiles == nullptr || out == nullptr || rows < 8 ||
      (rows % ROW_TILE) != 0 || rows > 64 || in_features <= 0 ||
      (in_features % QK_K) != 0 || (in_features % Q8_BLOCK) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / 8),
      static_cast<unsigned int>(rows / ROW_TILE));
  hipLaunchKernelGGL(
      (q4_k_qmicro_q8_1_dp4a_grouped_kernel<float>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      tiles,
      out,
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(in_features / Q8_GROUP));
  return static_cast<int>(hipGetLastError());
}
