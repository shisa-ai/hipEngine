// Grouped q8_1-dp4a Q5_K qmicro-planar kernel (C8-P2 residual candidate).
//
// Consumes the planar qmicro T16 tile layout produced by
// convert_gguf_q5_k_qmicro_tile16_to_planar (d/dmin planes + 24-bit
// scale/min metadata records + 12-byte dp4a records per (subblock, column
// quartet, 4-quant pack)) and the d4s4-f32 q8_1 activation producer used
// by the retained raw Q5 MMQ owners. Thread structure, load coalescing,
// and reduction mirror the proven Q6 grouped dp4a kernel: one thread per
// 4-k group across an 8-column tile, u32 record loads straight from
// global, no LDS staging, no per-k barriers.
//
// Arithmetic class: integer dp4a decode with f32 scale application and a
// changed f32 accumulation order versus the raw mmq32 owner (T2). The
// Q5_K minimum correction uses the producer's s4 (the f32 sum of the
// ORIGINAL pre-quantization group values), applied once per
// (row, subblock) by the pack-0 thread; the quantized integer sum is NOT
// equivalent (it carries the quantization residual).
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

constexpr int Q5_QMICRO_PLANAR_D_OFFSET = 0;
constexpr int Q5_QMICRO_PLANAR_DMIN_OFFSET =
    Q5_QMICRO_PLANAR_D_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_PLANAR_META_OFFSET =
    Q5_QMICRO_PLANAR_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_PLANAR_META_BYTES = 2 * SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q5_QMICRO_PLANAR_RECORDS_OFFSET =
    Q5_QMICRO_PLANAR_META_OFFSET + Q5_QMICRO_PLANAR_META_BYTES;
constexpr int Q5_QMICRO_PLANAR_RECORD_BYTES = 12;
constexpr int Q5_QMICRO_PLANAR_BLOCK_BYTES =
    Q5_QMICRO_PLANAR_RECORDS_OFFSET +
    SUBBLOCKS * (T16_COLS / 4) * (SUBBLOCK / 4) * Q5_QMICRO_PLANAR_RECORD_BYTES;

struct q8_1_d4s4_f32_block {
  float d4[4];
  float s4[4];
  int8_t qs[Q8_BLOCK];
};
static_assert(sizeof(q8_1_d4s4_f32_block) == 160,
              "unexpected Q8_1 D4S4-F32 block size");

using half_t = _Float16;

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
  // Q5_K qmicro meta records pack four 6-bit coefficients contiguously
  // (c0 | c1<<6 | c2<<12 | c3<<18), unlike the Q6_K byte-spaced layout.
  const uint32_t word = static_cast<uint32_t>(record[0]) |
                        (static_cast<uint32_t>(record[1]) << 8) |
                        (static_cast<uint32_t>(record[2]) << 16);
  return word & 0x00FFFFFFu;
}

__device__ inline uint32_t q5_planar_pack4_unsigned(
    uint32_t low01,
    uint32_t low23,
    uint32_t high,
    int col4) {
  const uint32_t low_pair = (col4 < 2) ? low01 : low23;
  const uint32_t low =
      ((col4 & 1) ? (low_pair >> 4) : low_pair) & 0x0F0F0F0Fu;
  const uint32_t high_bits = ((high >> col4) & 0x01010101u) << 4;
  return low | high_bits;
}

template <typename output_t>
__device__ inline void planar_store_output(
    output_t* out, int64_t index, float value) {
  out[index] = static_cast<output_t>(value);
}

template <>
__device__ inline void planar_store_output<uint16_t>(
    uint16_t* out, int64_t index, float value) {
  out[index] = float_to_bf16_bits(value);
}

template <typename output_t>
__global__ __launch_bounds__(128, 4) void
q5_k_qmicro_planar_q8_1_dp4a_grouped_kernel(
    const q8_1_d4s4_f32_block* __restrict__ xq,
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
            Q5_QMICRO_PLANAR_BLOCK_BYTES;

    // Scale/min metadata: 24-bit records, 4 x 6-bit coefficients per
    // (half, subblock, quartet); half 0 = scale, half 1 = min. The
    // coefficients are PER-COLUMN: T16 columns are distinct output rows,
    // each with its own d/dmin fp16 in the 16-entry planes (the q5_K
    // per-row scale is folded into the per-column planes).
    float wscale[kCols];
    float wmin[kCols];
#pragma unroll
    for (int qi = 0; qi < 2; ++qi) {
      const int quartet = quartet_base + qi;
      const uint32_t scale_word = qmicro_unpack_record_contiguous(
          tile + Q5_QMICRO_PLANAR_META_OFFSET + (sb * 4 + quartet) * 3);
      const uint32_t min_word = qmicro_unpack_record_contiguous(
          tile + Q5_QMICRO_PLANAR_META_OFFSET +
          (SUBBLOCKS * 4 + sb * 4 + quartet) * 3);
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        const int col = quartet * 4 + col4;
        const float d = fp16_bytes_to_float(
            tile + Q5_QMICRO_PLANAR_D_OFFSET + col * 2);
        const float dmin = fp16_bytes_to_float(
            tile + Q5_QMICRO_PLANAR_DMIN_OFFSET + col * 2);
        wscale[qi * 4 + col4] =
            d * static_cast<float>((scale_word >> (6 * col4)) & 0x3Fu);
        wmin[qi * 4 + col4] =
            dmin * static_cast<float>((min_word >> (6 * col4)) & 0x3Fu);
      }
    }

    // Weight quants: two 12-byte planar records (one per quartet).
    uint32_t wpack[kCols];
#pragma unroll
    for (int qi = 0; qi < 2; ++qi) {
      const int quartet = quartet_base + qi;
      const uint8_t* __restrict__ record =
          tile + Q5_QMICRO_PLANAR_RECORDS_OFFSET +
          ((sb * 4 + quartet) * 8 + pack) * Q5_QMICRO_PLANAR_RECORD_BYTES;
      const uint32_t low01 = *reinterpret_cast<const uint32_t*>(record);
      const uint32_t low23 = *reinterpret_cast<const uint32_t*>(record + 4);
      const uint32_t high = *reinterpret_cast<const uint32_t*>(record + 8);
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        wpack[qi * 4 + col4] =
            q5_planar_pack4_unsigned(low01, low23, high, col4);
      }
    }

#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const int row = row_base + r;
      const q8_1_d4s4_f32_block* __restrict__ xb =
          xq + static_cast<int64_t>(row) * q8_blocks_per_row +
          (group >> 5);
      const int g32 = (group >> 3) & 3;
      const int8_t* __restrict__ group_qs = xb->qs + g32 * Q8_GROUP;
      const uint32_t x_pack =
          *reinterpret_cast<const uint32_t*>(group_qs + pack * 4);
      const float xd = xb->d4[g32];
      const float s4_row = xb->s4[g32];
      const bool apply_min = (pack == 0);
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int dot = dp4a_u8_s8(wpack[c], x_pack, 0);
        float term = xd * wscale[c] * static_cast<float>(dot);
        if (apply_min) term -= wmin[c] * s4_row;
        acc[r][c] += term;
      }
    }
  }

  // Wave-level reduce then the cross-wave xchg roundtrip (the proven Q6
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
      planar_store_output<output_t>(
          out, row * static_cast<int64_t>(out_features) + col, v);
    }
  }
}

}  // namespace

extern "C" int hipengine_gguf_q5_k_qmicro_planar_dp4a_grouped_bf16_bf16_out(
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
      (q5_k_qmicro_planar_q8_1_dp4a_grouped_kernel<uint16_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      tiles,
      out,
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(in_features / Q8_BLOCK));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q5_k_qmicro_planar_dp4a_grouped_bf16_f32_out(
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
      (q5_k_qmicro_planar_q8_1_dp4a_grouped_kernel<float>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      tiles,
      out,
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(in_features / Q8_BLOCK));
  return static_cast<int>(hipGetLastError());
}
