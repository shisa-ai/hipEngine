// Pack8 grouped/selected MoE decode GEMV for raw GGUF Q5_1 down experts.
//
// Structure mirrors gguf_k_selected_pack8_gemv.hip (PARO-style pack8: one
// 128-thread block per (output pack, compact row), 4-wave32 reduction, raw
// compact-MoE ABI x[E-row slab], expert_start_compact[E+1], qweight rank-3,
// out row-major), but the inner product follows Q5_1's 32-element/24-byte
// blocks (fp16 d, fp16 m, uint32 qh, qs[16]) instead of QK_K=256:
//
//   w = d*q5 + m          (dequant_q5_1 from qwen4_exp_q5_1.hip, bit-exact)
//
// The block is deliberately the GGUF block size 32, NOT 256, so in_features
// only needs in_features % 32 == 0 — Gemma 4's FFN width 704 qualifies.
//
// Per 32-element block the four waves cooperate: each wave hoists its 8
// channels' d/m/qh into shared memory once (all 8 outputs of the pack share
// x), each lane dequantizes its own element for all 8 channels, and a
// wave32 shuffle reduces the 32-element sum per channel before the final
// 4-wave accumulation writes the packed outputs.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_Q5_1 = 32;
constexpr int Q5_1_BLOCK_BYTES = 24;
constexpr int PACK8 = 8;
constexpr int WAVES_PER_BLOCK = 4;  // 128 threads / 32 lanes

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

// Decode-shaped pack8 GEMV for raw GGUF Q5_1 selected single output (down).
//
// Grid:  blockIdx.x = output pack [0, out_features/8)
//        blockIdx.y = compact_row [0, compact_rows)
// Block: blockDim.x = 128 (4 wave32 waves).
__launch_bounds__(128, 4) __global__ void gguf_q5_1_selected_pack8_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_pack = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  int64_t expert_id = -1;
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      expert_id = e;
      break;
    }
  }
  if (expert_id < 0) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t row_bytes = (in_features / QK_Q5_1) * Q5_1_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = expert_base + (out_col + ch) * row_bytes;
  }

  const uint16_t* __restrict__ x_row = x + compact_row * in_features;

  const int64_t blocks_per_row = in_features / QK_Q5_1;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  __shared__ float s_d[WAVES_PER_BLOCK * PACK8];
  __shared__ float s_m[WAVES_PER_BLOCK * PACK8];
  __shared__ uint32_t s_qh[WAVES_PER_BLOCK * PACK8];
  __shared__ float s_sum[WAVES_PER_BLOCK * PACK8];

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < WAVES_PER_BLOCK * PACK8; ++i) s_sum[i] = 0.0f;
  }
  __syncthreads();

  for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES_PER_BLOCK) {
    // Cooperative hoist of this block's d/m/qh for all 8 packed channels.
    if (lane < PACK8) {
      const uint8_t* __restrict__ block = row_ptr[lane] + blk * Q5_1_BLOCK_BYTES;
      s_d[wave * PACK8 + lane] = fp16_bytes_to_float(block);
      s_m[wave * PACK8 + lane] = fp16_bytes_to_float(block + 2);
      s_qh[wave * PACK8 + lane] = *reinterpret_cast<const uint32_t*>(block + 4);
    }
    __syncthreads();

    const int64_t k_base = blk * QK_Q5_1 + lane;
    const float xv = bf16_bits_to_float(x_row[k_base]);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const uint8_t qh_byte_shift = static_cast<uint8_t>(lane);
#pragma unroll
    for (int ch = 0; ch < PACK8; ++ch) {
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q5_1_BLOCK_BYTES;
      const uint8_t packed = block[8 + (lane & 15)];
      const uint8_t low = lane < 16 ? (packed & 0x0Fu) : (packed >> 4);
      const uint8_t high =
          static_cast<uint8_t>((s_qh[wave * PACK8 + ch] >> qh_byte_shift) & 1U);
      const float w = s_d[wave * PACK8 + ch] * static_cast<float>(low | (high << 4)) +
                      s_m[wave * PACK8 + ch];
      if (ch == 0) acc0 = fmaf(xv, w, acc0);
      else if (ch == 1) acc1 = fmaf(xv, w, acc1);
      else if (ch == 2) acc2 = fmaf(xv, w, acc2);
      else if (ch == 3) acc3 = fmaf(xv, w, acc3);
      else if (ch == 4) acc4 = fmaf(xv, w, acc4);
      else if (ch == 5) acc5 = fmaf(xv, w, acc5);
      else if (ch == 6) acc6 = fmaf(xv, w, acc6);
      else acc7 = fmaf(xv, w, acc7);
    }

    // In-wave lane shuffle reduction across the 32 elements of this block.
#define Q51_PACK8_REDUCE_SHFL(acc)                        \
  do {                                                    \
    acc += __shfl_down(acc, 16, 32);                      \
    acc += __shfl_down(acc, 8, 32);                       \
    acc += __shfl_down(acc, 4, 32);                       \
    acc += __shfl_down(acc, 2, 32);                       \
    acc += __shfl_down(acc, 1, 32);                       \
  } while (0)

    Q51_PACK8_REDUCE_SHFL(acc0);
    Q51_PACK8_REDUCE_SHFL(acc1);
    Q51_PACK8_REDUCE_SHFL(acc2);
    Q51_PACK8_REDUCE_SHFL(acc3);
    Q51_PACK8_REDUCE_SHFL(acc4);
    Q51_PACK8_REDUCE_SHFL(acc5);
    Q51_PACK8_REDUCE_SHFL(acc6);
    Q51_PACK8_REDUCE_SHFL(acc7);
#undef Q51_PACK8_REDUCE_SHFL

    if (lane == 0) {
      s_sum[wave * PACK8 + 0] += acc0;
      s_sum[wave * PACK8 + 1] += acc1;
      s_sum[wave * PACK8 + 2] += acc2;
      s_sum[wave * PACK8 + 3] += acc3;
      s_sum[wave * PACK8 + 4] += acc4;
      s_sum[wave * PACK8 + 5] += acc5;
      s_sum[wave * PACK8 + 6] += acc6;
      s_sum[wave * PACK8 + 7] += acc7;
    }
    __syncthreads();  // protect s_d/s_m/s_qh/s_sum before the next block
  }

  if (tid == 0) {
    uint16_t* __restrict__ out_ptr = out + compact_row * out_features + out_col;
#pragma unroll
    for (int ch = 0; ch < PACK8; ++ch) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < WAVES_PER_BLOCK; ++w) {
        total += s_sum[w * PACK8 + ch];
      }
      out_ptr[ch] = float_to_bf16_bits(total);
    }
  }
}

bool valid_selected_pack8_gemv(
    const void* x,
    const void* expert_start_compact,
    const void* qweight,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  return x != nullptr && expert_start_compact != nullptr && qweight != nullptr &&
      out != nullptr && compact_rows > 0 && in_features > 0 && out_features > 0 &&
      num_experts > 0 && (out_features & 7) == 0 && (in_features % QK_Q5_1) == 0;
}

int launch_selected_pack8_gemv(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!valid_selected_pack8_gemv(
          x, expert_start_compact, qweight, out,
          compact_rows, in_features, out_features, num_experts)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_q5_1_selected_pack8_gemv_kernel),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(compact_rows)),
      dim3(128),
      0,
      stream,
      x,
      expert_start_compact,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_pack8_gemv(
      x, expert_start_compact, qweight, out,
      compact_rows, in_features, out_features, num_experts, stream);
}