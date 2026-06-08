// qwen35moe GGUF expert pack8 selected GEMV kernels.
//
// These kernels consume the explicit gguf_expert_pack8_v1 sidecar layout:
//   qweight_low [experts, out_features/8, in_features]
//   qweight_high optional [experts, out_features/8, in_features]
//   scales/mins [experts, groups, out_features]
// and a device-selected row vector of length rows = tokens * top_k.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int QK_K = 256;
constexpr int PACK8 = 8;
constexpr int Q4_Q5_GROUP = 32;
constexpr int Q6_GROUP = 16;
using half_t = _Float16;

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

__device__ inline int pack8_shift_for_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

#define PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
  for (int offset = 16; offset > 0; offset >>= 1) { \
    acc0 += __shfl_down(acc0, offset); \
    acc1 += __shfl_down(acc1, offset); \
    acc2 += __shfl_down(acc2, offset); \
    acc3 += __shfl_down(acc3, offset); \
    acc4 += __shfl_down(acc4, offset); \
    acc5 += __shfl_down(acc5, offset); \
    acc6 += __shfl_down(acc6, offset); \
    acc7 += __shfl_down(acc7, offset); \
  } \
  __shared__ float xchg[8 * 8]; \
  const int sub_id = tid >> 5; \
  const int sub_lane = tid & 31; \
  if (sub_lane == 0) { \
    xchg[sub_id * 8 + 0] = acc0; \
    xchg[sub_id * 8 + 1] = acc1; \
    xchg[sub_id * 8 + 2] = acc2; \
    xchg[sub_id * 8 + 3] = acc3; \
    xchg[sub_id * 8 + 4] = acc4; \
    xchg[sub_id * 8 + 5] = acc5; \
    xchg[sub_id * 8 + 6] = acc6; \
    xchg[sub_id * 8 + 7] = acc7; \
  } \
  __syncthreads(); \
  float r0 = 0.0f; \
  float r1 = 0.0f; \
  float r2 = 0.0f; \
  float r3 = 0.0f; \
  float r4 = 0.0f; \
  float r5 = 0.0f; \
  float r6 = 0.0f; \
  float r7 = 0.0f; \
  if (tid == 0) { \
    const int sub_groups = blockDim.x >> 5; \
    for (int g = 0; g < sub_groups; ++g) { \
      r0 += xchg[g * 8 + 0]; \
      r1 += xchg[g * 8 + 1]; \
      r2 += xchg[g * 8 + 2]; \
      r3 += xchg[g * 8 + 3]; \
      r4 += xchg[g * 8 + 4]; \
      r5 += xchg[g * 8 + 5]; \
      r6 += xchg[g * 8 + 6]; \
      r7 += xchg[g * 8 + 7]; \
    } \
  }

__device__ inline void accum_q4_pack8(
    uint32_t pw,
    float xv,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ min_ptr,
    float& acc0,
    float& acc1,
    float& acc2,
    float& acc3,
    float& acc4,
    float& acc5,
    float& acc6,
    float& acc7) {
  acc0 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(0)) & 0x0Fu) * scale_ptr[0] - min_ptr[0]);
  acc1 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(1)) & 0x0Fu) * scale_ptr[1] - min_ptr[1]);
  acc2 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(2)) & 0x0Fu) * scale_ptr[2] - min_ptr[2]);
  acc3 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(3)) & 0x0Fu) * scale_ptr[3] - min_ptr[3]);
  acc4 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(4)) & 0x0Fu) * scale_ptr[4] - min_ptr[4]);
  acc5 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(5)) & 0x0Fu) * scale_ptr[5] - min_ptr[5]);
  acc6 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(6)) & 0x0Fu) * scale_ptr[6] - min_ptr[6]);
  acc7 += xv * (static_cast<float>((pw >> pack8_shift_for_lane(7)) & 0x0Fu) * scale_ptr[7] - min_ptr[7]);
}

__device__ inline void accum_q5_pack8(
    uint32_t pw,
    uint8_t ph,
    float xv,
    const float* __restrict__ scale_ptr,
    const float* __restrict__ min_ptr,
    float& acc0,
    float& acc1,
    float& acc2,
    float& acc3,
    float& acc4,
    float& acc5,
    float& acc6,
    float& acc7) {
  const float q0 = static_cast<float>(((pw >> pack8_shift_for_lane(0)) & 0x0Fu) | (((ph >> 0) & 0x01u) << 4));
  const float q1 = static_cast<float>(((pw >> pack8_shift_for_lane(1)) & 0x0Fu) | (((ph >> 1) & 0x01u) << 4));
  const float q2 = static_cast<float>(((pw >> pack8_shift_for_lane(2)) & 0x0Fu) | (((ph >> 2) & 0x01u) << 4));
  const float q3 = static_cast<float>(((pw >> pack8_shift_for_lane(3)) & 0x0Fu) | (((ph >> 3) & 0x01u) << 4));
  const float q4 = static_cast<float>(((pw >> pack8_shift_for_lane(4)) & 0x0Fu) | (((ph >> 4) & 0x01u) << 4));
  const float q5 = static_cast<float>(((pw >> pack8_shift_for_lane(5)) & 0x0Fu) | (((ph >> 5) & 0x01u) << 4));
  const float q6 = static_cast<float>(((pw >> pack8_shift_for_lane(6)) & 0x0Fu) | (((ph >> 6) & 0x01u) << 4));
  const float q7 = static_cast<float>(((pw >> pack8_shift_for_lane(7)) & 0x0Fu) | (((ph >> 7) & 0x01u) << 4));
  acc0 += xv * (q0 * scale_ptr[0] - min_ptr[0]);
  acc1 += xv * (q1 * scale_ptr[1] - min_ptr[1]);
  acc2 += xv * (q2 * scale_ptr[2] - min_ptr[2]);
  acc3 += xv * (q3 * scale_ptr[3] - min_ptr[3]);
  acc4 += xv * (q4 * scale_ptr[4] - min_ptr[4]);
  acc5 += xv * (q5 * scale_ptr[5] - min_ptr[5]);
  acc6 += xv * (q6 * scale_ptr[6] - min_ptr[6]);
  acc7 += xv * (q7 * scale_ptr[7] - min_ptr[7]);
}

__device__ inline void accum_q6_pack8(
    uint32_t pw,
    uint16_t ph,
    float xv,
    const float* __restrict__ scale_ptr,
    float& acc0,
    float& acc1,
    float& acc2,
    float& acc3,
    float& acc4,
    float& acc5,
    float& acc6,
    float& acc7) {
  const float q0 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(0)) & 0x0Fu) | (((ph >> 0) & 0x03u) << 4)) - 32);
  const float q1 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(1)) & 0x0Fu) | (((ph >> 2) & 0x03u) << 4)) - 32);
  const float q2 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(2)) & 0x0Fu) | (((ph >> 4) & 0x03u) << 4)) - 32);
  const float q3 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(3)) & 0x0Fu) | (((ph >> 6) & 0x03u) << 4)) - 32);
  const float q4 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(4)) & 0x0Fu) | (((ph >> 8) & 0x03u) << 4)) - 32);
  const float q5 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(5)) & 0x0Fu) | (((ph >> 10) & 0x03u) << 4)) - 32);
  const float q6 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(6)) & 0x0Fu) | (((ph >> 12) & 0x03u) << 4)) - 32);
  const float q7 = static_cast<float>(static_cast<int>(((pw >> pack8_shift_for_lane(7)) & 0x0Fu) | (((ph >> 14) & 0x03u) << 4)) - 32);
  acc0 += xv * (q0 * scale_ptr[0]);
  acc1 += xv * (q1 * scale_ptr[1]);
  acc2 += xv * (q2 * scale_ptr[2]);
  acc3 += xv * (q3 * scale_ptr[3]);
  acc4 += xv * (q4 * scale_ptr[4]);
  acc5 += xv * (q5 * scale_ptr[5]);
  acc6 += xv * (q6 * scale_ptr[6]);
  acc7 += xv * (q7 * scale_ptr[7]);
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(128, 4)
__global__ void gguf_expert_pack8_selected_prefill_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight_low,
    const void* __restrict__ qweight_high,
    const float* __restrict__ scales,
    const float* __restrict__ mins,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int out_packed) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;

  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t out_col = out_pack << 3;
  const int64_t x_base = x_row * static_cast<int64_t>(in_features);
  const int64_t qw_base = (expert * static_cast<int64_t>(out_packed) + out_pack) * static_cast<int64_t>(in_features);
  const int group_size = (qtype == 6) ? Q6_GROUP : Q4_Q5_GROUP;
  const int64_t groups = in_features / group_size;
  const int64_t scale_expert_base = expert * groups * static_cast<int64_t>(out_features);

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  // Match the raw selected GGUF GEMV k-iteration order so sidecar kernels are
  // a parity-safe drop-in for recurrent long-prompt probes.
  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x[x_base + k]);
    const uint32_t pw = static_cast<uint32_t>(qweight_low[qw_base + k]);
    const int64_t group = k / group_size;
    const float* __restrict__ scale_ptr = scales + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
    if constexpr (qtype == 4) {
      const float* __restrict__ min_ptr = mins + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
      accum_q4_pack8(pw, xv, scale_ptr, min_ptr, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    } else if constexpr (qtype == 5) {
      const float* __restrict__ min_ptr = mins + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
      const uint8_t ph = static_cast<const uint8_t*>(qweight_high)[qw_base + k];
      accum_q5_pack8(pw, ph, xv, scale_ptr, min_ptr, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    } else {
      const uint16_t ph = static_cast<const uint16_t*>(qweight_high)[qw_base + k];
      accum_q6_pack8(pw, ph, xv, scale_ptr, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    }
  }

  PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = row * static_cast<int64_t>(out_features) + out_col;
    store_output<out_t>(out, out_base + 0, r0);
    store_output<out_t>(out, out_base + 1, r1);
    store_output<out_t>(out, out_base + 2, r2);
    store_output<out_t>(out, out_base + 3, r3);
    store_output<out_t>(out, out_base + 4, r4);
    store_output<out_t>(out, out_base + 5, r5);
    store_output<out_t>(out, out_base + 6, r6);
    store_output<out_t>(out, out_base + 7, r7);
  }
}

template <typename scalar_t, typename out_t, int qtype>
int launch_selected(
    const scalar_t* x,
    const int64_t* selected,
    const int32_t* qweight_low,
    const void* qweight_high,
    const float* scales,
    const float* mins,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_group = (qtype == 6) ? ((in_features % Q6_GROUP) == 0) : ((in_features % Q4_Q5_GROUP) == 0);
  if (x == nullptr || selected == nullptr || qweight_low == nullptr || scales == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (out_features % PACK8) != 0 || !valid_group ||
      (threads != 32 && threads != 64 && threads != 128)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if constexpr (qtype == 4 || qtype == 5) {
    if (mins == nullptr) return static_cast<int>(hipErrorInvalidValue);
  }
  if constexpr (qtype == 5 || qtype == 6) {
    if (qweight_high == nullptr) return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_expert_pack8_selected_prefill_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight_low,
      qweight_high,
      scales,
      mins,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(out_packed));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t>
__launch_bounds__(128, 4)
__global__ void gguf_expert_pack8_q4_dual_selected_prefill_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight_a,
    const float* __restrict__ scales_a,
    const float* __restrict__ mins_a,
    const int32_t* __restrict__ qweight_b,
    const float* __restrict__ scales_b,
    const float* __restrict__ mins_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int out_packed) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;
  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t out_col = out_pack << 3;
  const int64_t x_base = x_row * static_cast<int64_t>(in_features);
  const int64_t qw_base = (expert * static_cast<int64_t>(out_packed) + out_pack) * static_cast<int64_t>(in_features);
  const int64_t groups = in_features / Q4_Q5_GROUP;
  const int64_t scale_expert_base = expert * groups * static_cast<int64_t>(out_features);

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
  float b4 = 0.0f, b5 = 0.0f, b6 = 0.0f, b7 = 0.0f;

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x[x_base + k]);
    const int64_t group = k >> 5;
    const uint32_t pwa = static_cast<uint32_t>(qweight_a[qw_base + k]);
    const float* __restrict__ sa = scales_a + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
    const float* __restrict__ ma = mins_a + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
    accum_q4_pack8(pwa, xv, sa, ma, a0, a1, a2, a3, a4, a5, a6, a7);
    const uint32_t pwb = static_cast<uint32_t>(qweight_b[qw_base + k]);
    const float* __restrict__ sb = scales_b + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
    const float* __restrict__ mb = mins_b + scale_expert_base + group * static_cast<int64_t>(out_features) + out_col;
    accum_q4_pack8(pwb, xv, sb, mb, b0, b1, b2, b3, b4, b5, b6, b7);
  }

  {
    PACK8_REDUCE(a0, a1, a2, a3, a4, a5, a6, a7, ar0, ar1, ar2, ar3, ar4, ar5, ar6, ar7);
    if (tid == 0) {
      const int64_t out_base = row * static_cast<int64_t>(out_features) + out_col;
      store_output<out_t>(out_a, out_base + 0, ar0);
      store_output<out_t>(out_a, out_base + 1, ar1);
      store_output<out_t>(out_a, out_base + 2, ar2);
      store_output<out_t>(out_a, out_base + 3, ar3);
      store_output<out_t>(out_a, out_base + 4, ar4);
      store_output<out_t>(out_a, out_base + 5, ar5);
      store_output<out_t>(out_a, out_base + 6, ar6);
      store_output<out_t>(out_a, out_base + 7, ar7);
    }
  }
  __syncthreads();
  {
    PACK8_REDUCE(b0, b1, b2, b3, b4, b5, b6, b7, br0, br1, br2, br3, br4, br5, br6, br7);
    if (tid == 0) {
      const int64_t out_base = row * static_cast<int64_t>(out_features) + out_col;
      store_output<out_t>(out_b, out_base + 0, br0);
      store_output<out_t>(out_b, out_base + 1, br1);
      store_output<out_t>(out_b, out_base + 2, br2);
      store_output<out_t>(out_b, out_base + 3, br3);
      store_output<out_t>(out_b, out_base + 4, br4);
      store_output<out_t>(out_b, out_base + 5, br5);
      store_output<out_t>(out_b, out_base + 6, br6);
      store_output<out_t>(out_b, out_base + 7, br7);
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_q4_dual_selected(
    const scalar_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const float* scales_a,
    const float* mins_a,
    const int32_t* qweight_b,
    const float* scales_b,
    const float* mins_b,
    out_t* out_a,
    out_t* out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight_a == nullptr || scales_a == nullptr || mins_a == nullptr ||
      qweight_b == nullptr || scales_b == nullptr || mins_b == nullptr || out_a == nullptr || out_b == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % Q4_Q5_GROUP) != 0 || out_features <= 0 || (out_features % PACK8) != 0 ||
      (threads != 32 && threads != 64 && threads != 128)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_expert_pack8_q4_dual_selected_prefill_kernel<scalar_t, out_t><<<grid, block, 0, stream>>>(
      x, selected, qweight_a, scales_a, mins_a, qweight_b, scales_b, mins_b, out_a, out_b,
      static_cast<int>(x_rows), static_cast<int>(rows), static_cast<int>(num_experts),
      static_cast<int>(in_features), static_cast<int>(out_features), static_cast<int>(out_packed));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_expert_pack8_selected_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_low,
    const void* qweight_high,
    const float* scales,
    const float* mins,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected<uint16_t, uint16_t, 4>(
      x, selected, qweight_low, qweight_high, scales, mins, out, x_rows, rows,
      num_experts, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_expert_pack8_selected_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_low,
    const void* qweight_high,
    const float* scales,
    const float* mins,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected<uint16_t, uint16_t, 5>(
      x, selected, qweight_low, qweight_high, scales, mins, out, x_rows, rows,
      num_experts, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_expert_pack8_selected_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_low,
    const void* qweight_high,
    const float* scales,
    const float* mins,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected<uint16_t, uint16_t, 6>(
      x, selected, qweight_low, qweight_high, scales, mins, out, x_rows, rows,
      num_experts, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const float* scales_a,
    const float* mins_a,
    const int32_t* qweight_b,
    const float* scales_b,
    const float* mins_b,
    uint16_t* out_a,
    uint16_t* out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_q4_dual_selected<uint16_t, uint16_t>(
      x, selected, qweight_a, scales_a, mins_a, qweight_b, scales_b, mins_b, out_a, out_b,
      x_rows, rows, num_experts, in_features, out_features, threads, stream);
}
