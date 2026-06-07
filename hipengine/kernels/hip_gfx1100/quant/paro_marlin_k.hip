// Marlin-K v0 FP16 GEMV port for PARO/AWQ W4 decode.
// Source lineage: nano-vllm-amd Marlin-K retained path documented as @7718fff
// (vec8 FMA) and @1522293 (qweight-neutral dispatch).  The current parent
// checkout no longer retains those commits, so this file preserves the documented
// retained algorithm beside hipEngine's pack8 GEMV family: vec8 FP32-FMA inner
// loop, qweight_mk [N/8, K/128, 128], qzeros_mk [N/8, K/128], scales_mk
// [N/8, K/128, 8], wave32 reductions, and no Q8/SUDOT/half-FMA experiments.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

__device__ inline int awq_shift_for_pack_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

#define MARLIN_K_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
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

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <typename scalar_t>
__launch_bounds__(128, 4)
__global__ void gemv_paro_marlin_k_fma_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight_mk,
    const int32_t* __restrict__ qzeros_mk,
    const scalar_t* __restrict__ scales_mk,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_packed,
    int group_size) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t out_features = out_packed << 3;
  const int64_t groups = in_features / group_size;
  const int64_t x_base = row * in_features;
  const int64_t qw_base = out_pack * in_features;
  const int64_t meta_base = out_pack * groups;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_mk[meta_base + group]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales_mk[(meta_base + group) * 8];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x[x_base + k_base + j]);
      const uint32_t pw = static_cast<uint32_t>(qweight_mk[qw_base + k_base + j]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_mk[meta_base + group]);
    const uint32_t pw = static_cast<uint32_t>(qweight_mk[qw_base + in_col]);
    const scalar_t* scale_ptr = &scales_mk[(meta_base + group) * 8];
    const float xv = scalar_to_float(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  MARLIN_K_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

int launch_gemv_paro_marlin_k_fma_fp16(
    const half_t* x,
    const int32_t* qweight_mk,
    const int32_t* qzeros_mk,
    const half_t* scales_mk,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_mk == nullptr || qzeros_mk == nullptr || scales_mk == nullptr || out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (in_features % group_size != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (threads != 32 && threads != 64 && threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_paro_marlin_k_fma_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight_mk,
      qzeros_mk,
      scales_mk,
      out,
      in_features,
      out_packed,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

// M15.2: weight-amortized multi-row Marlin-K GEMV.  Grid is (out_packed); one
// block reads each weight element once and FMAs it into all ``rows`` row
// accumulators in the *same* k-order and per-row reduction as the single-row
// kernel above.  Each row's accumulation sequence is therefore byte-identical to
// gemv_paro_marlin_k_fma_kernel, so every verifier row is bit-exact vs AR's
// rows==1 Marlin-K output (no top-1 flips from reordered fp32 reductions).
template <typename scalar_t, int MAX_ROWS>
__launch_bounds__(128, 4)
__global__ void gemv_paro_marlin_k_fma_multi_row_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight_mk,
    const int32_t* __restrict__ qzeros_mk,
    const scalar_t* __restrict__ scales_mk,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int group_size) {
  const int64_t out_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t out_features = out_packed << 3;
  const int64_t groups = in_features / group_size;
  const int64_t qw_base = out_pack * in_features;
  const int64_t meta_base = out_pack * groups;

  float acc[MAX_ROWS][8];
#pragma unroll
  for (int r = 0; r < MAX_ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[r][j] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_mk[meta_base + group]);
    int z[8];
    const scalar_t* scale_ptr = &scales_mk[(meta_base + group) * 8];
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      z[j] = static_cast<int>((pz >> awq_shift_for_pack_lane(j)) & 0xF);
      s[j] = scalar_to_float(scale_ptr[j]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t pw = static_cast<uint32_t>(qweight_mk[qw_base + k_base + j]);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const float w = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(jj)) & 0xF) - z[jj]) * s[jj];
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) {
          if (r >= rows) break;
          const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + k_base + j]);
          acc[r][jj] = fmaf(xv, w, acc[r][jj]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_mk[meta_base + group]);
    const uint32_t pw = static_cast<uint32_t>(qweight_mk[qw_base + in_col]);
    const scalar_t* scale_ptr = &scales_mk[(meta_base + group) * 8];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const float w = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(jj)) & 0xF) -
                                         static_cast<int>((pz >> awq_shift_for_pack_lane(jj)) & 0xF)) *
                      scalar_to_float(scale_ptr[jj]);
#pragma unroll
      for (int r = 0; r < MAX_ROWS; ++r) {
        if (r >= rows) break;
        const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
        acc[r][jj] = fmaf(xv, w, acc[r][jj]);
      }
    }
  }

  // Per-row reduction, identical pattern to the single-row MARLIN_K_SHFL_REDUCE,
  // reusing one shared exchange buffer across rows with a barrier between rows.
  __shared__ float xchg[8 * 8];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  const int sub_groups = blockDim.x >> 5;
  for (int r = 0; r < MAX_ROWS; ++r) {
    if (r >= rows) break;
    float a0 = acc[r][0], a1 = acc[r][1], a2 = acc[r][2], a3 = acc[r][3];
    float a4 = acc[r][4], a5 = acc[r][5], a6 = acc[r][6], a7 = acc[r][7];
    for (int offset = 16; offset > 0; offset >>= 1) {
      a0 += __shfl_down(a0, offset);
      a1 += __shfl_down(a1, offset);
      a2 += __shfl_down(a2, offset);
      a3 += __shfl_down(a3, offset);
      a4 += __shfl_down(a4, offset);
      a5 += __shfl_down(a5, offset);
      a6 += __shfl_down(a6, offset);
      a7 += __shfl_down(a7, offset);
    }
    __syncthreads();
    if (sub_lane == 0) {
      xchg[sub_id * 8 + 0] = a0;
      xchg[sub_id * 8 + 1] = a1;
      xchg[sub_id * 8 + 2] = a2;
      xchg[sub_id * 8 + 3] = a3;
      xchg[sub_id * 8 + 4] = a4;
      xchg[sub_id * 8 + 5] = a5;
      xchg[sub_id * 8 + 6] = a6;
      xchg[sub_id * 8 + 7] = a7;
    }
    __syncthreads();
    if (tid == 0) {
      float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
      float r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        r0 += xchg[g * 8 + 0];
        r1 += xchg[g * 8 + 1];
        r2 += xchg[g * 8 + 2];
        r3 += xchg[g * 8 + 3];
        r4 += xchg[g * 8 + 4];
        r5 += xchg[g * 8 + 5];
        r6 += xchg[g * 8 + 6];
        r7 += xchg[g * 8 + 7];
      }
      scalar_t* out_ptr = &out[static_cast<int64_t>(r) * out_features + out_col];
      out_ptr[0] = float_to_scalar<scalar_t>(r0);
      out_ptr[1] = float_to_scalar<scalar_t>(r1);
      out_ptr[2] = float_to_scalar<scalar_t>(r2);
      out_ptr[3] = float_to_scalar<scalar_t>(r3);
      out_ptr[4] = float_to_scalar<scalar_t>(r4);
      out_ptr[5] = float_to_scalar<scalar_t>(r5);
      out_ptr[6] = float_to_scalar<scalar_t>(r6);
      out_ptr[7] = float_to_scalar<scalar_t>(r7);
    }
  }
}

int launch_gemv_paro_marlin_k_fma_multi_row_fp16(
    const half_t* x,
    const int32_t* qweight_mk,
    const int32_t* qzeros_mk,
    const half_t* scales_mk,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_mk == nullptr || qzeros_mk == nullptr || scales_mk == nullptr || out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || rows > 8 || in_features <= 0 || out_packed <= 0 || group_size != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (in_features % group_size != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (threads != 32 && threads != 64 && threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_packed));
  const dim3 block(static_cast<unsigned int>(threads));
#define MARLIN_K_MULTI_ROW_LAUNCH(MAX_ROWS)                                       \
  hipLaunchKernelGGL((gemv_paro_marlin_k_fma_multi_row_kernel<half_t, MAX_ROWS>), \
                     grid, block, 0, stream, x, qweight_mk, qzeros_mk, scales_mk, \
                     out, rows, in_features, out_packed, static_cast<int>(group_size))
  switch (static_cast<int>(rows)) {
    case 1: MARLIN_K_MULTI_ROW_LAUNCH(1); break;
    case 2: MARLIN_K_MULTI_ROW_LAUNCH(2); break;
    case 3: MARLIN_K_MULTI_ROW_LAUNCH(3); break;
    case 4: MARLIN_K_MULTI_ROW_LAUNCH(4); break;
    case 5: MARLIN_K_MULTI_ROW_LAUNCH(5); break;
    case 6: MARLIN_K_MULTI_ROW_LAUNCH(6); break;
    case 7: MARLIN_K_MULTI_ROW_LAUNCH(7); break;
    default: MARLIN_K_MULTI_ROW_LAUNCH(8); break;
  }
#undef MARLIN_K_MULTI_ROW_LAUNCH
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gemv_paro_marlin_k_fma_fp16(
    const half_t* x,
    const int32_t* qweight_mk,
    const int32_t* qzeros_mk,
    const half_t* scales_mk,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_gemv_paro_marlin_k_fma_fp16(
      x, qweight_mk, qzeros_mk, scales_mk, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_paro_marlin_k_fma_multi_row_fp16(
    const half_t* x,
    const int32_t* qweight_mk,
    const int32_t* qzeros_mk,
    const half_t* scales_mk,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_gemv_paro_marlin_k_fma_multi_row_fp16(
      x, qweight_mk, qzeros_mk, scales_mk, out, rows, in_features, out_packed, group_size, threads, stream);
}
