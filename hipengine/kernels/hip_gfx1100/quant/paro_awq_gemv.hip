// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// Selected and generic pack8 GEMV kernel bodies are preserved; wrappers are HIPENGINE C ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

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
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

__device__ inline int awq_shift_for_pack_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

// RDNA3 CU-mode wave64 shuffles only exchange within 32-lane half-waves.
// Reduce each half-wave with __shfl_down, then explicitly sum however many
// half-waves the launched block actually has. The pack8 wrappers allow
// 64/128/256 threads; the previous fixed 4-half-wave sum read uninitialized
// xchg slots for 64-thread launches and could write past xchg for 256.
#define PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
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

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_pack8_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size,
    bool has_bias) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t x_base = row * in_features;
  const int64_t qw_base = qweight_transposed ? out_pack * in_features : out_pack;

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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
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
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + j)
          : ((k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
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

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    if (has_bias) {
      out_ptr[0] = float_to_scalar<scalar_t>(r0 + scalar_to_float(bias[out_col + 0]));
      out_ptr[1] = float_to_scalar<scalar_t>(r1 + scalar_to_float(bias[out_col + 1]));
      out_ptr[2] = float_to_scalar<scalar_t>(r2 + scalar_to_float(bias[out_col + 2]));
      out_ptr[3] = float_to_scalar<scalar_t>(r3 + scalar_to_float(bias[out_col + 3]));
      out_ptr[4] = float_to_scalar<scalar_t>(r4 + scalar_to_float(bias[out_col + 4]));
      out_ptr[5] = float_to_scalar<scalar_t>(r5 + scalar_to_float(bias[out_col + 5]));
      out_ptr[6] = float_to_scalar<scalar_t>(r6 + scalar_to_float(bias[out_col + 6]));
      out_ptr[7] = float_to_scalar<scalar_t>(r7 + scalar_to_float(bias[out_col + 7]));
    } else {
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

template <typename scalar_t, bool qweight_transposed, bool separate_inputs>
__launch_bounds__(128, 4)
__global__ void gemv_awq_dual_pack8_kernel(
    const scalar_t* __restrict__ x_a,
    const scalar_t* __restrict__ x_b,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t x_base = row * in_features;
  const scalar_t* __restrict__ x = (separate_inputs && use_b) ? x_b : x_a;
  const int32_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;
  const int64_t qweight_base = qweight_transposed ? (local_pack * in_features) : local_pack;

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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
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
      const int64_t qw_idx = qweight_transposed
          ? (qweight_base + k_base + j)
          : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed
        ? (qweight_base + in_col)
        : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
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

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

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

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4) __global__ void gemv_awq_selected_dual_pack8_strided_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t x_row = (x_rows == 1) ? 0 : row;
  const int64_t x_base = x_row * in_features;
  const int32_t* __restrict__ qweight_base = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros_base = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales_base = use_b ? scales_b : scales_a;
  const int64_t expert_qweight_stride = qweight_transposed
      ? (local_out_packed * in_features)
      : (in_features * local_out_packed);
  const int32_t* __restrict__ qweight = qweight_base + expert_id * expert_qweight_stride
      + (qweight_transposed ? (local_pack * in_features) : 0);
  const int32_t* __restrict__ qzeros = qzeros_base + expert_id * groups * local_out_packed;
  const scalar_t* __restrict__ scales = scales_base + expert_id * groups * local_out_features;

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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
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
      const int64_t qw_idx = qweight_transposed ? (k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? in_col : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
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

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

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

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_pack8_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int64_t num_experts,
    int group_size) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t out_col = out_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t x_base = row * in_features;
  const int64_t expert_qweight_base = expert_id * in_features * out_packed;
  const int64_t expert_qweight_t_base = expert_id * out_packed * in_features;
  const int64_t qw_base = qweight_transposed ? (expert_qweight_t_base + out_pack * in_features) : (expert_qweight_base + out_pack);
  const int32_t* __restrict__ qzeros_e = qzeros + expert_id * groups * out_packed;
  const scalar_t* __restrict__ scales_e = scales + expert_id * groups * out_features;

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
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
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
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + j)
          : (expert_qweight_base + (k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
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
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (expert_qweight_base + in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
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

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

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


// Fused rotation + selected dual pack8 GEMV kernel
// Applies PARO pairwise rotation to input, then runs selected-expert dual W4 GEMV
// on the rotated result, all in one launch with no intermediate tensor.
template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_dual_pack8_strided_rotate_out_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ channel_scales,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size,
    int krot) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t x_row = (x_rows == 1) ? 0 : row;
  const int64_t x_base = x_row * in_features;
  const int32_t* __restrict__ qweight_base = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros_base = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales_base = use_b ? scales_b : scales_a;
  const int64_t expert_qweight_stride = qweight_transposed
      ? (local_out_packed * in_features)
      : (in_features * local_out_packed);
  const int32_t* __restrict__ qweight = qweight_base + expert_id * expert_qweight_stride
      + (qweight_transposed ? (local_pack * in_features) : 0);
  const int32_t* __restrict__ qzeros = qzeros_base + expert_id * groups * local_out_packed;
  const scalar_t* __restrict__ scales = scales_base + expert_id * groups * local_out_features;

  // ---- Fused rotation: apply channel scales + pairwise rotation in shared memory ----
  // Stage the full rotated input row in dynamic shared memory.
  // For in_features=2048, this is 8 KiB; for in_features=512, 2 KiB.
  extern __shared__ float rotated_buf[];

  for (int64_t idx = tid; idx < in_features; idx += blockDim.x) {
    rotated_buf[idx] = scalar_to_float(x[x_base + idx]) * scalar_to_float(channel_scales[idx]);
  }
  __syncthreads();

  // Apply krot pairwise rotations
  const int half_group = group_size / 2;
  for (int r = 0; r < krot; ++r) {
    for (int g = 0; g < groups; ++g) {
      const int base = g * group_size;
      for (int lane = tid; lane < half_group; lane += blockDim.x) {
        const int pair_base = r * static_cast<int>(in_features) + base + 2 * lane;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (in_features / 2) + g * half_group + lane]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = rotated_buf[base + i];
        const float xj = rotated_buf[base + j];
        rotated_buf[base + i] = fmaf(xj, s, xi * c);
        rotated_buf[base + j] = fmaf(xi, -s, xj * c);
      }
      __syncthreads();
    }
  }

  // ---- Selected dual pack8 GEMV on rotated result ----
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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
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
      const float xv = rotated_buf[k_base + j];
      const int64_t qw_idx = qweight_transposed ? (k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
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
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? in_col : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = rotated_buf[in_col];
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

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


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128;
}

}  // namespace


extern "C" int hipengine_gemv_awq_pack8_strided_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_pack8_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight,
      qzeros,
      scales,
      nullptr,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      static_cast<int>(group_size),
      false);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_pack8_transposed_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_pack8_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight,
      qzeros,
      scales,
      nullptr,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      static_cast<int>(group_size),
      false);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_dual_pack8_strided_bf16(
    const uint16_t* x,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_kernel<uint16_t, false, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      nullptr,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_bf16(
    const uint16_t* x_a,
    const uint16_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_kernel<uint16_t, true, true>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x_a,
      x_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int16_t* pairs,
    const uint16_t* theta,
    const uint16_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || krot < 0 || (in_features % group_size) != 0 ||
      (group_size % 8) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(in_features * sizeof(float)),
      stream,
      x,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_pack8_strided_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight,
      qzeros,
      scales,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_pack8_transposed_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight,
      qzeros,
      scales,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}
