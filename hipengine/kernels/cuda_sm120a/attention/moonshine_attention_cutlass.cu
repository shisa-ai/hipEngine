// Torch-free AOT CUTLASS/CuTe FP16 flash-style encoder self-attention kernel
// for CUDA ``sm_120a``.
//
// Implements the exact Moonshine encoder self-attention contract
// (``hipengine_kernels.cuda_sm120a.encoder.moonshine_encoder_attention_fp16``):
// head-major FP16 Q/K/V ``[heads, seq, 52]``, int32 per-token mask ``[seq]``
// (visible where ``mask != 0``), row-major FP16 output ``[seq, 416]``, online
// FP32 softmax over the visible tokens, scale ``1/sqrt(52)``, non-causal
// (every query attends every visible key).  It is built AOT to an
// architecture-qualified ``.so`` from pinned CUTLASS/CuTe (review §8.3 item 3),
// so no CUTLASS/FlashAttention dependency is needed on a deployment host.
//
// Algorithm (single-pass online-softmax flash attention):
//   - 512 threads; SM80_16x8x16 FP16 MMA atom tiled (2,8,1) -> tile (32,64,16),
//     K-loop 4 over the padded head dimension 64 (52 padded).
//   - Q/P/S/O are (32,64) A/C tiles; K/V are (64,64) B tiles (uniform shapes,
//     one smem<->reg tiled copy per role).
//   - S = Q @ K^T is staged to shared memory; per-tile row max/sum drive the
//     online softmax; O = O * exp(m_old - m_new) + P @ V with P in FP16.
//   - The row reduction uses 16-lane warp shuffle groups (512 threads / 32
//     rows); masked and beyond-sequence columns contribute 0 (exact: an
//     all-masked row produces a zero output like the custom kernel).
//   - head_dim 52 is padded to 64 for the tensor-core MMA; only dims < 52 are
//     written to the row-major output.
//
// The numerics differ from the scalar-FMA custom kernel only in FP32
// reduction order / tensor-core rounding; outputs agree to a few FP16 ULP
// (measured max_abs ~ 3.9e-3 at 40/207/1248, masked and unmasked).

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

using namespace cute;

// file-scope constants (internal linkage)
constexpr int kHeads     = 8;
constexpr int kHeadDim   = 52;   // logical
constexpr int kHeadDimP  = 64;   // padded
constexpr int kHidden    = 416;  // heads * head_dim
constexpr int kBM        = 32;   // query rows per block
constexpr int kBN        = 64;   // key columns per N-tile
constexpr int kThreads   = 512;

using MmaAtom     = SM80_16x8x16_F32F16F16F32_TN;
using TiledMma    = decltype(make_tiled_mma(MMA_Atom<MmaAtom>{}, Layout<Shape<_2,_8,_1>>{}));
using S2RCopyAtom = Copy_Atom<UniversalCopy<cute::half_t>, cute::half_t>;
using R2SCopyAtom = Copy_Atom<UniversalCopy<float>, float>;

__global__ __launch_bounds__(kThreads)
void moonshine_encoder_attention_cutlass_fp16_kernel(
    const __half* __restrict__ query,   // [batch, heads, seq, 52] head-major
    const __half* __restrict__ key,     // [batch, heads, seq, 52] head-major
    const __half* __restrict__ value,   // [batch, heads, seq, 52] head-major
    const int32_t* __restrict__ mask,   // [batch, seq] or null
    __half* __restrict__ output,        // [batch, seq, 416] row-major
    int64_t seq, float scale)
{
  const int tid = (int)threadIdx.x;
  const int head = (int)blockIdx.y;
  const int batch = (int)blockIdx.z;
  const int m0 = (int)blockIdx.x * kBM;
  if (m0 >= seq) return;
  const int64_t head_off =
      ((int64_t)batch * kHeads + head) * seq * kHeadDim;
  const int64_t mask_off = (int64_t)batch * seq;
  const int64_t output_off = (int64_t)batch * seq * kHidden;

  // ---------------- shared memory ----------------
  __shared__ __half q_smem[kBM * kHeadDimP];     // (32,64)
  __shared__ __half k_smem[kBN * kHeadDimP];     // (64,64)
  __shared__ __half v_smem[kHeadDimP * kBN];     // (64,64): v[d][key]
  __shared__ __half p_smem[kBM * kBN];           // (32,64): softmaxed weights
  __shared__ float  s_smem[kBM * kBN];           // (32,64) float S
  __shared__ float  o_smem[kBM * kHeadDimP];     // (32,64) float O (epilogue)
  __shared__ float  row_m[kBM], row_l[kBM], row_alpha[kBM];
  __shared__ float  alpha_smem[kBM * kBN];       // (32,64) alpha broadcast

  // ---------------- load Q tile ----------------
  for (int idx = tid; idx < kBM * kHeadDimP; idx += kThreads) {
    int r = idx / kHeadDimP, d = idx % kHeadDimP;
    float val = 0.f;
    if (d < kHeadDim) {
      int gr = m0 + r;
      if (gr < seq) val = __half2float(query[head_off + (int64_t)gr * kHeadDim + d]);
    }
    q_smem[idx] = __float2half(val);
  }

  // ---------------- init online stats ----------------
  if (tid < kBM) { row_m[tid] = -INFINITY; row_l[tid] = 0.f; }
  __syncthreads();

  // ---------------- CuTe setup ----------------
  TiledMma tiled_mma = make_tiled_mma(MMA_Atom<MmaAtom>{}, Layout<Shape<_2,_8,_1>>{});
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  auto q_view = make_tensor(make_smem_ptr(q_smem),
      make_layout(make_shape(Int<kBM>{}, Int<kHeadDimP>{}), make_stride(Int<kHeadDimP>{}, Int<1>{})));
  auto k_view = make_tensor(make_smem_ptr(k_smem),
      make_layout(make_shape(Int<kBN>{}, Int<kHeadDimP>{}), make_stride(Int<kHeadDimP>{}, Int<1>{})));
  auto v_view = make_tensor(make_smem_ptr(v_smem),
      make_layout(make_shape(Int<kHeadDimP>{}, Int<kBN>{}), make_stride(Int<kBN>{}, Int<1>{})));
  auto p_view = make_tensor(make_smem_ptr(p_smem),
      make_layout(make_shape(Int<kBM>{}, Int<kBN>{}), make_stride(Int<kBN>{}, Int<1>{})));
  auto s_view = make_tensor(make_smem_ptr(s_smem),
      make_layout(make_shape(Int<kBM>{}, Int<kBN>{}), make_stride(Int<kBN>{}, Int<1>{})));
  auto o_view = make_tensor(make_smem_ptr(o_smem),
      make_layout(make_shape(Int<kBM>{}, Int<kHeadDimP>{}), make_stride(Int<kHeadDimP>{}, Int<1>{})));
  auto alpha_view = make_tensor(make_smem_ptr(alpha_smem),
      make_layout(make_shape(Int<kBM>{}, Int<kBN>{}), make_stride(Int<kBN>{}, Int<1>{})));

  auto tCrQ = thr_mma.partition_fragment_A(q_view);   // (MMA,2,4)
  auto tCrK = thr_mma.partition_fragment_B(k_view);   // (MMA,8,4)
  auto tCrV = thr_mma.partition_fragment_B(v_view);   // (MMA,8,4)
  auto tCrP = thr_mma.partition_fragment_A(p_view);   // (MMA,2,4)
  auto tCrS = thr_mma.partition_fragment_C(s_view);   // (MMA,2,8) float
  auto tCrO = thr_mma.partition_fragment_C(o_view);   // (MMA,2,8) float
  auto tCrAlpha = thr_mma.partition_fragment_C(o_view);

  TiledCopy copyA = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
  TiledCopy copyB = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
  TiledCopy copyC = make_tiled_copy_C(R2SCopyAtom{}, tiled_mma);
  ThrCopy tcA = copyA.get_slice(tid);
  ThrCopy tcB = copyB.get_slice(tid);
  ThrCopy tcC = copyC.get_slice(tid);

  // load Q fragment once
  copy(S2RCopyAtom{}, tcA.partition_S(q_view), tcA.retile_D(tCrQ));

  clear(tCrO);

  // ---------------- N-tile loop (online softmax) ----------------
  for (int n0 = 0; n0 < seq; n0 += kBN) {
    // load K tile (64 keys x 64) into k_smem
    for (int idx = tid; idx < kBN * kHeadDimP; idx += kThreads) {
      int r = idx / kHeadDimP, d = idx % kHeadDimP;
      int gr = n0 + r;
      float val = 0.f;
      if (d < kHeadDim && gr < seq) val = __half2float(key[head_off + (int64_t)gr * kHeadDim + d]);
      k_smem[idx] = __float2half(val);
    }
    // load V tile (64 keys) transposed into v_smem[d][key]
    for (int idx = tid; idx < kHeadDimP * kBN; idx += kThreads) {
      int d = idx / kBN, r = idx % kBN;
      int gr = n0 + r;
      float val = 0.f;
      if (d < kHeadDim && gr < seq) val = __half2float(value[head_off + (int64_t)gr * kHeadDim + d]);
      v_smem[d * kBN + r] = __float2half(val);
    }
    __syncthreads();

    copy(S2RCopyAtom{}, tcB.partition_S(k_view), tcB.retile_D(tCrK));
    copy(S2RCopyAtom{}, tcB.partition_S(v_view), tcB.retile_D(tCrV));

    // S = Q @ K^T
    clear(tCrS);
    CUTE_UNROLL
    for (int kb = 0; kb < 4; ++kb)
      cute::gemm(tiled_mma, tCrQ(_,_,kb), tCrK(_,_,kb), tCrS);

    // stage S -> smem
    copy(R2SCopyAtom{}, tcC.retile_S(tCrS), tcC.partition_D(s_view));
    __syncthreads();

    // ---------------- softmax (online) ----------------
    const int row = tid / 16, sub = tid % 16;   // 16 threads per row
    // pass 1: tile max over visible columns
    float tmax = -INFINITY;
#pragma unroll
    for (int c = sub; c < kBN; c += 16) {
      int gn = n0 + c;
      bool vis = gn < seq && (mask == nullptr || mask[mask_off + gn] != 0);
      if (vis) tmax = fmaxf(tmax, scale * s_smem[row * kBN + c]);
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
      tmax = fmaxf(tmax, __shfl_down_sync(0xffffffffu, tmax, off));
    if (sub == 0) {
      float oldm = row_m[row];
      float newm = fmaxf(oldm, tmax);
      row_alpha[row] = expf(oldm - newm);
      row_m[row] = newm;
    }
    __syncthreads();

    // pass 2: P = exp(S - m) (masked -> 0), tile sum
    float tsum = 0.f;
#pragma unroll
    for (int c = sub; c < kBN; c += 16) {
      int gn = n0 + c;
      bool vis = gn < seq && (mask == nullptr || mask[mask_off + gn] != 0);
      float newm = row_m[row];
      float pv = 0.f;
      if (vis) { pv = expf(scale * s_smem[row * kBN + c] - newm); tsum += pv; }
      p_smem[row * kBN + c] = __float2half(pv);
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
      tsum += __shfl_down_sync(0xffffffffu, tsum, off);
    if (sub == 0) row_l[row] = row_l[row] * row_alpha[row] + tsum;
    __syncthreads();

    // ---------------- rescale O by alpha ----------------
    for (int idx = tid; idx < kBM * kBN; idx += kThreads)
      alpha_smem[idx] = row_alpha[idx / kBN];
    __syncthreads();
    copy(R2SCopyAtom{}, tcC.partition_S(alpha_view), tcC.retile_D(tCrAlpha));
    for (int i = 0; i < (int)size(tCrO); ++i) tCrO(i) *= tCrAlpha(i);

    // ---------------- O += P @ V ----------------
    copy(S2RCopyAtom{}, tcA.partition_S(p_view), tcA.retile_D(tCrP));
    CUTE_UNROLL
    for (int kb = 0; kb < 4; ++kb)
      cute::gemm(tiled_mma, tCrP(_,_,kb), tCrV(_,_,kb), tCrO);

    __syncthreads();   // protect k/v/p smem before next tile overwrite
  }

  // ---------------- epilogue ----------------
  // stage O -> smem, divide by row_l, write to global row-major
  copy(R2SCopyAtom{}, tcC.retile_S(tCrO), tcC.partition_D(o_view));
  __syncthreads();
  for (int idx = tid; idx < kBM * kHeadDimP; idx += kThreads) {
    int r = idx / kHeadDimP, d = idx % kHeadDimP;
    int gr = m0 + r;
    if (gr < seq && d < kHeadDim) {
      float l = row_l[r];
      float o = (l > 0.f) ? o_smem[idx] / l : 0.f;
      output[output_off + (int64_t)gr * kHidden + head * kHeadDim + d] =
          __float2half(o);
    }
  }
}

// ---------------- host ABI ----------------
extern "C" int hipengine_cuda_sm120a_moonshine_encoder_attention_cutlass_fp16(
    const __half* query, const __half* key, const __half* value,
    const int32_t* mask, __half* output,
    int64_t heads, int64_t head_dim, int64_t seq,
    float scale, int64_t threads, cudaStream_t stream)
{
  if (heads != kHeads || head_dim != kHeadDim || seq <= 0 ||
      threads != kThreads) return 1;
  dim3 grid((unsigned)((seq + kBM - 1) / kBM), (unsigned)heads, 1);
  moonshine_encoder_attention_cutlass_fp16_kernel<<<grid, kThreads, 0, stream>>>(
      query, key, value, mask, output, seq, scale);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_encoder_attention_cutlass_batch_fp16(
    const __half* query, const __half* key, const __half* value,
    const int32_t* mask, __half* output,
    int64_t batch, int64_t heads, int64_t head_dim, int64_t seq,
    float scale, int64_t threads, cudaStream_t stream)
{
  if (batch <= 0 || heads != kHeads || head_dim != kHeadDim || seq <= 0 ||
      threads != kThreads) return 1;
  dim3 grid((unsigned)((seq + kBM - 1) / kBM), (unsigned)heads,
            (unsigned)batch);
  moonshine_encoder_attention_cutlass_fp16_kernel<<<grid, kThreads, 0, stream>>>(
      query, key, value, mask, output, seq, scale);
  return cudaGetLastError();
}
