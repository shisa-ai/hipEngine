// QSA dense-attention flash prefill for Qwen4Exp (contiguous c1 chunks).
//
// Two-kernel design specialized for the QSA geometry (24 q-heads x 256 dim,
// 2 kv-heads GQA 12:1, paged bf16 KV with 256-token blocks, dense-equivalent
// context <= ~2051+chunk):
//
//   1. qwen4_exp_qsa_gather_kv_f32_kernel: gathers the live context K/V from
//      the paged bf16 cache into contiguous f32 scratch via the block table.
//   2. qwen4_exp_qsa_flash_prefill_kernel: block-per-Q-tile online-softmax
//      attention over the contiguous scratch (lineage: llama.cpp
//      flash_attn_tile scalar path, ggml-org/llama.cpp@17252c769
//      ggml/src/ggml-cuda/fattn-tile.cuh). Each block owns QT query rows of
//      one q-head; 8 warps split the Q tile rows; lanes split the K dim for
//      the QK dot and the V accumulation. All math f32.
//
// The layout contract follows the existing spans decode kernel so the
// runner can route between them per layer.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int QSA_QT = 32;   // query rows per block
constexpr int QSA_KT = 32;   // KV tokens per tile
constexpr int QSA_THREADS = 256;

__device__ inline float qsa_bf16_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__global__ void qwen4_exp_qsa_gather_kv_f32_kernel(
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    int64_t block_table_len,
    int64_t block_size,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t context_len,
    uint16_t* __restrict__ k_out,
    uint16_t* __restrict__ v_out) {
  // one thread per (token, dim); heads handled by looping kv_heads*? -> use
  // grid.y for the (kv_head) pair? Simplest: thread = (token*head_dim + dim)
  // over the full context; loop kv_heads inside.
  const int64_t token_dim = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t row_elems = kv_heads * head_dim;
  const int64_t total = context_len * row_elems;
  for (int64_t idx = token_dim; idx < total; idx += stride) {
    const int64_t token = idx / row_elems;
    const int64_t rem = idx % row_elems;
    const int64_t kv_head = rem / head_dim;
    const int64_t dim = rem % head_dim;
    const int64_t logical_block = token / block_size;
    const int64_t physical = block_table[logical_block];
    const int64_t block_offset = token % block_size;
    const int64_t src = (((physical * block_size) + block_offset) * kv_heads + kv_head) * head_dim + dim;
    const int64_t dst = (token * kv_heads + kv_head) * head_dim + dim;
    k_out[dst] = key_cache[src];
    v_out[dst] = value_cache[src];
  }
}

__global__ void __launch_bounds__(QSA_THREADS, 2)
qwen4_exp_qsa_flash_prefill_kernel(
    const float* __restrict__ query,       // [rows, q_heads, head_dim]
    const uint16_t* __restrict__ k_gathered,  // [context, kv_heads, head_dim] bf16
    const uint16_t* __restrict__ v_gathered,  // [context, kv_heads, head_dim] bf16
    const int64_t* __restrict__ positions, // [rows] absolute positions
    float* __restrict__ out,               // [rows, q_heads, head_dim] f32
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t context_len,
    float scale) {
  __shared__ uint16_t q_tile[QSA_QT * 256];
  __shared__ uint16_t k_tile[QSA_KT * 256];
  __shared__ uint16_t v_tile[QSA_KT * 256];

  const int64_t q_head = blockIdx.y;
  const int row0 = blockIdx.x * QSA_QT;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t kv_group = q_heads / kv_heads;
  const int64_t kv_head = q_head / kv_group;

  // Load the Q tile (rows clipped): [QT, 256] f32.
  for (int idx = threadIdx.x; idx < QSA_QT * 256; idx += QSA_THREADS) {
    const int r = idx / 256;
    const int d = idx % 256;
    const int row = row0 + r;
    const float v = (row < rows && d < head_dim)
        ? query[(static_cast<int64_t>(row) * q_heads + q_head) * head_dim + d]
        : 0.0f;
    union {
      float f32;
      uint32_t u32;
    } in;
    in.f32 = v;
    const uint32_t lsb = (in.u32 >> 16) & 1U;
    in.u32 += 0x7FFFU + lsb;
    q_tile[idx] = static_cast<uint16_t>(in.u32 >> 16);
  }
  __syncthreads();

  // Key-parallel layout: lane k owns key k. Per row: 256 serial-dim FMAs for
  // the QK dot (no per-key shuffles); softmax max/sum use one butterfly per
  // row; p values stage through LDS so the accumulation phase reads them by
  // (row, key). Accumulators live in LDS and survive tiles with the online
  // rescale.
  constexpr int ROWS_PER_WARP = QSA_QT / 8;
  const int my_row_base = warp * ROWS_PER_WARP;

  __shared__ float m_run[QSA_QT];
  __shared__ float l_run[QSA_QT];
  __shared__ float s_p[QSA_QT * QSA_KT];
  __shared__ float rescale[QSA_QT];

  // Per-thread register accumulators: the (row, dim) mapping below is stable
  // across tiles (idx = threadIdx.x + i*QSA_THREADS), so each thread owns the
  // same 16 (row, dim) cells throughout.
  constexpr int ACC_CELLS = QSA_QT * 256 / QSA_THREADS;  // 16
  float acc_reg[ACC_CELLS];
#pragma unroll
  for (int i = 0; i < ACC_CELLS; ++i) acc_reg[i] = 0.0f;

  for (int idx = threadIdx.x; idx < QSA_QT; idx += QSA_THREADS) {
    m_run[idx] = -3.4e38f;
    l_run[idx] = 0.0f;
  }
  __syncthreads();

  const int64_t max_context = context_len;
  const int64_t kv_tiles = (max_context + QSA_KT - 1) / QSA_KT;

  for (int64_t tile = 0; tile < kv_tiles; ++tile) {
    const int64_t k0 = tile * QSA_KT;
    const int64_t k_count = min(static_cast<int64_t>(QSA_KT), max_context - k0);
    // Cooperative K/V tile load (bf16 in LDS).
    for (int idx = threadIdx.x; idx < QSA_KT * 256; idx += QSA_THREADS) {
      const int t = idx / 256;
      const int d = idx % 256;
      const int64_t src = (k0 + t) * kv_heads * head_dim + kv_head * head_dim + d;
      const bool ok = t < k_count && d < head_dim;
      k_tile[idx] = ok ? k_gathered[src] : 0;
      v_tile[idx] = ok ? v_gathered[src] : 0;
    }
    __syncthreads();

    // Phase 1: per warp-row, lane k computes dot(row, key_k) serially.
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int q_row = my_row_base + r;
      const int row = row0 + q_row;
      const bool row_ok = row < rows;
      const int64_t pos = row_ok ? positions[row] : -1;
      float dot = -3.4e38f;  // masked
      if (row_ok && k0 + static_cast<int64_t>(lane) <= pos && lane < k_count) {
        dot = 0.0f;
        const uint16_t* qrow = q_tile + q_row * 256;
        const uint16_t* krow = k_tile + lane * 256;
#pragma unroll 8
        for (int d = 0; d < 256; ++d) {
          dot += qsa_bf16_to_float(qrow[d]) * qsa_bf16_to_float(krow[d]);
        }
        dot *= scale;
      }
      float mx = dot;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        mx = fmaxf(mx, __shfl_xor(mx, offset));
      }
      const float row_max = fmaxf(mx, m_run[q_row]);
      if (lane == 0) {
        rescale[q_row] = expf(m_run[q_row] - row_max);
        m_run[q_row] = row_max;
      }
      const float p = expf(dot - row_max);  // masked -> 0
      float l_partial = p;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        l_partial += __shfl_xor(l_partial, offset);
      }
      if (lane == 0) {
        l_run[q_row] = l_run[q_row] * rescale[q_row] + l_partial;
      }
      s_p[q_row * QSA_KT + lane] = p;
    }
    __syncthreads();

    // Phase 2: accumulation into register cells; serial over keys.
#pragma unroll
    for (int i = 0; i < ACC_CELLS; ++i) {
      const int idx = threadIdx.x + i * QSA_THREADS;
      const int q_row = idx / 256;
      const int d = idx % 256;
      float sum = 0.0f;
      for (int k = 0; k < QSA_KT; ++k) {
        sum += s_p[q_row * QSA_KT + k] * qsa_bf16_to_float(v_tile[k * 256 + d]);
      }
      acc_reg[i] = acc_reg[i] * rescale[q_row] + sum;
    }
    __syncthreads();
  }

  // Normalize and write f32 output from the register cells.
#pragma unroll
  for (int i = 0; i < ACC_CELLS; ++i) {
    const int idx = threadIdx.x + i * QSA_THREADS;
    const int q_row = idx / 256;
    const int d = idx % 256;
    const int row = row0 + q_row;
    if (row < rows && d < head_dim) {
      const float inv_l = l_run[q_row] > 0.0f ? 1.0f / l_run[q_row] : 0.0f;
      out[(static_cast<int64_t>(row) * q_heads + q_head) * head_dim + d] =
          acc_reg[i] * inv_l;
    }
  }

}

}  // namespace

extern "C" int hipengine_qwen4_exp_qsa_flash_prefill(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* block_table,
    const void* positions,
    void* k_scratch,
    void* v_scratch,
    void* out,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t block_table_len,
    int64_t context_len,
    float scale,
    hipStream_t stream) {
  if (!query || !key_cache || !value_cache || !block_table || !positions ||
      !k_scratch || !v_scratch || !out || rows <= 0 || q_heads <= 0 ||
      kv_heads <= 0 || q_heads % kv_heads || head_dim != 256 ||
      block_size <= 0 || block_table_len <= 0 || context_len <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  {
    const int64_t total = context_len * kv_heads * head_dim;
    const int64_t threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(
        qwen4_exp_qsa_gather_kv_f32_kernel,
        dim3(static_cast<uint32_t>(blocks)), dim3(threads), 0, stream,
        static_cast<const uint16_t*>(key_cache),
        static_cast<const uint16_t*>(value_cache),
        static_cast<const int32_t*>(block_table),
        block_table_len, block_size, kv_heads, head_dim, context_len,
        static_cast<uint16_t*>(k_scratch), static_cast<uint16_t*>(v_scratch));
    const int err = static_cast<int>(hipGetLastError());
    if (err != 0) return err;
  }
  {
    const int64_t q_blocks = (rows + QSA_QT - 1) / QSA_QT;
    hipLaunchKernelGGL(
        qwen4_exp_qsa_flash_prefill_kernel,
        dim3(static_cast<uint32_t>(q_blocks),
             static_cast<uint32_t>(q_heads)),
        dim3(QSA_THREADS), 0, stream,
        static_cast<const float*>(query),
        static_cast<const uint16_t*>(k_scratch),
        static_cast<const uint16_t*>(v_scratch),
        static_cast<const int64_t*>(positions),
        static_cast<float*>(out),
        rows, q_heads, kv_heads, head_dim, context_len, scale);
    return static_cast<int>(hipGetLastError());
  }
}
