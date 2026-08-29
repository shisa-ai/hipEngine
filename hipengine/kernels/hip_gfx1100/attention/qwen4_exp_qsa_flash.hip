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
  __shared__ float q_tile[QSA_QT * 256];
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
    q_tile[idx] = (row < rows && d < head_dim)
        ? query[(static_cast<int64_t>(row) * q_heads + q_head) * head_dim + d]
        : 0.0f;
  }
  __syncthreads();

  // Each warp owns QSA_QT/8 = 2 query rows; lanes split the head dim. The
  // XOR-butterfly reduction leaves each dot on every lane, so scores are
  // buffered in registers (no shared s_tile).
  constexpr int ROWS_PER_WARP = QSA_QT / 8;
  const int my_row_base = warp * ROWS_PER_WARP;

  // Per-row online softmax state and V accumulator (8 dims per lane).
  float m_val[ROWS_PER_WARP];
  float l_val[ROWS_PER_WARP];
  float acc[ROWS_PER_WARP][8];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    m_val[r] = -3.4e38f;
    l_val[r] = 0.0f;
#pragma unroll
    for (int d = 0; d < 8; ++d) acc[r][d] = 0.0f;
  }

  const int64_t max_context = context_len;
  const int64_t kv_tiles = (max_context + QSA_KT - 1) / QSA_KT;

  for (int64_t tile = 0; tile < kv_tiles; ++tile) {
    const int64_t k0 = tile * QSA_KT;
    const int64_t k_count = min(static_cast<int64_t>(QSA_KT), max_context - k0);
    // Load the K and V tiles cooperatively (V stays bf16 in LDS).
    for (int idx = threadIdx.x; idx < QSA_KT * 256; idx += QSA_THREADS) {
      const int t = idx / 256;
      const int d = idx % 256;
      const int64_t src = (k0 + t) * kv_heads * head_dim + kv_head * head_dim + d;
      const bool ok = t < k_count && d < head_dim;
      k_tile[idx] = ok ? k_gathered[src] : 0;
      v_tile[idx] = ok ? v_gathered[src] : 0;
    }
    __syncthreads();

    // Per row: compute all 32 dots (butterfly leaves the sum on every lane),
    // buffer them, then run the online-softmax update and V accumulation.
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int row = row0 + my_row_base + r;
      const bool row_ok = row < rows;
      const int64_t pos = row_ok ? positions[row] : -1;
      float dots[QSA_KT];
      float tile_max = m_val[r];
#pragma unroll
      for (int k = 0; k < QSA_KT; ++k) {
        const int64_t abs_k = k0 + k;
        float dot = -3.4e38f;  // masked keys must not inflate the max
        if (row_ok && abs_k <= pos && abs_k < max_context) {
          dot = 0.0f;
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            dot += q_tile[(my_row_base + r) * 256 + lane * 8 + d] *
                   qsa_bf16_to_float(k_tile[k * 256 + lane * 8 + d]);
          }
          dot *= scale;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          dot += __shfl_xor(dot, offset);
        }
        dots[k] = dot;
        tile_max = fmaxf(tile_max, dot);
      }
      const float exp_scale = expf(m_val[r] - tile_max);
      // Rescale the running accumulator once per tile (l is rescaled below
      // with the same factor; both must happen regardless of key hits).
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        acc[r][d] *= exp_scale;
      }
      float l_new = 0.0f;
#pragma unroll
      for (int k = 0; k < QSA_KT; ++k) {
        const int64_t abs_k = k0 + k;
        const float p = expf(dots[k] - tile_max);  // masked dots give 0
        l_new += p;
        if (p != 0.0f) {
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            const int vd = lane * 8 + d;
            const float vval = vd < head_dim
                ? qsa_bf16_to_float(v_tile[k * 256 + vd]) : 0.0f;
            acc[r][d] = fmaf(p, vval, acc[r][d]);
          }
        }
      }
      l_val[r] = l_val[r] * exp_scale + l_new;
      m_val[r] = tile_max;
    }
    __syncthreads();
  }

  // Normalize and write bf16 output for this warp's rows.
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    const int row = row0 + my_row_base + r;
    if (row >= rows) continue;
    const float inv_l = l_val[r] > 0.0f ? 1.0f / l_val[r] : 0.0f;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      const int64_t vd = lane * 8 + d;
      if (vd < head_dim) {
        out[(static_cast<int64_t>(row) * q_heads + q_head) * head_dim + vd] =
            acc[r][d] * inv_l;
      }
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
