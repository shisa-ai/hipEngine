// YuE2 NAR (acoustic flow matching) device kernels.
//
// The NAR stage attends bidirectionally over the cached AR conditioning keys
// plus all of its own keys, and its ODE loop is device-resident so the BF16
// rounding points match the reference exactly. Storage is bf16; accumulation is
// fp32, matching the upstream torch implementation (which keeps BF16 tensors and
// FP32 accumulation inside the GEMM/attention).
//
// The attention kernel tiles its key loop and runs an online (rescaled) softmax,
// so shared memory stays bounded by the tile rather than by the key count: a
// chunk's key count is `ar_length + frames + 2`, which upstream allows up to the
// full model context, and materializing all scores would exceed shared memory.

#include <hip/hip_runtime.h>
#include <cstdint>

__device__ inline float yue2_bf16_to_float(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ inline uint16_t yue2_float_to_bf16(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

// out[row, c] = bf16(bf16(x[row, c]) + bf16(table[position[row], c]))
//
// The audio-position embedding is a checkpoint buffer indexed by the *local*
// NAR position, clamped to the table length, and added to the projected latent
// hidden state in BF16 -- one rounding, as torch's `a + b` on BF16 tensors does.
__global__ void yue2_nar_gather_add_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ table,
    const int64_t* __restrict__ positions,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * width;
  if (idx >= total) return;
  const int64_t row = idx / width;
  const int64_t col = idx - row * width;
  const float xv = yue2_bf16_to_float(x[idx]);
  const float pv = yue2_bf16_to_float(table[positions[row] * width + col]);
  out[idx] = yue2_float_to_bf16(xv + pv);
}

// out[row, c] = bf16(bf16(x[row, c]) + bf16(vector[c]))
//
// Broadcast add used for the timestep embedding, which is one row repeated at
// every NAR position.
__global__ void yue2_nar_add_broadcast_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ vector,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * width;
  if (idx >= total) return;
  const int64_t col = idx % width;
  const float xv = yue2_bf16_to_float(x[idx]);
  const float vv = yue2_bf16_to_float(vector[col]);
  out[idx] = yue2_float_to_bf16(xv + vv);
}

// out = bf16(state - bf16(velocity * scale))
//
// Two roundings, in this order: the reference computes `state - v * (h/2)` as
// BF16 tensor ops, so the product is rounded to BF16 before the subtraction.
// A fused multiply-add would round once and is therefore not equivalent. The
// scale stays FP32 because the reference multiplies by a Python float, which
// torch applies in the tensor's FP32 compute type.
__global__ void yue2_nar_state_update_kernel(
    const uint16_t* __restrict__ state,
    const uint16_t* __restrict__ velocity,
    float scale,
    uint16_t* __restrict__ out,
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const float sv = yue2_bf16_to_float(state[idx]);
  const float vv = yue2_bf16_to_float(velocity[idx]);
  const float product = yue2_bf16_to_float(yue2_float_to_bf16(vv * scale));
  out[idx] = yue2_float_to_bf16(sv - product);
}

// Bidirectional grouped-query attention over [AR cache | NAR keys].
//
// One block per (query head, query row). Keys 0..ar_rows-1 come from the cached
// AR conditioning, keys ar_rows..ar_rows+rows-1 from this step's NAR projection;
// nothing is masked, because the NAR attends to every visible key. The key loop
// is tiled and uses an online softmax: each thread owns one head_dim lane of the
// running accumulator, and the tile's scores live in shared memory.
//
// Q is FP32 (the reference's SDPA consumes the FP32 projections after RoPE), K/V
// are BF16, and the output is FP32.
//
// Each block owns YUE2_NAR_ROWS_PER_BLOCK query rows for one query head and walks
// the key tiles once, reusing every K and V element it loads across those rows.
// Blocking is what took the kernel off its old per-(row, head) traffic wall, but
// it is not bandwidth-bound at four rows: `rocprofv3 --pmc` on a 1299-row solve
// measures 3.5e8 cycles for 3.7e9 VALU / 4.3e8 LDS / 3.4e8 global instructions at
// 19% occupancy, i.e. an instruction and latency problem, not a traffic one. Two
// shapes of that dominate. A lane owns one key, so walking the K row one scalar
// at a time asks the L1 for 32 sectors per request (lanes land 256 bytes apart)
// 128 times per tile; and every lane re-reads the whole query row from L1, which
// is 128 broadcast loads per row per tile. Both are packed below into 16-byte
// requests, and the V walk is split into its AR and NAR ranges so it steps by
// pointer instead of multiplying and branching per key.
//
// Per-row arithmetic is untouched by any of that: the same serial dot product in
// the same d-ascending order, the same tile walk, and the same key-ascending
// accumulation order, so the output is bit-identical to the unblocked kernel
// (tests/fixtures/yue2/operators/nar_attention_parent.npz).
#define YUE2_NAR_ROWS_PER_BLOCK 8

__global__ void yue2_nar_attention_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ nar_k,
    const uint16_t* __restrict__ nar_v,
    const uint16_t* __restrict__ ar_k,
    const uint16_t* __restrict__ ar_v,
    float* __restrict__ out,
    int64_t rows,
    int64_t ar_rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t row0 = blockIdx.y * YUE2_NAR_ROWS_PER_BLOCK;
  const int64_t q_head = blockIdx.x;
  const int64_t lane = threadIdx.x;
  if (row0 >= rows || q_head >= num_q_heads || lane >= head_dim) return;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t keys = ar_rows + rows;

  extern __shared__ float shared[];
  float* scores = shared;                                       // [rows][blockDim]
  float* weights = shared + YUE2_NAR_ROWS_PER_BLOCK * blockDim.x;

  float running_max[YUE2_NAR_ROWS_PER_BLOCK];
  float running_sum[YUE2_NAR_ROWS_PER_BLOCK];
  float accumulator[YUE2_NAR_ROWS_PER_BLOCK];
  for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
    running_max[r] = -3.4028234663852886e38f;
    running_sum[r] = 0.0f;
    accumulator[r] = 0.0f;
  }

  for (int64_t base = 0; base < keys; base += blockDim.x) {
    const int64_t key = base + lane;
    // One K row per lane, shared by every row this block owns.
    const uint16_t* k_row = 0;
    if (key < keys) {
      k_row = key < ar_rows
          ? ar_k + (key * num_kv_heads + kv_head) * head_dim
          : nar_k + ((key - ar_rows) * num_kv_heads + kv_head) * head_dim;
    }
    const int64_t owned = rows - row0;
    const int r_valid = static_cast<int>(
        owned < YUE2_NAR_ROWS_PER_BLOCK ? owned : YUE2_NAR_ROWS_PER_BLOCK);
    const float* q_rows[YUE2_NAR_ROWS_PER_BLOCK];
    float dots[YUE2_NAR_ROWS_PER_BLOCK];
    for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
      q_rows[r] = r < r_valid ? q + ((row0 + r) * num_q_heads + q_head) * head_dim : 0;
      dots[r] = 0.0f;
    }
    // Dimension outside, row inside: one K element is loaded and converted once
    // and then reused by every row this block owns. That reuse is the entire
    // point of the blocking -- reading it once per row would leave the K traffic
    // unchanged. Eight dimensions per request: `head_dim` a multiple of eight
    // (every NAR call) keeps every row 16-byte aligned, which is what lets the
    // 16-byte loads below replace the 128 scalar ones. The fallback keeps the
    // kernel correct for any head_dim.
    if (k_row != 0) {
      const bool packed = (head_dim & 7) == 0;
      if (packed) {
        for (int64_t d = 0; d < head_dim; d += 8) {
          const uint4 kpack = *reinterpret_cast<const uint4*>(k_row + d);
          const uint16_t* kk = reinterpret_cast<const uint16_t*>(&kpack);
          const float kv0 = yue2_bf16_to_float(kk[0]);
          const float kv1 = yue2_bf16_to_float(kk[1]);
          const float kv2 = yue2_bf16_to_float(kk[2]);
          const float kv3 = yue2_bf16_to_float(kk[3]);
          const float kv4 = yue2_bf16_to_float(kk[4]);
          const float kv5 = yue2_bf16_to_float(kk[5]);
          const float kv6 = yue2_bf16_to_float(kk[6]);
          const float kv7 = yue2_bf16_to_float(kk[7]);
          for (int r = 0; r < r_valid; ++r) {
            const float4 qlo = *reinterpret_cast<const float4*>(q_rows[r] + d);
            const float4 qhi = *reinterpret_cast<const float4*>(q_rows[r] + d + 4);
            dots[r] = fmaf(qlo.x, kv0, dots[r]);
            dots[r] = fmaf(qlo.y, kv1, dots[r]);
            dots[r] = fmaf(qlo.z, kv2, dots[r]);
            dots[r] = fmaf(qlo.w, kv3, dots[r]);
            dots[r] = fmaf(qhi.x, kv4, dots[r]);
            dots[r] = fmaf(qhi.y, kv5, dots[r]);
            dots[r] = fmaf(qhi.z, kv6, dots[r]);
            dots[r] = fmaf(qhi.w, kv7, dots[r]);
          }
        }
      } else {
        for (int64_t d = 0; d < head_dim; ++d) {
          const float kv = yue2_bf16_to_float(k_row[d]);
          for (int r = 0; r < r_valid; ++r) {
            dots[r] = fmaf(q_rows[r][d], kv, dots[r]);
          }
        }
      }
    }
    for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
      scores[r * blockDim.x + lane] =
          (r < r_valid && k_row != 0) ? dots[r] * scale : -3.4028234663852886e38f;
    }
    __syncthreads();

    float tile_sum[YUE2_NAR_ROWS_PER_BLOCK];
    float tile_acc[YUE2_NAR_ROWS_PER_BLOCK];
    float new_max[YUE2_NAR_ROWS_PER_BLOCK];
    float rescale[YUE2_NAR_ROWS_PER_BLOCK];
    for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
      const float* tile = scores + r * blockDim.x;
      // Four shared-memory entries per instruction, into four independent
      // accumulators. The tile maximum is a max over a set, and fmax is exact and
      // order-independent for finite scores (which the projections and the BF16
      // K/V guarantee), so the interleaved walk returns the same bits as the
      // serial one did -- it just asks for 32 requests per row instead of 128 and
      // shortens the dependent chain from 128 to 32. The scalar form stays for a
      // head_dim that is not a multiple of four, where the rows are not 16-byte
      // aligned.
      float tile_max = -3.4028234663852886e38f;
      if ((blockDim.x & 3) == 0) {
        const float4* quads = reinterpret_cast<const float4*>(tile);
        float m0 = -3.4028234663852886e38f;
        float m1 = -3.4028234663852886e38f;
        float m2 = -3.4028234663852886e38f;
        float m3 = -3.4028234663852886e38f;
        for (int64_t q = 0; q < (blockDim.x >> 2); ++q) {
          const float4 v = quads[q];
          m0 = fmaxf(m0, v.x);
          m1 = fmaxf(m1, v.y);
          m2 = fmaxf(m2, v.z);
          m3 = fmaxf(m3, v.w);
        }
        tile_max = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
      } else {
        for (int64_t t = 0; t < blockDim.x; ++t) {
          tile_max = fmaxf(tile_max, tile[t]);
        }
      }
      new_max[r] = fmaxf(running_max[r], tile_max);
      rescale[r] = expf(running_max[r] - new_max[r]);
      // Each lane evaluates its own key's weight exactly once. Every lane used to
      // recompute the whole tile's exponentials for itself -- blockDim.x times the
      // transcendental work per tile for identical values. The weight is still
      // expf(score - new_max) for the same score and the same tile maximum, and
      // the sums below still walk the tile in the same order, so this is
      // bit-exact against the parent kernel.
      weights[lane * YUE2_NAR_ROWS_PER_BLOCK + r] =
          k_row != 0 ? expf(tile[lane] - new_max[r]) : 0.0f;
    }
    __syncthreads();

    for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
      tile_sum[r] = 0.0f;
      tile_acc[r] = 0.0f;
    }
    // One V element per lane, shared by every row this block owns. The walk is
    // split at the AR/NAR boundary so each key costs a pointer step instead of a
    // 64-bit multiply plus an ar/nar compare, and both ranges run key-ascending
    // exactly as the single loop did.
    //
    // The weights are stored key-major -- [key][row] -- so one key's whole row
    // block sits in eight consecutive floats. The row index is then a
    // compile-time offset in an unrolled loop rather than `r * blockDim.x`
    // against a runtime block size, which the profiler counts as eight scalar
    // instructions per key, and the block arrives in two shared-memory requests
    // instead of eight. The sums still walk the tile key-ascending, so this is
    // bit-exact.
    auto accumulate = [&](const uint16_t* v_row, int64_t t) {
      const float v = yue2_bf16_to_float(v_row[lane]);
      const float* w = weights + t * YUE2_NAR_ROWS_PER_BLOCK;
      float ws[YUE2_NAR_ROWS_PER_BLOCK];
#if (YUE2_NAR_ROWS_PER_BLOCK % 4) == 0
      const float4* wq = reinterpret_cast<const float4*>(w);
      for (int q = 0; q < YUE2_NAR_ROWS_PER_BLOCK / 4; ++q) {
        const float4 quad = wq[q];
        ws[q * 4 + 0] = quad.x;
        ws[q * 4 + 1] = quad.y;
        ws[q * 4 + 2] = quad.z;
        ws[q * 4 + 3] = quad.w;
      }
#else
      for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
        ws[r] = w[r];
      }
#endif
      for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
        tile_sum[r] += ws[r];
        tile_acc[r] += ws[r] * v;
      }
    };
    const int64_t v_stride = num_kv_heads * head_dim;
    const int64_t tile_hi = base + blockDim.x < keys ? base + blockDim.x : keys;
    const int64_t ar_hi = ar_rows < tile_hi ? ar_rows : tile_hi;
    int64_t t = 0;
    if (base < ar_hi) {
      const uint16_t* v_row = ar_v + (base * num_kv_heads + kv_head) * head_dim;
      for (int64_t key_index = base; key_index < ar_hi; ++key_index, ++t) {
        accumulate(v_row, t);
        v_row += v_stride;
      }
    }
    const int64_t nar_lo = base > ar_rows ? base : ar_rows;
    t = nar_lo - base;
    if (nar_lo < tile_hi) {
      const uint16_t* nar_row =
          nar_v + ((nar_lo - ar_rows) * num_kv_heads + kv_head) * head_dim;
      for (int64_t key_index = nar_lo; key_index < tile_hi; ++key_index, ++t) {
        accumulate(nar_row, t);
        nar_row += v_stride;
      }
    }
    for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
      running_sum[r] = running_sum[r] * rescale[r] + tile_sum[r];
      accumulator[r] = accumulator[r] * rescale[r] + tile_acc[r];
      running_max[r] = new_max[r];
    }
    __syncthreads();
  }

  // Every lane walks the same key loop, so `running_sum` is already the whole
  // row's denominator -- reducing it across the block would multiply it by
  // blockDim.x.
  for (int r = 0; r < YUE2_NAR_ROWS_PER_BLOCK; ++r) {
    const int64_t row = row0 + r;
    if (row >= rows) continue;
    const float total = running_sum[r] > 0.0f ? running_sum[r] : 1.0f;
    out[(row * num_q_heads + q_head) * head_dim + lane] = accumulator[r] / total;
  }
}

// Rotate-half RoPE with per-row positions for the NAR projections.
//
// Same convention as the vibevoice/AR rope (and therefore the same table): the
// first half of a head is paired with the second half and negated. The
// difference is that either head count may be zero, which skips that side. The
// NAR needs that because its keys must be rotated for *every* row before any
// query tile attends to them, while its queries are rotated per tile -- and the
// cached AR keys arrive already rotated.
__global__ void yue2_nar_rope_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ positions,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t total_q = rows * num_q_heads * head_dim;
  const int64_t total_k = rows * num_kv_heads * head_dim;
  const int64_t half = head_dim / 2;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_q + total_k;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const bool is_q = idx < total_q;
    const int64_t local = is_q ? idx : idx - total_q;
    const int64_t heads = is_q ? num_q_heads : num_kv_heads;
    const int64_t row = local / (heads * head_dim);
    const int64_t within = local - row * heads * head_dim;
    const int64_t dim = within % head_dim;
    const float* src = is_q ? q : k;
    float* dst = is_q ? q_out : k_out;
    const float* cos_p = cos_table + positions[row] * head_dim;
    const float* sin_p = sin_table + positions[row] * head_dim;
    const int64_t base = local - dim;
    const int64_t pair_dim = dim < half ? dim + half : dim - half;
    const float paired = src[base + pair_dim];
    const float rotated = dim < half ? -paired : paired;
    dst[local] = src[local] * cos_p[dim] + rotated * sin_p[dim];
  }
}

extern "C" int hipengine_yue2_nar_rope_f32(
    const float* q, const float* k, const float* cos_table, const float* sin_table,
    const int64_t* positions, float* q_out, float* k_out,
    int64_t rows, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || num_q_heads < 0 || num_kv_heads < 0 || head_dim <= 0
      || head_dim % 2 || num_q_heads + num_kv_heads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * (num_q_heads + num_kv_heads) * head_dim;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(yue2_nar_rope_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
                     0, stream, q, k, cos_table, sin_table, positions, q_out, k_out,
                     rows, num_q_heads, num_kv_heads, head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_nar_gather_add_bf16(
    const uint16_t* x, const uint16_t* table, const int64_t* positions,
    uint16_t* out, int64_t rows, int64_t width, hipStream_t stream) {
  if (rows < 0 || width <= 0) return static_cast<int>(hipErrorInvalidValue);
  const int64_t total = rows * width;
  if (total == 0) return static_cast<int>(hipSuccess);
  const unsigned int blocks = static_cast<unsigned int>((total + 255) / 256);
  hipLaunchKernelGGL(yue2_nar_gather_add_kernel, dim3(blocks), dim3(256), 0, stream,
                     x, table, positions, out, rows, width);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_nar_add_broadcast_bf16(
    const uint16_t* x, const uint16_t* vector, uint16_t* out,
    int64_t rows, int64_t width, hipStream_t stream) {
  if (rows < 0 || width <= 0) return static_cast<int>(hipErrorInvalidValue);
  const int64_t total = rows * width;
  if (total == 0) return static_cast<int>(hipSuccess);
  const unsigned int blocks = static_cast<unsigned int>((total + 255) / 256);
  hipLaunchKernelGGL(yue2_nar_add_broadcast_kernel, dim3(blocks), dim3(256), 0, stream,
                     x, vector, out, rows, width);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_nar_state_update_bf16(
    const uint16_t* state, const uint16_t* velocity, float scale,
    uint16_t* out, int64_t rows, int64_t width, hipStream_t stream) {
  if (rows < 0 || width <= 0) return static_cast<int>(hipErrorInvalidValue);
  const int64_t total = rows * width;
  if (total == 0) return static_cast<int>(hipSuccess);
  const unsigned int blocks = static_cast<unsigned int>((total + 255) / 256);
  hipLaunchKernelGGL(yue2_nar_state_update_kernel, dim3(blocks), dim3(256), 0, stream,
                     state, velocity, scale, out, total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_nar_attention_f32(
    const float* q, const uint16_t* nar_k, const uint16_t* nar_v,
    const uint16_t* ar_k, const uint16_t* ar_v, float* out,
    int64_t rows, int64_t ar_rows, int64_t num_q_heads, int64_t num_kv_heads,
    int64_t head_dim, float scale, hipStream_t stream) {
  if (rows <= 0 || ar_rows < 0 || num_q_heads <= 0 || num_kv_heads <= 0
      || num_q_heads % num_kv_heads || head_dim <= 0 || head_dim > 1024
      || scale <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (ar_rows == 0 && rows == 0) return static_cast<int>(hipSuccess);
  const int threads = static_cast<int>(head_dim);
  const int64_t row_blocks =
      (rows + YUE2_NAR_ROWS_PER_BLOCK - 1) / YUE2_NAR_ROWS_PER_BLOCK;
  const size_t shared = static_cast<size_t>(threads) * 2 * YUE2_NAR_ROWS_PER_BLOCK
      * sizeof(float);
  hipLaunchKernelGGL(yue2_nar_attention_kernel, dim3(num_q_heads, row_blocks), dim3(threads),
                     shared, stream, q, nar_k, nar_v, ar_k, ar_v, out, rows, ar_rows,
                     num_q_heads, num_kv_heads, head_dim, scale);
  return static_cast<int>(hipGetLastError());
}
