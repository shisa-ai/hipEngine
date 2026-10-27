// Dense causal flash prefill attention for the dense Llama-family path.
//
// The exact block kernel this replaces stages one FP32 logit row per key in
// workgroup LDS and runs one block per (token, head). That sizes shared memory
// by the context, so the prompt bound was the LDS budget, and it re-reads the
// whole K/V row for every query row, so a 15856-token prompt cost hours.
//
// This kernel tiles query rows and keys instead: one block owns kWarps * RPW
// query rows of one head, stages K/V for a kKeyTile slice in LDS, keeps a
// (rows x kKeyTile) score tile, and folds each slice in with an online softmax.
// Shared memory therefore does not mention `keys`, and each K/V slice serves
// every row in the block.
//
// Arithmetic: scores are FP32 dot products of BF16 operands (exact products,
// FP32 accumulation), the running max/sum are FP32, and the output is
// accumulated in FP32 registers before one BF16 rounding. The reduction order
// differs from the exact block kernel -- that kernel normalizes once against a
// whole-row max, this one rescales per slice -- so outputs are equal within BF16
// representation, not bit-identical. The exact kernel stays registered as the
// strict fallback.
//
// Precondition, shared with the exact kernel's default caller: the mask is
// causal in the frame `position = keys - tokens + row`, so no key past a row's
// own position contributes. The walk is bounded by that, which is what keeps a
// causal prompt from paying for the masked half. Passing a null `keep_mask`
// asserts exactly that precondition and skips the mask reads, so a prompt does
// not have to materialize a tokens x keys byte array to run.

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

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kKeyTile = 64;

// RPW query rows per warp; the block owns kWarps * RPW of them.
template <int HD, int RPW>
__global__ void dense_flash_prefill_kernel(
    const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    uint16_t* __restrict__ out,
    int64_t tokens,
    int64_t keys,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale) {
  constexpr int kRows = kWarps * RPW;
  constexpr int kDimsPerLane = HD / 32;

  const int64_t head = blockIdx.x;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kRows;
  const int64_t gqa = num_heads / num_kv_heads;
  const int64_t kv_head = head / gqa;
  // Row r sits at absolute position `keys - tokens + row0 + r` in the key frame.
  const int64_t position_base = keys - tokens + row0;

  extern __shared__ char smem[];
  float* q_s = reinterpret_cast<float*>(smem);
  uint16_t* k_s = reinterpret_cast<uint16_t*>(q_s + kRows * HD);
  uint16_t* v_s = k_s + kKeyTile * HD;
  float* p_s = reinterpret_cast<float*>(v_s + kKeyTile * HD);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int64_t index = tid; index < kRows * HD; index += kThreads) {
    const int64_t row = row0 + index / HD;
    const int64_t dim = index % HD;
    q_s[index] = row < tokens
        ? bf16_bits_to_float(query[(row * num_heads + head) * HD + dim])
        : 0.0f;
  }
  __syncthreads();

  float acc[RPW][kDimsPerLane];
  float running_max[RPW];
  float running_sum[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    running_max[r] = -INFINITY;
    running_sum[r] = 0.0f;
#pragma unroll
    for (int d = 0; d < kDimsPerLane; ++d) {
      acc[r][d] = 0.0f;
    }
  }

  // No row in the block sees a key past the last row's own position.
  const int64_t key_end = keys < position_base + kRows ? keys : position_base + kRows;

  for (int64_t k0 = 0; k0 < key_end; k0 += kKeyTile) {
    for (int64_t index = tid; index < kKeyTile * HD; index += kThreads) {
      const int64_t kk = k0 + index / HD;
      const int64_t dim = index % HD;
      const int64_t offset = (kk * num_kv_heads + kv_head) * HD + dim;
      const bool live = kk < keys;
      k_s[index] = live ? key[offset] : static_cast<uint16_t>(0);
      v_s[index] = live ? value[offset] : static_cast<uint16_t>(0);
    }
    __syncthreads();

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
      const int r = warp * RPW + rr;
      const int64_t row = row0 + r;
      if (row >= tokens) {
        continue;
      }
      const int64_t row_limit = position_base + r + 1;
      // A null mask means "the causal mask": the walk bound above already
      // excludes every key past this row, so the mask read would only restate
      // it. That is what keeps a long prompt from allocating a tokens x keys
      // byte array, which is quadratic in the prompt.
      const uint8_t* mask_row = keep_mask == nullptr ? nullptr : keep_mask + row * keys;
      float* p_row = p_s + r * kKeyTile;

      for (int jj = lane; jj < kKeyTile; jj += 32) {
        const int64_t kk = k0 + jj;
        float score = -INFINITY;
        if (kk < row_limit) {
          const float* q_row = q_s + r * HD;
          const uint16_t* k_row = k_s + jj * HD;
          float dot = 0.0f;
#pragma unroll
          for (int d = 0; d < HD; ++d) {
            dot += q_row[d] * bf16_bits_to_float(k_row[d]);
          }
          score = (mask_row == nullptr || mask_row[kk]) ? dot * scale : -INFINITY;
        }
        p_row[jj] = score;
      }
      __syncwarp();

      float tile_max = -INFINITY;
      for (int jj = lane; jj < kKeyTile; jj += 32) {
        tile_max = fmaxf(tile_max, p_row[jj]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset));
      }
      // Every key in this slice is masked for this row: nothing to fold in.
      if (tile_max == -INFINITY) {
        continue;
      }

      const float new_max = fmaxf(running_max[rr], tile_max);
      const float correction = expf(running_max[rr] - new_max);
      float tile_sum = 0.0f;
      for (int jj = lane; jj < kKeyTile; jj += 32) {
        const float p = expf(p_row[jj] - new_max);
        p_row[jj] = p;
        tile_sum += p;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_sum += __shfl_xor(tile_sum, offset);
      }
      __syncwarp();

      running_max[rr] = new_max;
      running_sum[rr] = running_sum[rr] * correction + tile_sum;
#pragma unroll
      for (int d = 0; d < kDimsPerLane; ++d) {
        acc[rr][d] *= correction;
      }

      // Each lane owns dims `lane + 32 * t` for this row.
#pragma unroll
      for (int jj = 0; jj < kKeyTile; ++jj) {
        const float p = p_row[jj];
        const uint16_t* v_row = v_s + jj * HD;
#pragma unroll
        for (int t = 0; t < kDimsPerLane; ++t) {
          acc[rr][t] += p * bf16_bits_to_float(v_row[lane + 32 * t]);
        }
      }
    }
    // The next slice overwrites k_s and v_s.
    __syncthreads();
  }

#pragma unroll
  for (int rr = 0; rr < RPW; ++rr) {
    const int r = warp * RPW + rr;
    const int64_t row = row0 + r;
    if (row >= tokens) {
      continue;
    }
    const float inv = running_sum[rr] > 0.0f ? 1.0f / running_sum[rr] : 0.0f;
#pragma unroll
    for (int t = 0; t < kDimsPerLane; ++t) {
      const int64_t dim = lane + 32 * t;
      out[(row * num_heads + head) * HD + dim] = float_to_bf16_bits(acc[rr][t] * inv);
    }
  }
}

}  // namespace

// Return codes: 0 ok, 1 bad shape, 2 GQA ratio not integral, 3 head_dim the
// template does not carry, 4 launch failed.
extern "C" int hipengine_dense_flash_prefill_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* value,
    const uint8_t* keep_mask,
    uint16_t* out,
    int64_t tokens,
    int64_t keys,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return 1;
  }
  if (num_heads % num_kv_heads != 0) {
    return 2;
  }
  if (head_dim != 128) {
    return 3;
  }
  if (keys < tokens) {
    // The key frame starts at `keys - tokens`; a shorter key range cannot hold
    // the prompt's own positions.
    return 1;
  }

  // Wide prompts get two rows per warp, so a K/V slice serves 16 rows; short
  // ones get one, because a half-empty block would only add LDS pressure.
  if (tokens > 8) {
    constexpr int kRows = kWarps * 2;
    const size_t shared_bytes = static_cast<size_t>(kRows * 128) * sizeof(float) +
        static_cast<size_t>(2 * kKeyTile * 128) * sizeof(uint16_t) +
        static_cast<size_t>(kRows * kKeyTile) * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(num_heads),
                    static_cast<unsigned int>((tokens + kRows - 1) / kRows));
    hipLaunchKernelGGL((dense_flash_prefill_kernel<128, 2>), grid, dim3(kThreads),
                       shared_bytes, stream, query, key, value, keep_mask, out,
                       tokens, keys, num_heads, num_kv_heads, scale);
  } else {
    constexpr int kRows = kWarps;
    const size_t shared_bytes = static_cast<size_t>(kRows * 128) * sizeof(float) +
        static_cast<size_t>(2 * kKeyTile * 128) * sizeof(uint16_t) +
        static_cast<size_t>(kRows * kKeyTile) * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(num_heads),
                    static_cast<unsigned int>((tokens + kRows - 1) / kRows));
    hipLaunchKernelGGL((dense_flash_prefill_kernel<128, 1>), grid, dim3(kThreads),
                       shared_bytes, stream, query, key, value, keep_mask, out,
                       tokens, keys, num_heads, num_kv_heads, scale);
  }
  return hipGetLastError() == hipSuccess ? 0 : 4;
}
