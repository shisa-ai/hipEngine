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
// A fused multiply-add would round once and is therefore not equivalent.
__global__ void yue2_nar_state_update_kernel(
    const uint16_t* __restrict__ state,
    const uint16_t* __restrict__ velocity,
    const uint16_t* __restrict__ scale,
    uint16_t* __restrict__ out,
    int64_t total,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t col = idx % width;
  const float sv = yue2_bf16_to_float(state[idx]);
  const float vv = yue2_bf16_to_float(velocity[idx]);
  const float gv = yue2_bf16_to_float(scale[col]);
  const float product = yue2_bf16_to_float(yue2_float_to_bf16(vv * gv));
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
  const int64_t row = blockIdx.y;
  const int64_t q_head = blockIdx.x;
  if (row >= rows || q_head >= num_q_heads || threadIdx.x >= head_dim) return;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t lane = threadIdx.x;
  const int64_t keys = ar_rows + rows;
  const float* q_row = q + (row * num_q_heads + q_head) * head_dim;

  extern __shared__ float shared[];
  float* scores = shared;                       // tile of key scores

  float running_max = -3.4028234663852886e38f;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  for (int64_t base = 0; base < keys; base += blockDim.x) {
    const int64_t key = base + lane;
    float score = -3.4028234663852886e38f;
    if (key < keys) {
      const uint16_t* k_row = key < ar_rows
          ? ar_k + (key * num_kv_heads + kv_head) * head_dim
          : nar_k + ((key - ar_rows) * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < head_dim; ++d) {
        dot = fmaf(q_row[d], yue2_bf16_to_float(k_row[d]), dot);
      }
      score = dot * scale;
    }
    scores[lane] = score;
    __syncthreads();

    float tile_max = -3.4028234663852886e38f;
    for (int64_t t = 0; t < blockDim.x; ++t) {
      tile_max = fmaxf(tile_max, scores[t]);
    }
    const float new_max = fmaxf(running_max, tile_max);
    const float rescale = expf(running_max - new_max);
    float tile_sum = 0.0f;
    float tile_acc = 0.0f;
    for (int64_t t = 0; t < blockDim.x; ++t) {
      const int64_t key_index = base + t;
      if (key_index >= keys) break;
      const float weight = expf(scores[t] - new_max);
      tile_sum += weight;
      const uint16_t* v_row = key_index < ar_rows
          ? ar_v + (key_index * num_kv_heads + kv_head) * head_dim
          : nar_v + ((key_index - ar_rows) * num_kv_heads + kv_head) * head_dim;
      tile_acc += weight * yue2_bf16_to_float(v_row[lane]);
    }
    running_sum = running_sum * rescale + tile_sum;
    accumulator = accumulator * rescale + tile_acc;
    running_max = new_max;
    __syncthreads();
  }

  // Every lane walks the same key loop, so `running_sum` is already the whole
  // row's denominator -- reducing it across the block would multiply it by
  // blockDim.x.
  const float total = running_sum > 0.0f ? running_sum : 1.0f;
  out[(row * num_q_heads + q_head) * head_dim + lane] = accumulator / total;
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
    const uint16_t* state, const uint16_t* velocity, const uint16_t* scale,
    uint16_t* out, int64_t rows, int64_t width, hipStream_t stream) {
  if (rows < 0 || width <= 0) return static_cast<int>(hipErrorInvalidValue);
  const int64_t total = rows * width;
  if (total == 0) return static_cast<int>(hipSuccess);
  const unsigned int blocks = static_cast<unsigned int>((total + 255) / 256);
  hipLaunchKernelGGL(yue2_nar_state_update_kernel, dim3(blocks), dim3(256), 0, stream,
                     state, velocity, scale, out, total, width);
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
  const size_t shared = static_cast<size_t>(threads) * sizeof(float);
  hipLaunchKernelGGL(yue2_nar_attention_kernel, dim3(num_q_heads, rows), dim3(threads),
                     shared, stream, q, nar_k, nar_v, ar_k, ar_v, out, rows, ar_rows,
                     num_q_heads, num_kv_heads, head_dim, scale);
  return static_cast<int>(hipGetLastError());
}
