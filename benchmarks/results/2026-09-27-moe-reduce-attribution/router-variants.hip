// Router logits screen: what actually limits the shipped kernel?
//
// Measured baseline: the shipped token-tile-4 kernel runs 1024x512x2560 in
// 3.02 ms = 889 GFLOP/s, against a 0.164 ms FMA floor. A 2D-tiled variant that
// cuts traffic 10x is 3.9x SLOWER, so traffic is not the constraint (the
// shipped kernel already sustains 2222 GB/s of L2-resident re-reads).
//
// The remaining suspects are the 8-barrier block reduction, which is issued
// once per (block, token) for only 4 tokens, and the short per-thread k-loop
// (2 iterations) that cannot hide L2 latency. This file varies both while
// keeping the shipped arithmetic order.

#include <hip/hip_runtime.h>
#include <cstdint>

// Faithful copy of the shipped structure with TOKENS_PER_BLOCK templated, so a
// timing difference is attributable to the tile width and nothing else.
template <int TOKENS_PER_BLOCK>
__global__ void __launch_bounds__(256)
router_token_tile_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    float* __restrict__ logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_experts) {
  extern __shared__ float partial[];
  const int64_t expert = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(blockIdx.y) * TOKENS_PER_BLOCK;
  if (expert >= num_experts) return;
  float acc[TOKENS_PER_BLOCK] = {};
  const int64_t weight_row = expert * hidden_size;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden_size / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(threadIdx.x) * 8;
       k_base < main_end; k_base += vec_stride) {
    const float* w_ptr = weight + weight_row + k_base;
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) {
        const float* x_ptr = hidden + token * hidden_size + k_base;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t] = fmaf(x_ptr[i], w_ptr[i], acc[t]);
      }
    }
  }
  for (int64_t k = main_end + threadIdx.x; k < hidden_size; k += blockDim.x) {
    const float w = weight[weight_row + k];
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) acc[t] = fmaf(hidden[token * hidden_size + k], w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    partial[t * blockDim.x + threadIdx.x] = acc[t];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
        partial[t * blockDim.x + threadIdx.x] +=
            partial[t * blockDim.x + threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) {
        logits[token * num_experts + expert] = partial[t * blockDim.x];
      }
    }
  }
}

// Warp-per-token variant: each of the 8 warps in the block owns one token, so
// the reduction is a 5-step shuffle instead of an 8-barrier LDS tree, and the
// weight row is read once per warp rather than once per block.
__global__ void __launch_bounds__(256)
router_warp_token_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    float* __restrict__ logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_experts) {
  const int64_t expert = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t token = static_cast<int64_t>(blockIdx.y) * 8 + warp;
  if (expert >= num_experts || token >= tokens) return;
  const float* w_row = weight + expert * hidden_size;
  const float* x_row = hidden + token * hidden_size;
  float acc = 0.0f;
  for (int64_t k = lane * 4; k + 3 < hidden_size; k += 128) {
    acc = fmaf(x_row[k + 0], w_row[k + 0], acc);
    acc = fmaf(x_row[k + 1], w_row[k + 1], acc);
    acc = fmaf(x_row[k + 2], w_row[k + 2], acc);
    acc = fmaf(x_row[k + 3], w_row[k + 3], acc);
  }
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down(acc, s);
  if (lane == 0) logits[token * num_experts + expert] = acc;
}

#define ENTRY_T(NAME, T)                                                      \
extern "C" int NAME(const void* h, const void* w, void* o, int64_t tok,       \
    int64_t hid, int64_t exp, hipStream_t s) {                                \
  if (!h || !w || !o || tok <= 0 || hid <= 0 || exp <= 0)                     \
    return static_cast<int>(hipErrorInvalidValue);                            \
  hipLaunchKernelGGL((router_token_tile_kernel<T>),                           \
      dim3(static_cast<unsigned>(exp),                                        \
           static_cast<unsigned>((tok + T - 1) / T)),                         \
      dim3(256), 256 * T * sizeof(float), s,                                  \
      static_cast<const float*>(h), static_cast<const float*>(w),             \
      static_cast<float*>(o), tok, hid, exp);                                 \
  return static_cast<int>(hipGetLastError());                                 \
}

ENTRY_T(router_token_tile4, 4)
ENTRY_T(router_token_tile8, 8)
ENTRY_T(router_token_tile16, 16)
ENTRY_T(router_token_tile32, 32)

extern "C" int router_warp_token(
    const void* h, const void* w, void* o,
    int64_t tok, int64_t hid, int64_t exp, hipStream_t s) {
  if (!h || !w || !o || tok <= 0 || hid <= 0 || exp <= 0)
    return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(router_warp_token_kernel,
      dim3(static_cast<unsigned>(exp),
           static_cast<unsigned>((tok + 7) / 8)),
      dim3(256), 0, s,
      static_cast<const float*>(h), static_cast<const float*>(w),
      static_cast<float*>(o), tok, hid, exp);
  return static_cast<int>(hipGetLastError());
}
