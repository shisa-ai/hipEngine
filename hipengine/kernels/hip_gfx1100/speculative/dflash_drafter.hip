// DFlash drafter root/query preparation kernels.
//
// The DFlash BF16 drafter reuses the target embedding table for the root token
// plus mask-token query rows.  This kernel materializes the compact native
// input slab on device: token ids, absolute query positions, and BF16 embedding
// rows.  It intentionally does not materialize context K/V; that remains the
// separate DFlash context-cache task.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;

namespace {

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

template <typename embed_t>
__device__ inline uint16_t embed_to_bf16(embed_t value) {
  return float_to_bf16_bits(static_cast<float>(value));
}

template <>
__device__ inline uint16_t embed_to_bf16<uint16_t>(uint16_t value) {
  return value;
}

template <typename embed_t>
__global__ void dflash_prepare_noise_inputs_bf16_i32_kernel(
    const int32_t* __restrict__ root_tokens,
    const int32_t* __restrict__ root_positions,
    const embed_t* __restrict__ embed_tokens,
    int32_t* __restrict__ noise_token_ids,
    int32_t* __restrict__ position_ids,
    uint16_t* __restrict__ noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col = static_cast<int64_t>(blockIdx.y);
  if (row >= request_count || col >= block_size) {
    return;
  }
  const int64_t slab_row = row * block_size + col;
  const int32_t token = (col == 0) ? root_tokens[row] : mask_token_id;
  const int32_t position = root_positions[row] + static_cast<int32_t>(col);
  if (threadIdx.x == 0) {
    noise_token_ids[slab_row] = token;
    position_ids[slab_row] = position;
  }
  const bool valid_token = token >= 0 && static_cast<int64_t>(token) < vocab_size;
  for (int64_t h = static_cast<int64_t>(threadIdx.x); h < hidden_size; h += blockDim.x) {
    uint16_t value = 0;
    if (valid_token) {
      value = embed_to_bf16(embed_tokens[static_cast<int64_t>(token) * hidden_size + h]);
    }
    noise_embeddings[slab_row * hidden_size + h] = value;
  }
}

bool valid_shape(int64_t request_count, int64_t block_size, int64_t hidden_size, int64_t vocab_size) {
  return request_count > 0 && block_size > 0 && hidden_size > 0 && vocab_size > 0;
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

}  // namespace

extern "C" int hipengine_dflash_prepare_noise_inputs_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const uint16_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_prepare_noise_inputs_f16_to_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const half_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<half_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}
