// hipEngine-native Qwen3.8-Flash-Next PLE strict primitives.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void ple_signed_sqrt_gate_f32_kernel(
    const float* __restrict__ key,
    const float* __restrict__ query,
    float* __restrict__ gate,
    int64_t hidden) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float dot = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    dot += key[offset + column] * query[offset + column];
  }
  partial[threadIdx.x] = dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float score = partial[0] * rsqrtf(static_cast<float>(hidden));
    const float transformed = copysignf(sqrtf(fmaxf(fabsf(score), 1.0e-6f)), score);
    gate[group] = 1.0f / (1.0f + expf(-transformed));
  }
}

__global__ void ple_repeat_gated_value_f32_kernel(
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int64_t offset = group * hidden;
  const float scale = gate[group];
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] = value[row * hidden + column] * scale;
  }
}

__global__ void ple_dilated_depthwise_conv_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ history,
    float* __restrict__ output,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    int64_t history_rows) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    float value = 0.0f;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
      const int64_t logical = row - (kernel_size - 1 - tap) * dilation;
      const float input_value = logical >= 0
          ? input[logical * channels + channel]
          : history[(history_rows + logical) * channels + channel];
      value += input_value * weights[channel * kernel_size + tap];
    }
    output[row * channels + channel] = value;
  }
  for (int64_t state_row = 0; state_row < history_rows; ++state_row) {
    const int64_t logical = rows - history_rows + state_row;
    const float value = logical >= 0
        ? input[logical * channels + channel]
        : history[(history_rows + logical) * channels + channel];
    history[state_row * channels + channel] = value;
  }
}

// Request-indexed PLE dilated depthwise Conv decode: one token per row, each row
// bound to its own request's history slot.
//
// The single-row-owner relation is exact. For one token the owner's row-local
// index is 0, so tap `t` reads history row `t * dilation` for `t < kernel_size - 1`
// and the token itself for the last tap; the owner's ascending state update
// becomes shift-by-one-row with the token appended. The statements below are the
// owner's, with the row index substituted.

constexpr int64_t kQwen4ExpPleIndexedMaxRows = 64;
constexpr int64_t kQwen4ExpPleIndexedMaxStateSlots = 4096;
constexpr int64_t kQwen4ExpPleIndexedMaxChannels = 1048576;
constexpr int64_t kQwen4ExpPleIndexedMaxKernelSize = 1024;
constexpr int64_t kQwen4ExpPleIndexedMaxDilation = 1024;
constexpr int64_t kQwen4ExpPleIndexedMaxHistoryRows = 65536;

__global__ void ple_dilated_depthwise_conv_decode_indexed_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ history,
    float* __restrict__ output,
    const int64_t* __restrict__ state_indices,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    int64_t history_rows,
    int64_t state_slots) {
  constexpr int kWords = static_cast<int>(kQwen4ExpPleIndexedMaxStateSlots / 32);
  __shared__ unsigned int s_slot_bitmap[kWords];
  __shared__ int s_batch_rejected;

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t row = blockIdx.y;
  const int64_t channel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = channel < channels;

  // Read phase. An out-of-range slot is redirected to slot 0 so no load leaves
  // the slab; the scan below then rejects the whole batch before any write.
  //
  // ``slot_history`` and the owner's ``history`` name the same allocation and are
  // read and written here, so this alias deliberately carries no ``__restrict__``:
  // a second restricted pointer to the same slab would promise they do not alias.
  const int64_t requested_slot = state_indices[row];
  const bool slot_in_range = requested_slot >= 0 && requested_slot < state_slots;
  const int64_t slot = slot_in_range ? requested_slot : 0;
  float* slot_history = history + slot * history_rows * channels;

  float value = 0.0f;
  if (active) {
    const float token = input[row * channels + channel];
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
      const int64_t logical = 0 - (kernel_size - 1 - tap) * dilation;
      const float input_value = logical >= 0
          ? token
          : slot_history[(history_rows + logical) * channels + channel];
      value += input_value * weights[channel * kernel_size + tap];
    }
  }

  // Whole-batch admission, ahead of every write. The decode body shifts its slot
  // as it runs, so two rows naming one slot would interleave two requests' state.
  // Warp 0 clears and scans; the clear -> scan dependency stays inside the warp,
  // so only publication of the verdict needs a block barrier.
  if (warp == 0) {
    const int64_t words = (state_slots + 31) / 32;
    for (int64_t word = lane; word < words; word += 32) {
      s_slot_bitmap[word] = 0u;
    }
    if (lane == 0) {
      atomicExch(&s_batch_rejected, 0);
    }
    __syncwarp();
    for (int64_t other = lane; other < rows; other += 32) {
      const int64_t index = state_indices[other];
      if (index < 0 || index >= state_slots) {
        atomicExch(&s_batch_rejected, 1);
        continue;
      }
      const unsigned int bit = 1u << (static_cast<unsigned int>(index) & 31u);
      if (atomicOr(&s_slot_bitmap[index >> 5], bit) & bit) {
        atomicExch(&s_batch_rejected, 1);
      }
    }
  }
  __syncthreads();
  if (s_batch_rejected) {
    return;
  }

  // Write phase: this row's output, then the owner's ascending state update.
  if (active) {
    const float token = input[row * channels + channel];
    output[row * channels + channel] = value;
    for (int64_t state_row = 0; state_row < history_rows; ++state_row) {
      const int64_t logical = 1 - history_rows + state_row;
      const float input_value = logical >= 0
          ? token
          : slot_history[(history_rows + logical) * channels + channel];
      slot_history[state_row * channels + channel] = input_value;
    }
  }
}

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union { float f32; uint32_t u32; } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__global__ void ple_add_delta_bf16_f32_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ gated_value,
    const float* __restrict__ conv_output,
    uint16_t* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = float_to_bf16_bits(
        bf16_bits_to_float(residual[index]) + gated_value[index] + conv_output[index]);
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_ple_signed_sqrt_gate_f32(
    const void* key,
    const void* query,
    void* gate,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!key || !query || !gate || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      ple_signed_sqrt_gate_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(key), static_cast<const float*>(query),
      static_cast<float*>(gate), hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_repeat_gated_value_f32(
    const void* value,
    const void* gate,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!value || !gate || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      ple_repeat_gated_value_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), 0, stream,
      static_cast<const float*>(value), static_cast<const float*>(gate),
      static_cast<float*>(output), branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_dilated_depthwise_conv_indexed_f32(
    const void* input,
    const void* weights,
    void* history,
    void* output,
    const void* state_indices,
    int64_t rows,
    int64_t state_slots,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    hipStream_t stream) {
  if (!input || !weights || !history || !output || !state_indices) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || rows > kQwen4ExpPleIndexedMaxRows) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (state_slots <= 0 || state_slots > kQwen4ExpPleIndexedMaxStateSlots) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (state_slots < rows) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (channels <= 0 || channels > kQwen4ExpPleIndexedMaxChannels) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (kernel_size <= 0 || kernel_size > kQwen4ExpPleIndexedMaxKernelSize) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (dilation <= 0 || dilation > kQwen4ExpPleIndexedMaxDilation) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // The derived slot stride is what the body indexes, so it is bounded directly:
  // slot * history_rows * channels stays inside int64 for every admitted shape.
  const int64_t history_rows = (kernel_size - 1) * dilation;
  if (history_rows > kQwen4ExpPleIndexedMaxHistoryRows) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (reinterpret_cast<uintptr_t>(state_indices) % 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      ple_dilated_depthwise_conv_decode_indexed_f32_kernel,
      dim3(blocks, static_cast<uint32_t>(rows)), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<const float*>(weights),
      static_cast<float*>(history), static_cast<float*>(output),
      static_cast<const int64_t*>(state_indices),
      rows, channels, kernel_size, dilation, history_rows, state_slots);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_add_delta_bf16_f32(
    const void* residual,
    const void* gated_value,
    const void* conv_output,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!residual || !gated_value || !conv_output || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      ple_add_delta_bf16_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(gated_value),
      static_cast<const float*>(conv_output), static_cast<uint16_t*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_dilated_depthwise_conv_f32(
    const void* input,
    const void* weights,
    void* history,
    void* output,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    hipStream_t stream) {
  if (!input || !weights || !history || !output || rows <= 0 || channels <= 0 ||
      kernel_size <= 0 || dilation <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t history_rows = (kernel_size - 1) * dilation;
  const uint32_t blocks = static_cast<uint32_t>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      ple_dilated_depthwise_conv_f32_kernel,
      dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<const float*>(weights),
      static_cast<float*>(history), static_cast<float*>(output),
      rows, channels, kernel_size, dilation, history_rows);
  return static_cast<int>(hipGetLastError());
}
