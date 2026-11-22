#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
constexpr int wave = 32;
constexpr int waves_per_block = 4;
constexpr int values_per_block = 64;
constexpr int bytes_per_block = 36;

__device__ __forceinline__ float decode_scale(uint8_t code) {
    // GGUF NVFP4 uses unsigned E4M3; its integer FP4 table is doubled.
    const int exponent = (code >> 3) & 15;
    const int significand = (code & 7) + (exponent ? 8 : 0);
    const float power = __uint_as_float(static_cast<uint32_t>(116 + (exponent ? exponent : 1)) << 23);
    return code == 0x7f ? 0.0f : significand * power;
}

__device__ __forceinline__ int decode_fp4(unsigned code) {
    constexpr int magnitude[] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int value = magnitude[code & 7];
    return code & 8 ? -value : value;
}

template<bool selected>
__global__ __launch_bounds__(wave * waves_per_block) void gguf_nvfp4_gemv_f32_kernel(
        const float * x, const uint8_t * weights, float * out, int in_features, int out_features,
        size_t output_values, float outer_scale, const int32_t * ids, const float * scales, int experts) {
    const size_t index = static_cast<size_t>(blockIdx.x) * waves_per_block + threadIdx.x / wave;
    if (index >= output_values) {
        return;
    }
    const int lane = threadIdx.x % wave;
    const int batch = index / out_features;
    const int row = index % out_features;
    const int blocks = in_features / values_per_block;
    size_t weight_row = row;
    if constexpr (selected) {
        const int expert = ids[batch];
        if (expert < 0 || expert >= experts) {
            if (lane == 0) {
                out[index] = 0;
            }
            return;
        }
        weight_row += static_cast<size_t>(expert) * out_features;
        outer_scale = scales ? scales[batch] : 1.0f;
    }
    const uint8_t * packed = weights + weight_row * blocks * bytes_per_block;
    const float * input = x + static_cast<size_t>(batch) * in_features;
    const int group = lane / 8;
    const int first = group * 16 + lane % 8;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float scale = decode_scale(packed[group]);
        const unsigned codes = packed[4 + lane];
        sum = fmaf(scale * decode_fp4(codes & 15), input[first], sum);
        sum = fmaf(scale * decode_fp4(codes >> 4), input[first + 8], sum);
        packed += bytes_per_block;
        input += values_per_block;
    }
    for (int shift = wave / 2; shift; shift /= 2) {
        sum += __shfl_down(sum, shift, wave);
    }
    if (lane == 0) {
        out[index] = sum * outer_scale;
    }
}

template<bool selected>
int launch(const float * x, const uint8_t * weights, float * out, int batch, int in_features, int out_features,
           float outer_scale, hipStream_t stream, const int32_t * ids, const float * scales, int experts) {
    if (!x || !weights || !out || batch <= 0 || in_features <= 0 || in_features % values_per_block
            || out_features <= 0 || (selected && (!ids || experts <= 0))) {
        return hipErrorInvalidValue;
    }
    const size_t output_values = static_cast<size_t>(batch) * out_features;
    const size_t blocks = (output_values + waves_per_block - 1) / waves_per_block;
    if (blocks > INT32_MAX) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL((gguf_nvfp4_gemv_f32_kernel<selected>), dim3(blocks), dim3(wave * waves_per_block),
                       0, stream, x, weights, out, in_features, out_features, output_values, outer_scale,
                       ids, scales, experts);
    return hipGetLastError();
}
}  // namespace

extern "C" int hipengine_gguf_nvfp4_gemv_f32_f32_out(
        const float * x, const uint8_t * weights, float * out, int batch,
        int in_features, int out_features, float outer_scale, hipStream_t stream) {
    return launch<false>(x, weights, out, batch, in_features, out_features, outer_scale, stream, nullptr, nullptr, 0);
}

extern "C" int hipengine_gguf_nvfp4_selected_f32_f32_out(
        const float * x, const uint8_t * weights, float * out, const int32_t * ids, const float * scales,
        int batch, int experts, int in_features, int out_features, hipStream_t stream) {
    return launch<true>(x, weights, out, batch, in_features, out_features, 1.0f, stream, ids, scales, experts);
}
