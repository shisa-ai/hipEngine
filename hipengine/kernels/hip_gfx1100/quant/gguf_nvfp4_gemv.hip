#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
constexpr int wave = 32;
constexpr int waves_per_block = 4;
constexpr int values_per_block = 64;
constexpr int bytes_per_block = 36;
constexpr int route_tile = 4;

__device__ __forceinline__ float decode_scale(uint8_t code) {
    // GGUF NVFP4 uses unsigned E4M3; its integer FP4 table is doubled.
    const int exponent = (code >> 3) & 15;
    const int significand = (code & 7) + (exponent ? 8 : 0);
    const float power = __uint_as_float(static_cast<uint32_t>(116 + (exponent ? exponent : 1)) << 23);
    return code == 0x7f ? 0.0f : significand * power;
}

template<bool register_fp4 = false>
__device__ __forceinline__ int decode_fp4(unsigned code) {
    constexpr int magnitude[] = {0, 1, 2, 3, 4, 6, 8, 12};
    // Four-bit magnitudes in ascending code order; no per-lane table load.
    const int value = register_fp4 ? (0xc8643210u >> (4 * (code & 7))) & 15 : magnitude[code & 7];
    return code & 8 ? -value : value;
}

template<bool selected, bool register_fp4 = false>
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
        sum = fmaf(scale * decode_fp4<register_fp4>(codes & 15), input[first], sum);
        sum = fmaf(scale * decode_fp4<register_fp4>(codes >> 4), input[first + 8], sum);
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

template<bool shared_input, int active_routes>
__device__ __forceinline__ void selected_tile_f32(
        const float* x, const uint8_t* weights, float* out, const int32_t* records, const float* scales,
        int routes, int experts, int in_features, int out_features, int routes_per_input) {
    const int row = blockIdx.x * waves_per_block + threadIdx.x / wave;
    if (row >= out_features) return;
    const int32_t* tile = records + blockIdx.y * (route_tile + 1);
    const int expert = tile[0], first_route = tile[1];
    if (expert < 0 || expert >= experts || first_route < 0 || first_route >= routes) return;
    const int blocks = in_features / values_per_block;
    const size_t weight_row = static_cast<size_t>(expert) * out_features + row;
    const uint8_t* packed = weights + weight_row * blocks * bytes_per_block;
    const int lane = threadIdx.x % wave;
    const int group = lane / 8;
    const int first = group * 16 + lane % 8;
    const float* inputs[active_routes];
    float sum[active_routes] = {};
    int route[active_routes];
    #pragma unroll
    for (int t = 0; t < active_routes; ++t) {
        route[t] = tile[t + 1];
        const bool valid = route[t] >= 0 && route[t] < routes;
        int input_row = valid ? route[t] : first_route;
        if constexpr (shared_input) input_row /= routes_per_input;
        inputs[t] = x + static_cast<size_t>(input_row) * in_features + first;
    }
    // Expose independent loads from successive blocks without reordering the dot products.
    #pragma unroll 4
    for (int b = 0; b < blocks; ++b) {
        const float scale = decode_scale(packed[group]);
        const unsigned codes = packed[4 + lane];
        const float low = scale * decode_fp4<true>(codes & 15);
        const float high = scale * decode_fp4<true>(codes >> 4);
        #pragma unroll
        for (int t = 0; t < active_routes; ++t) {
            sum[t] = fmaf(low, inputs[t][0], sum[t]);
            sum[t] = fmaf(high, inputs[t][8], sum[t]);
            inputs[t] += values_per_block;
        }
        packed += bytes_per_block;
    }
    #pragma unroll
    for (int t = 0; t < active_routes; ++t) {
        for (int shift = wave / 2; shift; shift /= 2) sum[t] += __shfl_down(sum[t], shift, wave);
        if (lane == 0 && route[t] >= 0 && route[t] < routes) {
            out[static_cast<size_t>(route[t]) * out_features + row] = sum[t] * (scales ? scales[route[t]] : 1.0f);
        }
    }
}

template<bool shared_input>
__global__ __launch_bounds__(wave * waves_per_block) void gguf_nvfp4_selected_tile4_f32_kernel(
        const float* x, const uint8_t* weights, float* out, const int32_t* records, const float* scales,
        int routes, int experts, int in_features, int out_features, int routes_per_input) {
    // Skip trailing padded routes; interior holes still use the per-route validity check.
    const auto* tile = records + blockIdx.y * (route_tile + 1);
    if (tile[4] >= 0) {
        selected_tile_f32<shared_input, 4>(x, weights, out, records, scales, routes, experts,
                                           in_features, out_features, routes_per_input);
    } else if (tile[3] >= 0) {
        selected_tile_f32<shared_input, 3>(x, weights, out, records, scales, routes, experts,
                                           in_features, out_features, routes_per_input);
    } else if (tile[2] >= 0) {
        selected_tile_f32<shared_input, 2>(x, weights, out, records, scales, routes, experts,
                                           in_features, out_features, routes_per_input);
    } else {
        selected_tile_f32<shared_input, 1>(x, weights, out, records, scales, routes, experts,
                                           in_features, out_features, routes_per_input);
    }
}

template<bool selected, bool register_fp4 = false>
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
    hipLaunchKernelGGL((gguf_nvfp4_gemv_f32_kernel<selected, register_fp4>), dim3(blocks), dim3(wave * waves_per_block),
                       0, stream, x, weights, out, in_features, out_features, output_values, outer_scale,
                       ids, scales, experts);
    return hipGetLastError();
}
template<bool shared_input>
int launch_tile4(const float* x, const uint8_t* weights, float* out, const int32_t* tiles, const float* scales,
                 int tile_count, int routes, int experts, int in_features, int out_features,
                 int routes_per_input, hipStream_t stream) {
    if (!x || !weights || !out || !tiles || tile_count <= 0 || tile_count > 65535 || routes <= 0 || experts <= 0 ||
            in_features <= 0 || in_features % values_per_block || out_features <= 0 ||
            routes_per_input <= 0 || routes % routes_per_input) return hipErrorInvalidValue;
    const dim3 grid(1 + (out_features - 1) / waves_per_block, tile_count);
    hipLaunchKernelGGL((gguf_nvfp4_selected_tile4_f32_kernel<shared_input>), grid, dim3(wave * waves_per_block),
                       0, stream, x, weights, out, tiles, scales, routes, experts, in_features, out_features,
                       routes_per_input);
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

extern "C" int hipengine_gguf_nvfp4_selected_register_fp4_f32_f32_out(
        const float* x, const uint8_t* weights, float* out, const int32_t* ids, const float* scales,
        int batch, int experts, int in_features, int out_features, hipStream_t stream) {
    return launch<true, true>(x, weights, out, batch, in_features, out_features, 1.0f, stream, ids, scales, experts);
}

extern "C" int hipengine_gguf_nvfp4_selected_tile4_f32_f32_out(
        const float* x, const uint8_t* weights, float* out, const int32_t* tiles, const float* scales,
        int tile_count, int routes, int experts, int in_features, int out_features, hipStream_t stream) {
    return launch_tile4<false>(x, weights, out, tiles, scales, tile_count, routes, experts, in_features,
                              out_features, 1, stream);
}

extern "C" int hipengine_gguf_nvfp4_selected_tile4_shared_input_f32_f32_out(
        const float* x, const uint8_t* weights, float* out, const int32_t* tiles, const float* scales,
        int tile_count, int routes, int experts, int in_features, int out_features, int routes_per_input,
        hipStream_t stream) {
    return launch_tile4<true>(x, weights, out, tiles, scales, tile_count, routes, experts, in_features,
                             out_features, routes_per_input, stream);
}
