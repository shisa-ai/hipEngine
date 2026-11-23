#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

// Pack 32 activations with codes chosen before rounding the stored scale.
template <bool SPLIT_ACTIVATION>
__global__ void pack_q8(const float* x, int8_t* codes, float* scales, int groups,
                        int width, float gate_cap, float up_cap) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= groups) return;
    const int i = 32 * group + lane;
    float v;
    if constexpr (SPLIT_ACTIVATION) {
        const int j = i + (i / width) * width;
        const float gate = x[j], up = x[j + width];
        const float e = expf(-fabsf(gate));
        const float sigmoid = (gate >= 0.0f ? 1.0f : e) / (1.0f + e);
        v = (gate_cap * tanhf(gate / gate_cap) * sigmoid) * (up_cap * tanhf(up / up_cap));
    } else {
        v = x[i];
    }
    float amax = fabsf(v);
    for (int offset = 16; offset; offset /= 2) amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    const float d = amax / 127.0f;
    const int code = d > 0.0f ? int(roundf(v / d)) : 0;
    codes[i] = static_cast<int8_t>(code);
    if (lane == 0) scales[group] = __half2float(__float2half_rn(d));
}

__device__ __forceinline__ float weight_scale(uint8_t code) {
    const int e = (code >> 3) & 15;
    const int significand = (code & 7) + (e ? 8 : 0);
    const float power = __uint_as_float(uint32_t(116 + (e ? e : 1)) << 23);
    return code == 0x7f ? 0.0f : significand * power;
}

__device__ __forceinline__ uint32_t fp4_to_byte(unsigned code) {
    const int value = (0xc8643210u >> (4 * (code & 7))) & 15;
    return uint32_t(code & 8 ? -value : value) & 255u;
}

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int sum) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), sum, false);
#else
    return sum;
#endif
}

template <bool SHARED_INPUT>
__global__ __launch_bounds__(128) void nvfp4_q8_selected(
        const int8_t* x, const float* xs, const uint8_t* weights, float* output,
        const int32_t* ids, const float* outer, int n, int m, int experts, size_t output_count) {
    const int lane = threadIdx.x & 31;
    const size_t i = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (i >= output_count) return;
    const int route = i / m, channel = i % m, blocks = n / 64;
    const int expert = ids[route];
    if (expert < 0 || expert >= experts) {
        if (!lane) output[i] = 0.0f;
        return;
    }
    const size_t row = size_t(expert) * m + channel;
    const uint8_t* packed = weights + row * blocks * 36;
    const int group = (lane % 8) / 2, part = lane & 1;
    if constexpr (!SHARED_INPUT) {
        x += size_t(route) * n;
        xs += size_t(route) * (n / 32);
    }
    float sum = 0.0f;
    // Eight lanes cover each 64-element block, interleaving four blocks per wave.
    for (int b = lane / 8; b < blocks; b += 4) {
        const uint8_t* w = packed + b * 36;
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(w + 4 + group * 8 + part * 4);
        uint32_t lo = 0, hi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned byte = (codes >> (8 * j)) & 255u;
            lo |= fp4_to_byte(byte & 15) << (8 * j);
            hi |= fp4_to_byte(byte >> 4) << (8 * j);
        }
        const int8_t* in = x + b * 64 + group * 16 + part * 4;
        const uint32_t xlo = *reinterpret_cast<const uint32_t*>(in);
        const uint32_t xhi = *reinterpret_cast<const uint32_t*>(in + 8);
        const int sum_int = dot4(hi, xhi, dot4(lo, xlo, 0));
        sum = fmaf(weight_scale(w[group]) * xs[2 * b + group / 2], float(sum_int), sum);
    }
    for (int shift = 16; shift; shift /= 2) sum += __shfl_down(sum, shift, 32);
    if (!lane) output[i] = sum * (outer ? outer[route] : 1.0f);
}

// Process only populated slots in each fixed-width expert record. Each
// route retains the scalar shared-input projection's accumulation order.
template<int COUNT>
__device__ __forceinline__ void counted_q8_tile(
        const int8_t* x, const float* xs, const uint8_t* weights, float* output,
        const int32_t* record, const float* outer, int n, int m, int routes, int channel, int lane) {
    const int blocks = n / 64;
    const int group = (lane % 8) / 2, part = lane & 1;
    const size_t row = size_t(record[0]) * m + channel;
    const uint8_t* packed = weights + row * blocks * 36;
    const int8_t* inputs[COUNT];
    const float* scales[COUNT];
    int route[COUNT];
    float sum[COUNT] = {};
    #pragma unroll
    for (int r = 0; r < COUNT; ++r) {
        route[r] = record[1 + r];
        const int safe = route[r] >= 0 && route[r] < routes ? route[r] : 0;
        inputs[r] = x + size_t(safe / 8) * n + group * 16 + part * 4;
        scales[r] = xs + size_t(safe / 8) * (n / 32) + group / 2;
    }
    for (int b = lane / 8; b < blocks; b += 4) {
        const uint8_t* w = packed + b * 36;
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(w + 4 + group * 8 + part * 4);
        uint32_t lo = 0, hi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned code = codes >> (8 * j);
            lo |= fp4_to_byte(code & 15) << (8 * j);
            hi |= fp4_to_byte((code >> 4) & 15) << (8 * j);
        }
        const float d = weight_scale(w[group]);
        #pragma unroll
        for (int r = 0; r < COUNT; ++r) {
            const int8_t* in = inputs[r] + b * 64;
            const uint32_t xlo = *reinterpret_cast<const uint32_t*>(in);
            const uint32_t xhi = *reinterpret_cast<const uint32_t*>(in + 8);
            const int acc = dot4(hi, xhi, dot4(lo, xlo, 0));
            sum[r] = fmaf(d * scales[r][2 * b], float(acc), sum[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < COUNT; ++r) {
        for (int offset = 16; offset; offset /= 2) sum[r] += __shfl_down(sum[r], offset, 32);
        if (!lane && route[r] >= 0 && route[r] < routes) {
            output[size_t(route[r]) * m + channel] = sum[r] * (outer ? outer[route[r]] : 1.0f);
        }
    }
}

__global__ __launch_bounds__(128) void nvfp4_q8_counted_gate(
        const int8_t* x, const float* xs, const uint8_t* weights, float* output,
        const int32_t* records, const float* outer, int n, int m, int routes, int experts) {
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.x * 4 + threadIdx.x / 32;
    const int32_t* record = records + 5 * blockIdx.y;
    if (channel >= m || record[0] < 0 || record[0] >= experts || record[1] < 0) return;
    if (record[2] < 0) counted_q8_tile<1>(x, xs, weights, output, record, outer, n, m, routes, channel, lane);
    else if (record[3] < 0) counted_q8_tile<2>(x, xs, weights, output, record, outer, n, m, routes, channel, lane);
    else if (record[4] < 0) counted_q8_tile<3>(x, xs, weights, output, record, outer, n, m, routes, channel, lane);
    else counted_q8_tile<4>(x, xs, weights, output, record, outer, n, m, routes, channel, lane);
}

extern "C" int hipengine_gguf_nvfp4_selected_counted_shared_input_q8_dot(
        const float* x, int8_t* codes, float* scales, const uint8_t* weights, float* out,
        const int32_t* records, const float* outer, int tiles, int rows, int n, int m, int experts) {
    if (!x || !codes || !scales || !weights || !out || !records || rows < 2 || rows > 4 ||
        tiles < rows * 2 || tiles > rows * 8 || m != 2304 || (n != 1152 && n != 2304) || experts <= 0)
        return hipErrorInvalidValue;
    const int groups = rows * n / 32;
    hipLaunchKernelGGL((pack_q8<false>), dim3((groups + 3) / 4), dim3(128), 0, 0, x, codes, scales,
                       groups, n, 0.0f, 0.0f);
    if (const int status = hipGetLastError()) return status;
    hipLaunchKernelGGL(nvfp4_q8_counted_gate, dim3((m + 3) / 4, tiles), dim3(128), 0, 0,
                       codes, scales, weights, out, records, outer, n, m, rows * 8, experts);
    return hipGetLastError();
}

template <bool SPLIT_ACTIVATION, bool SHARED_INPUT = false>
static int project_q8(const float* x, int8_t* codes, float* scales, const uint8_t* weights, float* out,
                      const int32_t* ids, const float* outer, int routes, int n, int m, int experts,
                      float gate_cap = 0.0f, float up_cap = 0.0f) {
    if (!x || !codes || !scales || !weights || !out || !ids || routes != 8 || m != 2304 ||
        (n != 1152 && n != 2304) || experts <= 0) return hipErrorInvalidValue;
    const int groups = (SHARED_INPUT ? 1 : routes) * (n / 32);
    hipLaunchKernelGGL((pack_q8<SPLIT_ACTIVATION>), dim3((groups + 3) / 4), dim3(128), 0, 0,
                       x, codes, scales, groups, n, gate_cap, up_cap);
    if (const int status = hipGetLastError()) return status;
    const size_t total = size_t(routes) * m;
    hipLaunchKernelGGL((nvfp4_q8_selected<SHARED_INPUT>), dim3((total + 3) / 4), dim3(128), 0, 0,
                       codes, scales, weights, out, ids, outer, n, m, experts, total);
    return hipGetLastError();
}

extern "C" int hipengine_gguf_nvfp4_selected_q8_dot(
        const float* x, int8_t* codes, float* scales, const uint8_t* weights, float* out,
        const int32_t* ids, const float* outer, int routes, int n, int m, int experts) {
    return project_q8<false>(x, codes, scales, weights, out, ids, outer, routes, n, m, experts);
}

extern "C" int hipengine_gguf_nvfp4_selected_shared_input_q8_dot(
        const float* x, int8_t* codes, float* scales, const uint8_t* weights, float* out,
        const int32_t* ids, const float* outer, int routes, int n, int m, int experts) {
    return project_q8<false, true>(x, codes, scales, weights, out, ids, outer, routes, n, m, experts);
}

extern "C" int hipengine_gguf_nvfp4_selected_split_activation_q8_dot(
        const float* x, int8_t* codes, float* scales, const uint8_t* weights, float* out,
        const int32_t* ids, const float* outer, int routes, int n, int m, int experts,
        float gate_cap, float up_cap) {
    if (n != 1152 || !(gate_cap > 0.0f) || !(up_cap > 0.0f)) return hipErrorInvalidValue;
    return project_q8<true>(x, codes, scales, weights, out, ids, outer, routes, n, m, experts, gate_cap, up_cap);
}
