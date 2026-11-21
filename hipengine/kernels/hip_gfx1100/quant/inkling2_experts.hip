#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ float inkling_e4m3(uint8_t x) {
    const int e = (x >> 3) & 15;
    const int m = x & 7;
    if (e == 15 && m == 7) return NAN;
    const float a = e == 0 ? ldexpf(float(m), -9) : ldexpf(float(8 + m), e - 10);
    return x & 128 ? -a : a;
}

template <bool NVFP4, bool GROUPED = false>
__global__ void inkling2_expert_projection(const uint8_t* w, const uint8_t* s, const float* x,
                                           float* out, int rows, int cols, float outer,
                                           const int32_t* selected, const float* outers, int input_stride) {
    constexpr int tile = 128;
    constexpr int group = NVFP4 ? 16 : 32;
    const int row = blockIdx.x;
    size_t physical_row = row;
    if constexpr (GROUPED) {
        const int expert = selected[blockIdx.y];
        physical_row += size_t(expert) * rows;
        x += size_t(blockIdx.y) * input_stride;
        out += size_t(blockIdx.y) * rows;
        if constexpr (NVFP4) outer = outers[expert];
    }
    w += physical_row * (NVFP4 ? cols / 2 : cols);
    s += physical_row * (cols / group);
    float acc = 0;
    for (int col = threadIdx.x; col < cols; col += tile) {
        float a;
        if constexpr (NVFP4) {
            const float levels[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
            const int q = (w[col / 2] >> ((col & 1) * 4)) & 15;
            a = (q & 8 ? -levels[q & 7] : levels[q & 7]) * inkling_e4m3(s[col / group]);
            a *= outer;
        } else {
            const int scale = s[col / group];
            const float factor = scale == 255 ? NAN : ldexpf(1.0f, scale - 127);
            a = inkling_e4m3(w[col]) * factor;
        }
        acc = fmaf(a, x[col], acc);
    }
    __shared__ float reduction[tile];
    reduction[threadIdx.x] = acc;
    __syncthreads();
    for (int step = tile / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = reduction[0];
}

extern "C" int hipengine_inkling2_expert_projection(const uint8_t* w, const uint8_t* s, const float* x,
                                                     float* out, int rows, int cols, int nvfp4,
                                                     float outer, hipStream_t stream) {
    if (!w || !s || !x || !out || rows <= 0 || cols <= 0 || cols % (nvfp4 ? 16 : 32)) {
        return hipErrorInvalidValue;
    }
    if (nvfp4) {
        hipLaunchKernelGGL((inkling2_expert_projection<true>), dim3(rows), dim3(128), 0, stream,
                           w, s, x, out, rows, cols, outer, nullptr, nullptr, 0);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection<false>), dim3(rows), dim3(128), 0, stream,
                           w, s, x, out, rows, cols, outer, nullptr, nullptr, 0);
    }
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_grouped_expert_projection(
    const uint8_t* w, const uint8_t* s, const float* x, float* out,
    const int32_t* selected, const float* outers, int count, int rows, int cols, int input_stride, int nvfp4) {
    if (!w || !s || !x || !out || !selected || (nvfp4 && !outers) || count <= 0 || count > 65535 ||
        rows <= 0 || cols <= 0 || cols % (nvfp4 ? 16 : 32) || (input_stride != 0 && input_stride != cols)) {
        return hipErrorInvalidValue;
    }
    if (nvfp4) {
        hipLaunchKernelGGL((inkling2_expert_projection<true, true>), dim3(rows, count), dim3(128), 0, 0,
                           w, s, x, out, rows, cols, 1.0f, selected, outers, input_stride);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection<false, true>), dim3(rows, count), dim3(128), 0, 0,
                           w, s, x, out, rows, cols, 1.0f, selected, outers, input_stride);
    }
    return hipGetLastError();
}

__global__ void inkling2_expert_activate(const float* paired, float* activated, size_t size,
                                         float gate_cap, float up_cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const float gate = paired[2 * i], up = paired[2 * i + 1];
    const float e = expf(-fabsf(gate));
    const float sigmoid = (gate >= 0 ? 1.0f : e) / (1.0f + e);
    activated[i] = gate_cap * tanhf(gate / gate_cap) * sigmoid * up_cap * tanhf(up / up_cap);
}

__global__ void inkling2_expert_mix(const float* down, const float* mix, float* out, int count, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float value = 0.0f;
    for (int j = 0; j < count; ++j) value = __fadd_rn(value, __fmul_rn(mix[j], down[size_t(j) * width + i]));
    out[i] = value;
}

extern "C" int hipengine_inkling2_expert_pointwise(const float* x, const float* mix, float* out,
                                                    int count, int width, float gate_cap, float up_cap) {
    if (!x || !out || count <= 0 || width <= 0) return hipErrorInvalidValue;
    if (mix) {
        hipLaunchKernelGGL(inkling2_expert_mix, dim3(1 + (width - 1) / 256), dim3(256), 0, 0,
                           x, mix, out, count, width);
    } else {
        if (!(gate_cap > 0.0f) || !(up_cap > 0.0f)) return hipErrorInvalidValue;
        const size_t size = size_t(count) * width;
        hipLaunchKernelGGL(inkling2_expert_activate, dim3(1 + (size - 1) / 256), dim3(256), 0, 0,
                           x, out, size, gate_cap, up_cap);
    }
    return hipGetLastError();
}
