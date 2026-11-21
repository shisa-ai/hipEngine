#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ float inkling_e4m3(uint8_t x) {
    const int e = (x >> 3) & 15;
    const int m = x & 7;
    if (e == 15 && m == 7) return NAN;
    const float a = e == 0 ? ldexpf(float(m), -9) : ldexpf(float(8 + m), e - 10);
    return x & 128 ? -a : a;
}

template <bool NVFP4>
__global__ void inkling2_expert_projection(const uint8_t* w, const uint8_t* s, const float* x,
                                           float* out, int cols, float outer) {
    constexpr int tile = 128;
    constexpr int group = NVFP4 ? 16 : 32;
    const int row = blockIdx.x;
    w += size_t(row) * (NVFP4 ? cols / 2 : cols);
    s += size_t(row) * (cols / group);
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
        hipLaunchKernelGGL((inkling2_expert_projection<true>), dim3(rows), dim3(128), 0, stream, w, s, x, out, cols, outer);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection<false>), dim3(rows), dim3(128), 0, stream, w, s, x, out, cols, outer);
    }
    return hipGetLastError();
}
