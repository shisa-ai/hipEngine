#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ float inkling_e4m3(uint8_t x) {
    const int e = (x >> 3) & 15;
    const int m = x & 7;
    if (e == 15 && m == 7) return NAN;
    const float a = e == 0 ? ldexpf(float(m), -9) : ldexpf(float(8 + m), e - 10);
    return x & 128 ? -a : a;
}

__device__ __forceinline__ float inkling_nvfp4_scale(uint8_t x) {
    const unsigned mag = x & 127;
    const unsigned normal = (120u << 23) + (mag << 20);
    const unsigned subnormal = __float_as_uint(float(mag) * 0x1p-9f);
    const unsigned bits = mag >= 8 ? normal : subnormal;
    return mag == 127 ? NAN : __uint_as_float(bits | ((unsigned(x) & 128) << 24));
}

__device__ __forceinline__ float inkling_e2m1(unsigned q) {
    const unsigned mag = q & 7;
    // Nonzero levels occupy consecutive FP32 half-exponent steps, except 0.5.
    const unsigned bits = mag ? (mag + 252 - (mag == 1)) << 22 : 0;
    return __uint_as_float(bits | ((q & 8) << 28));
}

template <bool NVFP4, bool WIDENED>
__device__ __forceinline__ float inkling_expert_value(const uint8_t* w, const uint8_t* s, int col, float outer) {
    if constexpr (NVFP4) {
        const unsigned q = (w[col / 2] >> ((col & 1) * 4)) & 15;
        const float scale = WIDENED
            ? __uint_as_float(unsigned(reinterpret_cast<const uint16_t*>(s)[col / 16]) << 16)
            : inkling_nvfp4_scale(s[col / 16]);
        const float a = inkling_e2m1(q) * scale;
        return a * outer;
    } else {
        const int scale = s[col / 32];
        const float factor = scale == 255 ? NAN : ldexpf(1.0f, scale - 127);
        return inkling_e4m3(w[col]) * factor;
    }
}

template <bool NVFP4, bool WIDENED>
__device__ __forceinline__ float inkling_expert_lane(const uint8_t* w, const uint8_t* s, const float* x,
                                                    int cols, float outer, int lane) {
    float acc = 0.0f;
    for (int col = lane; col < cols; col += 128) {
        acc = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col, outer), x[col], acc);
    }
    return acc;
}

template <bool NVFP4, bool GROUPED = false, bool WAVE_ROWS = false, bool BATCHED = false, bool WIDENED = false>
__global__ void inkling2_expert_projection(const uint8_t* w, const uint8_t* s, const float* x,
                                           float* out, int rows, int cols, float outer,
                                           const int32_t* selected, const float* outers, int input_stride) {
    constexpr int tile = 128;
    constexpr int group = NVFP4 ? 16 : 32;
    const int row = WAVE_ROWS ? 4 * blockIdx.x + threadIdx.x / 32 : blockIdx.x;
    if (row >= rows) return;
    size_t physical_row = row;
    if constexpr (GROUPED) {
        size_t group_index = blockIdx.y;
        if constexpr (BATCHED) group_index += size_t(blockIdx.z) * gridDim.y;
        const int expert = selected[group_index];
        physical_row += size_t(expert) * rows;
        if constexpr (BATCHED) {
            x += input_stride ? group_index * input_stride : size_t(blockIdx.z) * cols;
        } else {
            x += group_index * input_stride;
        }
        out += group_index * rows;
        if constexpr (NVFP4) outer = outers[expert];
    }
    w += physical_row * (NVFP4 ? cols / 2 : cols);
    s += physical_row * (cols / group) * (WIDENED ? 2 : 1);
    if constexpr (WAVE_ROWS) {
        const int lane = threadIdx.x % 32;
        const int full = cols / tile * tile;
        float a = 0, b = 0, c = 0, d = 0;
        int col = lane;
        for (; col < full; col += tile) {
            a = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col, outer), x[col], a);
            b = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 32, outer), x[col + 32], b);
            c = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 64, outer), x[col + 64], c);
            d = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 96, outer), x[col + 96], d);
        }
        if (col < cols) a = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col, outer), x[col], a);
        if (col + 32 < cols) b = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 32, outer), x[col + 32], b);
        if (col + 64 < cols) c = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 64, outer), x[col + 64], c);
        if (col + 96 < cols) d = fmaf(inkling_expert_value<NVFP4, WIDENED>(w, s, col + 96, outer), x[col + 96], d);
        // Match the original 128-lane +64, +32, +16, ... tree.
        float sum = (a + c) + (b + d);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
        if (lane == 0) out[row] = sum;
    } else {
        __shared__ float reduction[tile];
        reduction[threadIdx.x] = inkling_expert_lane<NVFP4, WIDENED>(w, s, x, cols, outer, threadIdx.x);
        __syncthreads();
        for (int step = tile / 2; step > 0; step /= 2) {
            if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[row] = reduction[0];
    }
}

extern "C" int hipengine_inkling2_expert_projection(const uint8_t* w, const uint8_t* s, const float* x,
                                                     float* out, int rows, int cols, int nvfp4,
                                                     float outer, hipStream_t stream, int widened) {
    if (!w || !s || !x || !out || rows <= 0 || cols <= 0 || cols % (nvfp4 ? 16 : 32) || (widened && !nvfp4)) {
        return hipErrorInvalidValue;
    }
    if (widened) {
        hipLaunchKernelGGL((inkling2_expert_projection<true, false, false, false, true>), dim3(rows), dim3(128), 0, stream,
                           w, s, x, out, rows, cols, outer, nullptr, nullptr, 0);
    } else if (nvfp4) {
        hipLaunchKernelGGL((inkling2_expert_projection<true>), dim3(rows), dim3(128), 0, stream,
                           w, s, x, out, rows, cols, outer, nullptr, nullptr, 0);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection<false>), dim3(rows), dim3(128), 0, stream,
                           w, s, x, out, rows, cols, outer, nullptr, nullptr, 0);
    }
    return hipGetLastError();
}

template <bool NVFP4, bool WAVE_ROWS, bool WIDENED = false>
static void launch_grouped(const uint8_t* w, const uint8_t* s, const float* x, float* out,
                           const int32_t* selected, const float* outers, int count, int rows,
                           int cols, int input_stride, int tokens) {
    const dim3 grid(WAVE_ROWS ? 1 + (rows - 1) / 4 : rows, count, tokens);
    if (tokens == 1) {
        hipLaunchKernelGGL((inkling2_expert_projection<NVFP4, true, WAVE_ROWS, false, WIDENED>), grid, dim3(128), 0, 0,
                           w, s, x, out, rows, cols, 1.0f, selected, outers, input_stride);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection<NVFP4, true, WAVE_ROWS, true, WIDENED>), grid, dim3(128), 0, 0,
                           w, s, x, out, rows, cols, 1.0f, selected, outers, input_stride);
    }
}

extern "C" int hipengine_inkling2_grouped_expert_projection(
    const uint8_t* w, const uint8_t* s, const float* x, float* out,
    const int32_t* selected, const float* outers, int count, int rows, int cols, int input_stride, int nvfp4,
    int wave_rows, int tokens, int widened) {
    if (!w || !s || !x || !out || !selected || (nvfp4 && !outers) || count <= 0 || count > 65535 ||
        tokens <= 0 || tokens > 65535 || rows <= 0 || cols <= 0 || cols % (nvfp4 ? 16 : 32) ||
        (input_stride != 0 && input_stride != cols) || (widened && !nvfp4)) {
        return hipErrorInvalidValue;
    }
    if (widened && wave_rows) {
        launch_grouped<true, true, true>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    } else if (widened) {
        launch_grouped<true, false, true>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    } else if (wave_rows && nvfp4) {
        launch_grouped<true, true>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    } else if (wave_rows) {
        launch_grouped<false, true>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    } else if (nvfp4) {
        launch_grouped<true, false>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    } else {
        launch_grouped<false, false>(w, s, x, out, selected, outers, count, rows, cols, input_stride, tokens);
    }
    return hipGetLastError();
}

template <bool NVFP4, bool WIDENED, int Tile>
__global__ void inkling2_expert_projection_tiled(const uint8_t* w, const uint8_t* s, const float* x,
                                                 float* out, const int32_t* routes, const float* outers,
                                                 int count, int rows, int cols, int input_stride) {
    const int row = 4 * blockIdx.x + threadIdx.x / 32;
    if (row >= rows) return;
    const int32_t* item = routes + (Tile + 1) * blockIdx.y;
    const int expert = item[0];
    const size_t physical = size_t(expert) * rows + row;
    w += physical * (NVFP4 ? cols / 2 : cols);
    s += physical * (cols / (NVFP4 ? 16 : 32)) * (WIDENED ? 2 : 1);
    const float outer = NVFP4 ? outers[expert] : 1.0f;
    const float* inputs[Tile];
    float accum[Tile][4] = {};
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        const int route = item[t + 1] < 0 ? item[1] : item[t + 1];
        inputs[t] = x + (input_stride ? size_t(route) * input_stride : size_t(route / count) * cols);
    }
    const int lane = threadIdx.x % 32;
    const int full = cols / 128 * 128;
    int col = lane;
    for (; col < full; col += 128) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = col + 32 * i;
            const float value = inkling_expert_value<NVFP4, WIDENED>(w, s, c, outer);
            #pragma unroll
            for (int t = 0; t < Tile; ++t) accum[t][i] = fmaf(value, inputs[t][c], accum[t][i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = col + 32 * i;
        if (c >= cols) continue;
        const float value = inkling_expert_value<NVFP4, WIDENED>(w, s, c, outer);
        #pragma unroll
        for (int t = 0; t < Tile; ++t) accum[t][i] = fmaf(value, inputs[t][c], accum[t][i]);
    }
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        float sum = (accum[t][0] + accum[t][2]) + (accum[t][1] + accum[t][3]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
        if (lane == 0 && item[t + 1] >= 0) out[size_t(item[t + 1]) * rows + row] = sum;
    }
}

template <bool NVFP4, bool WIDENED>
static void launch_tiled_experts(const uint8_t* w, const uint8_t* s, const float* x, float* out,
                                const int32_t* routes, const float* outers, int count, int rows, int cols,
                                int input_stride, int tiles, int tile_width) {
    const dim3 grid(1 + (rows - 1) / 4, tiles);
    if (tile_width == 4) {
        hipLaunchKernelGGL((inkling2_expert_projection_tiled<NVFP4, WIDENED, 4>), grid, dim3(128), 0, 0,
                           w, s, x, out, routes, outers, count, rows, cols, input_stride);
    } else {
        hipLaunchKernelGGL((inkling2_expert_projection_tiled<NVFP4, WIDENED, 2>), grid, dim3(128), 0, 0,
                           w, s, x, out, routes, outers, count, rows, cols, input_stride);
    }
}

extern "C" int hipengine_inkling2_tiled_experts(
    const uint8_t* w, const uint8_t* s, const float* x, float* out, const int32_t* routes, const float* outers,
    int count, int rows, int cols, int input_stride, int nvfp4, int widened, int tiles, int tile_width) {
    if (!w || !s || !x || !out || !routes || (nvfp4 && !outers) || count <= 0 || rows <= 0 ||
        cols <= 0 || cols % (nvfp4 ? 16 : 32) || (input_stride != 0 && input_stride != cols) ||
        (widened && !nvfp4) || tiles <= 0 || tiles > 65535 || (tile_width != 2 && tile_width != 4)) {
        return hipErrorInvalidValue;
    }
    if (widened) {
        launch_tiled_experts<true, true>(w, s, x, out, routes, outers, count, rows, cols, input_stride, tiles, tile_width);
    } else if (nvfp4) {
        launch_tiled_experts<true, false>(w, s, x, out, routes, outers, count, rows, cols, input_stride, tiles, tile_width);
    } else {
        launch_tiled_experts<false, false>(w, s, x, out, routes, outers, count, rows, cols, input_stride, tiles, tile_width);
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

template <bool BATCHED, bool EXACT = false>
__global__ void inkling2_expert_mix(const float* down, const float* mix, float* out, int count, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if constexpr (BATCHED) {
        const size_t row = blockIdx.y;
        mix += row * count;
        down += row * count * width;
        out += row * width;
    }
    float value = 0.0f;
    if constexpr (EXACT) {
        // Match separate NumPy multiply/add ufuncs, including cancellation.
        #pragma clang fp contract(off)
        for (int j = 0; j < count; ++j) value = value + mix[j] * down[size_t(j) * width + i];
    } else {
        for (int j = 0; j < count; ++j) value = __fadd_rn(value, __fmul_rn(mix[j], down[size_t(j) * width + i]));
    }
    out[i] = value;
}

extern "C" int hipengine_inkling2_expert_mix_exact(const float* down, const float* mix, float* out,
                                                    int count, int width, int tokens) {
    if (!down || !mix || !out || count <= 0 || width <= 0 || tokens <= 0 || tokens > 65535) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL((inkling2_expert_mix<true, true>), dim3(1 + (width - 1) / 256, tokens),
                       dim3(256), 0, 0, down, mix, out, count, width);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_expert_pointwise(const float* x, const float* mix, float* out,
                                                    int count, int width, float gate_cap, float up_cap, int tokens) {
    if (!x || !out || count <= 0 || width <= 0 || tokens <= 0 || tokens > 65535) return hipErrorInvalidValue;
    if (mix) {
        const dim3 grid(1 + (width - 1) / 256, tokens);
        if (tokens == 1) {
            hipLaunchKernelGGL((inkling2_expert_mix<false>), grid, dim3(256), 0, 0, x, mix, out, count, width);
        } else {
            hipLaunchKernelGGL((inkling2_expert_mix<true>), grid, dim3(256), 0, 0, x, mix, out, count, width);
        }
    } else {
        if (!(gate_cap > 0.0f) || !(up_cap > 0.0f)) return hipErrorInvalidValue;
        const size_t size = size_t(tokens) * count * width;
        hipLaunchKernelGGL(inkling2_expert_activate, dim3(1 + (size - 1) / 256), dim3(256), 0, 0,
                           x, out, size, gate_cap, up_cap);
    }
    return hipGetLastError();
}
