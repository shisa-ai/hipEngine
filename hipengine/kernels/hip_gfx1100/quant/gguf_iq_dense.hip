// Raw dense IQ strict projections. Codec math: llama.cpp@17252c769a63c1cb650ce98ae309cf4de0da7778
// ggml/src/ggml-quants.c; IQ3_S table from ggml-common.h (MIT).
// Fixed 128-thread reduction, no contraction, BF16 input and BF16/F32 output.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {
#include "gguf_iq_dense_tables.h"
#include "gguf_iq2_xs_dense_table.h"
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__device__ __constant__ uint8_t iq3_grid[2048] = {
1,1,1,1,
3,1,1,1,
5,1,1,1,
11,1,1,1,
15,1,1,1,
1,3,1,1,
3,3,1,1,
5,3,1,1,
9,3,1,1,
13,3,1,1,
1,5,1,1,
3,5,1,1,
11,5,1,1,
7,7,1,1,
1,9,1,1,
5,9,1,1,
11,9,1,1,
15,9,1,1,
3,11,1,1,
7,11,1,1,
1,13,1,1,
5,13,1,1,
3,15,1,1,
9,15,1,1,
15,15,1,1,
1,1,3,1,
3,1,3,1,
5,1,3,1,
9,1,3,1,
1,3,3,1,
3,3,3,1,
11,3,3,1,
1,5,3,1,
7,5,3,1,
15,5,3,1,
3,7,3,1,
11,7,3,1,
9,9,3,1,
3,13,3,1,
11,13,3,1,
5,15,3,1,
1,1,5,1,
3,1,5,1,
11,1,5,1,
15,1,5,1,
1,3,5,1,
7,3,5,1,
13,3,5,1,
3,5,5,1,
11,5,5,1,
1,7,5,1,
9,7,5,1,
5,9,5,1,
11,9,5,1,
15,9,5,1,
3,11,5,1,
7,11,5,1,
1,15,5,1,
7,15,5,1,
7,1,7,1,
3,3,7,1,
11,3,7,1,
1,5,7,1,
5,5,7,1,
3,7,7,1,
7,7,7,1,
13,7,7,1,
9,9,7,1,
1,11,7,1,
5,11,7,1,
15,13,7,1,
3,15,7,1,
11,15,7,1,
1,1,9,1,
7,3,9,1,
15,3,9,1,
3,5,9,1,
9,5,9,1,
5,7,9,1,
1,9,9,1,
7,9,9,1,
3,11,9,1,
1,15,9,1,
5,1,11,1,
9,1,11,1,
1,5,11,1,
5,5,11,1,
13,5,11,1,
7,7,11,1,
3,9,11,1,
11,9,11,1,
15,9,11,1,
13,13,11,1,
7,15,11,1,
13,1,13,1,
3,3,13,1,
7,3,13,1,
3,7,13,1,
5,11,13,1,
3,15,13,1,
1,1,15,1,
5,1,15,1,
9,1,15,1,
1,5,15,1,
5,5,15,1,
13,5,15,1,
7,7,15,1,
1,11,15,1,
9,11,15,1,
1,1,1,3,
3,1,1,3,
5,1,1,3,
9,1,1,3,
1,3,1,3,
3,3,1,3,
7,3,1,3,
11,3,1,3,
15,3,1,3,
1,5,1,3,
5,5,1,3,
3,7,1,3,
9,7,1,3,
13,7,1,3,
9,11,1,3,
13,11,1,3,
3,13,1,3,
5,15,1,3,
1,1,3,3,
3,1,3,3,
7,1,3,3,
13,1,3,3,
1,3,3,3,
9,3,3,3,
3,5,3,3,
1,7,3,3,
7,7,3,3,
3,9,3,3,
1,11,3,3,
5,11,3,3,
1,15,3,3,
13,15,3,3,
1,1,5,3,
5,3,5,3,
11,3,5,3,
15,3,5,3,
1,5,5,3,
9,5,5,3,
5,7,5,3,
1,9,5,3,
7,9,5,3,
11,11,5,3,
1,13,5,3,
5,15,5,3,
3,1,7,3,
9,1,7,3,
15,1,7,3,
1,3,7,3,
7,3,7,3,
3,5,7,3,
15,5,7,3,
1,7,7,3,
9,7,7,3,
3,9,7,3,
5,13,7,3,
1,15,7,3,
7,1,9,3,
11,1,9,3,
5,3,9,3,
9,3,9,3,
3,7,9,3,
7,7,9,3,
5,9,9,3,
13,9,9,3,
1,11,9,3,
9,11,9,3,
3,1,11,3,
1,3,11,3,
7,3,11,3,
3,5,11,3,
1,7,11,3,
5,7,11,3,
3,11,11,3,
1,5,13,3,
9,5,13,3,
15,5,13,3,
9,9,13,3,
13,9,13,3,
3,1,15,3,
7,1,15,3,
1,3,15,3,
5,3,15,3,
3,5,15,3,
11,7,15,3,
3,9,15,3,
5,13,15,3,
1,15,15,3,
1,1,1,5,
3,1,1,5,
7,1,1,5,
11,1,1,5,
15,1,1,5,
1,3,1,5,
5,3,1,5,
9,3,1,5,
13,3,1,5,
3,5,1,5,
7,5,1,5,
15,5,1,5,
1,7,1,5,
5,7,1,5,
3,9,1,5,
7,9,1,5,
11,9,1,5,
1,11,1,5,
5,11,1,5,
15,13,1,5,
1,15,1,5,
7,15,1,5,
11,15,1,5,
1,1,3,5,
5,1,3,5,
1,3,3,5,
7,3,3,5,
15,3,3,5,
5,5,3,5,
11,5,3,5,
3,7,3,5,
9,7,3,5,
5,9,3,5,
3,11,3,5,
3,1,5,5,
9,1,5,5,
15,1,5,5,
3,5,5,5,
7,5,5,5,
1,7,5,5,
15,7,5,5,
3,9,5,5,
7,11,5,5,
15,11,5,5,
3,15,5,5,
9,15,5,5,
1,1,7,5,
5,1,7,5,
11,1,7,5,
3,3,7,5,
5,5,7,5,
9,5,7,5,
3,7,7,5,
7,7,7,5,
5,9,7,5,
1,11,7,5,
13,13,7,5,
3,1,9,5,
15,1,9,5,
1,5,9,5,
7,5,9,5,
5,7,9,5,
11,7,9,5,
3,9,9,5,
5,15,9,5,
11,15,9,5,
9,1,11,5,
3,3,11,5,
5,5,11,5,
15,7,11,5,
1,9,11,5,
7,11,11,5,
1,15,11,5,
1,1,13,5,
5,1,13,5,
15,1,13,5,
3,5,13,5,
11,11,13,5,
3,13,13,5,
11,1,15,5,
3,3,15,5,
13,5,15,5,
1,7,15,5,
7,9,15,5,
1,11,15,5,
5,1,1,7,
3,3,1,7,
7,3,1,7,
11,3,1,7,
15,3,1,7,
5,5,1,7,
3,7,1,7,
7,7,1,7,
11,7,1,7,
5,9,1,7,
9,9,1,7,
15,9,1,7,
3,11,1,7,
7,13,1,7,
3,15,1,7,
3,1,3,7,
7,1,3,7,
11,1,3,7,
9,3,3,7,
3,5,3,7,
7,5,3,7,
1,9,3,7,
1,13,3,7,
5,15,3,7,
13,15,3,7,
1,1,5,7,
5,3,5,7,
1,5,5,7,
5,7,5,7,
9,7,5,7,
1,11,5,7,
3,1,7,7,
1,3,7,7,
9,3,7,7,
3,5,7,7,
7,5,7,7,
15,5,7,7,
1,7,7,7,
3,9,7,7,
7,9,7,7,
15,9,7,7,
11,11,7,7,
7,15,7,7,
7,1,9,7,
3,3,9,7,
13,3,9,7,
5,5,9,7,
3,7,9,7,
5,11,9,7,
1,13,9,7,
9,13,9,7,
3,1,11,7,
1,3,11,7,
5,3,11,7,
11,5,11,7,
5,7,11,7,
9,9,11,7,
13,11,11,7,
7,15,11,7,
13,3,13,7,
3,9,13,7,
3,1,15,7,
7,1,15,7,
1,5,15,7,
5,5,15,7,
11,7,15,7,
1,1,1,9,
9,1,1,9,
5,3,1,9,
1,5,1,9,
9,5,1,9,
15,5,1,9,
5,7,1,9,
3,9,1,9,
1,11,1,9,
1,15,1,9,
5,1,3,9,
15,1,3,9,
3,3,3,9,
7,3,3,9,
5,5,3,9,
1,7,3,9,
11,7,3,9,
7,9,3,9,
3,11,3,9,
11,11,3,9,
3,1,5,9,
7,1,5,9,
1,3,5,9,
11,3,5,9,
3,5,5,9,
7,7,5,9,
1,9,5,9,
15,11,5,9,
5,13,5,9,
1,15,5,9,
9,1,7,9,
3,3,7,9,
7,3,7,9,
1,5,7,9,
5,5,7,9,
3,7,7,9,
11,7,7,9,
1,1,9,9,
5,1,9,9,
9,5,9,9,
15,7,9,9,
1,9,9,9,
3,15,9,9,
11,1,11,9,
15,1,11,9,
3,5,11,9,
5,13,11,9,
7,3,13,9,
9,7,13,9,
1,13,13,9,
1,3,15,9,
11,3,15,9,
1,7,15,9,
7,9,15,9,
3,11,15,9,
5,1,1,11,
1,3,1,11,
9,3,1,11,
5,5,1,11,
1,9,1,11,
9,9,1,11,
15,9,1,11,
5,11,1,11,
13,13,1,11,
9,15,1,11,
3,1,3,11,
7,1,3,11,
11,1,3,11,
5,3,3,11,
3,5,3,11,
5,7,3,11,
5,15,3,11,
1,1,5,11,
3,3,5,11,
7,5,5,11,
1,7,5,11,
13,7,5,11,
7,11,5,11,
5,1,7,11,
15,1,7,11,
1,3,7,11,
15,5,7,11,
9,9,7,11,
3,11,7,11,
11,13,7,11,
7,15,7,11,
3,1,9,11,
9,1,9,11,
1,5,9,11,
5,7,9,11,
13,9,9,11,
5,3,11,11,
13,5,11,11,
3,11,11,11,
7,11,11,11,
5,9,13,11,
5,1,15,11,
9,1,15,11,
5,5,15,11,
3,3,1,13,
7,3,1,13,
11,3,1,13,
3,7,1,13,
7,7,1,13,
1,13,1,13,
1,1,3,13,
1,5,3,13,
15,5,3,13,
9,13,3,13,
5,3,5,13,
9,7,5,13,
5,9,5,13,
11,11,5,13,
5,13,5,13,
1,15,5,13,
1,1,7,13,
9,3,7,13,
3,5,7,13,
1,9,7,13,
11,5,9,13,
7,9,9,13,
5,13,9,13,
1,1,11,13,
7,1,11,13,
9,7,11,13,
1,13,11,13,
11,1,13,13,
1,9,13,13,
3,3,15,13,
7,3,15,13,
1,1,1,15,
9,1,1,15,
15,1,1,15,
1,5,1,15,
5,5,1,15,
13,7,1,15,
1,9,1,15,
9,11,1,15,
5,13,1,15,
5,1,3,15,
3,3,3,15,
9,5,3,15,
7,9,3,15,
11,9,3,15,
3,1,5,15,
9,1,5,15,
1,3,5,15,
13,3,5,15,
3,5,5,15,
1,7,5,15,
3,11,5,15,
5,1,7,15,
5,7,7,15,
11,7,7,15,
7,11,7,15,
3,1,9,15,
11,1,9,15,
7,3,9,15,
1,5,9,15,
1,11,9,15,
5,5,11,15,
5,9,11,15,
5,1,13,15,
3,7,13,15,
1,1,15,15
};
__device__ __constant__ int8_t iq4_values[16] = {
    -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// The IQ4 codebook stays a __constant__ table. Replacing it with packed
// immediates removes 16 global_load_i8 per inner body but costs ~110 VALU
// there, and measured -5.0% on the IQ4_XS leaf (2026-09-08 interleaved A/B,
// gfx1151): at 16 waves/SIMD this kernel hides VMEM latency and is VALU
// bound, so an L1-resident byte load is the cheaper operand. The candidate
// was measured and dropped, not deferred; do not re-derive it.
// IQ3_XXS / IQ2_XS sign masks. The published iq_signs[128] table is exactly
// `i` carrying its own odd-parity flag in bit 7, so the lookup is a popcount.
// Asserted entry-for-entry in tests/test_gguf_iq_dense_tables_closed_form.py.
__device__ inline int iq_sign_bits(int i) {
    return i | ((__popc(i) & 1) << 7);
}
// IQ2_XS magnitudes: code 0/1/2/3 select 8/25/43/43. One packed immediate
// replaces a two-deep compare/select chain per decoded element.
__device__ inline float iq2_xs_magnitude(int code) {
    return float((0x2B2B1908u >> (8 * code)) & 0xffu);
}
__device__ float half_load(const uint8_t* p) {
    union { uint16_t u; _Float16 h; } v;
    v.u = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return float(v.h);
}
__device__ float bf16_load(uint16_t x) {
    union { uint32_t u; float f; } v;
    v.u = uint32_t(x) << 16; return v.f;
}
__device__ uint16_t bf16_store(float x) {
    union { uint32_t u; float f; } v;
    v.f = x; return uint16_t((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16);
}
template<int Q> __device__ float weight(const uint8_t* b, int k) {
    if constexpr (Q == 1) {
        int code = (b[2+(k&15)] >> (4*(k>>4))) & 15;
        return half_load(b) * float(iq4_values[code]);
    } else if constexpr (Q == 0) {
        int g = k >> 5, j = k & 31;
        int hi = int(b[2]) | (int(b[3]) << 8);
        int scale = (((b[4+g/2] >> (4*(g&1))) & 15) | (((hi >> (2*g)) & 3) << 4)) - 32;
        int code = (b[8+g*16+(j&15)] >> (4*(j>>4))) & 15;
        return (half_load(b) * float(scale)) * float(iq4_values[code]);
    } else if constexpr (Q == 4) {
        int g = k >> 5, j = k & 31;
        const uint8_t* p = b+66+4*g;
        uint32_t aux = uint32_t(p[0]) | (uint32_t(p[1])<<8)
                     | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
        int index = b[2+g*8+j/4];
        int signs = iq_sign_bits((aux >> (7*(j/8))) & 127);
        float d = (half_load(b) * (0.5f+float(aux>>28))) * 0.5f;
        float mag = d * float(iq3_xxs_grid[index*4+(j&3)]);
        return mag * ((signs & (1<<(j&7))) ? -1.0f : 1.0f);
    } else if constexpr (Q == 6) {
        int g = k >> 5, j = k & 31, slot = j >> 3;
        const uint8_t* p = b + 2 + 2*(g*4+slot);
        int word = int(p[0]) | (int(p[1]) << 8);
        int code = (IQ2_XS_GRID_PACKED[word & 511] >> (2*(j&7))) & 3;
        int scale = (b[66+g] >> (4*(j>>4))) & 15;
        float d = (half_load(b) * (0.5f+float(scale))) * 0.25f;
        float mag = d * iq2_xs_magnitude(code);
        return mag * ((iq_sign_bits(word >> 9) & (1<<(j&7))) ? -1.0f : 1.0f);
    } else if constexpr (Q == 5) {
        int g = k >> 5, j = k & 31, slot = j >> 3;
        int index = b[2+g*4+slot] | (((b[66+g] >> (2*slot)) & 3) << 8);
        int scale = (b[74+g] >> (4*(j>>4))) & 15;
        float d = (half_load(b) * (0.5f+float(scale))) * 0.25f;
        float mag = d * float(iq2_s_grid[index*8+(j&7)]);
        return mag * ((b[34+g*4+slot] & (1<<(j&7))) ? -1.0f : 1.0f);
    } else if constexpr (Q == 3) {
        // block_q3_K: hmask[32], qs[64], scales[12], fp16 d.
        // Same indexing as the in-tree selected Q3_K leaf. Multiply d by
        // the signed scale before the quant, matching the pinned C decoder.
        int group = k >> 4;
        int scale = ((b[96+(group&7)] >> (4*(group>>3))) & 15)
                  | (((b[104+(group&3)] >> (2*(group>>2))) & 3) << 4);
        int q = (b[32+(k>>7)*32+(k&31)] >> (2*((k&127)>>5))) & 3;
        q -= (b[k&31] & (1 << (k>>5))) ? 0 : 4;
        return (half_load(b+108) * float(scale-32)) * float(q);
    } else {
        int g = k >> 5, j = k & 31, grid_slot = j >> 2;
        int index = b[2+g*8+grid_slot] | (((b[66+g] >> grid_slot)&1) << 8);
        int scale = 1+2*((b[106+g/2] >> (4*(g&1))) & 15);
        float mag = (half_load(b) * float(scale)) * float(iq3_grid[index*4+(j&3)]);
        return mag * ((b[74+g*4+j/8] & (1 << (j&7))) ? -1.0f : 1.0f);
    }
}
__global__ void gguf_q3_k_embedding_kernel(const int64_t* ids,
    const uint8_t* w, uint16_t* out, int64_t K, int64_t vocab) {
    int64_t row = blockIdx.y, k = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t token = ids[row];
    if (k >= K || token < 0 || token >= vocab) return;
    float value = weight<3>(w + (token*(K/256)+k/256)*110, k%256);
    uint32_t bits = __float_as_uint(value);
    out[row*K+k] = uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}

// Strict T-column x R-row tile with per-quant-block header preloading. Each
// 128-thread block owns T consecutive output columns and R consecutive rows,
// decoding each weight once and reusing it across the R activations: weight
// traffic for a prompt of `rows` rows falls by a factor of R. Because R only
// changes which rows share a block, the per-(row, column) arithmetic is
// unchanged and every R is bit-identical to R=1.
//
// The block walks the 256-element
// quant blocks in lockstep: phase A hoists each column's (d * scale) header
// products into shared memory cooperatively (128 threads cover at most
// T*16 header slots), phase B decodes the two payload elements this thread
// owns per contract. Per-column arithmetic is bit-identical to the declared
// single-column contract (128 strided accumulators over k = tid + 128*i in
// increasing i, wave32 shuffle tree, four serial wave sums): phase A
// evaluates exactly the first multiplication of each weight expression and
// phase B evaluates the remainder, so the FP sequence per element is
// unchanged while the global-load count per element drops to the payload
// byte(s) only. IQ4_NL (Q=1, 32-element blocks) keeps the plain per-element
// decode path inside the same tile.
template<int Q, bool BF16, int T, int R>
__launch_bounds__(128, R == 1 ? 16 : 2) __global__ void gguf_iq_dense_strict_kernel(
    const uint16_t* x, const uint8_t* w, void* out, int64_t K, int64_t N, int64_t rows) {
    constexpr int B = Q == 1 ? 32 : 256;
    constexpr int S = Q == 0 ? 136 : (Q == 1 ? 18 : (Q == 4 ? 98 : (Q == 5 ? 82 : (Q == 6 ? 74 : 110))));
    constexpr int GPB = Q == 3 ? 16 : 8;   // 32-element groups per 256-block
    constexpr int HDR = Q == 3 ? T*16 : T*8; // header slots filled by phase A
    int64_t n0 = int64_t(blockIdx.x)*T, row0 = int64_t(blockIdx.y)*R;
    const int tid = threadIdx.x;
    const int64_t col_bytes = (K/B) * S;
    const uint8_t* wcol[T];
    #pragma unroll
    for (int t=0;t<T;++t) wcol[t] = w + (n0+t)*col_bytes;
    float acc[R][T];
    #pragma unroll
    for (int r=0;r<R;++r)
        #pragma unroll
        for (int t=0;t<T;++t) acc[r][t] = 0.0f;
    constexpr int RB = 4;                        // quant blocks per barrier round
    __shared__ float s_d[RB][T*16];               // d or d*scale per (column, group)
    __shared__ uint32_t s_aux[RB][T*8];           // IQ3_XXS per-group sign/khigh word
    __shared__ uint8_t s_qs[RB][T*8];             // IQ2_S per-group index high bits
    if constexpr (Q == 1) {
        for (int64_t k = tid; k < K; k += 128) {
            float xv[R];
            #pragma unroll
            for (int r=0;r<R;++r)
                xv[r] = row0+r < rows ? bf16_load(x[(row0+r)*K+k]) : 0.0f;
            #pragma unroll
            for (int t=0;t<T;++t)
                if (n0+t < N) {
                    const float wv = weight<Q>(wcol[t] + (k/B)*S, k%B);
                    #pragma unroll
                    for (int r=0;r<R;++r) acc[r][t] += xv[r] * wv;
                }
        }
    } else {
        const int64_t nblk = K / 256;
        for (int64_t rbase = 0; rbase < nblk; rbase += RB) {
            const int nb = int(nblk - rbase < RB ? nblk - rbase : RB);
            // Phase A: header preload for all T columns of nb quant blocks.
            for (int slot = tid; slot < nb*HDR; slot += 128) {
                const int bi = slot / HDR, hslot = slot % HDR;
                const int t = hslot / GPB, g = hslot % GPB;
                if (n0+t < N) {
                const uint8_t* b = wcol[t] + (rbase+bi)*S;
                if constexpr (Q == 0) {
                    int hi = int(b[2]) | (int(b[3]) << 8);
                    int scale = (((b[4+g/2] >> (4*(g&1))) & 15) | (((hi >> (2*g)) & 3) << 4)) - 32;
                    s_d[bi][t*16+g] = half_load(b) * float(scale);
                } else if constexpr (Q == 4) {
                    const uint8_t* p = b+66+4*g;
                    uint32_t aux = uint32_t(p[0]) | (uint32_t(p[1])<<8)
                                 | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
                    s_aux[bi][t*8+g] = aux;
                    s_d[bi][t*16+g] = (half_load(b) * (0.5f+float(aux>>28))) * 0.5f;
                } else if constexpr (Q == 6) {
                    // j<16 reads the low nibble (shift 0), j>=16 the high one.
                    int scale = b[66+g] & 15;
                    s_d[bi][t*16+g] = (half_load(b) * (0.5f+float(scale))) * 0.25f;
                    scale = (b[66+g] >> 4) & 15;
                    s_d[bi][t*16+8+g] = (half_load(b) * (0.5f+float(scale))) * 0.25f;
                } else if constexpr (Q == 5) {
                    s_qs[bi][t*8+g] = b[66+g];
                    int scale = b[74+g] & 15;
                    s_d[bi][t*16+g] = (half_load(b) * (0.5f+float(scale))) * 0.25f;
                    scale = (b[74+g] >> 4) & 15;
                    s_d[bi][t*16+8+g] = (half_load(b) * (0.5f+float(scale))) * 0.25f;
                } else if constexpr (Q == 2) {
                    int scale = 1+2*((b[106+g/2] >> (4*(g&1))) & 15);
                    s_d[bi][t*16+g] = half_load(b) * float(scale);
                } else if constexpr (Q == 3) {
                    int scale = ((b[96+(g&7)] >> (4*(g>>3))) & 15)
                              | (((b[104+(g&3)] >> (2*(g>>2))) & 3) << 4);
                    s_d[bi][t*16+g] = half_load(b+108) * float(scale-32);
                }
                }
            }
            __syncthreads();
            // Phase B: this thread's two contract elements per quant block.
            for (int bi=0; bi<nb; ++bi) {
            #pragma unroll
            for (int h=0;h<2;++h) {
                const int k = int(rbase+bi)*256 + h*128 + tid;
                // R activations, one decoded weight: the whole point of the
                // row tile. Padding rows load 0.0f and are never stored.
                float xv[R];
                #pragma unroll
                for (int r=0;r<R;++r)
                    xv[r] = row0+r < rows ? bf16_load(x[(row0+r)*K+k]) : 0.0f;
                #pragma unroll
                for (int t=0;t<T;++t) {
                    if (n0+t < N) {
                        const uint8_t* b = wcol[t] + (rbase+bi)*S;
                        const int j = k & 31;          // element within its 32-group
                        const int g = (k & 255) >> 5;  // group within the quant block
                        float wv;
                        if constexpr (Q == 0) {
                            int code = (b[8+g*16+(j&15)] >> (4*(j>>4))) & 15;
                            wv = s_d[bi][t*16+g] * float(iq4_values[code]);
                        } else if constexpr (Q == 4) {
                            const uint32_t aux = s_aux[bi][t*8+g];
                            int index = b[2+g*8+j/4];
                            int signs = iq_sign_bits((aux >> (7*(j/8))) & 127);
                            float mag = s_d[bi][t*16+g] * float(iq3_xxs_grid[index*4+(j&3)]);
                            wv = mag * ((signs & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else if constexpr (Q == 6) {
                            const int slot = j >> 3;
                            const uint8_t* p = b + 2 + 2*(g*4+slot);
                            int word = int(p[0]) | (int(p[1]) << 8);
                            int code = (IQ2_XS_GRID_PACKED[word & 511] >> (2*(j&7))) & 3;
                            float mag = s_d[bi][t*16+((j>>4)?8:0)+g] * iq2_xs_magnitude(code);
                            wv = mag * ((iq_sign_bits(word >> 9) & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else if constexpr (Q == 5) {
                            const int slot = j >> 3;
                            int index = b[2+g*4+slot] | (((s_qs[bi][t*8+g] >> (2*slot)) & 3) << 8);
                            float mag = s_d[bi][t*16+((j>>4)?8:0)+g] * float(iq2_s_grid[index*8+(j&7)]);
                            wv = mag * ((b[34+g*4+slot] & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else if constexpr (Q == 2) {
                            const int grid_slot = j >> 2;
                            int index = b[2+g*8+grid_slot] | (((b[66+g] >> grid_slot)&1) << 8);
                            float mag = s_d[bi][t*16+g] * float(iq3_grid[index*4+(j&3)]);
                            wv = mag * ((b[74+g*4+j/8] & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else { // Q == 3: payload indexing is block-local.
                            const int kk = k & 255;
                            int q = (b[32+(kk>>7)*32+(kk&31)] >> (2*((kk&127)>>5))) & 3;
                            q -= (b[kk&31] & (1 << (kk>>5))) ? 0 : 4;
                            wv = s_d[bi][t*16+(kk>>4)] * float(q);
                        }
                        #pragma unroll
                        for (int r=0;r<R;++r) acc[r][t] += xv[r] * wv;
                    }
                }
            }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int r=0;r<R;++r)
        #pragma unroll
        for (int t=0;t<T;++t)
            for (int delta=16; delta; delta>>=1) acc[r][t] += __shfl_down(acc[r][t], delta);
    __shared__ float sums[4][R][T];
    if ((threadIdx.x & 31)==0)
        #pragma unroll
        for (int r=0;r<R;++r)
            #pragma unroll
            for (int t=0;t<T;++t) sums[threadIdx.x>>5][r][t] = acc[r][t];
    __syncthreads();
    // One thread per live output instead of a serial tail in lane 0. The four
    // wave partials are still summed in index order, so each output keeps the
    // declared serial wave-0..3 accumulation.
    for (int slot = threadIdx.x; slot < R*T; slot += 128) {
        const int r = slot / T, t = slot % T;
        if (row0+r < rows && n0+t < N) {
            float total=0;
            for (int i=0;i<4;++i) total += sums[i][r][t];
            const int64_t index = (row0+r)*N + n0 + t;
            if constexpr (BF16) static_cast<uint16_t*>(out)[index] = bf16_store(total);
            else static_cast<float*>(out)[index] = total;
        }
    }
}
template<int Q, int R> void run_rows(const void* x, const void* w, void* out,
    int64_t rows, int64_t K, int64_t N, int output, hipStream_t stream) {
    constexpr int T = 8;
    dim3 grid((unsigned)((N+T-1)/T), (unsigned)((rows+R-1)/R));
    if (output) hipLaunchKernelGGL((gguf_iq_dense_strict_kernel<Q,true,T,R>),grid,dim3(128),0,stream,
        static_cast<const uint16_t*>(x),static_cast<const uint8_t*>(w),out,K,N,rows);
    else hipLaunchKernelGGL((gguf_iq_dense_strict_kernel<Q,false,T,R>),grid,dim3(128),0,stream,
        static_cast<const uint16_t*>(x),static_cast<const uint8_t*>(w),out,K,N,rows);
}
// Row slab selection. R only changes how many prompt rows one block owns; the
// per-(row, column) k ownership, FMA order, shuffle tree and serial wave-0..3
// sum are all unchanged, so every R produces bit-identical outputs. The caller
// owns the choice; 0 means "pick from rows".
template<int Q> void run(const void* x, const void* w, void* out,
    int64_t rows, int64_t K, int64_t N, int output, int rowbatch, hipStream_t stream) {
    switch (rowbatch) {
        case 8: run_rows<Q,8>(x,w,out,rows,K,N,output,stream); return;
        case 4: run_rows<Q,4>(x,w,out,rows,K,N,output,stream); return;
        case 2: run_rows<Q,2>(x,w,out,rows,K,N,output,stream); return;
        default: run_rows<Q,1>(x,w,out,rows,K,N,output,stream); return;
    }
}
}
extern "C" int hipengine_gguf_q3_k_embedding(const void* ids, const void* w,
    void* out, int64_t rows, int64_t K, int64_t vocab, void* stream) {
    if (!ids || !w || !out || rows<=0 || rows>65535 || K<=0 || K%256 || vocab<=0)
        return int(hipErrorInvalidValue);
    gguf_q3_k_embedding_kernel<<<dim3((K+255)/256,rows),256,0,(hipStream_t)stream>>>(
        (const int64_t*)ids,(const uint8_t*)w,(uint16_t*)out,K,vocab);
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq_dense(const void* x,const void* w,void* out,
    int64_t rows,int64_t K,int64_t N,int quant,int output,int rowbatch,void* stream) {
    switch(quant) {
        case 0: run<0>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 1: run<1>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 2: run<2>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 3: run<3>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 4: run<4>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 5: run<5>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        case 6: run<6>(x,w,out,rows,K,N,output,rowbatch,(hipStream_t)stream); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}
