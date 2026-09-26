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
#include "gguf_iq3_s_grid.h"
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
// Same conversion as half_load, from a value already in a register. Used by the
// T16 owner, which reads a column's d as one half of a wide u32 header load.
__device__ float half_from_bits(uint16_t bits) {
    union { uint16_t u; _Float16 h; } v;
    v.u = bits;
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
    constexpr int RB = 4;
    __shared__ float s_d[RB][T*16];               // d or d*scale per (column, group)
    __shared__ uint32_t s_aux[RB][T*8];           // IQ3_XXS per-group sign/khigh word
    __shared__ uint8_t s_qs[RB][T*8];             // IQ2_S per-group index high bits
    // Codebook grid staged in LDS. Under a lane-varying index a __constant__
    // grid is a global byte load per decoded element -- 64 per inner body for
    // IQ3_S. Staging swaps each for a ds_read, which issues on a different
    // pipe and adds no arithmetic, so it does not repeat the rejected IQ4
    // codebook trade. iq2_s_grid (8192 B) is excluded by measurement, not by
    // guess: staging it drops IQ2_S from 16 to 12 waves/SIMD and measured
    // -10.6% (2026-09-08 interleaved A/B, gfx1151), so the occupancy loss
    // outweighs the 16 loads it removes.
    constexpr int GRID_BYTES = Q == 2 ? 2048 : (Q == 4 ? 1024 : (Q == 6 ? 1024 : 0));
    __shared__ uint8_t s_grid[GRID_BYTES ? GRID_BYTES : 1];
    if constexpr (GRID_BYTES) {
        const uint8_t* src;
        if constexpr (Q == 2) src = iq3_grid;
        else if constexpr (Q == 4) src = iq3_xxs_grid;
        else src = reinterpret_cast<const uint8_t*>(IQ2_XS_GRID_PACKED);
        for (int i = tid; i < GRID_BYTES; i += 128) s_grid[i] = src[i];
        __syncthreads();
    }
    // IQ4_XS fused codebook LUT. The strict phase-B dequant is a dependent
    // load chain per element: payload byte -> nibble extract -> __constant__
    // iq4_values[code]. Measured on the W7900 (2026-09-09 mutation bisection,
    // blk.0.ffn_gate 5120->17408 at rows=1) the codebook load alone costs
    // 24% of the kernel. The nibble half is thread-invariant (j = tid & 31
    // for both h phases), so a 512-entry byte-indexed LDS table fuses the
    // extract and the codebook into one ds_read: s_cb[byte] is the low
    // nibble's codebook value, s_cb[256+byte] the high one. Values are
    // entry-for-entry iq4_values, and the multiply order is unchanged, so the
    // FP sequence per element is identical.
    constexpr int CB_FLOATS = Q == 0 ? 512 : 0;
    __shared__ float s_cb[CB_FLOATS ? CB_FLOATS : 1];
    if constexpr (CB_FLOATS) {
        for (int i = tid; i < CB_FLOATS; i += 128) {
            const int byte = i & 255;
            const int nib = i >= 256 ? (byte >> 4) : (byte & 15);
            s_cb[i] = float(iq4_values[nib]);
        }
        __syncthreads();
    }
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
                            const int bytev = b[8+g*16+(j&15)];
                            wv = s_d[bi][t*16+g] * s_cb[bytev + ((j>>4) ? 256 : 0)];
                        } else if constexpr (Q == 4) {
                            const uint32_t aux = s_aux[bi][t*8+g];
                            int index = b[2+g*8+j/4];
                            int signs = iq_sign_bits((aux >> (7*(j/8))) & 127);
                            float mag = s_d[bi][t*16+g] * float(s_grid[index*4+(j&3)]);
                            wv = mag * ((signs & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else if constexpr (Q == 6) {
                            const int slot = j >> 3;
                            const uint8_t* p = b + 2 + 2*(g*4+slot);
                            int word = int(p[0]) | (int(p[1]) << 8);
                            int code = (reinterpret_cast<const uint16_t*>(s_grid)[word & 511] >> (2*(j&7))) & 3;
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
                            float mag = s_d[bi][t*16+g] * float(s_grid[index*4+(j&3)]);
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

// Output columns per local32 block. The four local32 owners and their seven
// launchers must agree on this: the kernel indexes columns from
// blockIdx.x * COLS, so a launcher that divides N by a different value
// silently computes only a fraction of the output. Measured 2026-09-13 on the
// 117 IQ4_XS verifier tensors at rows=3 (per-tensor, W7900): 4 -> 442.1,
// 8 -> 440.9, 2 -> 421.4, 16 -> 149.3 GB/s, all bit-identical, so 8 stays.
#define GGUF_IQ_LOCAL32_COLS 8

// Local32 IQ4_XS decode GEMV: one wave per 8 output columns, each lane owns
// 8 contiguous K values, and the whole 8-column payload for one 256-element
// block is read as two u32 loads per column. This is the raw-layout analogue
// of the retained pack8/T16 local32 decode owner: the IQ4_XS in-block
// geometry gives every lane a lane-constant nibble half and a consecutive
// 8-byte payload window (element k lives in byte 8+g*16+(k&15) at nibble
// (k&31)>>4; a lane's k = tid*8..tid*8+7 spans exactly one 32-group, one
// nibble half and one 8-byte window), and the byte-indexed fused codebook
// LUT from the strict kernel is reused so each element costs one BFE, one
// ds_read, one mul and one FMA. No phase-A/B split, no per-round barriers,
// and 4x fewer L1 requests per block than the strict owner (64 u32 vs 256 u8
// per column-block; each payload u32 is shared by two lanes as an L1 hit).
// The accumulation order differs from the strict contract (lane-grouped
// contiguous k, wave32 shuffle tree, cross-wave serial sum), so this is a
// gated approximate owner, not a strict replacement: routes admitting it owe
// the production-referenced accuracy gate.
//
// WAVES > 1 is split-K across the block's waves for the narrow-N shapes
// (ffn_down at N=5120 leaves only N/8 = 640 single-wave blocks on a 96-CU
// part, which starves the machine). Each wave strides the K block range;
// per-column partials cross-wave reduce through LDS in a fixed wave order,
// so the owner stays deterministic. The K-range partition changes the sum
// order again, which the gate already covers.
// ROWS > 1 is the verifier sibling (2026-09-12): the same owner geometry with
// one block owning ROWS prompt rows. The weight decode is shared across the
// rows and every per-(row, column) k ownership, FMA order, shuffle tree and
// serial wave-0..WAVES-1 sum is unchanged, so each row's output is
// bit-identical to the ROWS == 1 owner's output for that row. The launch
// bounds relax to one resident block for ROWS > 1 because the row accumulators
// and activations need the registers that the rows == 1 variant spends on
// occupancy.
template<int WAVES, int ROWS = 1>
__global__ __launch_bounds__(32 * WAVES,
    ROWS > 1 ? 1 : (16 / WAVES > 0 ? 16 / WAVES : 1))
void gguf_iq4_xs_local32_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    uint16_t* __restrict__ out, int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = blocks_per_row * 136;

    // Lane-constant in-block geometry (see the header comment).
    const int g = lane >> 2;                // this lane's 32-element group
    const int nib = (lane >> 1) & 1;         // its nibble half
    const int byteoff = 8 + g * 16 + (lane & 1) * 8;  // payload window base

    // Fused byte-indexed codebook LUT: [v] is iq4_values[v&15] and [256+v] is
    // iq4_values[v>>4], so one ds_read replaces the extract-and-codebook-load
    // pair. Built once; the leading barrier of this kernel guards it.
    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc[ROWS][COLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t) acc[r][t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        // This lane's 8 activation values (contiguous k window) for every row.
        float xv[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint16_t* xp = x + r * K + blk * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[r][j] = bf16_load(xp[j]);
        }
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* b = w + (n0 + t) * col_bytes + blk * 136;
            // Header: d (f16) and hi (u16) share one u32; the scale nibbles
            // for all 8 groups share the next one.
            const uint32_t hdr = *reinterpret_cast<const uint32_t*>(b);
            const uint32_t scl = *reinterpret_cast<const uint32_t*>(b + 4);
            const int scale = (((scl >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
                               (((hdr >> (16 + 2 * g)) & 3) << 4)) - 32;
            const float s_d = half_load(b) * float(scale);
            const uint32_t p0 = *reinterpret_cast<const uint32_t*>(b + byteoff);
            const uint32_t p1 = *reinterpret_cast<const uint32_t*>(b + byteoff + 4);
            const uint32_t p[2] = {p0, p1};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int code_byte = int((p[j >> 2] >> (8 * (j & 3))) & 0xffu);
                const float wv = s_d * s_cb[code_byte + (nib << 8)];
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
                    acc[r][t] += xv[r][j] * wv;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t)
            for (int delta = 16; delta; delta >>= 1)
                acc[r][t] += __shfl_down(acc[r][t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][ROWS][COLS];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int t = 0; t < COLS; ++t) xwave[wave][r][t] = acc[r][t];
    }
    __syncthreads();
    if (tid < ROWS * COLS) {
        const int r = tid / COLS;
        const int t = tid % COLS;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][r][t];
            out[int64_t(r) * N + n0 + t] = bf16_store(total);
        }
    }
}
}
// Fused IQ4_XS gate/up dual + SiLU decode GEMV (2026-09-11): the decode
// analogue of the Q4 T16 dense dual local32 SiLU owner. One block computes
// the same 8 output columns for BOTH matrices from one activation window
// (the x reads, codebook LUT and per-lane geometry are shared), halving the
// gate/up launch count and folding the SiLU multiply. Both accumulators are
// bf16-rounded before the SiLU exactly like the unfused
// single/single/silu_mul path (the separate kernel reads the two bf16
// buffers and computes 1/(1+expf(-g))*g*u in f32), so the fused owner is
// bit-exact with it: same per-column accumulation as the single, same
// rounding boundaries, same SiLU arithmetic.
template<int WAVES>
__global__ __launch_bounds__(32 * WAVES, 16 / WAVES > 0 ? 16 / WAVES : 1)
void gguf_iq4_xs_local32_dual_silu_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ wa,
    const uint8_t* __restrict__ wb, uint16_t* __restrict__ out,
    int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = blocks_per_row * 136;

    const int g = lane >> 2;
    const int nib = (lane >> 1) & 1;
    const int byteoff = 8 + g * 16 + (lane & 1) * 8;

    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc_a[COLS];
    float acc_b[COLS];
#pragma unroll
    for (int t = 0; t < COLS; ++t) { acc_a[t] = 0.0f; acc_b[t] = 0.0f; }

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        const uint16_t* xp = x + blk * 256 + lane * 8;
        float xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) xv[j] = bf16_load(xp[j]);
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* ba = wa + (n0 + t) * col_bytes + blk * 136;
            const uint8_t* bb = wb + (n0 + t) * col_bytes + blk * 136;
            const uint32_t hdr_a = *reinterpret_cast<const uint32_t*>(ba);
            const uint32_t scl_a = *reinterpret_cast<const uint32_t*>(ba + 4);
            const int scale_a = (((scl_a >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
                               (((hdr_a >> (16 + 2 * g)) & 3) << 4)) - 32;
            const float sd_a = half_load(ba) * float(scale_a);
            const uint32_t hdr_b = *reinterpret_cast<const uint32_t*>(bb);
            const uint32_t scl_b = *reinterpret_cast<const uint32_t*>(bb + 4);
            const int scale_b = (((scl_b >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
                               (((hdr_b >> (16 + 2 * g)) & 3) << 4)) - 32;
            const float sd_b = half_load(bb) * float(scale_b);
            const uint32_t pa[2] = {
                *reinterpret_cast<const uint32_t*>(ba + byteoff),
                *reinterpret_cast<const uint32_t*>(ba + byteoff + 4)};
            const uint32_t pb[2] = {
                *reinterpret_cast<const uint32_t*>(bb + byteoff),
                *reinterpret_cast<const uint32_t*>(bb + byteoff + 4)};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int code_a = int((pa[j >> 2] >> (8 * (j & 3))) & 0xffu);
                const int code_b = int((pb[j >> 2] >> (8 * (j & 3))) & 0xffu);
                acc_a[t] += xv[j] * (sd_a * s_cb[code_a + (nib << 8)]);
                acc_b[t] += xv[j] * (sd_b * s_cb[code_b + (nib << 8)]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < COLS; ++t)
        for (int delta = 16; delta; delta >>= 1) {
            acc_a[t] += __shfl_down(acc_a[t], delta);
            acc_b[t] += __shfl_down(acc_b[t], delta);
        }
    __shared__ float xwave_a[WAVES > 1 ? WAVES : 1][COLS];
    __shared__ float xwave_b[WAVES > 1 ? WAVES : 1][COLS];
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            xwave_a[wave][t] = acc_a[t];
            xwave_b[wave][t] = acc_b[t];
        }
    }
    __syncthreads();
    if (tid < COLS) {
        const int t = tid;
        if (n0 + t < N) {
            float ta = 0.0f, tb = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) {
                ta += xwave_a[wv2][t];
                tb += xwave_b[wv2][t];
            }
            // bf16-round both accumulators exactly like the unfused path,
            // then the silu_mul kernel's arithmetic verbatim.
            const float ga = bf16_load(bf16_store(ta));
            const float ub = bf16_load(bf16_store(tb));
            const float sig = 1.0f / (1.0f + expf(-ga));
            out[n0 + t] = bf16_store(ga * sig * ub);
        }
    }
}

// Local32 IQ4_NL decode GEMV: the sibling of the IQ4_XS owner above.
// IQ4_NL packs one f16 scale and 16 payload bytes per 32-element block
// (18 bytes, K%32), with the SAME element-to-byte/nibble mapping as
// IQ4_XS (byte 2+(k&15), nibble (k>>4)&1), so the local32 geometry
// transfers directly: each lane's 8 contiguous k values span exactly one
// nibble half and one 8-byte payload window per column per 256-window,
// and the byte-indexed fused codebook LUT is shared. The block scale is
// the plain f16 d (no hi/scale header). Same accumulation-order contract
// as the IQ4_XS owner: lane-grouped contiguous k, wave32 shuffle tree,
// fixed-order cross-wave sum - a gated approximate owner, not a strict
// replacement.
// ROWS > 1 is the verifier sibling (2026-09-12), exactly as on the IQ4_XS
// owner: one block owns ROWS prompt rows, the weight decode is shared across
// them, and each row's per-(row, column) k ownership, FMA order, shuffle tree
// and cross-wave sum are unchanged, so every row is bit-identical to the
// ROWS == 1 owner's output for that row.
template<int WAVES, int ROWS = 1>
__global__ __launch_bounds__(32 * WAVES,
    ROWS > 1 ? 1 : (16 / WAVES > 0 ? 16 / WAVES : 1))
void gguf_iq4_nl_local32_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    uint16_t* __restrict__ out, int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = (K / 32) * 18;

    const int g = lane >> 2;                // this lane's 32-element block
    const int nib = (lane >> 1) & 1;         // its nibble half
    const int byteoff = 2 + (lane & 1) * 8; // payload window base

    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc[ROWS][COLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t) acc[r][t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        float xv[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint16_t* xp = x + r * K + blk * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[r][j] = bf16_load(xp[j]);
        }
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* b =
                w + (n0 + t) * col_bytes + (blk * 8 + g) * 18;
            const float s_d = half_load(b);
            // The 18-byte block stride leaves the payload only 2-aligned
            // (unlike the XS owner's 136-byte stride), so the window is
            // read as four u16s - each 2-aligned by construction - instead
            // of two u32s.
            const uint16_t q[4] = {
                *reinterpret_cast<const uint16_t*>(b + byteoff),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 2),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 4),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 6)};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int code_byte = int((q[j >> 1] >> (8 * (j & 1))) & 0xffu);
                const float wv = s_d * s_cb[code_byte + (nib << 8)];
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
                    acc[r][t] += xv[r][j] * wv;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t)
            for (int delta = 16; delta; delta >>= 1)
                acc[r][t] += __shfl_down(acc[r][t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][ROWS][COLS];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int t = 0; t < COLS; ++t) xwave[wave][r][t] = acc[r][t];
    }
    __syncthreads();
    if (tid < ROWS * COLS) {
        const int r = tid / COLS;
        const int t = tid % COLS;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][r][t];
            out[int64_t(r) * N + n0 + t] = bf16_store(total);
        }
    }
}

extern "C" int hipengine_gguf_iq4_nl_local32_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    if (!x || !w || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (gguf_iq4_nl_local32_gemv_kernel<1><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        case 2:
            err = (gguf_iq4_nl_local32_gemv_kernel<2><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        case 4:
            err = (gguf_iq4_nl_local32_gemv_kernel<4><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}

extern "C" int hipengine_gguf_iq4_xs_local32_dual_silu(const void* x,
    const void* wa, const void* wb, void* out, int64_t rows,
    int64_t K, int64_t N, int waves, void* stream) {
    if (!x || !wa || !wb || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (gguf_iq4_xs_local32_dual_silu_kernel<1><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 2:
            err = (gguf_iq4_xs_local32_dual_silu_kernel<2><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        case 4:
            err = (gguf_iq4_xs_local32_dual_silu_kernel<4><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)wa, (const uint8_t*)wb,
                (uint16_t*)out, K, N), int(hipGetLastError()));
            return err;
        default:
            return int(hipErrorInvalidValue);
    }
}

// Local32 decode GEMV for the split/scale IQ quants (decode lever 2b):
// IQ3_S (Q=2), IQ3_XXS (Q=4), IQ2_S (Q=5), IQ2_XS (Q=6) - the siblings of
// the IQ4_XS/IQ4_NL owners. Same geometry: one wave per 8 output columns,
// each lane owns 8 contiguous k, WAVES-way K split, and per quant a lane's
// window maps to exactly one 8-element slot (IQ2_S/XS, IQ3_XXS) or two
// 4-element grid slots (IQ3_S) of its 32-element group, so the per-lane
// metadata is a handful of bytes per column per 256-block.
//
// Grid staging follows the strict kernel's own measurements: iq3_grid
// (2 KB) and iq3_xxs_grid (1 KB) stage in LDS; iq2_s_grid (8 KB) is
// excluded by measurement (occupancy loss outweighs the loads it
// removes) and stays __constant__; IQ2_XS_GRID_PACKED (1 KB) stages.
//
// Same accumulation-order contract as the other local32 owners:
// lane-grouped contiguous k, wave32 shuffle tree, fixed-order cross-wave
// sum - a gated approximate owner, not a strict replacement.
// ROWS > 1 is the verifier sibling (2026-09-12), exactly as on the IQ4_XS
// owner: one block owns ROWS prompt rows, the weight decode is shared across
// them, and each row's per-(row, column) k ownership, FMA order, shuffle tree
// and cross-wave sum are unchanged, so every row is bit-identical to the
// ROWS == 1 owner's output for that row.
template<int Q, int WAVES, int ROWS = 1>
__global__ __launch_bounds__(32 * WAVES,
    ROWS > 1 ? 1 : (16 / WAVES > 0 ? 16 / WAVES : 1))
void gguf_iq_split_local32_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    uint16_t* __restrict__ out, int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    constexpr int S = Q == 2 ? 110 : (Q == 4 ? 98 : (Q == 5 ? 82 : 74));
    constexpr int GRID_BYTES =
        Q == 2 ? 2048 : (Q == 4 ? 1024 : (Q == 6 ? 1024 : 0));
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = blocks_per_row * S;

    const int g = lane >> 2;            // this lane's 32-element group
    const int slot = lane & 3;          // its 8-element slot in the group
    const int half = (lane & 3) >> 1;  // its 16-element scale half

    __shared__ uint8_t s_grid[GRID_BYTES ? GRID_BYTES : 1];
    if constexpr (GRID_BYTES) {
        const uint8_t* src = Q == 2 ? iq3_grid
            : (Q == 4 ? iq3_xxs_grid
                      : reinterpret_cast<const uint8_t*>(IQ2_XS_GRID_PACKED));
        for (int i = tid; i < GRID_BYTES; i += 32 * WAVES)
            s_grid[i] = src[i];
        __syncthreads();
    }

    float acc[ROWS][COLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t) acc[r][t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        float xv[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint16_t* xp = x + r * K + blk * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[r][j] = bf16_load(xp[j]);
        }
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* b = w + (n0 + t) * col_bytes + blk * S;
            if constexpr (Q == 2) {
                // IQ3_S: d + per-group scale nibble, 2 grid-slot index
                // bytes (+high bits) and 1 sign byte for this lane's window.
                const float d = half_load(b);
                const int scale =
                    1 + 2 * ((b[106 + g / 2] >> (4 * (g & 1))) & 15);
                const float sd = d * float(scale);
                const int hb = b[66 + g];
                const int i0 = b[2 + g * 8 + 2 * slot]
                             | (((hb >> (2 * slot)) & 1) << 8);
                const int i1 = b[2 + g * 8 + 2 * slot + 1]
                             | (((hb >> (2 * slot + 1)) & 1) << 8);
                const int signs = b[74 + g * 4 + slot];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int index = j < 4 ? i0 : i1;
                    const float mag =
                        float(s_grid[index * 4 + (j & 3)]);
                    const float sgn = (signs & (1 << j)) ? -1.0f : 1.0f;
                    const float wv = sd * mag * sgn;
#pragma unroll
                    for (int r = 0; r < ROWS; ++r)
                        acc[r][t] += xv[r][j] * wv;
                }
            } else if constexpr (Q == 4) {
                // IQ3_XXS: d + 4-byte per-group aux (scale in the top
                // nibble, 7-bit sign patterns below). The 98-byte stride
                // leaves aux only 2-aligned, so it reads as two u16s.
                const float d = half_load(b);
                const uint16_t a0 = *reinterpret_cast<const uint16_t*>(
                    b + 66 + 4 * g);
                const uint16_t a1 = *reinterpret_cast<const uint16_t*>(
                    b + 66 + 4 * g + 2);
                const uint32_t aux =
                    uint32_t(a0) | (uint32_t(a1) << 16);
                const float sd = (d * (0.5f + float(aux >> 28))) * 0.5f;
                // The grid index is per 4 elements (j/4), so this lane's
                // window spans two grid slots; signs stay per 8.
                const int i0 = b[2 + g * 8 + 2 * slot];
                const int i1 = b[2 + g * 8 + 2 * slot + 1];
                const int signs =
                    iq_sign_bits(int((aux >> (7 * slot)) & 127));
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float mag = float(
                        s_grid[(j < 4 ? i0 : i1) * 4 + (j & 3)]);
                    const float sgn = (signs & (1 << j)) ? -1.0f : 1.0f;
                    const float wv = sd * mag * sgn;
#pragma unroll
                    for (int r = 0; r < ROWS; ++r)
                        acc[r][t] += xv[r][j] * wv;
                }
            } else if constexpr (Q == 5) {
                // IQ2_S: d + per-group scale/index-high bytes and one
                // sign byte; the 8 KB grid stays __constant__ (staging it
                // was measured -10.6% on the strict owner).
                const float d = half_load(b);
                const int index = b[2 + g * 4 + slot]
                                | (((b[66 + g] >> (2 * slot)) & 3) << 8);
                const int scale = (b[74 + g] >> (4 * half)) & 15;
                const float sd = (d * (0.5f + float(scale))) * 0.25f;
                const int signs = b[34 + g * 4 + slot];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float mag = float(iq2_s_grid[index * 8 + j]);
                    const float sgn = (signs & (1 << j)) ? -1.0f : 1.0f;
                    const float wv = sd * mag * sgn;
#pragma unroll
                    for (int r = 0; r < ROWS; ++r)
                        acc[r][t] += xv[r][j] * wv;
                }
            } else {
                // IQ2_XS: d + one 16-bit word per slot (9-bit packed-grid
                // index + 7-bit sign pattern) and a per-16 scale nibble.
                const float d = half_load(b);
                const uint16_t word = *reinterpret_cast<const uint16_t*>(
                    b + 2 + 2 * (g * 4 + slot));
                const int scale = (b[66 + g] >> (4 * half)) & 15;
                const float sd = (d * (0.5f + float(scale))) * 0.25f;
                const uint16_t packed = *reinterpret_cast<const uint16_t*>(
                    s_grid + 2 * (word & 511));
                const int signs = iq_sign_bits(word >> 9);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int code = (packed >> (2 * j)) & 3;
                    const float mag = iq2_xs_magnitude(code);
                    const float sgn = (signs & (1 << j)) ? -1.0f : 1.0f;
                    const float wv = sd * mag * sgn;
#pragma unroll
                    for (int r = 0; r < ROWS; ++r)
                        acc[r][t] += xv[r][j] * wv;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t)
            for (int delta = 16; delta; delta >>= 1)
                acc[r][t] += __shfl_down(acc[r][t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][ROWS][COLS];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int t = 0; t < COLS; ++t) xwave[wave][r][t] = acc[r][t];
    }
    __syncthreads();
    if (tid < ROWS * COLS) {
        const int r = tid / COLS;
        const int t = tid % COLS;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][r][t];
            out[int64_t(r) * N + n0 + t] = bf16_store(total);
        }
    }
}

extern "C" int hipengine_gguf_iq_split_local32_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int quant, int waves, void* stream) {
    if (!x || !w || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    if (quant != 2 && quant != 4 && quant != 5 && quant != 6)
        return int(hipErrorInvalidValue);
    if (waves != 1 && waves != 2 && waves != 4)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    uint16_t* op = (uint16_t*)out;
#define UD_LAUNCH(QQ, WW)                                                     \
    gguf_iq_split_local32_gemv_kernel<QQ, WW>                                 \
        <<<grid, 32 * WW, 0, s>>>(xp, wp, op, K, N)
    if (quant == 2) {
        if (waves == 1) { UD_LAUNCH(2, 1); }
        else if (waves == 2) { UD_LAUNCH(2, 2); }
        else { UD_LAUNCH(2, 4); }
    } else if (quant == 4) {
        if (waves == 1) { UD_LAUNCH(4, 1); }
        else if (waves == 2) { UD_LAUNCH(4, 2); }
        else { UD_LAUNCH(4, 4); }
    } else if (quant == 5) {
        if (waves == 1) { UD_LAUNCH(5, 1); }
        else if (waves == 2) { UD_LAUNCH(5, 2); }
        else { UD_LAUNCH(5, 4); }
    } else {
        if (waves == 1) { UD_LAUNCH(6, 1); }
        else if (waves == 2) { UD_LAUNCH(6, 2); }
        else { UD_LAUNCH(6, 4); }
    }
#undef UD_LAUNCH
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_q3_k_embedding(const void* ids, const void* w,
    void* out, int64_t rows, int64_t K, int64_t vocab, void* stream) {
    if (!ids || !w || !out || rows<=0 || rows>65535 || K<=0 || K%256 || vocab<=0)
        return int(hipErrorInvalidValue);
    gguf_q3_k_embedding_kernel<<<dim3((K+255)/256,rows),256,0,(hipStream_t)stream>>>(
        (const int64_t*)ids,(const uint8_t*)w,(uint16_t*)out,K,vocab);
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_local32_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    if (!x || !w || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    int err;
    switch (waves) {
        case 1:
            err = (gguf_iq4_xs_local32_gemv_kernel<1><<<grid, 32, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        case 2:
            err = (gguf_iq4_xs_local32_gemv_kernel<2><<<grid, 64, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        case 4:
            err = (gguf_iq4_xs_local32_gemv_kernel<4><<<grid, 128, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        case 8:
            err = (gguf_iq4_xs_local32_gemv_kernel<8><<<grid, 256, 0, s>>>(
                (const uint16_t*)x, (const uint8_t*)w, (uint16_t*)out, K, N),
                int(hipGetLastError()));
            return err;
        default: return int(hipErrorInvalidValue);
    }
}

// Verifier sibling (2026-09-12): rows 2-4 of the same local32 owners. ROWS is
// a template parameter, so every row keeps the rows == 1 accumulation order and
// each row's output is bit-identical to the rows == 1 owner's output for that
// row. The host keeps the strict per-row GEMV registered as the fallback.
template<int ROWS>
int launch_iq4_xs_local32_rows(const void* x, const void* w, void* out,
    int64_t K, int64_t N, int waves, hipStream_t s) {
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1: gguf_iq4_xs_local32_gemv_kernel<1, ROWS>
            <<<grid, 32, 0, s>>>(xp, wp, op, K, N); break;
        case 2: gguf_iq4_xs_local32_gemv_kernel<2, ROWS>
            <<<grid, 64, 0, s>>>(xp, wp, op, K, N); break;
        case 4: gguf_iq4_xs_local32_gemv_kernel<4, ROWS>
            <<<grid, 128, 0, s>>>(xp, wp, op, K, N); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

template<int ROWS>
int launch_iq_split_local32_rows(const void* x, const void* w, void* out,
    int64_t K, int64_t N, int quant, int waves, hipStream_t s) {
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1:
            if (quant == 2) { gguf_iq_split_local32_gemv_kernel<2, 1, ROWS><<<grid, 32, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 4) { gguf_iq_split_local32_gemv_kernel<4, 1, ROWS><<<grid, 32, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 5) { gguf_iq_split_local32_gemv_kernel<5, 1, ROWS><<<grid, 32, 0, s>>>(xp, wp, op, K, N); }
            else { gguf_iq_split_local32_gemv_kernel<6, 1, ROWS><<<grid, 32, 0, s>>>(xp, wp, op, K, N); }
            break;
        case 2:
            if (quant == 2) { gguf_iq_split_local32_gemv_kernel<2, 2, ROWS><<<grid, 64, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 4) { gguf_iq_split_local32_gemv_kernel<4, 2, ROWS><<<grid, 64, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 5) { gguf_iq_split_local32_gemv_kernel<5, 2, ROWS><<<grid, 64, 0, s>>>(xp, wp, op, K, N); }
            else { gguf_iq_split_local32_gemv_kernel<6, 2, ROWS><<<grid, 64, 0, s>>>(xp, wp, op, K, N); }
            break;
        case 4:
            if (quant == 2) { gguf_iq_split_local32_gemv_kernel<2, 4, ROWS><<<grid, 128, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 4) { gguf_iq_split_local32_gemv_kernel<4, 4, ROWS><<<grid, 128, 0, s>>>(xp, wp, op, K, N); }
            else if (quant == 5) { gguf_iq_split_local32_gemv_kernel<5, 4, ROWS><<<grid, 128, 0, s>>>(xp, wp, op, K, N); }
            else { gguf_iq_split_local32_gemv_kernel<6, 4, ROWS><<<grid, 128, 0, s>>>(xp, wp, op, K, N); }
            break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_local32_rows_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    if (!x || !w || !out || rows < 2 || rows > 4 || K <= 0 || K % 256
        || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const hipStream_t s = (hipStream_t)stream;
    switch (rows) {
        case 2: return launch_iq4_xs_local32_rows<2>(x, w, out, K, N, waves, s);
        case 3: return launch_iq4_xs_local32_rows<3>(x, w, out, K, N, waves, s);
        default: return launch_iq4_xs_local32_rows<4>(x, w, out, K, N, waves, s);
    }
}

extern "C" int hipengine_gguf_iq_split_local32_rows_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int quant, int waves, void* stream) {
    if (!x || !w || !out || rows < 2 || rows > 4 || K <= 0 || K % 256
        || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    if (quant != 2 && quant != 4 && quant != 5 && quant != 6)
        return int(hipErrorInvalidValue);
    if (waves != 1 && waves != 2 && waves != 4)
        return int(hipErrorInvalidValue);
    const hipStream_t s = (hipStream_t)stream;
    switch (rows) {
        case 2: return launch_iq_split_local32_rows<2>(x, w, out, K, N, quant, waves, s);
        case 3: return launch_iq_split_local32_rows<3>(x, w, out, K, N, quant, waves, s);
        default: return launch_iq_split_local32_rows<4>(x, w, out, K, N, quant, waves, s);
    }
}

template<int ROWS>
int launch_iq4_nl_local32_rows(const void* x, const void* w, void* out,
    int64_t K, int64_t N, int waves, hipStream_t s) {
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1: gguf_iq4_nl_local32_gemv_kernel<1, ROWS>
            <<<grid, 32, 0, s>>>(xp, wp, op, K, N); break;
        case 2: gguf_iq4_nl_local32_gemv_kernel<2, ROWS>
            <<<grid, 64, 0, s>>>(xp, wp, op, K, N); break;
        case 4: gguf_iq4_nl_local32_gemv_kernel<4, ROWS>
            <<<grid, 128, 0, s>>>(xp, wp, op, K, N); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_nl_local32_rows_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    if (!x || !w || !out || rows < 2 || rows > 4 || K <= 0 || K % 256
        || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const hipStream_t s = (hipStream_t)stream;
    switch (rows) {
        case 2: return launch_iq4_nl_local32_rows<2>(x, w, out, K, N, waves, s);
        case 3: return launch_iq4_nl_local32_rows<3>(x, w, out, K, N, waves, s);
        default: return launch_iq4_nl_local32_rows<4>(x, w, out, K, N, waves, s);
    }
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

// ===========================================================================
// IQ4_XS T16-layout local32 decode GEMV (2026-09-13).
//
// Same owner geometry as gguf_iq4_xs_local32_gemv_kernel -- one wave per 8
// output columns, lane k = lane*8..lane*8+7, wave32 shuffle tree, serial
// wave-0..WAVES-1 cross-wave sum -- but it reads the IQ4_XS T16 tile instead of
// raw GGUF columns.
//
// Why the T16 tile: in the raw layout a lane's 8-byte payload window is read by
// two lanes at opposite nibbles (lane 4g+0 reads bytes 0..7 for the low nibble
// and lane 4g+2 reads the same bytes 0..7 for the high nibble), so a
// 256-element, 8-column block moves 2048 weight bytes. The T16 qs plane is
// [group][element][column pair], so one u32 load at a fixed (group, element)
// yields all four column pairs -- eight output columns -- for that element.
// The same block then moves 1024 weight bytes. Decode is weight-bandwidth
// bound, so this is the structural lever the layout was built for.
//
// Arithmetic is unchanged per column: the accumulation still runs j = 0..7 over
// this lane's eight elements in order, each weight is still
// s_d * s_cb[code] with s_d = half_load(d) * float(scale), and the reduction
// tree and cross-wave order are identical. So every output is bit-identical to
// the raw-layout owner's output, which the parity test asserts rather than
// assumes.
//
// T16 tile geometry (GGUF_IQ4_XS_T16_BLOCK_BYTES = 2176 per 16-column tile):
//   d        at 2*t                      (f16, one per tile-local column t)
//   scales_h at 32 + 2*t                 (u16, two bits per group)
//   scales_l at 64 + 4*t                 (u32, one nibble per group)
//   qs       at 128 + g*256 + e*8 + cp   (one byte per (group, element, pair))
// A qs byte's low nibble belongs to column 2*cp and its high nibble to
// column 2*cp + 1.
#define IQ4XS_T16_BLOCK_BYTES 2176
#define IQ4XS_T16_HI_OFFSET 32
#define IQ4XS_T16_LO_OFFSET 64
#define IQ4XS_T16_QS_OFFSET 128
#define IQ4XS_T16_QS_GROUP_STRIDE 256
#define IQ4XS_T16_QS_ELEMENT_STRIDE 8

template<int WAVES, int ROWS = 1>
__global__ __launch_bounds__(32 * WAVES,
    ROWS > 1 ? 1 : (16 / WAVES > 0 ? 16 / WAVES : 1))
void gguf_iq4_xs_t16_local32_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    uint16_t* __restrict__ out, int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    // One T16 tile holds 16 columns, so this block's columns live in the tile
    // at n0/16 and start at tile-local column (n0 & 15).
    const uint8_t* tile = w + (n0 >> 4) * (blocks_per_row * IQ4XS_T16_BLOCK_BYTES);
    const int ct0 = int(n0 & 15);

    // Lane-constant in-block geometry: group, and the lane's first element
    // within that group. This is the raw owner's lane ownership unchanged.
    const int g = lane >> 2;
    const int el = (lane & 3) * 8;
    // A u32 at the element's qs slot covers column pairs 0..3 (tile-local
    // columns 0..7) or 4..7 (tile-local columns 8..15).
    const int qs_pair_base = (ct0 >= 8) ? 4 : 0;

    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc[ROWS][COLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t) acc[r][t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        float xv[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint16_t* xp = x + r * K + blk * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) xv[r][j] = bf16_load(xp[j]);
        }
        const uint8_t* blkp = tile + blk * IQ4XS_T16_BLOCK_BYTES;
        // Per-column d*scale for this lane's group, read from this block's own
        // header. The header is inside the 2176-byte block, so it cannot be
        // hoisted out of the K loop. d and scales_h are 16 contiguous bytes for
        // the eight columns and scales_l is 32, so both are read as wide u32
        // vectors rather than three mixed-width loads per column.
        uint32_t dv[4], hv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            dv[i] = *reinterpret_cast<const uint32_t*>(
                blkp + 2 * ct0 + 4 * i);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            hv[i] = *reinterpret_cast<const uint32_t*>(
                blkp + IQ4XS_T16_HI_OFFSET + 2 * ct0 + 4 * i);
        float sd[COLS];
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            const uint32_t lo = *reinterpret_cast<const uint32_t*>(
                blkp + IQ4XS_T16_LO_OFFSET + 4 * (ct0 + t));
            const uint16_t d16 = uint16_t((dv[t >> 1] >> (16 * (t & 1))) & 0xffffu);
            const uint16_t hi = uint16_t((hv[t >> 1] >> (16 * (t & 1))) & 0xffffu);
            const int scale = (((lo >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
                               (((hi >> (2 * g)) & 3) << 4)) - 32;
            sd[t] = half_from_bits(d16) * float(scale);
        }
        const uint8_t* qsb = blkp
            + IQ4XS_T16_QS_OFFSET + g * IQ4XS_T16_QS_GROUP_STRIDE
            + el * IQ4XS_T16_QS_ELEMENT_STRIDE
            + qs_pair_base;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Eight columns' nibbles for element el+j, in one u32.
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                qsb + j * IQ4XS_T16_QS_ELEMENT_STRIDE);
#pragma unroll
            for (int cp = 0; cp < 4; ++cp) {
                const uint32_t byte = (packed >> (8 * cp)) & 0xffu;
                // The LUT takes the whole byte: s_cb[v] is iq4_values[v & 15]
                // and s_cb[256 + v] is iq4_values[v >> 4], so the nibble is
                // selected inside the LUT and must not be shifted out here.
                const float wl = sd[2 * cp] * s_cb[byte];
                const float wh = sd[2 * cp + 1] * s_cb[256 + byte];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[r][2 * cp] += xv[r][j] * wl;
                    acc[r][2 * cp + 1] += xv[r][j] * wh;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < COLS; ++t)
            for (int delta = 16; delta; delta >>= 1)
                acc[r][t] += __shfl_down(acc[r][t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][ROWS][COLS];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int t = 0; t < COLS; ++t) xwave[wave][r][t] = acc[r][t];
    }
    __syncthreads();
    if (tid < ROWS * COLS) {
        const int r = tid / COLS;
        const int t = tid % COLS;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][r][t];
            out[int64_t(r) * N + n0 + t] = bf16_store(total);
        }
    }
}

template<int ROWS>
int launch_iq4_xs_t16_local32_rows(const void* x, const void* w, void* out,
    int64_t K, int64_t N, int waves, hipStream_t s) {
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1: gguf_iq4_xs_t16_local32_gemv_kernel<1, ROWS>
            <<<grid, 32, 0, s>>>(xp, wp, op, K, N); break;
        case 2: gguf_iq4_xs_t16_local32_gemv_kernel<2, ROWS>
            <<<grid, 64, 0, s>>>(xp, wp, op, K, N); break;
        case 4: gguf_iq4_xs_t16_local32_gemv_kernel<4, ROWS>
            <<<grid, 128, 0, s>>>(xp, wp, op, K, N); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_t16_local32_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    // N % 16 because a T16 tile is 16 columns wide and this block reads a whole
    // tile's qs plane; N % 8 alone would let the last half tile read past it.
    if (!x || !w || !out || rows != 1 || K <= 0 || K % 256 || N <= 0 || N % 16)
        return int(hipErrorInvalidValue);
    return launch_iq4_xs_t16_local32_rows<1>(
        x, w, out, K, N, waves, (hipStream_t)stream);
}

extern "C" int hipengine_gguf_iq4_xs_t16_local32_rows_gemv(const void* x,
    const void* w, void* out, int64_t rows, int64_t K, int64_t N,
    int waves, void* stream) {
    if (!x || !w || !out || rows < 2 || rows > 4 || K <= 0 || K % 256
        || N <= 0 || N % 16)
        return int(hipErrorInvalidValue);
    const hipStream_t s = (hipStream_t)stream;
    switch (rows) {
        case 2: return launch_iq4_xs_t16_local32_rows<2>(x, w, out, K, N, waves, s);
        case 3: return launch_iq4_xs_t16_local32_rows<3>(x, w, out, K, N, waves, s);
        case 4: return launch_iq4_xs_t16_local32_rows<4>(x, w, out, K, N, waves, s);
        default: return int(hipErrorInvalidValue);
    }
}

// ===========================================================================
// E6c-2: raw dense-IQ rows-1 FFN-down residual composites. Each kernel below
// is a verbatim copy of its resident parent owner -- same geometry, same
// accumulation order, same BF16 store rounding -- with the ONLY change being
// the output epilogue, which replays the exact standalone gguf_bf16_add_kernel
// the runner pays per layer: round the projection to BF16 first, then FP32-add
// the already-BF16 residual and round again (float_to_bf16_bits and bf16_store
// are the same RNE conversion gguf_bf16_add_kernel performs). The composite
// therefore replaces [owner, gguf_bf16_add] bit-for-bit at rows=1.
//
// Host validation keeps rows==1: the parents' verifier siblings (rows 2-4)
// have no residual variant, and GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT caps
// these quants at rows 1 so the residual path never asks for one.

// IQ4_XS local32 decode owner + rounded residual. Parent:
// gguf_iq4_xs_local32_gemv_kernel (verbatim through the cross-wave sum).
template<int WAVES>
__global__ __launch_bounds__(32 * WAVES, 16 / WAVES > 0 ? 16 / WAVES : 1)
void gguf_iq4_xs_local32_residual_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    const uint16_t* __restrict__ residual, uint16_t* __restrict__ out,
    int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = blocks_per_row * 136;

    const int g = lane >> 2;
    const int nib = (lane >> 1) & 1;
    const int byteoff = 8 + g * 16 + (lane & 1) * 8;

    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc[COLS];
#pragma unroll
    for (int t = 0; t < COLS; ++t) acc[t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        float xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            xv[j] = bf16_load(x[blk * 256 + lane * 8 + j]);
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* b = w + (n0 + t) * col_bytes + blk * 136;
            const uint32_t hdr = *reinterpret_cast<const uint32_t*>(b);
            const uint32_t scl = *reinterpret_cast<const uint32_t*>(b + 4);
            const int scale = (((scl >> (8 * (g >> 1) + 4 * (g & 1))) & 15) |
                               (((hdr >> (16 + 2 * g)) & 3) << 4)) - 32;
            const float s_d = half_load(b) * float(scale);
            const uint32_t p0 = *reinterpret_cast<const uint32_t*>(b + byteoff);
            const uint32_t p1 = *reinterpret_cast<const uint32_t*>(b + byteoff + 4);
            const uint32_t p[2] = {p0, p1};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int code_byte = int((p[j >> 2] >> (8 * (j & 3))) & 0xffu);
                const float wv = s_d * s_cb[code_byte + (nib << 8)];
                acc[t] += xv[j] * wv;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < COLS; ++t)
        for (int delta = 16; delta; delta >>= 1)
            acc[t] += __shfl_down(acc[t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][COLS];
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < COLS; ++t) xwave[wave][t] = acc[t];
    }
    __syncthreads();
    if (tid < COLS) {
        const int t = tid;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][t];
            // Exact replay of [owner bf16 store] then gguf_bf16_add_kernel.
            const int64_t idx = n0 + t;
            const uint16_t down = bf16_store(total);
            out[idx] = bf16_store(bf16_load(down) + bf16_load(residual[idx]));
        }
    }
}

// IQ4_NL local32 decode owner + rounded residual. Parent:
// gguf_iq4_nl_local32_gemv_kernel (verbatim through the cross-wave sum).
template<int WAVES>
__global__ __launch_bounds__(32 * WAVES, 16 / WAVES > 0 ? 16 / WAVES : 1)
void gguf_iq4_nl_local32_residual_gemv_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    const uint16_t* __restrict__ residual, uint16_t* __restrict__ out,
    int64_t K, int64_t N) {
    constexpr int COLS = GGUF_IQ_LOCAL32_COLS;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int64_t n0 = int64_t(blockIdx.x) * COLS;
    const int64_t blocks_per_row = K / 256;
    const int64_t col_bytes = (K / 32) * 18;

    const int g = lane >> 2;
    const int nib = (lane >> 1) & 1;
    const int byteoff = 2 + (lane & 1) * 8;

    __shared__ float s_cb[512];
    for (int i = tid; i < 512; i += 32 * WAVES)
        s_cb[i] = float(iq4_values[i >= 256 ? ((i & 255) >> 4) : (i & 15)]);
    __syncthreads();

    float acc[COLS];
#pragma unroll
    for (int t = 0; t < COLS; ++t) acc[t] = 0.0f;

    for (int64_t blk = wave; blk < blocks_per_row; blk += WAVES) {
        float xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            xv[j] = bf16_load(x[blk * 256 + lane * 8 + j]);
#pragma unroll
        for (int t = 0; t < COLS; ++t) {
            if (n0 + t >= N) break;
            const uint8_t* b =
                w + (n0 + t) * col_bytes + (blk * 8 + g) * 18;
            const float s_d = half_load(b);
            const uint16_t q[4] = {
                *reinterpret_cast<const uint16_t*>(b + byteoff),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 2),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 4),
                *reinterpret_cast<const uint16_t*>(b + byteoff + 6)};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int code_byte = int((q[j >> 1] >> (8 * (j & 1))) & 0xffu);
                const float wv = s_d * s_cb[code_byte + (nib << 8)];
                acc[t] += xv[j] * wv;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < COLS; ++t)
        for (int delta = 16; delta; delta >>= 1)
            acc[t] += __shfl_down(acc[t], delta);
    __shared__ float xwave[WAVES > 1 ? WAVES : 1][COLS];
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < COLS; ++t) xwave[wave][t] = acc[t];
    }
    __syncthreads();
    if (tid < COLS) {
        const int t = tid;
        if (n0 + t < N) {
            float total = 0.0f;
#pragma unroll
            for (int wv2 = 0; wv2 < WAVES; ++wv2) total += xwave[wv2][t];
            const int64_t idx = n0 + t;
            const uint16_t down = bf16_store(total);
            out[idx] = bf16_store(bf16_load(down) + bf16_load(residual[idx]));
        }
    }
}

// Strict per-row owner + rounded residual (T=8, R=1, BF16 out: the rows-1
// decode shape the fold admits). Parent: gguf_iq_dense_strict_kernel,
// verbatim through the serial wave-0..3 sum; only the store folds.
template<int Q>
__launch_bounds__(128, 16) __global__ void gguf_iq_dense_strict_residual_kernel(
    const uint16_t* x, const uint8_t* w, const uint16_t* residual,
    uint16_t* out, int64_t K, int64_t N, int64_t rows) {
    constexpr int T = 8;
    constexpr int R = 1;
    constexpr int B = Q == 1 ? 32 : 256;
    constexpr int S = Q == 0 ? 136 : (Q == 1 ? 18 : (Q == 4 ? 98 : (Q == 5 ? 82 : (Q == 6 ? 74 : 110))));
    constexpr int GPB = Q == 3 ? 16 : 8;
    constexpr int HDR = Q == 3 ? T*16 : T*8;
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
    constexpr int RB = 4;
    __shared__ float s_d[RB][T*16];
    __shared__ uint32_t s_aux[RB][T*8];
    __shared__ uint8_t s_qs[RB][T*8];
    constexpr int GRID_BYTES = Q == 2 ? 2048 : (Q == 4 ? 1024 : (Q == 6 ? 1024 : 0));
    __shared__ uint8_t s_grid[GRID_BYTES ? GRID_BYTES : 1];
    if constexpr (GRID_BYTES) {
        const uint8_t* src;
        if constexpr (Q == 2) src = iq3_grid;
        else if constexpr (Q == 4) src = iq3_xxs_grid;
        else src = reinterpret_cast<const uint8_t*>(IQ2_XS_GRID_PACKED);
        for (int i = tid; i < GRID_BYTES; i += 128) s_grid[i] = src[i];
        __syncthreads();
    }
    constexpr int CB_FLOATS = Q == 0 ? 512 : 0;
    __shared__ float s_cb[CB_FLOATS ? CB_FLOATS : 1];
    if constexpr (CB_FLOATS) {
        for (int i = tid; i < CB_FLOATS; i += 128) {
            const int byte = i & 255;
            const int nib = i >= 256 ? (byte >> 4) : (byte & 15);
            s_cb[i] = float(iq4_values[nib]);
        }
        __syncthreads();
    }
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
            for (int bi=0; bi<nb; ++bi) {
            #pragma unroll
            for (int h=0;h<2;++h) {
                const int k = int(rbase+bi)*256 + h*128 + tid;
                float xv[R];
                #pragma unroll
                for (int r=0;r<R;++r)
                    xv[r] = row0+r < rows ? bf16_load(x[(row0+r)*K+k]) : 0.0f;
                #pragma unroll
                for (int t=0;t<T;++t) {
                    if (n0+t < N) {
                        const uint8_t* b = wcol[t] + (rbase+bi)*S;
                        const int j = k & 31;
                        const int g = (k & 255) >> 5;
                        float wv;
                        if constexpr (Q == 0) {
                            const int bytev = b[8+g*16+(j&15)];
                            wv = s_d[bi][t*16+g] * s_cb[bytev + ((j>>4) ? 256 : 0)];
                        } else if constexpr (Q == 4) {
                            const uint32_t aux = s_aux[bi][t*8+g];
                            int index = b[2+g*8+j/4];
                            int signs = iq_sign_bits((aux >> (7*(j/8))) & 127);
                            float mag = s_d[bi][t*16+g] * float(s_grid[index*4+(j&3)]);
                            wv = mag * ((signs & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else if constexpr (Q == 6) {
                            const int slot = j >> 3;
                            const uint8_t* p = b + 2 + 2*(g*4+slot);
                            int word = int(p[0]) | (int(p[1]) << 8);
                            int code = (reinterpret_cast<const uint16_t*>(s_grid)[word & 511] >> (2*(j&7))) & 3;
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
                            float mag = s_d[bi][t*16+g] * float(s_grid[index*4+(j&3)]);
                            wv = mag * ((b[74+g*4+j/8] & (1<<(j&7))) ? -1.0f : 1.0f);
                        } else { // Q == 3
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
    for (int slot = threadIdx.x; slot < R*T; slot += 128) {
        const int r = slot / T, t = slot % T;
        if (row0+r < rows && n0+t < N) {
            float total=0;
            for (int i=0;i<4;++i) total += sums[i][r][t];
            const int64_t index = (row0+r)*N + n0 + t;
            // Exact replay of the parent's bf16 store, then gguf_bf16_add_kernel.
            const uint16_t down = bf16_store(total);
            out[index] = bf16_store(bf16_load(down) + bf16_load(residual[index]));
        }
    }
}

extern "C" int hipengine_gguf_iq4_xs_local32_gemv_residual(const void* x,
    const void* w, const void* residual, void* out, int64_t rows, int64_t K,
    int64_t N, int waves, void* stream) {
    if (!x || !w || !residual || !out || rows != 1 || K <= 0 || K % 256
        || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    const uint16_t* rp = (const uint16_t*)residual;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1: gguf_iq4_xs_local32_residual_gemv_kernel<1>
            <<<grid, 32, 0, s>>>(xp, wp, rp, op, K, N); break;
        case 2: gguf_iq4_xs_local32_residual_gemv_kernel<2>
            <<<grid, 64, 0, s>>>(xp, wp, rp, op, K, N); break;
        case 4: gguf_iq4_xs_local32_residual_gemv_kernel<4>
            <<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N); break;
        case 8: gguf_iq4_xs_local32_residual_gemv_kernel<8>
            <<<grid, 256, 0, s>>>(xp, wp, rp, op, K, N); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_nl_local32_gemv_residual(const void* x,
    const void* w, const void* residual, void* out, int64_t rows, int64_t K,
    int64_t N, int waves, void* stream) {
    if (!x || !w || !residual || !out || rows != 1 || K <= 0 || K % 256
        || N <= 0 || N % 8)
        return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + GGUF_IQ_LOCAL32_COLS - 1) / GGUF_IQ_LOCAL32_COLS));
    const hipStream_t s = (hipStream_t)stream;
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    const uint16_t* rp = (const uint16_t*)residual;
    uint16_t* op = (uint16_t*)out;
    switch (waves) {
        case 1: gguf_iq4_nl_local32_residual_gemv_kernel<1>
            <<<grid, 32, 0, s>>>(xp, wp, rp, op, K, N); break;
        case 2: gguf_iq4_nl_local32_residual_gemv_kernel<2>
            <<<grid, 64, 0, s>>>(xp, wp, rp, op, K, N); break;
        case 4: gguf_iq4_nl_local32_residual_gemv_kernel<4>
            <<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}

extern "C" int hipengine_gguf_iq_dense_residual(const void* x, const void* w,
    const void* residual, void* out, int64_t rows, int64_t K, int64_t N,
    int quant, void* stream) {
    if (rows != 1) return int(hipErrorInvalidValue);
    const dim3 grid(unsigned((N + 8 - 1) / 8), 1u);
    const hipStream_t s = (hipStream_t)stream;
    const uint16_t* xp = (const uint16_t*)x;
    const uint8_t* wp = (const uint8_t*)w;
    const uint16_t* rp = (const uint16_t*)residual;
    uint16_t* op = (uint16_t*)out;
    switch (quant) {
        case 0: gguf_iq_dense_strict_residual_kernel<0><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 1: gguf_iq_dense_strict_residual_kernel<1><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 2: gguf_iq_dense_strict_residual_kernel<2><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 3: gguf_iq_dense_strict_residual_kernel<3><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 4: gguf_iq_dense_strict_residual_kernel<4><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 5: gguf_iq_dense_strict_residual_kernel<5><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        case 6: gguf_iq_dense_strict_residual_kernel<6><<<grid, 128, 0, s>>>(xp, wp, rp, op, K, N, rows); break;
        default: return int(hipErrorInvalidValue);
    }
    return int(hipGetLastError());
}
