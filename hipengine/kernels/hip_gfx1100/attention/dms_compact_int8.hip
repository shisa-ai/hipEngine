// Compact DMS INT8 payloads with FP32 per-token/head K/V scales.
// Control follows the in-tree BF16 compact kernels. Inputs are transient BF16;
// only INT8 vectors and their scales survive pack/append. No BF16 shadow.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>

namespace {
constexpr int Threads = 256;
__device__ float from_bf16(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }

// All lanes cooperate on one vector pair. The maximum is exact for finite
// BF16 inputs; division and round-to-nearest-even match the CPU codec.
__device__ void quantize_pair(const uint16_t* k, const uint16_t* v,
    int8_t* ko, int8_t* vo, float* ks, float* vs, int dim,
    float* reduction) {
  const int tid = threadIdx.x;
  float km = 0, vm = 0;
  for (int d = tid; d < dim; d += Threads) {
    km = fmaxf(km, fabsf(from_bf16(k[d])));
    vm = fmaxf(vm, fabsf(from_bf16(v[d])));
  }
  reduction[tid] = km; reduction[Threads + tid] = vm;
  __syncthreads();
  for (int s = Threads / 2; s; s >>= 1) {
    if (tid < s) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid+s]);
      reduction[Threads+tid] = fmaxf(reduction[Threads+tid], reduction[Threads+tid+s]);
    }
    __syncthreads();
  }
  const float sk = fmaxf(reduction[0] / 127.0f, FLT_MIN);
  const float sv = fmaxf(reduction[Threads] / 127.0f, FLT_MIN);
  for (int d = tid; d < dim; d += Threads) {
    ko[d] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(from_bf16(k[d]) / sk))));
    vo[d] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(from_bf16(v[d]) / sv))));
  }
  if (tid == 0) { *ks = sk; *vs = sv; }
  __syncthreads();
}

__global__ void dms_int8_pack_kernel(const uint16_t* k, const uint16_t* v,
    const uint8_t* evict, const int32_t* bases, const int32_t* capacities,
    int32_t* lives, const int32_t* starts, const int32_t* lengths,
    int8_t* ko, int8_t* vo, int32_t* positions, uint8_t* flags,
    float* ks, float* vs, int heads, int dim, int window) {
  const int rh = blockIdx.x, row = rh / heads, head = rh % heads;
  const int n = lengths[row], base = bases[rh];
  int input_base = 0;
  for (int r = 0; r < row; ++r) input_base += lengths[r];
  int kept = 0;
  for (int t = 0; t < n; ++t)
    kept += !evict[(int64_t(input_base)+t)*heads+head] || n-1-t <= window;
  if (kept > capacities[rh]) return; // no writes on overflow
  __shared__ float reduction[2*Threads];
  int rank = 0;
  for (int t = 0; t < n; ++t) {
    const int64_t source = (int64_t(input_base)+t)*heads+head;
    const uint8_t e = evict[source];
    if (e && n-1-t > window) continue;
    const int dst = base + rank++;
    quantize_pair(k+source*dim, v+source*dim, ko+int64_t(dst)*dim,
                  vo+int64_t(dst)*dim, ks+dst, vs+dst, dim, reduction);
    if (threadIdx.x == 0) { positions[dst] = starts[row]+t; flags[dst] = e; }
  }
  if (threadIdx.x == 0) lives[rh] = kept;
}

__global__ void dms_int8_append_kernel(const uint16_t* k, const uint16_t* v,
    const uint8_t* evict, const int32_t* row_positions, const int32_t* bases,
    const int32_t* capacities, int32_t* lives, int8_t* ko, int8_t* vo,
    int32_t* positions, uint8_t* flags, int32_t* status, float* ks, float* vs,
    int heads, int dim, int window) {
  const int rh = blockIdx.x, row = rh/heads, base = bases[rh];
  const int p = row_positions[row], n = lives[rh], tid = threadIdx.x;
  // Chunked keep-scan (BF16 family's proven structure): two passes over the
  // live prefix with log-depth prefix scans per 256-token chunk instead of a
  // serial per-token loop with two barriers per token. Compaction moves
  // entries strictly leftward within and across chunks, so chunk i's writes
  // always land before chunk i+1's sources - the same ownership rule the
  // row-by-row version preserved, now in parallel.
  __shared__ int s_flag[Threads];
  __shared__ int s_chunk_total;
  int keepers = 0;
  for (int chunk0 = 0; chunk0 < n; chunk0 += Threads) {
    const int in_chunk = tid < (n - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int src = base + chunk0 + tid;
      if (!flags[src] || p - positions[src] <= window) s_flag[tid] = 1;
    }
    __syncthreads();
    for (int off = 1; off < Threads; off <<= 1) {
      const int delta = (tid >= off) ? s_flag[tid - off] : 0;
      __syncthreads();
      s_flag[tid] += delta;
      __syncthreads();
    }
    s_chunk_total = s_flag[Threads - 1];
    __syncthreads();
    keepers += s_chunk_total;
  }
  const int final_live = keepers + 1;
  if (final_live > capacities[rh]) {
    if (tid == 0) status[rh] = 1;
    return;
  }
  __shared__ float reduction[2*Threads];
  int chunk_rank = 0;
  for (int chunk0 = 0; chunk0 < n; chunk0 += Threads) {
    const int in_chunk = tid < (n - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int src = base + chunk0 + tid;
      if (!flags[src] || p - positions[src] <= window) s_flag[tid] = 1;
    }
    __syncthreads();
    for (int off = 1; off < Threads; off <<= 1) {
      const int delta = (tid >= off) ? s_flag[tid - off] : 0;
      __syncthreads();
      s_flag[tid] += delta;
      __syncthreads();
    }
    if (in_chunk && s_flag[tid] > (tid > 0 ? s_flag[tid - 1] : 0)) {
      const int dst = base + chunk_rank + s_flag[tid] - 1;
      const int src = base + chunk0 + tid;
      if (dst != src) {
        // The owning thread moves the whole row (payload bytes and scales
        // without requantization), matching the BF16 family's ownership rule.
        for (int d = 0; d < dim; ++d) {
          ko[int64_t(dst)*dim+d] = ko[int64_t(src)*dim+d];
          vo[int64_t(dst)*dim+d] = vo[int64_t(src)*dim+d];
        }
        ks[dst] = ks[src]; vs[dst] = vs[src];
        positions[dst] = positions[src]; flags[dst] = flags[src];
      }
    }
    s_chunk_total = s_flag[Threads - 1];
    __syncthreads();
    chunk_rank += s_chunk_total;
    __syncthreads();
  }
  const int dst = base+keepers;
  quantize_pair(k+int64_t(rh)*dim, v+int64_t(rh)*dim,
                ko+int64_t(dst)*dim, vo+int64_t(dst)*dim, ks+dst, vs+dst, dim, reduction);
  if (tid == 0) {
    positions[dst] = p; flags[dst] = evict[rh]; lives[rh] = final_live; status[rh] = 0;
  }
}

// Wave-grouped GQA producer for the 24q/4kv/256d geometry: one block per
// (kv-head, split), the 6 q-heads of the group share one K/V pass through
// shared Q staging, 4-wide int8 loads with the per-token scale dequantized in
// registers, and warp-shuffle reductions (the BF16 wave6 treatment).
__global__ void dms_int8_attn_split_wave_kernel(
    const float* __restrict__ q, const int8_t* __restrict__ k,
    const int8_t* __restrict__ v, const int32_t* __restrict__ bases,
    const int32_t* __restrict__ lives, float* __restrict__ po,
    float* __restrict__ pm, float* __restrict__ pl,
    const float* __restrict__ ks, const float* __restrict__ vs,
    const int32_t rows, const int32_t qheads, const int32_t heads,
    const int32_t dim, const float scale, const int32_t chunk,
    const int32_t splits) {
  constexpr int32_t QPerKv = 6;
  constexpr int32_t kHeadDim = 256;
  constexpr int32_t kWaveSize = 32;
  constexpr int32_t kNumWaves = Threads / kWaveSize;
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t split = gid % splits;
  const int32_t rowh = gid / splits;
  const int32_t r = rowh / heads;
  const int32_t h = rowh - r * heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid & (kWaveSize - 1);
  const int32_t wave = tid / kWaveSize;
  const int32_t rowq_base = r * qheads + h * QPerKv;
  const int32_t live = lives[rowh];
  const int32_t begin = split * chunk;
  const int32_t count = max(0, min(chunk, live - begin));
  const int32_t base = bases[rowh] + begin;
  __shared__ float scores[QPerKv][Threads];
  __shared__ float wave_partials[QPerKv][kNumWaves];
  __shared__ float block_max[QPerKv];
  __shared__ float q_shared[QPerKv][kHeadDim];

  if (count <= 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat = static_cast<int64_t>(rowq_base + j) * splits + split;
      float* part = po + stat * kHeadDim;
      for (int32_t d = tid; d < kHeadDim; d += Threads) part[d] = 0.0f;
      if (tid == 0) {
        pm[stat] = -3.4028234663852886e38f;
        pl[stat] = 0.0f;
      }
    }
    return;
  }

#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    q_shared[j][tid] = q[static_cast<int64_t>(rowq_base + j) * kHeadDim + tid];
  }
  __syncthreads();

  float local_max[QPerKv];
#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    local_max[j] = -3.4028234663852886e38f;
  }
  for (int32_t t = wave; t < count; t += kNumWaves) {
    const int64_t token = static_cast<int64_t>(base) + t;
    const int8_t* krow = k + token * kHeadDim;
    const float ks_row = ks[token];
    float dots[QPerKv] = {};
    for (int32_t d = lane * 4; d < kHeadDim; d += kWaveSize * 4) {
      const float k0 = static_cast<float>(krow[d + 0]) * ks_row;
      const float k1 = static_cast<float>(krow[d + 1]) * ks_row;
      const float k2 = static_cast<float>(krow[d + 2]) * ks_row;
      const float k3 = static_cast<float>(krow[d + 3]) * ks_row;
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        dots[j] += q_shared[j][d + 0] * k0;
        dots[j] += q_shared[j][d + 1] * k1;
        dots[j] += q_shared[j][d + 2] * k2;
        dots[j] += q_shared[j][d + 3] * k3;
      }
    }
    for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        dots[j] += __shfl_down(dots[j], offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        const float score = dots[j] * scale;
        scores[j][t] = score;
        local_max[j] = fmaxf(local_max[j], score);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      wave_partials[j][wave] = local_max[j];
    }
  }
  __syncthreads();

  if (tid < kWaveSize) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      float value = tid < kNumWaves ? wave_partials[j][tid]
                                    : -3.4028234663852886e38f;
      for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down(value, offset));
      }
      if (tid == 0) {
        wave_partials[j][0] = value;
        block_max[j] = value;
      }
    }
  }
  __syncthreads();

  float local_sum[QPerKv] = {};
#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    const float maximum = block_max[j];
    for (int32_t t = tid; t < count; t += Threads) {
      const float value = expf(scores[j][t] - maximum);
      scores[j][t] = value;
      local_sum[j] += value;
    }
  }
  for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      local_sum[j] += __shfl_down(local_sum[j], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      wave_partials[j][wave] = local_sum[j];
    }
  }
  __syncthreads();

  if (tid < kWaveSize) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      float value = tid < kNumWaves ? wave_partials[j][tid] : 0.0f;
      for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
      }
      if (tid == 0) {
        wave_partials[j][0] = value;
      }
    }
  }
  __syncthreads();

  if (tid < kHeadDim) {
    float values[QPerKv] = {};
    const int8_t* vrow = v + static_cast<int64_t>(base) * kHeadDim;
    for (int32_t t = 0; t < count; ++t) {
      const float value =
          static_cast<float>(vrow[static_cast<int64_t>(t) * kHeadDim + tid]) *
          vs[static_cast<int64_t>(base) + t];
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        values[j] += scores[j][t] * value;
      }
    }
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat =
          static_cast<int64_t>(rowq_base + j) * splits + split;
      po[stat * kHeadDim + tid] = values[j];
    }
  }
  if (tid == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat =
          static_cast<int64_t>(rowq_base + j) * splits + split;
      pm[stat] = block_max[j];
      pl[stat] = wave_partials[j][0];
    }
  }
}

// Bounded 256-token producers use the BF16 family's partial-output ABI and
// deterministic reducer. Dequantization happens in registers at each read.
__global__ void dms_int8_attn_split_kernel(const float* q, const int8_t* k,
    const int8_t* v, const int32_t* bases, const int32_t* lives,
    float* po, float* pm, float* pl, const float* ks, const float* vs,
    int qheads, int heads, int dim, float scale, int chunk, int splits) {
  const int split = blockIdx.x % splits, rq = blockIdx.x / splits;
  const int rh = (rq/qheads)*heads + (rq%qheads)/(qheads/heads);
  const int begin = split*chunk, n = max(0, min(chunk, lives[rh]-begin));
  const int base = bases[rh]+begin, tid = threadIdx.x;
  const int64_t stat = int64_t(rq)*splits+split;
  __shared__ float scores[Threads], reduction[Threads];
  float score = -FLT_MAX;
  if (tid < n) {
    float dot = 0;
    for (int d = 0; d < dim; ++d)
      dot += q[int64_t(rq)*dim+d] * (float(k[int64_t(base+tid)*dim+d])*ks[base+tid]);
    score = dot*scale;
  }
  reduction[tid] = score;
  __syncthreads();
  for (int s = Threads/2; s; s >>= 1) {
    if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid+s]);
    __syncthreads();
  }
  const float maximum = reduction[0];
  // Every warp must finish reading the maximum before lane 0 reuses its
  // shared slot for the probability reduction (a write-after-read hazard).
  __syncthreads();
  const float probability = tid < n ? expf(score-maximum) : 0;
  scores[tid] = probability; reduction[tid] = probability;
  __syncthreads();
  for (int s = Threads/2; s; s >>= 1) {
    if (tid < s) reduction[tid] += reduction[tid+s];
    __syncthreads();
  }
  if (tid == 0) { pm[stat] = maximum; pl[stat] = reduction[0]; }
  for (int d = tid; d < dim; d += Threads) {
    float acc = 0;
    for (int t = 0; t < n; ++t)
      acc += scores[t] * (float(v[int64_t(base+t)*dim+d])*vs[base+t]);
    po[stat*dim+d] = acc;
  }
}
} // namespace

extern "C" int hipengine_dms_streaming_pack_int8(
    const uint16_t* k, const uint16_t* v, const uint8_t* evict,
    const int32_t* base, const int32_t* cap, int32_t* live,
    const int32_t* starts, const int32_t* tokens, int8_t* ko, int8_t* vo,
    int32_t* pos, uint8_t* flags, float* ks, float* vs,
    int rows, int heads, int dim, int window, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim <= 0 || window < 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dms_int8_pack_kernel, dim3(rows*heads), dim3(Threads), 0, stream,
                    k,v,evict,base,cap,live,starts,tokens,ko,vo,pos,flags,ks,vs,heads,dim,window);
  return hipGetLastError();
}
extern "C" int hipengine_dms_append_decode_int8(
    const uint16_t* k, const uint16_t* v, const uint8_t* evict, const int32_t* rowpos,
    const int32_t* base, const int32_t* cap, int32_t* live, int8_t* ko, int8_t* vo,
    int32_t* pos, uint8_t* flags, int32_t* status, float* ks, float* vs,
    int rows, int heads, int dim, int window, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim <= 0 || window < 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dms_int8_append_kernel, dim3(rows*heads), dim3(Threads), 0, stream,
                    k,v,evict,rowpos,base,cap,live,ko,vo,pos,flags,status,ks,vs,heads,dim,window);
  return hipGetLastError();
}
extern "C" int hipengine_dms_compact_attn_decode_splitk_reduce_bf16(
    const float*, const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t, hipStream_t);
extern "C" int hipengine_dms_compact_attn_decode_splitk_int8(
    const float* q, const int8_t* k, const int8_t* v, const int32_t* base, const int32_t* live,
    float* po, float* pm, float* pl, float* out, const float* ks, const float* vs,
    int rows, int qheads, int heads, int dim, float scale, int chunk, int splits, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || qheads <= 0 || qheads%heads || dim <= 0 ||
      chunk <= 0 || chunk > Threads || splits <= 0) return hipErrorInvalidValue;
  if (qheads == 24 && heads == 4 && dim == 256 && rows == 1) {
    hipLaunchKernelGGL(dms_int8_attn_split_wave_kernel, dim3(rows*heads*splits), dim3(Threads), 0, stream,
                      q,k,v,base,live,po,pm,pl,ks,vs,rows,qheads,heads,dim,scale,chunk,splits);
  } else {
    hipLaunchKernelGGL(dms_int8_attn_split_kernel, dim3(rows*qheads*splits), dim3(Threads), 0, stream,
                      q,k,v,base,live,po,pm,pl,ks,vs,qheads,heads,dim,scale,chunk,splits);
  }
  const auto error = hipGetLastError();
  if (error != hipSuccess) return error;
  return hipengine_dms_compact_attn_decode_splitk_reduce_bf16(po,pm,pl,out,rows,qheads,dim,splits,stream);
}
