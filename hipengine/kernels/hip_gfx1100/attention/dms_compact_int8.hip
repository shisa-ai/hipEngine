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
  int kept = 0;
  for (int t = 0; t < n; ++t)
    kept += !flags[base+t] || p-positions[base+t] <= window;
  if (kept+1 > capacities[rh]) {
    if (tid == 0) status[rh] = 1;
    return;
  }
  __shared__ int keep;
  __shared__ float reduction[2*Threads];
  int rank = 0;
  // Row-by-row ordering prevents backward compaction from overwriting a
  // source another lane has not yet read. Payload bytes and scales move
  // without requantization. Later tuning must preserve this ownership rule.
  for (int t = 0; t < n; ++t) {
    const int src = base+t;
    if (tid == 0) keep = !flags[src] || p-positions[src] <= window;
    __syncthreads();
    if (keep) {
      const int dst = base+rank++;
      if (dst != src) {
        for (int d = tid; d < dim; d += Threads) {
          ko[int64_t(dst)*dim+d] = ko[int64_t(src)*dim+d];
          vo[int64_t(dst)*dim+d] = vo[int64_t(src)*dim+d];
        }
        if (tid == 0) {
          ks[dst] = ks[src]; vs[dst] = vs[src];
          positions[dst] = positions[src]; flags[dst] = flags[src];
        }
      }
    }
    __syncthreads();
  }
  const int dst = base+kept;
  quantize_pair(k+int64_t(rh)*dim, v+int64_t(rh)*dim,
                ko+int64_t(dst)*dim, vo+int64_t(dst)*dim, ks+dst, vs+dst, dim, reduction);
  if (tid == 0) {
    positions[dst] = p; flags[dst] = evict[rh]; lives[rh] = kept+1; status[rh] = 0;
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
  hipLaunchKernelGGL(dms_int8_attn_split_kernel, dim3(rows*qheads*splits), dim3(Threads), 0, stream,
                    q,k,v,base,live,po,pm,pl,ks,vs,qheads,heads,dim,scale,chunk,splits);
  const auto error = hipGetLastError();
  if (error != hipSuccess) return error;
  return hipengine_dms_compact_attn_decode_splitk_reduce_bf16(po,pm,pl,out,rows,qheads,dim,splits,stream);
}
