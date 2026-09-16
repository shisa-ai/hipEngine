// Identical-operand replay adapter for the pinned comparator's MMB dense matmul.
//
// The comparator is a pinned llama.cpp tree built as libggml-hip.so. Its dense
// MMB entry point, ggml_cuda_mul_mat_mmb, is an exported C++ symbol that takes
// ggml tensors. This shim drives that entry point directly on operands supplied
// by the caller, so a hipEngine capture and the comparator operation run on
// byte-identical weights and activations.
//
// The shim includes the comparator's own common.cuh rather than re-declaring its
// backend context. That is deliberate: ggml creates its streams with
// cudaStreamNonBlocking, so events recorded on the legacy default stream do not
// order against the kernels at all and measure only launch overhead. Reaching
// the context through common.cuh is what makes the GPU-event timings real. It
// also means this file is pinned to the comparator revision it is built against;
// build_shim.sh compiles it with that revision's own flags for the same reason.
//
// What is and is not measured (all times in ms, GPU events on the comparator's
// stream unless the name says wall):
//   * upload_ms      host->device staging of weights and activations, untimed
//   * convert_ms     F32->BF16 activation conversion, from the cold/warm delta
//   * matmul_ms      dense MMB kernel on warm (already converted) activations
//   * total_ms       complete ggml_cuda_mul_mat_mmb call, cold activations
//   * wall_*_ms      host-wall equivalent of the same two regions
//
// Publication is fused into the comparator's matmul kernel: the Q8_0 dispatch
// leaves store_f32 true and writes the F32 result directly, so there is no
// separable publication stage to report.

#include <hip/hip_runtime.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

// Brings in the comparator's real ggml_backend_cuda_context, including the
// stream() accessor that makes correct event timing possible.
#include "common.cuh"
// Declarations of the MMB entry points, taken from the pinned tree so they
// cannot drift from the library this links against.
#include "mmb.cuh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void hip_check(hipError_t err, const char * what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "shim: %s failed: %s\n", what, hipGetErrorString(err));
        std::exit(3);
    }
}

double event_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    hip_check(hipEventElapsedTime(&ms, a, b), "hipEventElapsedTime");
    return static_cast<double>(ms);
}

double wall_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct stats {
    double mean = 0.0;
    double lo = 0.0;
    double hi = 0.0;
};

stats summarize(const std::vector<double> & v) {
    stats s;
    if (v.empty()) return s;
    s.lo = s.hi = v[0];
    double sum = 0.0;
    for (double x : v) {
        s.lo = x < s.lo ? x : s.lo;
        s.hi = x > s.hi ? x : s.hi;
        sum += x;
    }
    s.mean = sum / static_cast<double>(v.size());
    return s;
}

}  // namespace

// Result block, plain C layout so ctypes can read it without guessing.
struct he_replay_result {
    double upload_ms;
    double convert_ms;
    double matmul_ms;
    double total_ms;
    double wall_matmul_ms;
    double wall_total_ms;
    double spread_ms;    // max - min over the cold event samples
    int    supported;    // ggml_cuda_mmb_supported_mm verdict
    int    samples;
    int    wtype;        // ggml_type actually used
    char   note[256];
};

extern "C" {

// Returns 1 when the comparator's MMB dense path is reachable at all, 0 when
// the pinned build has MMB compiled out. Callers treat 0 as a hard failure
// rather than silently reporting a fallback kernel as the comparator result.
int he_replay_mmb_available(void) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        return 0;
    }
    ggml_backend_free(backend);
    return 1;
}

// Host-side byte size of a (K, M) weight tensor of the given ggml type, so the
// caller can allocate the exact buffer the comparator expects without
// reimplementing ggml's block layout.
size_t he_replay_mmb_weight_bytes(int wtype, int K, int M) {
    ggml_init_params ip = { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) return 0;
    ggml_tensor * t = ggml_new_tensor_2d(gctx, static_cast<ggml_type>(wtype), K, M);
    const size_t n = t ? ggml_nbytes(t) : 0;
    ggml_free(gctx);
    return n;
}

// Number of distinct activation buffers that guarantee a conversion cache miss
// on every call. The comparator caches converted activations in a 4-entry list
// keyed by data pointer, so anything above 4 rotates out.
int he_replay_mmb_min_rotate(void) {
    return 5;
}

// Replays one dense projection through the comparator's MMB entry point.
//
//   w_bytes / x_f32 / out_f32 : host buffers, sizes implied by K, M, T
//   wtype                     : ggml type of the weight (1 == Q8_0)
//   reps                      : timed repetitions per pass
//   rotate                    : distinct activation buffers cycled through to
//                               force conversion cache misses; >= 5 to miss
//   strict                    : when nonzero, an unsupported operand set is a
//                               hard error instead of a reportable result
//
// On return out_f32 holds the comparator's complete-operation result, so the
// caller can diff it against hipEngine's output on the same operands.
int he_replay_mmb_run(const void * w_bytes,
                      const float * x_f32,
                      float * out_f32,
                      int K, int M, int T,
                      int wtype,
                      int reps,
                      int rotate,
                      int strict,
                      struct he_replay_result * res) {
    if (res == nullptr || K <= 0 || M <= 0 || T <= 0 || reps <= 0 || rotate < 1) {
        return 1;
    }
    std::memset(res, 0, sizeof(*res));
    res->wtype = wtype;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::snprintf(res->note, sizeof(res->note), "ggml_backend_cuda_init returned null");
        return 2;
    }
    ggml_backend_cuda_context * bctx =
        reinterpret_cast<ggml_backend_cuda_context *>(backend->context);
    if (bctx == nullptr) {
        std::snprintf(res->note, sizeof(res->note), "backend context is null");
        ggml_backend_free(backend);
        return 2;
    }
    // The stream the comparator will actually launch on. Every event below is
    // recorded here; on the legacy default stream these timings would be empty.
    hipStream_t stream = reinterpret_cast<hipStream_t>(bctx->stream());

    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) {
        std::snprintf(res->note, sizeof(res->note), "ggml_init failed");
        ggml_backend_free(backend);
        return 2;
    }

    // ne0 is the contraction dim for src0/src1, matching ggml's layout.
    ggml_tensor * w   = ggml_new_tensor_2d(gctx, static_cast<ggml_type>(wtype), K, M);
    ggml_tensor * x   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32,  K, T);
    ggml_tensor * dst = ggml_new_tensor_2d(gctx, GGML_TYPE_F32,  M, T);
    if (w == nullptr || x == nullptr || dst == nullptr) {
        std::snprintf(res->note, sizeof(res->note), "tensor creation failed");
        ggml_free(gctx);
        ggml_backend_free(backend);
        return 2;
    }

    const size_t w_bytes_n = ggml_nbytes(w);
    const size_t x_bytes_n = ggml_nbytes(x);
    const size_t d_bytes_n = ggml_nbytes(dst);

    hipEvent_t e0 = nullptr, e1 = nullptr;
    hip_check(hipEventCreate(&e0), "hipEventCreate");
    hip_check(hipEventCreate(&e1), "hipEventCreate");

    void * dw = nullptr;
    hip_check(hipMalloc(&dw, w_bytes_n), "hipMalloc(w)");

    std::vector<float *> dx(static_cast<size_t>(rotate), nullptr);
    for (int i = 0; i < rotate; ++i) {
        hip_check(hipMalloc(&dx[static_cast<size_t>(i)], x_bytes_n), "hipMalloc(x)");
    }
    void * dd = nullptr;
    hip_check(hipMalloc(&dd, d_bytes_n), "hipMalloc(dst)");

    // Staging, outside every timed region. Synchronized on the comparator's
    // stream so the following events cannot observe the copies.
    const double up_h0 = wall_ms();
    hip_check(hipEventRecord(e0, stream), "eventRecord");
    hip_check(hipMemcpyAsync(dw, w_bytes, w_bytes_n, hipMemcpyHostToDevice, stream), "memcpy w");
    for (int i = 0; i < rotate; ++i) {
        hip_check(hipMemcpyAsync(dx[static_cast<size_t>(i)], x_f32, x_bytes_n,
                                 hipMemcpyHostToDevice, stream), "memcpy x");
    }
    hip_check(hipEventRecord(e1, stream), "eventRecord");
    hip_check(hipEventSynchronize(e1), "eventSynchronize");
    res->upload_ms = event_ms(e0, e1);
    (void) up_h0;

    w->data = dw;
    x->data = dx[0];
    dst->data = dd;

    // Dispatch guard. If the pinned build would not select MMB for these
    // operands, the caller must not treat this as a comparator measurement.
    res->supported = ggml_cuda_mmb_supported_mm(w, x, dst) ? 1 : 0;
    if (!res->supported && strict) {
        std::snprintf(res->note, sizeof(res->note),
                      "MMB-NOT-SELECTED K=%d M=%d T=%d (strict)", K, M, T);
        hip_check(hipEventDestroy(e0), "hipEventDestroy");
        hip_check(hipEventDestroy(e1), "hipEventDestroy");
        for (float * p : dx) (void) hipFree(p);
        (void) hipFree(dw);
        (void) hipFree(dd);
        ggml_free(gctx);
        ggml_backend_free(backend);
        return 4;
    }

    // Warm-up: populates the pool and the conversion cache for dx[0], and pays
    // any lazily created handle cost. Nothing here is timed.
    for (int i = 0; i < 3; ++i) {
        ggml_cuda_mul_mat_mmb(*bctx, w, x, dst);
    }
    hip_check(hipStreamSynchronize(stream), "sync after warmup");

    // Warm pass: one activation buffer, so the conversion is a cache hit and
    // the timed region is the dense matmul kernel plus launch overhead.
    x->data = dx[0];
    hip_check(hipEventRecord(e0, stream), "eventRecord");
    const double warm_h0 = wall_ms();
    for (int i = 0; i < reps; ++i) {
        ggml_cuda_mul_mat_mmb(*bctx, w, x, dst);
    }
    hip_check(hipEventRecord(e1, stream), "eventRecord");
    hip_check(hipEventSynchronize(e1), "eventSynchronize");
    const double warm_wall = wall_ms();
    res->matmul_ms = event_ms(e0, e1) / reps;
    res->wall_matmul_ms = (warm_wall - warm_h0) / reps;
    res->total_ms = res->matmul_ms;
    res->wall_total_ms = res->wall_matmul_ms;

    // Cold pass: cycle through `rotate` buffers so every call misses the
    // conversion cache. The delta against the warm pass is the conversion cost.
    if (rotate > 1) {
        std::vector<double> ev;
        std::vector<double> wl;
        ev.reserve(static_cast<size_t>(reps));
        wl.reserve(static_cast<size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            x->data = dx[static_cast<size_t>(i % rotate)];
            hip_check(hipEventRecord(e0, stream), "eventRecord");
            const double h0 = wall_ms();
            ggml_cuda_mul_mat_mmb(*bctx, w, x, dst);
            hip_check(hipEventRecord(e1, stream), "eventRecord");
            hip_check(hipEventSynchronize(e1), "eventSynchronize");
            const double h1 = wall_ms();
            ev.push_back(event_ms(e0, e1));
            wl.push_back(h1 - h0);
        }
        const stats se = summarize(ev);
        const stats sw = summarize(wl);
        res->spread_ms = se.hi - se.lo;
        res->samples = static_cast<int>(ev.size());
        res->total_ms = se.mean;
        res->convert_ms = se.mean - res->matmul_ms;
        res->wall_total_ms = sw.mean;
    } else {
        res->samples = reps;
    }

    hip_check(hipMemcpy(out_f32, dd, d_bytes_n, hipMemcpyDeviceToHost), "memcpy out");

    std::snprintf(res->note, sizeof(res->note),
                  "K=%d M=%d T=%d wtype=%d rotate=%d reps=%d %s",
                  K, M, T, wtype, rotate, reps,
                  res->supported ? "mmb-selected" : "MMB-NOT-SELECTED");

    hip_check(hipEventDestroy(e0), "hipEventDestroy");
    hip_check(hipEventDestroy(e1), "hipEventDestroy");
    for (float * p : dx) {
        (void) hipFree(p);
    }
    (void) hipFree(dw);
    (void) hipFree(dd);
    ggml_free(gctx);
    ggml_backend_free(backend);
    return 0;
}

}  // extern "C"
