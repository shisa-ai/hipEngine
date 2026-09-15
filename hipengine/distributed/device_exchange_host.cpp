// Two-rank device-side staged exchange: the reduction moves out of the host
// dependency chain entirely.
//
// Each rank owns one host-mapped staging slot per exchange point (per layer)
// and one host-mapped flag per slot. The exchange for one slot is three
// stream-ordered operations on each rank's own stream, all safe to capture in
// a HIP graph:
//
//   1. a D2H memcpy node staging this rank's bf16 partial row into its slot;
//   2. a publish kernel writing this rank's system-visible flag for the slot
//      to the rank's step counter (a device counter bumped once per step);
//   3. a spin-sum kernel that waits for the OTHER rank's flag for the slot to
//      reach this rank's counter, then adds the other rank's staged row
//      (read zero-copy over the bus) to this rank's device partial in f32 and
//      narrows once to bf16 - the same widen/add/narrow arithmetic the host
//      driver performs, so the output is bit-identical.
//
// Every rank publishes before it spins, so the two ranks lockstep at every
// slot with no host synchronization and no deadlock. Slot reuse across steps
// is safe by the same lockstep: a producer cannot rewrite a slot until both
// ranks have passed the slot's consumers in the previous step.
//
// Ownership: the handle owns its mapped arenas and device counters for its
// lifetime; nothing is allocated or freed per step or per reduce. The bump,
// publish, and spin kernels are launched only through this module's entry
// points.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

inline int blocks_of(int n) { return (n + 255) / 256; }

constexpr int32_t kOk = 0;
constexpr int32_t kErrArg = -1;
constexpr int32_t kErrHip = -2;

struct Tp2DeviceExchange {
  int world = 0;
  int num_layers = 0;
  int hidden = 0;
  int row_bytes = 0;
  std::vector<int> devices;
  std::vector<hipStream_t> streams;
  // Per owner rank: one mapped staging arena of num_layers rows, one mapped
  // flag array of num_layers uint32 entries (host pointers), and one device
  // step counter.
  std::vector<unsigned char*> staging_host;
  std::vector<unsigned int*> flags_host;
  std::vector<unsigned int*> counters;
  // Per VIEWING rank, per owner rank: the device-visible address of the
  // owner's arena as mapped for the viewing rank's device. A host-mapped
  // pointer returned by hipHostGetDevicePointer on one device is not
  // necessarily valid from another device's kernel, so every rank keeps its
  // own view of both arenas.
  std::vector<std::vector<unsigned char*>> staging_views;
  std::vector<std::vector<unsigned int*>> flags_views;
  // Per rank: one device timeout flag, set by a spin kernel that exceeded
  // its spin budget (a missing/stalled peer); checked and cleared by the
  // host's wait/step_begin.
  std::vector<unsigned int*> timeout_flags;
  unsigned int max_spins = 0;
  std::string error;
};

int32_t set_hip_error(Tp2DeviceExchange* ex, hipError_t code, const char* what) {
  if (ex != nullptr) {
    ex->error = std::string(what) + ": " + hipGetErrorString(code);
  }
  return kErrHip;
}

__global__ void tp2_dev_bump_counter(unsigned int* counter) {
  if (threadIdx.x == 0) {
    atomicAdd(counter, 1u);
  }
}

__global__ void tp2_dev_publish_flag(
    const unsigned int* counter,
    volatile unsigned int* host_flag) {
  if (threadIdx.x == 0) {
    // The staging memcpy for this slot is earlier on this stream, so the
    // copy is visible once this flag store lands at system scope.
    *host_flag = *counter;
    __threadfence_system();
  }
}

__device__ inline float bf16_bits_to_float_value(unsigned short bits) {
  // The same widen the boundary-cast kernel uses
  // (kernels/hip_gfx1100/convert/cast.hip bf16_bits_to_float): the bf16 bit
  // pattern is the high half of an f32, NOT an integer value.
  union {
    float f32;
    uint32_t u32;
  } out;
  out.u32 = static_cast<uint32_t>(bits) << 16;
  return out.f32;
}

__device__ inline unsigned short bf16_bits_to_float_round(float value) {
  // The same RNE bit arithmetic the boundary-cast kernel uses
  // (kernels/hip_gfx1100/convert/cast.hip float_to_bf16_bits), so the
  // narrowed output is bit-identical to the host driver's path.
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<unsigned short>(in.u32 >> 16);
}

__global__ void tp2_dev_spin_add_bf16(
    const unsigned short* __restrict__ own_partial,
    const unsigned short* __restrict__ remote_staged,
    unsigned short* __restrict__ out,
    const volatile unsigned int* remote_flag,
    const unsigned int* own_counter,
    volatile unsigned int* timeout_flag,
    unsigned int max_spins,
    int n) {
  const unsigned int expected = *own_counter;
  if (threadIdx.x == 0) {
    unsigned int spins = 0;
    while (*remote_flag < expected) {
      if (++spins >= max_spins) {
        // Bounded failure: exit instead of hanging the group. The host's
        // wait checks this flag after the streams sync.
        atomicExch(const_cast<unsigned int*>(timeout_flag), 1u);
        return;
      }
      __builtin_amdgcn_s_sleep(64);
    }
  }
  // Every block's lead thread waits for the flag (or times out), then the
  // block barriers behind it, so no thread reads the staged row before the
  // remote's copy is visible at system scope - and a timeout poisons this
  // rank's output instead of publishing one.
  __syncthreads();
  if (*timeout_flag != 0u) {
    return;
  }
  __threadfence();
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float va = bf16_bits_to_float_value(own_partial[i]);
    const float vb = bf16_bits_to_float_value(remote_staged[i]);
    out[i] = bf16_bits_to_float_round(va + vb);
  }
}

}  // namespace

extern "C" {

void* tp2_dev_exchange_create(
    const int32_t* devices,
    int32_t world,
    const uint64_t* streams,
    int32_t num_layers,
    int32_t hidden,
    uint32_t max_spins,
    int32_t* error_code) {
  *error_code = kOk;
  if (devices == nullptr || streams == nullptr || world != 2 || num_layers < 1 ||
      hidden < 1 || max_spins == 0) {
    *error_code = kErrArg;
    return nullptr;
  }
  auto* ex = new Tp2DeviceExchange();
  ex->world = world;
  ex->num_layers = num_layers;
  ex->hidden = hidden;
  ex->row_bytes = hidden * 2;  // bf16 partials
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }
  for (int rank = 0; rank < world; ++rank) {
    ex->devices.push_back(static_cast<int>(devices[rank]));
    ex->streams.push_back(reinterpret_cast<hipStream_t>(streams[rank]));
    if (hipSetDevice(ex->devices[rank]) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipSetDevice");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    const size_t arena_bytes =
        static_cast<size_t>(num_layers) * static_cast<size_t>(ex->row_bytes);
    unsigned char* sh = nullptr;
    if (hipHostAlloc(reinterpret_cast<void**>(&sh), arena_bytes,
                     hipHostAllocMapped) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipHostAlloc staging");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    std::memset(sh, 0, arena_bytes);
    unsigned char* sd = nullptr;
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&sd), sh, 0) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipHostGetDevicePointer");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    unsigned int* fh = nullptr;
    if (hipHostAlloc(reinterpret_cast<void**>(&fh),
                     static_cast<size_t>(num_layers) * sizeof(unsigned int),
                     hipHostAllocMapped) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipHostAlloc flags");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    std::memset(fh, 0, static_cast<size_t>(num_layers) * sizeof(unsigned int));
    unsigned int* fd = nullptr;
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&fd), fh, 0) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipHostGetDevicePointer flags");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    unsigned int* counter = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&counter), sizeof(unsigned int)) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipMalloc counter");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    if (hipMemset(counter, 0, sizeof(unsigned int)) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipMemset counter");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    unsigned int* timeout_flag = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&timeout_flag), sizeof(unsigned int)) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipMalloc timeout flag");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    if (hipMemset(timeout_flag, 0, sizeof(unsigned int)) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipMemset timeout flag");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    ex->staging_host.push_back(sh);
    ex->flags_host.push_back(fh);
    ex->counters.push_back(counter);
    ex->timeout_flags.push_back(timeout_flag);
  }
  ex->max_spins = max_spins;
  // Per-rank device views of both ranks' arenas: with device r current,
  // map each owner's host pointers into device r's address space.
  for (int rank = 0; rank < world; ++rank) {
    if (hipSetDevice(ex->devices[rank]) != hipSuccess) {
      *error_code = set_hip_error(ex, hipGetLastError(), "hipSetDevice views");
      hipSetDevice(previous_device);
      delete ex;
      return nullptr;
    }
    std::vector<unsigned char*> srow;
    std::vector<unsigned int*> frow;
    for (int owner = 0; owner < world; ++owner) {
      unsigned char* sd = nullptr;
      if (hipHostGetDevicePointer(reinterpret_cast<void**>(&sd),
                                  ex->staging_host[owner], 0) != hipSuccess) {
        *error_code = set_hip_error(ex, hipGetLastError(), "hipHostGetDevicePointer view");
        hipSetDevice(previous_device);
        delete ex;
        return nullptr;
      }
      unsigned int* fd = nullptr;
      if (hipHostGetDevicePointer(reinterpret_cast<void**>(&fd),
                                  ex->flags_host[owner], 0) != hipSuccess) {
        *error_code = set_hip_error(ex, hipGetLastError(), "hipHostGetDevicePointer flags view");
        hipSetDevice(previous_device);
        delete ex;
        return nullptr;
      }
      srow.push_back(sd);
      frow.push_back(fd);
    }
    ex->staging_views.push_back(srow);
    ex->flags_views.push_back(frow);
  }
  (void)hipSetDevice(previous_device);
  return ex;
}

// One bump per rank per step, enqueued eagerly on each rank's stream before
// the step's graphs run. The spin/publish kernels read the counter from
// device memory, so no per-step argument is baked into a graph.
int32_t tp2_dev_exchange_step_begin(void* handle) {
  auto* ex = static_cast<Tp2DeviceExchange*>(handle);
  if (ex == nullptr) {
    return kErrArg;
  }
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }
  for (int rank = 0; rank < ex->world; ++rank) {
    if (hipSetDevice(ex->devices[rank]) != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, hipGetLastError(), "hipSetDevice step_begin");
    }
    if (hipMemsetAsync(ex->timeout_flags[rank], 0, sizeof(unsigned int),
                       ex->streams[rank]) != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, hipGetLastError(), "hipMemsetAsync timeout flag");
    }
    hipLaunchKernelGGL(tp2_dev_bump_counter, dim3(1), dim3(1), 0, ex->streams[rank],
                       ex->counters[rank]);
    hipError_t code = hipGetLastError();
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, code, "bump launch");
    }
  }
  (void)hipSetDevice(previous_device);
  return kOk;
}

// The rank-scoped building block: stage THIS rank's partial, publish this
// rank's flag, and run this rank's spin-sum against the other rank's staged
// row and flag. Stream-ordered on this rank's own stream, no host
// synchronization, safe inside this rank's HIP graph capture. Both ranks run
// this for the same slot; each publishes before it spins.
int32_t tp2_dev_exchange_enqueue_rank(
    void* handle,
    int32_t rank,
    const void* own_partial,
    int32_t slot,
    uint64_t out_payload) {
  auto* ex = static_cast<Tp2DeviceExchange*>(handle);
  if (ex == nullptr || own_partial == nullptr) {
    return kErrArg;
  }
  if (rank < 0 || rank >= ex->world) {
    if (ex != nullptr) {
      ex->error = "rank " + std::to_string(rank) + " outside this handle's world";
    }
    return kErrArg;
  }
  if (slot < 0 || slot >= ex->num_layers) {
    if (ex != nullptr) {
      ex->error = "exchange slot " + std::to_string(slot) + " outside this handle's " +
                  std::to_string(ex->num_layers) + " slots";
    }
    return kErrArg;
  }
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }
  if (hipSetDevice(ex->devices[rank]) != hipSuccess) {
    (void)hipSetDevice(previous_device);
    return set_hip_error(ex, hipGetLastError(), "hipSetDevice enqueue_rank");
  }
  const int other = 1 - rank;
  unsigned char* slot_staging =
      ex->staging_views[rank][rank] +
      static_cast<size_t>(slot) * static_cast<size_t>(ex->row_bytes);
  volatile unsigned int* slot_flag =
      ex->flags_views[rank][rank] + static_cast<size_t>(slot);
  volatile unsigned int* remote_flag =
      ex->flags_views[rank][other] + static_cast<size_t>(slot);
  const unsigned short* remote_staged =
      reinterpret_cast<const unsigned short*>(
          ex->staging_views[rank][other] +
          static_cast<size_t>(slot) * static_cast<size_t>(ex->row_bytes));

  hipError_t code = hipMemcpyAsync(
      slot_staging,
      const_cast<void*>(own_partial),
      static_cast<size_t>(ex->row_bytes),
      hipMemcpyDeviceToHost,
      ex->streams[rank]);
  if (code != hipSuccess) {
    (void)hipSetDevice(previous_device);
    return set_hip_error(ex, code, "staging memcpy");
  }
  hipLaunchKernelGGL(
      tp2_dev_publish_flag, dim3(1), dim3(1), 0, ex->streams[rank],
      ex->counters[rank],
      const_cast<unsigned int*>(slot_flag));
  code = hipGetLastError();
  if (code != hipSuccess) {
    (void)hipSetDevice(previous_device);
    return set_hip_error(ex, code, "publish launch");
  }
  hipLaunchKernelGGL(
      tp2_dev_spin_add_bf16, dim3(blocks_of(ex->hidden)), dim3(256), 0,
      ex->streams[rank],
      reinterpret_cast<const unsigned short*>(own_partial),
      remote_staged,
      reinterpret_cast<unsigned short*>(out_payload),
      remote_flag,
      ex->counters[rank],
      ex->timeout_flags[rank],
      ex->max_spins,
      ex->hidden);
  code = hipGetLastError();
  if (code != hipSuccess) {
    (void)hipSetDevice(previous_device);
    return set_hip_error(ex, code, "spin-add launch");
  }
  (void)hipSetDevice(previous_device);
  return kOk;
}

// Host-side wait for one slot's exchange: both ranks' spin-sums are
// stream-ordered behind their launches, so syncing both streams once each is
// the only wait. Consumers that read the output eagerly (the eager schedule's
// tail) need this; captured-graph consumers need nothing.
int32_t tp2_dev_exchange_wait(void* handle) {
  auto* ex = static_cast<Tp2DeviceExchange*>(handle);
  if (ex == nullptr) {
    return kErrArg;
  }
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }
  int timeouts = 0;
  for (int rank = 0; rank < ex->world; ++rank) {
    if (hipSetDevice(ex->devices[rank]) != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, hipGetLastError(), "hipSetDevice wait");
    }
    hipError_t code = hipStreamSynchronize(ex->streams[rank]);
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, code, "hipStreamSynchronize");
    }
    unsigned int flag = 0;
    if (hipMemcpy(&flag, ex->timeout_flags[rank], sizeof(unsigned int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_hip_error(ex, hipGetLastError(), "timeout flag readback");
    }
    if (flag != 0u) {
      if (ex != nullptr) {
        ex->error = "spin timeout on rank " + std::to_string(rank) +
                    ": the peer did not publish within the spin budget";
      }
      timeouts = 1;
    }
  }
  (void)hipSetDevice(previous_device);
  return timeouts ? kErrHip : kOk;
}

const char* tp2_dev_exchange_last_error(void* handle) {
  if (handle == nullptr) {
    return "device exchange: unknown error slot";
  }
  return static_cast<Tp2DeviceExchange*>(handle)->error.c_str();
}

void tp2_dev_exchange_destroy(void* handle) {
  auto* ex = static_cast<Tp2DeviceExchange*>(handle);
  if (ex == nullptr) {
    return;
  }
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }
  for (int rank = 0; rank < ex->world; ++rank) {
    (void)hipSetDevice(ex->devices[rank]);
    if (rank < static_cast<int>(ex->staging_host.size()) && ex->staging_host[rank] != nullptr) {
      (void)hipHostFree(ex->staging_host[rank]);
    }
    if (rank < static_cast<int>(ex->flags_host.size()) && ex->flags_host[rank] != nullptr) {
      (void)hipHostFree(ex->flags_host[rank]);
    }
    if (rank < static_cast<int>(ex->counters.size()) && ex->counters[rank] != nullptr) {
      (void)hipFree(ex->counters[rank]);
    }
  }
  (void)hipSetDevice(previous_device);
  delete ex;
}

}  // extern "C"
