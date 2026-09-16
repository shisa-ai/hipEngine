// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Compiled host driver for the two-rank staged exchange.
//
// Structure lineage: this implements the same batched protocol as the Python
// route in `hipengine/distributed/staged.py` (both D2H submitted before either
// wait, one wait per stream, host f32 sum) and the native A/B arm in
// `benchmarks/micro/runners/hip_staged_exchange.hip` (commit 0627ecae, measured
// 20.8 us per reduction against ~40 us for the Python loop on idle buffers).
// The Python loop's gap is protocol overhead, not copy bandwidth: twelve
// device-scoped switches and eight ctypes submissions per reduction. This
// driver removes the protocol overhead and the H2D return path: the reduced
// f32 payload lives in mapped pinned host memory, so both ranks' consumer
// kernels read it zero-copy over the bus and no H2D copy is submitted at all.
//
// Bit-parity contract with the Python route: for world == 2 the host sum is
// the same single f32 addition per element in the same rank order (row 0 +
// row 1), and bf16 staging widens by the same bit shift into the high half,
// so a reduced payload is bit-identical to the Python route's. World != 2 is
// rejected at create: the Python route stays the general fallback.
//
// Slot discipline: `slot_sets` alternating slot sets, exactly like the Python
// route. The next reduce's per-stream waits are queued after this reduce's
// consumer kernels on each rank's stream, so the host cannot rewrite a payload
// slot while a consumer still reads it. On any HIP error the driver records
// the message, restores the caller's current device, and returns a negative
// code without advancing the slot; the Python wrapper poisons the transport.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kErrArg = -1;
constexpr int32_t kErrHip = -2;
constexpr int32_t kStagingF32 = 0;
constexpr int32_t kStagingBf16 = 1;
constexpr unsigned kHostMallocMapped = 0x02;

// The last create failure is retrievable through tp2_staged_last_error(nullptr).
// Construction is once per session on one host thread; a static slot is
// deliberate and single-consumer.
std::string& create_error_slot() {
  static std::string slot;
  return slot;
}

struct Tp2StagedExchange {
  int world = 0;
  int hidden = 0;
  int capacity_rows = 0;
  int staging_dtype = 0;
  int slot_sets = 0;
  size_t staging_row_bytes = 0;  // per-rank capacity row block in the staging dtype
  size_t payload_row_bytes = 0;  // capacity_rows * hidden * 4: the published f32 block
  size_t staging_nbytes = 0;
  size_t payload_nbytes = 0;
  std::vector<int> devices;
  std::vector<hipStream_t> streams;
  unsigned char* staging = nullptr;         // pinned arena: slot_sets * world rows
  unsigned char* payload = nullptr;         // pinned mapped: slot_sets f32 rows
  uint64_t payload_device_base = 0;         // device-visible address of payload
  int slot = 0;
  std::string error;
};

hipError_t set_error(Tp2StagedExchange* ex, hipError_t code, const char* what) {
  if (ex != nullptr) {
    ex->error = std::string(what) + ": " + hipGetErrorString(code);
  }
  return code;
}

}  // namespace

extern "C" {

// Allocate the mapped pinned arenas and validate the fixed configuration.
// Returns an opaque handle, or nullptr with an error code and message.
void* tp2_staged_create(
    const int32_t* devices,
    int32_t world,
    void* const* streams,
    int32_t hidden,
    int32_t staging_dtype,
    int32_t slot_sets,
    int32_t capacity_rows,
    int32_t* out_error) {
  create_error_slot().clear();
  if (out_error != nullptr) {
    *out_error = kOk;
  }
  auto fail = [&](int32_t code, const std::string& message) {
    create_error_slot() = message;
    if (out_error != nullptr) {
      *out_error = code;
    }
    return static_cast<void*>(nullptr);
  };
  if (devices == nullptr || streams == nullptr) {
    return fail(kErrArg, "devices and streams arrays are required");
  }
  if (world != 2) {
    return fail(kErrArg, "the compiled driver is the two-rank transport; "
                         "world " + std::to_string(world) + " must use the Python route");
  }
  if (devices[0] == devices[1]) {
    return fail(kErrArg, "staged-exchange ranks must be distinct devices");
  }
  if (hidden <= 0 || slot_sets < 2 || capacity_rows < 1) {
    return fail(kErrArg, "hidden must be positive, slot_sets at least 2, and "
                         "capacity_rows at least 1");
  }
  if (staging_dtype != kStagingF32 && staging_dtype != kStagingBf16) {
    return fail(kErrArg, "staging_dtype must be 0 (f32) or 1 (bf16)");
  }
  // Stream handles are raw: 0 is the per-device default stream, exactly what
  // the model-owning loop passes, and it is valid. There is no null check.

  auto* ex = new (std::nothrow) Tp2StagedExchange();
  if (ex == nullptr) {
    return fail(kErrArg, "driver state allocation failed");
  }
  ex->world = static_cast<int>(world);
  ex->hidden = static_cast<int>(hidden);
  ex->capacity_rows = static_cast<int>(capacity_rows);
  ex->staging_dtype = static_cast<int>(staging_dtype);
  ex->slot_sets = static_cast<int>(slot_sets);
  ex->staging_row_bytes =
      static_cast<size_t>(capacity_rows) * static_cast<size_t>(hidden) *
      (staging_dtype == kStagingBf16 ? 2u : 4u);
  ex->payload_row_bytes =
      static_cast<size_t>(capacity_rows) * static_cast<size_t>(hidden) * 4u;
  ex->staging_nbytes =
      static_cast<size_t>(slot_sets) * ex->world * ex->staging_row_bytes;
  ex->payload_nbytes = static_cast<size_t>(slot_sets) * ex->payload_row_bytes;
  for (int rank = 0; rank < world; ++rank) {
    ex->devices.push_back(static_cast<int>(devices[rank]));
    ex->streams.push_back(static_cast<hipStream_t>(streams[rank]));
  }

  // Mapped pinned arenas: the staging rows are the D2H destinations and the
  // payload rows are what both ranks' consumer kernels read zero-copy.
  hipError_t code = hipHostMalloc(
      reinterpret_cast<void**>(&ex->staging), ex->staging_nbytes, kHostMallocMapped);
  if (code != hipSuccess) {
    delete ex;
    return fail(kErrHip, std::string("hipHostMalloc staging: ") + hipGetErrorString(code));
  }
  code = hipHostMalloc(
      reinterpret_cast<void**>(&ex->payload), ex->payload_nbytes, kHostMallocMapped);
  if (code != hipSuccess) {
    (void)hipHostFree(ex->staging);
    delete ex;
    return fail(kErrHip, std::string("hipHostMalloc payload: ") + hipGetErrorString(code));
  }
  const void* device_view = nullptr;
  code = hipHostGetDevicePointer(
      const_cast<void**>(&device_view), ex->payload, 0);
  if (code != hipSuccess) {
    (void)hipHostFree(ex->payload);
    (void)hipHostFree(ex->staging);
    delete ex;
    return fail(
        kErrHip,
        std::string("hipHostGetDevicePointer: ") + hipGetErrorString(code));
  }
  ex->payload_device_base = reinterpret_cast<uint64_t>(device_view);
  return static_cast<void*>(ex);
}

void tp2_staged_destroy(void* handle) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return;
  }
  if (ex->staging != nullptr) {
    (void)hipHostFree(ex->staging);
  }
  if (ex->payload != nullptr) {
    (void)hipHostFree(ex->payload);
  }
  delete ex;
}

// One staged reduction into a caller-chosen payload slot. `partials` holds
// one device partial pointer per rank; on success `out_payload` receives the
// device-visible address of the f32 row this reduce published (the same
// address for both ranks). No H2D copy is submitted: the consumers read the
// mapped host row directly. The slot is not advanced.
static int32_t reduce_into(
    Tp2StagedExchange* ex,
    void* const* partials,
    int32_t slot,
    int32_t active_rows,
    uint64_t* out_payload) {
  if (ex == nullptr || partials == nullptr) {
    return kErrArg;
  }
  if (slot < 0 || slot >= ex->slot_sets) {
    if (ex != nullptr) {
      ex->error = "payload slot " + std::to_string(slot) +
                  " outside this transport's " + std::to_string(ex->slot_sets) +
                  " slot sets";
    }
    return kErrArg;
  }
  if (active_rows < 1 || active_rows > ex->capacity_rows) {
    if (ex != nullptr) {
      ex->error = "active rows " + std::to_string(active_rows) +
                  " outside this transport's 1.." +
                  std::to_string(ex->capacity_rows);
    }
    return kErrArg;
  }
  for (int rank = 0; rank < ex->world; ++rank) {
    if (partials[rank] == nullptr) {
      if (ex != nullptr) {
        ex->error = "no partial given for rank " + std::to_string(rank);
      }
      return kErrArg;
    }
  }
  int previous_device = 0;
  if (hipGetDevice(&previous_device) != hipSuccess) {
    previous_device = 0;
  }

  const size_t element_bytes =
      static_cast<size_t>(ex->staging_dtype == kStagingBf16 ? 2 : 4);
  const size_t active_bytes =
      static_cast<size_t>(active_rows) * static_cast<size_t>(ex->hidden) * element_bytes;

  hipError_t code = hipSuccess;
  unsigned char* staging_slot =
      ex->staging + static_cast<size_t>(slot) * ex->world * ex->staging_row_bytes;
  for (int rank = 0; rank < ex->world; ++rank) {
    code = hipSetDevice(ex->devices[rank]);
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_error(ex, code, "hipSetDevice");
    }
    code = hipMemcpyAsync(
        staging_slot + static_cast<size_t>(rank) * ex->staging_row_bytes,
        partials[rank],
        active_bytes,
        hipMemcpyDeviceToHost,
        ex->streams[rank]);
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_error(ex, code, "hipMemcpyAsync D2H");
    }
  }
  for (int rank = 0; rank < ex->world; ++rank) {
    code = hipSetDevice(ex->devices[rank]);
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_error(ex, code, "hipSetDevice");
    }
    code = hipStreamSynchronize(ex->streams[rank]);
    if (code != hipSuccess) {
      (void)hipSetDevice(previous_device);
      return set_error(ex, code, "hipStreamSynchronize");
    }
  }

  // The host sum: the same single f32 addition per element, in rank order,
  // as the Python route's (ranks, active) reduction. bf16 staging widens by
  // the same bit shift into the high half before the f32 sum. Only the active
  // rows are summed; the trailing capacity rows are zeroed so an inactive tail
  // is never published as a valid partial.
  const int active_floats = active_rows * ex->hidden;
  const int capacity_floats = ex->capacity_rows * ex->hidden;
  float* out = reinterpret_cast<float*>(
      ex->payload + static_cast<size_t>(slot) * ex->payload_row_bytes);
  if (ex->staging_dtype == kStagingBf16) {
    const uint16_t* row0 =
        reinterpret_cast<const uint16_t*>(staging_slot);
    const uint16_t* row1 =
        reinterpret_cast<const uint16_t*>(staging_slot + ex->staging_row_bytes);
    for (int j = 0; j < active_floats; ++j) {
      const uint32_t bits0 = static_cast<uint32_t>(row0[j]) << 16;
      const uint32_t bits1 = static_cast<uint32_t>(row1[j]) << 16;
      float value0;
      float value1;
      std::memcpy(&value0, &bits0, sizeof(value0));
      std::memcpy(&value1, &bits1, sizeof(value1));
      out[j] = value0 + value1;
    }
  } else {
    const float* row0 = reinterpret_cast<const float*>(staging_slot);
    const float* row1 =
        reinterpret_cast<const float*>(staging_slot + ex->staging_row_bytes);
    for (int j = 0; j < active_floats; ++j) {
      out[j] = row0[j] + row1[j];
    }
  }
  for (int j = active_floats; j < capacity_floats; ++j) {
    out[j] = 0.0f;
  }

  if (out_payload != nullptr) {
    *out_payload =
        ex->payload_device_base + static_cast<uint64_t>(slot) * ex->payload_row_bytes;
  }
  (void)hipSetDevice(previous_device);
  return kOk;
}

int32_t tp2_staged_reduce(void* handle, void* const* partials, uint64_t* out_payload) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return kErrArg;
  }
  const int32_t slot = ex->slot;
  const int32_t code = reduce_into(ex, partials, slot, ex->capacity_rows, out_payload);
  if (code == kOk) {
    ex->slot = 1 - slot;
  }
  return code;
}

int32_t tp2_staged_reduce_at(
    void* handle,
    void* const* partials,
    int32_t slot,
    uint64_t* out_payload) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return kErrArg;
  }
  return reduce_into(ex, partials, slot, ex->capacity_rows, out_payload);
}

// Batched siblings: stage and sum exactly `active_rows` rows (1..capacity) and
// zero the trailing capacity rows. The single-row ABI above is unchanged.
int32_t tp2_staged_reduce_rows(
    void* handle,
    void* const* partials,
    int32_t active_rows,
    uint64_t* out_payload) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return kErrArg;
  }
  const int32_t slot = ex->slot;
  const int32_t code = reduce_into(ex, partials, slot, active_rows, out_payload);
  if (code == kOk) {
    ex->slot = 1 - slot;
  }
  return code;
}

int32_t tp2_staged_reduce_at_rows(
    void* handle,
    void* const* partials,
    int32_t slot,
    int32_t active_rows,
    uint64_t* out_payload) {
  return reduce_into(
      static_cast<Tp2StagedExchange*>(handle), partials, slot, active_rows, out_payload);
}

const char* tp2_staged_last_error(void* handle) {
  if (handle == nullptr) {
    return create_error_slot().c_str();
  }
  return static_cast<Tp2StagedExchange*>(handle)->error.c_str();
}

// The device-visible base of the mapped reduced-payload arena and the byte
// stride between payload slot sets, so a caller can precompute the fixed
// per-slot address a captured graph's consumer reads.
uint64_t tp2_staged_payload_base(void* handle) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return 0;
  }
  return ex->payload_device_base;
}

size_t tp2_staged_slot_stride(void* handle) {
  auto* ex = static_cast<Tp2StagedExchange*>(handle);
  if (ex == nullptr) {
    return 0;
  }
  return ex->payload_row_bytes;
}

}  // extern "C"
