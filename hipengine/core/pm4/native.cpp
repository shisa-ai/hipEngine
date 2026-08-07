// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Minimal public-ROCr AQL and retained-PM4 owner for hipEngine.
//
// Packet/register and lifecycle shapes were independently reduced from the
// Apache-2.0 Redline reference at
// 33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e, especially
// redline-rocr/{pm4_gfx10.rs,packet.rs,runtime.rs}. The vendor packet matches
// ROCm/rocm-systems@c0430a50286200ab0562f4733445cdee6e48d416
// aqlprofile/src/core/amd_aql_pm4_ib_packet.h. This file deliberately supports
// only exact gfx1100, zero-scratch wave32 kernels and fails closed otherwise.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr uint32_t kAbiVersion = 2;
constexpr uint32_t kExecutableFlagTimestamps = 1u << 0;
constexpr uint32_t kExecutableFlagStatefulRegisters = 1u << 1;
constexpr size_t kPacketBytes = 64;
constexpr uint64_t kNanosPerSecond = 1000000000ull;
constexpr uint32_t kPacket3SetShReg = 0x76;
constexpr uint32_t kPacket3DispatchDirect = 0x15;
constexpr uint32_t kPacket3EventWrite = 0x46;
constexpr uint32_t kPacket3AcquireMem = 0x58;
constexpr uint32_t kPacket3CopyData = 0x40;
constexpr uint32_t kPacket3ReleaseMem = 0x49;
constexpr uint32_t kPacket3IndirectBuffer = 0x3f;
constexpr uint32_t kComputeNumThreadX = 0x207;
constexpr uint32_t kComputePgmLo = 0x20c;
constexpr uint32_t kComputePgmRsrc1 = 0x212;
constexpr uint32_t kComputeResourceLimits = 0x215;
constexpr uint32_t kComputeTmpRingSize = 0x218;
constexpr uint32_t kComputePgmRsrc3 = 0x228;
constexpr uint32_t kComputeUserData0 = 0x240;
constexpr uint32_t kLdsSizeMask = 0x00ff8000;
constexpr uint32_t kLdsSizeShift = 15;
constexpr uint32_t kLdsGranule = 512;
constexpr uint16_t kEnablePrivateSegmentBuffer = 1u << 0;
constexpr uint16_t kEnableDispatchPtr = 1u << 1;
constexpr uint16_t kEnableQueuePtr = 1u << 2;
constexpr uint16_t kEnableKernargPtr = 1u << 3;
constexpr uint16_t kEnableDispatchId = 1u << 4;
constexpr uint16_t kEnableFlatScratch = 1u << 5;
constexpr uint16_t kEnablePrivateSegmentSize = 1u << 6;
constexpr uint16_t kEnableWave32 = 1u << 10;
constexpr uint16_t kSupportedProperties =
    kEnablePrivateSegmentBuffer | kEnableKernargPtr | kEnableWave32;
using SteadyClock = std::chrono::steady_clock;

uint64_t elapsed_ns(SteadyClock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - start).count());
}

static_assert(sizeof(hsa_kernel_dispatch_packet_t) == kPacketBytes,
              "public HSA kernel packet ABI must remain 64 bytes");

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string hsa_error(const char* operation, hsa_status_t status) {
  const char* detail = nullptr;
  (void)hsa_status_string(status, &detail);
  std::ostringstream out;
  out << operation << " failed with HSA status " << static_cast<uint32_t>(status);
  if (detail != nullptr) out << ": " << detail;
  return out.str();
}

void check(hsa_status_t status, const char* operation) {
  if (status != HSA_STATUS_SUCCESS) throw Error(hsa_error(operation, status));
}

void set_error(char* error, size_t error_size, const std::string& message) {
  if (error == nullptr || error_size == 0) return;
  const size_t copy = std::min(error_size - 1, message.size());
  std::memcpy(error, message.data(), copy);
  error[copy] = '\0';
}

template <typename Function>
int guarded(char* error, size_t error_size, Function&& function) {
  try {
    function();
    set_error(error, error_size, "");
    return 0;
  } catch (const std::exception& exc) {
    set_error(error, error_size, exc.what());
    return 1;
  } catch (...) {
    set_error(error, error_size, "unknown native PM4 failure");
    return 1;
  }
}

struct PciAddress {
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t device = 0;
  uint32_t function = 0;
};

PciAddress parse_pci(const char* value) {
  if (value == nullptr) throw Error("PCI BDF is null");
  PciAddress result;
  char trailing = 0;
  const int count = std::sscanf(value, "%x:%x:%x.%x%c", &result.domain, &result.bus,
                                &result.device, &result.function, &trailing);
  if (count != 4 || result.domain > 0xffff || result.bus > 0xff ||
      result.device > 0x1f || result.function > 7) {
    throw Error(std::string("invalid PCI BDF: ") + value);
  }
  return result;
}

std::string format_pci(const PciAddress& value) {
  char output[32] = {};
  std::snprintf(output, sizeof(output), "%04x:%02x:%02x.%x", value.domain, value.bus,
                value.device, value.function);
  return output;
}

std::mutex g_runtime_mutex;
uint32_t g_runtime_leases = 0;

void acquire_runtime() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_runtime_leases == 0) check(hsa_init(), "hsa_init");
  ++g_runtime_leases;
}

void release_runtime() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_runtime_leases == 0) return;
  --g_runtime_leases;
  if (g_runtime_leases == 0) check(hsa_shut_down(), "hsa_shut_down");
}

struct AgentSearch {
  PciAddress wanted;
  std::string gfx;
  hsa_agent_t gpu{0};
  hsa_agent_t cpu{0};
  uint32_t matches = 0;
  hsa_status_t error = HSA_STATUS_SUCCESS;
};

hsa_status_t find_agent(hsa_agent_t agent, void* data) {
  auto* search = static_cast<AgentSearch*>(data);
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    search->error = status;
    return status;
  }
  if (type == HSA_DEVICE_TYPE_CPU && search->cpu.handle == 0) {
    search->cpu = agent;
    return HSA_STATUS_SUCCESS;
  }
  if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;

  uint32_t domain = 0;
  uint32_t bdfid = 0;
  char name[64] = {};
  if ((status = hsa_agent_get_info(
           agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain)) !=
          HSA_STATUS_SUCCESS ||
      (status = hsa_agent_get_info(
           agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdfid)) !=
          HSA_STATUS_SUCCESS ||
      (status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name)) != HSA_STATUS_SUCCESS) {
    search->error = status;
    return status;
  }
  PciAddress actual{domain, (bdfid >> 8) & 0xff, (bdfid >> 3) & 0x1f, bdfid & 7};
  if (actual.domain == search->wanted.domain && actual.bus == search->wanted.bus &&
      actual.device == search->wanted.device && actual.function == search->wanted.function) {
    if (search->gfx != name) {
      search->error = HSA_STATUS_ERROR_INVALID_AGENT;
      return search->error;
    }
    search->gpu = agent;
    ++search->matches;
  }
  return HSA_STATUS_SUCCESS;
}

struct PoolSearch {
  hsa_amd_memory_pool_t pool{0};
  size_t granule = 0;
  size_t alignment = 0;
  hsa_status_t error = HSA_STATUS_SUCCESS;
};

hsa_status_t find_kernarg_pool(hsa_amd_memory_pool_t pool, void* data) {
  auto* search = static_cast<PoolSearch*>(data);
  hsa_amd_segment_t segment;
  uint32_t flags = 0;
  bool allowed = false;
  hsa_status_t status = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    search->error = status;
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  if ((status = hsa_amd_memory_pool_get_info(
           pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags)) != HSA_STATUS_SUCCESS ||
      (status = hsa_amd_memory_pool_get_info(
           pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed)) !=
          HSA_STATUS_SUCCESS) {
    search->error = status;
    return status;
  }
  const uint32_t required = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  if (!allowed || (flags & required) != required) return HSA_STATUS_SUCCESS;
  size_t granule = 0;
  size_t alignment = 0;
  if ((status = hsa_amd_memory_pool_get_info(
           pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &granule)) !=
          HSA_STATUS_SUCCESS ||
      (status = hsa_amd_memory_pool_get_info(
           pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &alignment)) !=
          HSA_STATUS_SUCCESS) {
    search->error = status;
    return status;
  }
  if (granule != 0 && alignment != 0 && (alignment & (alignment - 1)) == 0 &&
      search->pool.handle == 0) {
    search->pool = pool;
    search->granule = granule;
    search->alignment = alignment;
  }
  return HSA_STATUS_SUCCESS;
}

struct Allocation {
  hsa_amd_memory_pool_t pool{0};
  void* pointer = nullptr;
  size_t length = 0;
  size_t allocated = 0;

  Allocation() = default;
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&& other) noexcept { *this = std::move(other); }
  Allocation& operator=(Allocation&& other) noexcept {
    if (this != &other) {
      reset_noexcept();
      pool = other.pool;
      pointer = other.pointer;
      length = other.length;
      allocated = other.allocated;
      other.pointer = nullptr;
      other.length = 0;
      other.allocated = 0;
    }
    return *this;
  }
  ~Allocation() { reset_noexcept(); }

  void release_checked() {
    if (pointer == nullptr) return;
    check(hsa_amd_memory_pool_free(pointer), "hsa_amd_memory_pool_free");
    pointer = nullptr;
    length = 0;
    allocated = 0;
  }

  void reset_noexcept() noexcept {
    if (pointer != nullptr) (void)hsa_amd_memory_pool_free(pointer);
    pointer = nullptr;
    length = 0;
    allocated = 0;
  }
};

Allocation allocate(hsa_amd_memory_pool_t pool, size_t granule, size_t pool_alignment,
                    hsa_agent_t gpu, size_t length, size_t required_alignment,
                    uint32_t flags) {
  Allocation allocation;
  allocation.pool = pool;
  allocation.length = length;
  if (length == 0) return allocation;
  if (required_alignment == 0 || (required_alignment & (required_alignment - 1)) != 0)
    throw Error("allocation alignment is not a power of two");
  if (length > std::numeric_limits<size_t>::max() - (granule - 1))
    throw Error("allocation size overflows");
  allocation.allocated = ((length + granule - 1) / granule) * granule;
  check(hsa_amd_memory_pool_allocate(pool, allocation.allocated, flags, &allocation.pointer),
        "hsa_amd_memory_pool_allocate");
  if (allocation.pointer == nullptr) throw Error("HSA memory-pool allocation returned null");
  const uintptr_t address = reinterpret_cast<uintptr_t>(allocation.pointer);
  if (address % required_alignment != 0 || address % pool_alignment != 0)
    throw Error("HSA memory-pool allocation did not satisfy alignment");
  check(hsa_amd_agents_allow_access(1, &gpu, nullptr, allocation.pointer),
        "hsa_amd_agents_allow_access");
  std::memset(allocation.pointer, 0, allocation.allocated);
  return allocation;
}

struct QueueFault {
  std::atomic<uint32_t> status{0};
};

void queue_error_callback(hsa_status_t status, hsa_queue_t*, void* data) {
  if (data == nullptr) return;
  auto* fault = static_cast<QueueFault*>(data);
  uint32_t expected = 0;
  (void)fault->status.compare_exchange_strong(expected, static_cast<uint32_t>(status),
                                               std::memory_order_release,
                                               std::memory_order_relaxed);
}

struct Context {
  bool runtime_lease = false;
  hsa_agent_t gpu{0};
  hsa_agent_t cpu{0};
  std::string gfx;
  std::string pci;
  hsa_profile_t profile = HSA_PROFILE_FULL;
  hsa_default_float_rounding_mode_t rounding = HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
  hsa_amd_memory_pool_t pool{0};
  size_t pool_granule = 0;
  size_t pool_alignment = 0;
  hsa_queue_t* queue = nullptr;
  std::unique_ptr<QueueFault> fault;
  hsa_signal_t completion{0};
  uint64_t timestamp_frequency = 0;
  uint16_t hsa_version_major = 0;
  uint16_t hsa_version_minor = 0;
  std::mutex mutex;
  uint64_t generation = 1;
  uint64_t submissions = 0;
  uint64_t last_doorbell_value = 0;
  uint32_t children = 0;
  uint32_t unretired_submissions = 0;
  bool usable = true;
  bool queue_active = false;

  ~Context() noexcept {
    // Construction-failure cleanup only. Explicit destruction performs the
    // checked path first and clears these handles. If queue destruction fails,
    // leak callback/runtime ownership rather than permit a callback UAF.
    if (queue != nullptr) {
      if (queue_active) (void)hsa_queue_inactivate(queue);
      const hsa_status_t status = hsa_queue_destroy(queue);
      if (status != HSA_STATUS_SUCCESS) {
        (void)fault.release();
        runtime_lease = false;
        return;
      }
      queue = nullptr;
      queue_active = false;
    }
    if (completion.handle != 0) {
      (void)hsa_signal_destroy(completion);
      completion.handle = 0;
    }
    fault.reset();
    if (runtime_lease) {
      try {
        release_runtime();
      } catch (...) {
      }
      runtime_lease = false;
    }
  }
};

void close_context_resources(Context* context) {
  if (context->queue != nullptr) {
    if (context->queue_active) {
      check(hsa_queue_inactivate(context->queue), "hsa_queue_inactivate");
      context->queue_active = false;
    }
    check(hsa_queue_destroy(context->queue), "hsa_queue_destroy");
    context->queue = nullptr;
  }
  if (context->completion.handle != 0) {
    check(hsa_signal_destroy(context->completion), "hsa_signal_destroy");
    context->completion.handle = 0;
  }
  context->fault.reset();
  if (context->runtime_lease) {
    release_runtime();
    context->runtime_lease = false;
  }
}

struct KernelInfo {
  uint64_t kernel_object = 0;
  uint32_t kernarg_size = 0;
  uint32_t kernarg_align = 0;
  uint32_t group_size = 0;
  uint32_t private_size = 0;
  bool dynamic_stack = false;
  uint64_t code_entry = 0;
  uint32_t rsrc1 = 0;
  uint32_t rsrc2 = 0;
  uint32_t rsrc3 = 0;
  uint16_t properties = 0;
};

struct Module {
  std::vector<uint8_t> bytes;
  hsa_code_object_reader_t reader{0};
  hsa_executable_t executable{0};
  std::unordered_map<std::string, KernelInfo> kernels;

  Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&& other) noexcept { *this = std::move(other); }
  Module& operator=(Module&& other) noexcept {
    if (this != &other) {
      reset_noexcept();
      bytes = std::move(other.bytes);
      reader = other.reader;
      executable = other.executable;
      kernels = std::move(other.kernels);
      other.reader.handle = 0;
      other.executable.handle = 0;
    }
    return *this;
  }
  ~Module() { reset_noexcept(); }

  void release_checked() {
    if (executable.handle != 0) {
      check(hsa_executable_destroy(executable), "hsa_executable_destroy");
      executable.handle = 0;
    }
    if (reader.handle != 0) {
      check(hsa_code_object_reader_destroy(reader), "hsa_code_object_reader_destroy");
      reader.handle = 0;
    }
    bytes.clear();
    kernels.clear();
  }

  void reset_noexcept() noexcept {
    if (executable.handle != 0) (void)hsa_executable_destroy(executable);
    if (reader.handle != 0) (void)hsa_code_object_reader_destroy(reader);
    executable.handle = 0;
    reader.handle = 0;
  }
};

uint16_t read_u16(const std::vector<uint8_t>& bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 2) throw Error("truncated HSACO u16");
  uint16_t value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}
uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4) throw Error("truncated HSACO u32");
  uint32_t value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}
uint64_t read_u64(const std::vector<uint8_t>& bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8) throw Error("truncated HSACO u64");
  uint64_t value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

size_t checked_offset(uint64_t value, const char* label) {
  if (value > std::numeric_limits<size_t>::max()) throw Error(std::string(label) + " overflows");
  return static_cast<size_t>(value);
}

KernelInfo descriptor_metadata(const std::vector<uint8_t>& elf, const std::string& symbol_name,
                               uint64_t loaded_descriptor, KernelInfo info) {
  if (elf.size() < 64 || std::memcmp(elf.data(), "\x7f" "ELF", 4) != 0 ||
      elf[4] != 2 || elf[5] != 1)
    throw Error("code object is not little-endian ELF64");
  const size_t phoff = checked_offset(read_u64(elf, 32), "program table");
  const size_t shoff = checked_offset(read_u64(elf, 40), "section table");
  const uint16_t phentsize = read_u16(elf, 54);
  const uint16_t phnum = read_u16(elf, 56);
  const uint16_t shentsize = read_u16(elf, 58);
  const uint16_t shnum = read_u16(elf, 60);
  if (phentsize < 56 || shentsize < 64 || phnum == 0 || shnum == 0)
    throw Error("unsupported HSACO ELF table layout");
  if (phoff > elf.size() || static_cast<uint64_t>(phentsize) * phnum > elf.size() - phoff ||
      shoff > elf.size() || static_cast<uint64_t>(shentsize) * shnum > elf.size() - shoff)
    throw Error("HSACO ELF table exceeds image");

  auto va_to_offset = [&](uint64_t va) -> size_t {
    for (uint16_t index = 0; index < phnum; ++index) {
      const size_t base = phoff + static_cast<size_t>(index) * phentsize;
      if (read_u32(elf, base) != 1) continue;
      const uint64_t offset = read_u64(elf, base + 8);
      const uint64_t vaddr = read_u64(elf, base + 16);
      const uint64_t filesz = read_u64(elf, base + 32);
      if (va >= vaddr && va - vaddr < filesz) return checked_offset(offset + (va - vaddr), "VA");
    }
    throw Error("kernel descriptor VA is outside PT_LOAD segments");
  };

  bool found = false;
  uint64_t descriptor_va = 0;
  for (uint16_t section = 0; section < shnum && !found; ++section) {
    const size_t base = shoff + static_cast<size_t>(section) * shentsize;
    const uint32_t type = read_u32(elf, base + 4);
    if (type != 2 && type != 11) continue;
    const size_t symbols_offset = checked_offset(read_u64(elf, base + 24), "symbol table");
    const size_t symbols_size = checked_offset(read_u64(elf, base + 32), "symbol table size");
    const uint32_t string_section = read_u32(elf, base + 40);
    const size_t entry_size = checked_offset(read_u64(elf, base + 56), "symbol entry size");
    if (entry_size < 24 || string_section >= shnum || symbols_offset > elf.size() ||
        symbols_size > elf.size() - symbols_offset)
      continue;
    const size_t string_base = shoff + static_cast<size_t>(string_section) * shentsize;
    const size_t strings_offset = checked_offset(read_u64(elf, string_base + 24), "strings");
    const size_t strings_size = checked_offset(read_u64(elf, string_base + 32), "strings size");
    if (strings_offset > elf.size() || strings_size > elf.size() - strings_offset) continue;
    for (size_t offset = 0; offset + entry_size <= symbols_size; offset += entry_size) {
      const size_t symbol = symbols_offset + offset;
      const uint32_t name_offset = read_u32(elf, symbol);
      if (name_offset >= strings_size) continue;
      const char* name = reinterpret_cast<const char*>(elf.data() + strings_offset + name_offset);
      const size_t remaining = strings_size - name_offset;
      const void* terminator = std::memchr(name, 0, remaining);
      if (terminator == nullptr) continue;
      const size_t length = static_cast<const char*>(terminator) - name;
      if (length == symbol_name.size() && std::memcmp(name, symbol_name.data(), length) == 0) {
        descriptor_va = read_u64(elf, symbol + 8);
        found = true;
        break;
      }
    }
  }
  if (!found) throw Error("HSACO symbol table has no requested kernel descriptor");
  const size_t descriptor = va_to_offset(descriptor_va);
  if (descriptor > elf.size() || elf.size() - descriptor < 64)
    throw Error("kernel descriptor is truncated");
  const int64_t entry_offset = static_cast<int64_t>(read_u64(elf, descriptor + 16));
  if (entry_offset >= 0) {
    if (loaded_descriptor > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(entry_offset))
      throw Error("loaded code entry overflows");
    info.code_entry = loaded_descriptor + static_cast<uint64_t>(entry_offset);
  } else {
    const uint64_t magnitude = static_cast<uint64_t>(-(entry_offset + 1)) + 1;
    if (loaded_descriptor < magnitude) throw Error("loaded code entry underflows");
    info.code_entry = loaded_descriptor - magnitude;
  }
  info.rsrc3 = read_u32(elf, descriptor + 44);
  info.rsrc1 = read_u32(elf, descriptor + 48);
  info.rsrc2 = read_u32(elf, descriptor + 52);
  info.properties = read_u16(elf, descriptor + 56);
  return info;
}

Module load_module(Context* context, const uint8_t* bytes, size_t size) {
  if (bytes == nullptr || size == 0) throw Error("HSACO input is null or empty");
  Module module;
  module.bytes.assign(bytes, bytes + size);
  check(hsa_code_object_reader_create_from_memory(module.bytes.data(), module.bytes.size(),
                                                  &module.reader),
        "hsa_code_object_reader_create_from_memory");
  check(hsa_executable_create_alt(context->profile, context->rounding, nullptr,
                                  &module.executable),
        "hsa_executable_create_alt");
  hsa_loaded_code_object_t loaded{0};
  check(hsa_executable_load_agent_code_object(module.executable, context->gpu, module.reader,
                                               nullptr, &loaded),
        "hsa_executable_load_agent_code_object");
  check(hsa_executable_freeze(module.executable, nullptr), "hsa_executable_freeze");
  return module;
}

KernelInfo load_kernel(Context* context, Module* module, const std::string& name) {
  auto existing = module->kernels.find(name);
  if (existing != module->kernels.end()) return existing->second;
  hsa_executable_symbol_t symbol{0};
  check(hsa_executable_get_symbol_by_name(module->executable, name.c_str(), &context->gpu,
                                          &symbol),
        "hsa_executable_get_symbol_by_name");
  hsa_symbol_kind_t kind = HSA_SYMBOL_KIND_VARIABLE;
  check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind),
        "hsa_executable_symbol_get_info(type)");
  if (kind != HSA_SYMBOL_KIND_KERNEL) throw Error("resolved HSA symbol is not a kernel");
  KernelInfo info;
  check(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &info.kernel_object),
        "hsa_executable_symbol_get_info(kernel_object)");
  if (info.kernel_object == 0) throw Error("HSA loader returned a null kernel object");
  check(hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info.kernarg_size),
        "hsa_executable_symbol_get_info(kernarg_size)");
  check(hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
            &info.kernarg_align),
        "hsa_executable_symbol_get_info(kernarg_align)");
  check(hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &info.group_size),
        "hsa_executable_symbol_get_info(group_size)");
  check(hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &info.private_size),
        "hsa_executable_symbol_get_info(private_size)");
  check(hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &info.dynamic_stack),
        "hsa_executable_symbol_get_info(dynamic_stack)");
  info = descriptor_metadata(module->bytes, name, info.kernel_object, info);
  module->kernels.emplace(name, info);
  return info;
}

uint32_t packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
  if (body_dwords == 0) throw Error("PACKET3 body is empty");
  return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
         (compute ? (1u << 1) : 0);
}

void append_acquire_system(std::vector<uint32_t>* words) {
  const uint32_t gcr = (1u << 16) | (1u << 15) | (1u << 14) | (1u << 9) |
                       (1u << 8) | (1u << 7) | (1u << 6) | (1u << 5) |
                       (1u << 4) | 1u;
  const uint32_t values[] = {packet3(kPacket3AcquireMem, 7, false), 0, 0xffffffff,
                             0xff, 0, 0, 4, gcr};
  words->insert(words->end(), std::begin(values), std::end(values));
}

void append_wait_compute_idle(std::vector<uint32_t>* words) {
  words->push_back(packet3(kPacket3EventWrite, 1, false));
  words->push_back(0x407);
}

void append_dependency_global(std::vector<uint32_t>* words) {
  append_wait_compute_idle(words);
  const uint32_t values[] = {packet3(kPacket3AcquireMem, 7, false), 0, 0xffffffff,
                             0x00ffffff, 0, 0, 10, 0x0c380};
  words->insert(words->end(), std::begin(values), std::end(values));
}

using RegisterState = std::unordered_map<uint32_t, uint32_t>;

void emit_sh_regs(std::vector<uint32_t>* words, uint32_t first,
                  const uint32_t* values, size_t count) {
  if (count == 0) return;
  words->push_back(packet3(kPacket3SetShReg, static_cast<uint32_t>(count) + 1, true));
  words->push_back(first);
  words->insert(words->end(), values, values + count);
}

void set_sh_regs_vector(std::vector<uint32_t>* words, uint32_t first,
                        const std::vector<uint32_t>& values, RegisterState* state) {
  if (values.empty()) return;
  if (state == nullptr) {
    emit_sh_regs(words, first, values.data(), values.size());
    return;
  }
  size_t run_start = 0;
  size_t run_count = 0;
  const auto flush = [&] {
    if (run_count != 0)
      emit_sh_regs(words, first + static_cast<uint32_t>(run_start),
                   values.data() + run_start, run_count);
    run_count = 0;
  };
  for (size_t offset = 0; offset < values.size(); ++offset) {
    const uint32_t reg = first + static_cast<uint32_t>(offset);
    const uint32_t value = values[offset];
    const auto existing = state->find(reg);
    if (existing != state->end() && existing->second == value) {
      flush();
      continue;
    }
    (*state)[reg] = value;
    if (run_count == 0) run_start = offset;
    ++run_count;
  }
  flush();
}

void set_sh_regs(std::vector<uint32_t>* words, uint32_t first,
                 std::initializer_list<uint32_t> values, RegisterState* state) {
  set_sh_regs_vector(words, first, std::vector<uint32_t>(values), state);
}

struct Dispatch {
  KernelInfo kernel;
  Allocation kernarg;
  std::string symbol;
  size_t module_index = 0;
  std::array<uint32_t, 3> grid{};
  std::array<uint32_t, 3> block{};
  uint32_t dynamic_lds = 0;
};

void append_dispatch(std::vector<uint32_t>* words, const Dispatch& dispatch,
                     RegisterState* state) {
  const KernelInfo& image = dispatch.kernel;
  if (image.private_size != 0 || image.dynamic_stack)
    throw Error("gfx1100 retained PM4 does not support scratch or dynamic call stacks");
  const uint16_t unsupported = image.properties & ~kSupportedProperties;
  if (unsupported != 0) throw Error("gfx1100 kernel uses unsupported implicit SGPR properties");
  if ((image.properties & kEnableWave32) == 0)
    throw Error("initial gfx1100 retained PM4 path requires wave32");
  if (image.code_entry == 0 || (image.code_entry & 0xff) != 0)
    throw Error("gfx1100 loaded code entry is not 256-byte aligned");
  if (dispatch.dynamic_lds > std::numeric_limits<uint32_t>::max() - image.group_size)
    throw Error("group-segment size overflows");
  const uint32_t group = image.group_size + dispatch.dynamic_lds;
  const uint32_t lds_blocks = (group + kLdsGranule - 1) / kLdsGranule;
  if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift))
    throw Error("group-segment size cannot be encoded");
  const uint32_t rsrc2 = (image.rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);
  std::array<uint32_t, 3> workgroups{};
  for (size_t axis = 0; axis < 3; ++axis) {
    if (dispatch.block[axis] == 0 || dispatch.grid[axis] == 0 ||
        dispatch.grid[axis] % dispatch.block[axis] != 0)
      throw Error("gfx1100 direct dispatch requires integral workgroups");
    workgroups[axis] = dispatch.grid[axis] / dispatch.block[axis];
  }
  std::vector<uint32_t> user_sgprs;
  if ((image.properties & kEnablePrivateSegmentBuffer) != 0)
    user_sgprs.insert(user_sgprs.end(), {0, 0, 0, 0});
  if ((image.properties & kEnableKernargPtr) != 0) {
    if (dispatch.kernarg.pointer == nullptr) throw Error("kernel requires non-null kernarg");
    const uint64_t address = reinterpret_cast<uintptr_t>(dispatch.kernarg.pointer);
    user_sgprs.push_back(static_cast<uint32_t>(address));
    user_sgprs.push_back(static_cast<uint32_t>(address >> 32));
  }
  if (user_sgprs.size() > 16) throw Error("kernel requires too many user SGPRs");

  set_sh_regs(words, kComputePgmLo,
              {static_cast<uint32_t>(image.code_entry >> 8),
               static_cast<uint32_t>(image.code_entry >> 40)}, state);
  set_sh_regs(words, kComputePgmRsrc1, {image.rsrc1, rsrc2}, state);
  set_sh_regs(words, kComputePgmRsrc3, {image.rsrc3}, state);
  set_sh_regs(words, kComputeTmpRingSize, {0}, state);
  set_sh_regs(words, kComputeNumThreadX,
              {dispatch.block[0], dispatch.block[1], dispatch.block[2]}, state);
  set_sh_regs(words, kComputeResourceLimits, {0}, state);
  set_sh_regs_vector(words, kComputeUserData0, user_sgprs, state);
  const uint32_t initiator = (1u << 0) | (1u << 2) | (1u << 3) | (1u << 15);
  words->push_back(packet3(kPacket3DispatchDirect, 4, true));
  words->insert(words->end(), workgroups.begin(), workgroups.end());
  words->push_back(initiator);
}

std::array<uint8_t, kPacketBytes> vendor_packet(void* address, uint32_t dwords,
                                                hsa_signal_t completion) {
  if (address == nullptr || (reinterpret_cast<uintptr_t>(address) & 3) != 0)
    throw Error("PM4 IB address is null or unaligned");
  if (dwords == 0 || dwords > 0x000fffff) throw Error("PM4 IB dword count is invalid");
  std::array<uint8_t, kPacketBytes> bytes{};
  const uint16_t aql_header = 1u << HSA_PACKET_HEADER_BARRIER;
  const uint16_t setup = 1;
  const uint32_t pm4_header = packet3(kPacket3IndirectBuffer, 3, false);
  const uint64_t ib = reinterpret_cast<uintptr_t>(address);
  const uint32_t low = static_cast<uint32_t>(ib) & 0xfffffffc;
  const uint32_t high = static_cast<uint32_t>(ib >> 32);
  const uint32_t control = dwords | (1u << 23) | (3u << 28);
  const uint32_t valid = 10;
  std::memcpy(bytes.data(), &aql_header, sizeof(aql_header));
  std::memcpy(bytes.data() + 2, &setup, sizeof(setup));
  std::memcpy(bytes.data() + 4, &pm4_header, sizeof(pm4_header));
  std::memcpy(bytes.data() + 8, &low, sizeof(low));
  std::memcpy(bytes.data() + 12, &high, sizeof(high));
  std::memcpy(bytes.data() + 16, &control, sizeof(control));
  std::memcpy(bytes.data() + 20, &valid, sizeof(valid));
  std::memcpy(bytes.data() + 56, &completion.handle, sizeof(completion.handle));
  return bytes;
}

uint16_t kernel_header() {
  return (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
         (1u << HSA_PACKET_HEADER_BARRIER) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
}

struct Executable {
  Context* context = nullptr;
  std::vector<Module> modules;
  std::vector<Dispatch> dispatches;
  std::vector<hsa_kernel_dispatch_packet_t> aql_packets;
  std::vector<uint32_t> pm4_words;
  Allocation indirect;
  Allocation timestamps;
  std::array<uint8_t, kPacketBytes> pm4_packet{};
  uint64_t generation = 0;
  uint64_t aql_submissions = 0;
  uint64_t pm4_submissions = 0;
  uint64_t last_packet_id = 0;
  uint64_t last_packet_count = 0;
  uint64_t last_timeout_ns = 0;
  uint64_t module_load_ns = 0;
  uint64_t kernel_resolve_ns = 0;
  uint64_t kernarg_allocate_ns = 0;
  uint64_t aql_packet_build_ns = 0;
  uint64_t pm4_encode_ns = 0;
  uint64_t ib_allocate_ns = 0;
  hsa_signal_value_t last_completion_value = 1;
  std::string last_transport = "none";
  bool stateful_registers = false;
  bool usable = true;
  bool in_flight = false;
  bool retired = true;
};

void append_copy_timestamp(std::vector<uint32_t>* words, uint64_t address) {
  constexpr uint32_t control = 9u | (5u << 8) | (1u << 16) | (1u << 20);
  const uint32_t values[] = {packet3(kPacket3CopyData, 5, false), control, 0, 0,
                             static_cast<uint32_t>(address),
                             static_cast<uint32_t>(address >> 32)};
  words->insert(words->end(), std::begin(values), std::end(values));
}

void append_release_timestamp(std::vector<uint32_t>* words, uint64_t address) {
  constexpr uint32_t event = 40u | (5u << 8);
  constexpr uint32_t control = (3u << 24) | (3u << 29);
  const uint32_t values[] = {packet3(kPacket3ReleaseMem, 7, false), event, control,
                             static_cast<uint32_t>(address),
                             static_cast<uint32_t>(address >> 32), 0, 0, 0};
  words->insert(words->end(), std::begin(values), std::end(values));
}

uint64_t timeout_ticks(Context* context, uint64_t timeout_ns) {
  if (timeout_ns == 0) throw Error("submission timeout must be positive");
  const unsigned __int128 product =
      static_cast<unsigned __int128>(timeout_ns) * context->timestamp_frequency;
  const unsigned __int128 ticks = (product + kNanosPerSecond - 1) / kNanosPerSecond;
  return ticks > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(ticks);
}

uint64_t reserve_packets(Context* context, size_t count, uint64_t timeout_ns) {
  if (count == 0 || count > context->queue->size)
    throw Error("submission packet count exceeds persistent queue capacity");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
  while (true) {
    if (context->fault->status.load(std::memory_order_acquire) != 0)
      throw Error("HSA queue callback reported a fault before publication");
    const uint64_t write = hsa_queue_load_write_index_relaxed(context->queue);
    const uint64_t read = hsa_queue_load_read_index_scacquire(context->queue);
    if (write - read + count <= context->queue->size)
      return hsa_queue_add_write_index_relaxed(context->queue, count);
    if (std::chrono::steady_clock::now() >= deadline)
      throw Error("timed out waiting for HSA queue capacity");
  }
}

void publish_packet(Context* context, uint64_t packet_id, const uint8_t* bytes,
                    uint32_t publication) {
  const size_t slot = static_cast<size_t>(packet_id & (context->queue->size - 1));
  auto* destination = static_cast<uint8_t*>(context->queue->base_address) + slot * kPacketBytes;
  std::memcpy(destination + 4, bytes + 4, kPacketBytes - 4);
  __atomic_store_n(reinterpret_cast<uint32_t*>(destination), publication, __ATOMIC_RELEASE);
}

void wait_completion(Executable* executable, uint64_t timeout_ns) {
  Context* context = executable->context;
  const hsa_signal_value_t observed = hsa_signal_wait_scacquire(
      context->completion, HSA_SIGNAL_CONDITION_LT, 1, timeout_ticks(context, timeout_ns),
      HSA_WAIT_STATE_BLOCKED);
  executable->last_completion_value = observed;
  const uint32_t fault = context->fault->status.load(std::memory_order_acquire);
  if (observed >= 1 || fault != 0) {
    executable->usable = false;
    executable->retired = false;
    context->usable = false;
    if (context->queue_active) {
      const hsa_status_t status = hsa_queue_inactivate(context->queue);
      if (status == HSA_STATUS_SUCCESS) context->queue_active = false;
    }
    std::ostringstream error;
    error << "HSA submission did not prove completion (signal=" << observed
          << ", callback_status=" << fault << ")";
    throw Error(error.str());
  }
  executable->in_flight = false;
  executable->retired = true;
  if (context->unretired_submissions == 0)
    throw Error("PM4 context retirement ledger underflow");
  --context->unretired_submissions;
  ++context->submissions;
}

void begin_submission(Executable* executable) {
  Context* context = executable->context;
  if (!context->usable || !context->queue_active || !executable->usable)
    throw Error("PM4 context or executable is inactive");
  if (executable->in_flight || !executable->retired)
    throw Error("previous submission has not retired");
  if (context->fault->status.load(std::memory_order_acquire) != 0)
    throw Error("HSA queue callback has recorded a fault");
  hsa_signal_store_screlease(context->completion, 1);
  executable->in_flight = true;
  executable->retired = false;
  ++context->unretired_submissions;
}

void quarantine_failed_submission(Executable* executable) noexcept {
  Context* context = executable->context;
  executable->usable = false;
  executable->retired = false;
  context->usable = false;
  if (context->queue_active) {
    const hsa_status_t status = hsa_queue_inactivate(context->queue);
    if (status == HSA_STATUS_SUCCESS) context->queue_active = false;
  }
}

void launch_aql(Executable* executable, uint64_t timeout_ns) {
  std::lock_guard<std::mutex> lock(executable->context->mutex);
  executable->last_transport = "aql";
  executable->last_timeout_ns = timeout_ns;
  begin_submission(executable);
  Context* context = executable->context;
  try {
    const uint64_t first = reserve_packets(context, executable->aql_packets.size(), timeout_ns);
    executable->last_packet_id = first;
    executable->last_packet_count = executable->aql_packets.size();
    for (size_t index = 0; index < executable->aql_packets.size(); ++index) {
      hsa_kernel_dispatch_packet_t packet = executable->aql_packets[index];
      packet.completion_signal =
          index + 1 == executable->aql_packets.size() ? context->completion : hsa_signal_t{0};
      const uint32_t publication = packet.full_header;
      std::array<uint8_t, kPacketBytes> bytes{};
      std::memcpy(bytes.data(), &packet, sizeof(packet));
      publish_packet(context, first + index, bytes.data(), publication);
    }
    context->last_doorbell_value = first + executable->aql_packets.size() - 1;
    hsa_signal_store_relaxed(
        context->queue->doorbell_signal,
        static_cast<hsa_signal_value_t>(context->last_doorbell_value));
    wait_completion(executable, timeout_ns);
    ++executable->aql_submissions;
  } catch (...) {
    if (!executable->retired) quarantine_failed_submission(executable);
    throw;
  }
}

void launch_pm4(Executable* executable, uint64_t timeout_ns) {
  std::lock_guard<std::mutex> lock(executable->context->mutex);
  executable->last_transport = "pm4";
  executable->last_timeout_ns = timeout_ns;
  begin_submission(executable);
  Context* context = executable->context;
  try {
    const uint64_t packet_id = reserve_packets(context, 1, timeout_ns);
    executable->last_packet_id = packet_id;
    executable->last_packet_count = 1;
    uint32_t publication = 0;
    std::memcpy(&publication, executable->pm4_packet.data(), sizeof(publication));
    publish_packet(context, packet_id, executable->pm4_packet.data(), publication);
    context->last_doorbell_value = packet_id;
    hsa_signal_store_relaxed(context->queue->doorbell_signal,
                             static_cast<hsa_signal_value_t>(context->last_doorbell_value));
    wait_completion(executable, timeout_ns);
    ++executable->pm4_submissions;
  } catch (...) {
    if (!executable->retired) quarantine_failed_submission(executable);
    throw;
  }
}

std::string context_json(Context* context) {
  std::lock_guard<std::mutex> lock(context->mutex);
  const uint64_t read = context->queue == nullptr ? 0 : hsa_queue_load_read_index_relaxed(context->queue);
  const uint64_t write = context->queue == nullptr ? 0 : hsa_queue_load_write_index_relaxed(context->queue);
  const hsa_signal_value_t completion_value =
      context->completion.handle == 0 ? 0 : hsa_signal_load_scacquire(context->completion);
  const hsa_signal_value_t doorbell_value =
      context->queue == nullptr ? 0 : hsa_signal_load_relaxed(context->queue->doorbell_signal);
  std::ostringstream out;
  out << "{\"abi_version\":" << kAbiVersion << ",\"process_id\":"
      << static_cast<uint64_t>(getpid()) << ",\"hsa_version_major\":"
      << context->hsa_version_major << ",\"hsa_version_minor\":"
      << context->hsa_version_minor << ",\"gfx\":\"" << context->gfx
      << "\",\"pci_bdf\":\"" << context->pci << "\",\"agent_handle\":"
      << context->gpu.handle << ",\"queue_id\":"
      << (context->queue == nullptr ? 0 : context->queue->id) << ",\"queue_size\":"
      << (context->queue == nullptr ? 0 : context->queue->size) << ",\"queue_base\":"
      << (context->queue == nullptr ? 0 : reinterpret_cast<uintptr_t>(context->queue->base_address))
      << ",\"queue_type\":\"multi\",\"doorbell_handle\":"
      << (context->queue == nullptr ? 0 : context->queue->doorbell_signal.handle)
      << ",\"doorbell_value\":" << doorbell_value
      << ",\"last_doorbell_value\":" << context->last_doorbell_value
      << ",\"read_index\":" << read << ",\"write_index\":" << write
      << ",\"completion_handle\":" << context->completion.handle
      << ",\"completion_value\":" << completion_value
      << ",\"generation\":" << context->generation << ",\"submissions\":"
      << context->submissions << ",\"children\":" << context->children
      << ",\"unretired_submissions\":" << context->unretired_submissions
      << ",\"callback_status\":"
      << (context->fault == nullptr ? 0 : context->fault->status.load(std::memory_order_acquire))
      << ",\"usable\":" << (context->usable ? "true" : "false") << "}";
  return out.str();
}

void append_json_string(std::ostringstream* out, const std::string& value) {
  *out << '"';
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': *out << "\\\""; break;
      case '\\': *out << "\\\\"; break;
      case '\b': *out << "\\b"; break;
      case '\f': *out << "\\f"; break;
      case '\n': *out << "\\n"; break;
      case '\r': *out << "\\r"; break;
      case '\t': *out << "\\t"; break;
      default:
        if (byte < 0x20) {
          char escaped[7] = {};
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
          *out << escaped;
        } else {
          *out << static_cast<char>(byte);
        }
    }
  }
  *out << '"';
}

std::string executable_json(Executable* executable) {
  std::lock_guard<std::mutex> lock(executable->context->mutex);
  uint32_t pm4_publication = 0;
  std::memcpy(&pm4_publication, executable->pm4_packet.data(), sizeof(pm4_publication));
  const uint32_t aql_publication = executable->aql_packets.empty()
                                       ? 0
                                       : executable->aql_packets.front().full_header;
  std::ostringstream out;
  out << "{\"abi_version\":" << kAbiVersion << ",\"generation\":"
      << executable->generation << ",\"nodes\":" << executable->dispatches.size()
      << ",\"modules\":" << executable->modules.size() << ",\"pm4_dwords\":"
      << executable->pm4_words.size() << ",\"ib_address\":"
      << reinterpret_cast<uintptr_t>(executable->indirect.pointer) << ",\"ib_bytes\":"
      << executable->indirect.length << ",\"aql_publication\":" << aql_publication
      << ",\"pm4_publication\":" << pm4_publication << ",\"timestamp_address\":"
      << reinterpret_cast<uintptr_t>(executable->timestamps.pointer)
      << ",\"timestamp_bytes\":" << executable->timestamps.length
      << ",\"stateful_registers\":"
      << (executable->stateful_registers ? "true" : "false")
      << ",\"aql_submissions\":" << executable->aql_submissions
      << ",\"pm4_submissions\":" << executable->pm4_submissions
      << ",\"last_packet_id\":" << executable->last_packet_id
      << ",\"last_packet_count\":" << executable->last_packet_count
      << ",\"last_timeout_ns\":" << executable->last_timeout_ns
      << ",\"module_load_ns\":" << executable->module_load_ns
      << ",\"kernel_resolve_ns\":" << executable->kernel_resolve_ns
      << ",\"kernarg_allocate_ns\":" << executable->kernarg_allocate_ns
      << ",\"aql_packet_build_ns\":" << executable->aql_packet_build_ns
      << ",\"pm4_encode_ns\":" << executable->pm4_encode_ns
      << ",\"ib_allocate_ns\":" << executable->ib_allocate_ns
      << ",\"last_completion_value\":" << executable->last_completion_value
      << ",\"last_transport\":";
  append_json_string(&out, executable->last_transport);
  out << ",\"in_flight\":" << (executable->in_flight ? "true" : "false")
      << ",\"retired\":" << (executable->retired ? "true" : "false")
      << ",\"usable\":" << (executable->usable ? "true" : "false")
      << ",\"module_records\":[";
  for (size_t index = 0; index < executable->modules.size(); ++index) {
    if (index != 0) out << ',';
    const Module& module = executable->modules[index];
    out << "{\"index\":" << index << ",\"reader_handle\":" << module.reader.handle
        << ",\"executable_handle\":" << module.executable.handle
        << ",\"hsaco_bytes\":" << module.bytes.size() << '}';
  }
  out << "],\"dispatch_records\":[";
  for (size_t index = 0; index < executable->dispatches.size(); ++index) {
    if (index != 0) out << ',';
    const Dispatch& dispatch = executable->dispatches[index];
    out << "{\"index\":" << index << ",\"module_index\":" << dispatch.module_index
        << ",\"symbol\":";
    append_json_string(&out, dispatch.symbol);
    out << ",\"kernel_object\":" << dispatch.kernel.kernel_object
        << ",\"code_entry\":" << dispatch.kernel.code_entry
        << ",\"kernarg_address\":"
        << reinterpret_cast<uintptr_t>(dispatch.kernarg.pointer)
        << ",\"kernarg_bytes\":" << dispatch.kernarg.length
        << ",\"kernarg_allocated_bytes\":" << dispatch.kernarg.allocated
        << ",\"kernarg_align\":" << dispatch.kernel.kernarg_align
        << ",\"group_segment_bytes\":"
        << dispatch.kernel.group_size + dispatch.dynamic_lds
        << ",\"private_segment_bytes\":" << dispatch.kernel.private_size
        << ",\"grid\":[" << dispatch.grid[0] << ',' << dispatch.grid[1] << ','
        << dispatch.grid[2] << "],\"block\":[" << dispatch.block[0] << ','
        << dispatch.block[1] << ',' << dispatch.block[2] << "]}";
  }
  out << "]}";
  return out.str();
}

void copy_json(const std::string& value, char* output, size_t output_size, size_t* required) {
  if (required != nullptr) *required = value.size() + 1;
  if (output == nullptr || output_size == 0) return;
  if (output_size <= value.size()) throw Error("JSON output buffer is too small");
  std::memcpy(output, value.data(), value.size());
  output[value.size()] = '\0';
}

}  // namespace

extern "C" {

struct he_pm4_context {
  Context value;
};

struct he_pm4_executable {
  Executable value;
};

struct he_pm4_buffer {
  Context* context = nullptr;
  Allocation allocation;
  uint64_t generation = 0;
};

struct he_pm4_node {
  const uint8_t* hsaco;
  size_t hsaco_size;
  const char* symbol;
  const uint8_t* kernarg;
  uint32_t kernarg_size;
  uint32_t kernarg_align;
  uint32_t grid[3];
  uint32_t block[3];
  uint32_t dynamic_lds;
  uint32_t expected_group_segment_size;
  uint32_t expected_private_segment_size;
  uint32_t expected_dynamic_stack;
  uint32_t expected_wavefront_size;
};

uint32_t he_pm4_native_abi_version() { return kAbiVersion; }
size_t he_pm4_node_size() { return sizeof(he_pm4_node); }

int he_pm4_context_create(const char* pci_bdf, const char* gfx_target,
                          he_pm4_context** output, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (output == nullptr) throw Error("context output pointer is null");
    *output = nullptr;
    if (gfx_target == nullptr || std::string(gfx_target) != "gfx1100")
      throw Error("native PM4 core currently admits exact gfx1100 only");
    std::unique_ptr<he_pm4_context> owner(new he_pm4_context());
    Context* context = &owner->value;
    acquire_runtime();
    context->runtime_lease = true;
    context->gfx = gfx_target;
    const PciAddress wanted = parse_pci(pci_bdf);
    context->pci = format_pci(wanted);
    AgentSearch search{wanted, context->gfx};
    check(hsa_iterate_agents(find_agent, &search), "hsa_iterate_agents");
    if (search.error != HSA_STATUS_SUCCESS) throw Error(hsa_error("agent selection", search.error));
    if (search.matches != 1 || search.gpu.handle == 0)
      throw Error("physical PCI BDF did not resolve exactly one matching HSA GPU agent");
    if (search.cpu.handle == 0) throw Error("HSA runtime exposed no CPU agent for kernarg memory");
    context->gpu = search.gpu;
    context->cpu = search.cpu;
    check(hsa_agent_get_info(context->gpu, HSA_AGENT_INFO_PROFILE, &context->profile),
          "hsa_agent_get_info(profile)");
    check(hsa_agent_get_info(context->gpu, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                             &context->rounding),
          "hsa_agent_get_info(rounding)");
    uint32_t queue_min = 0;
    uint32_t queue_max = 0;
    check(hsa_agent_get_info(context->gpu, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &queue_min),
          "hsa_agent_get_info(queue_min)");
    check(hsa_agent_get_info(context->gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max),
          "hsa_agent_get_info(queue_max)");
    uint32_t queue_size = std::max(queue_min, 4096u);
    queue_size = std::min(queue_size, queue_max);
    if (queue_size < queue_min || queue_size == 0 || (queue_size & (queue_size - 1)) != 0)
      throw Error("HSA agent has no usable power-of-two queue size");
    context->fault = std::make_unique<QueueFault>();
    check(hsa_queue_create(context->gpu, queue_size, HSA_QUEUE_TYPE_MULTI, queue_error_callback,
                           context->fault.get(), std::numeric_limits<uint32_t>::max(),
                           std::numeric_limits<uint32_t>::max(), &context->queue),
          "hsa_queue_create");
    if (context->queue == nullptr || context->queue->base_address == nullptr ||
        context->queue->size != queue_size ||
        (context->queue->features & HSA_QUEUE_FEATURE_KERNEL_DISPATCH) == 0)
      throw Error("HSA queue descriptor failed validation");
    context->queue_active = true;
    check(hsa_signal_create(1, 1, &context->gpu, &context->completion), "hsa_signal_create");
    check(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                              &context->timestamp_frequency),
          "hsa_system_get_info(timestamp_frequency)");
    check(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR,
                              &context->hsa_version_major),
          "hsa_system_get_info(version_major)");
    check(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR,
                              &context->hsa_version_minor),
          "hsa_system_get_info(version_minor)");
    PoolSearch pools;
    check(hsa_amd_agent_iterate_memory_pools(context->cpu, find_kernarg_pool, &pools),
          "hsa_amd_agent_iterate_memory_pools");
    if (pools.error != HSA_STATUS_SUCCESS)
      throw Error(hsa_error("memory-pool selection", pools.error));
    if (pools.pool.handle == 0) throw Error("no fine-grained kernarg-init HSA memory pool");
    context->pool = pools.pool;
    context->pool_granule = pools.granule;
    context->pool_alignment = pools.alignment;
    *output = owner.release();
  });
}

int he_pm4_context_retire_queue(he_pm4_context* opaque, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) throw Error("PM4 context is null");
    Context* context = &opaque->value;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->unretired_submissions != 0)
      throw Error("cannot retire HSA queue with unretired submissions");
    if (context->queue == nullptr) return;
    if (context->queue_active) {
      check(hsa_queue_inactivate(context->queue), "hsa_queue_inactivate(retire)");
      context->queue_active = false;
    }
    check(hsa_queue_destroy(context->queue), "hsa_queue_destroy(retire)");
    context->queue = nullptr;
    context->usable = false;
  });
}

int he_pm4_context_destroy(he_pm4_context* opaque, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) return;
    Context* context = &opaque->value;
    {
      std::lock_guard<std::mutex> lock(context->mutex);
      if (context->children != 0)
        throw Error("cannot destroy PM4 context with live executables or buffers");
      close_context_resources(context);
    }
    delete opaque;
  });
}

int he_pm4_buffer_create(he_pm4_context* context_opaque, size_t size,
                         he_pm4_buffer** output, uint64_t* address,
                         char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (context_opaque == nullptr || output == nullptr || address == nullptr)
      throw Error("null HSA buffer creation input");
    *output = nullptr;
    *address = 0;
    if (size == 0 || size > (1ull << 34)) throw Error("HSA buffer size is invalid");
    Context* context = &context_opaque->value;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (!context->usable || !context->queue_active) throw Error("PM4 context is inactive");
    std::unique_ptr<he_pm4_buffer> owner(new he_pm4_buffer());
    owner->context = context;
    owner->generation = ++context->generation;
    owner->allocation = allocate(context->pool, context->pool_granule,
                                 context->pool_alignment, context->gpu, size, 16,
                                 HSA_AMD_MEMORY_POOL_STANDARD_FLAG);
    *address = reinterpret_cast<uintptr_t>(owner->allocation.pointer);
    ++context->children;
    *output = owner.release();
  });
}

int he_pm4_buffer_write(he_pm4_buffer* opaque, size_t offset, const void* source,
                        size_t size, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr || (size != 0 && source == nullptr))
      throw Error("null HSA buffer write input");
    Context* context = opaque->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (!context->usable || !context->queue_active)
      throw Error("cannot write HSA buffer after queue retirement");
    if (context->unretired_submissions != 0)
      throw Error("cannot write HSA buffer with unretired submissions");
    if (offset > opaque->allocation.length || size > opaque->allocation.length - offset)
      throw Error("HSA buffer write range exceeds allocation");
    if (size != 0)
      std::memcpy(static_cast<uint8_t*>(opaque->allocation.pointer) + offset, source, size);
  });
}

int he_pm4_buffer_read(he_pm4_buffer* opaque, size_t offset, void* destination,
                       size_t size, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr || (size != 0 && destination == nullptr))
      throw Error("null HSA buffer read input");
    Context* context = opaque->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->unretired_submissions != 0)
      throw Error("cannot read HSA buffer with unretired submissions");
    if (offset > opaque->allocation.length || size > opaque->allocation.length - offset)
      throw Error("HSA buffer read range exceeds allocation");
    if (size != 0)
      std::memcpy(destination, static_cast<uint8_t*>(opaque->allocation.pointer) + offset, size);
  });
}

int he_pm4_buffer_destroy(he_pm4_buffer* opaque, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) return;
    Context* context = opaque->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->unretired_submissions != 0)
      throw Error("cannot free HSA buffer with unretired submissions");
    if (context->children == 0) throw Error("PM4 buffer/context child ledger underflow");
    opaque->allocation.release_checked();
    --context->children;
    delete opaque;
  });
}

int he_pm4_executable_create_ex(he_pm4_context* context_opaque, const he_pm4_node* input,
                                size_t count, uint32_t flags,
                                he_pm4_executable** output, char* error,
                                size_t error_size) {
  return guarded(error, error_size, [&] {
    if (context_opaque == nullptr || output == nullptr) throw Error("null executable input");
    *output = nullptr;
    if (flags & ~(kExecutableFlagTimestamps | kExecutableFlagStatefulRegisters))
      throw Error("unsupported PM4 executable creation flags");
    if (input == nullptr || count == 0) throw Error("cannot instantiate an empty graph");
    Context* context = &context_opaque->value;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->queue == nullptr || count > context->queue->size)
      throw Error("direct-AQL graph exceeds persistent queue capacity");
    if (!context->usable || !context->queue_active) throw Error("PM4 context is inactive");
    std::unique_ptr<he_pm4_executable> owner(new he_pm4_executable());
    Executable* executable = &owner->value;
    executable->context = context;
    executable->generation = ++context->generation;
    executable->stateful_registers = (flags & kExecutableFlagStatefulRegisters) != 0;
    executable->modules.reserve(count);
    executable->dispatches.reserve(count);

    for (size_t index = 0; index < count; ++index) {
      const he_pm4_node& source = input[index];
      if (source.hsaco == nullptr || source.hsaco_size == 0 || source.symbol == nullptr ||
          source.symbol[0] == '\0')
        throw Error("graph node has missing HSACO or symbol");
      size_t module_index = executable->modules.size();
      for (size_t candidate = 0; candidate < executable->modules.size(); ++candidate) {
        const auto& bytes = executable->modules[candidate].bytes;
        if (bytes.size() == source.hsaco_size &&
            std::memcmp(bytes.data(), source.hsaco, source.hsaco_size) == 0) {
          module_index = candidate;
          break;
        }
      }
      if (module_index == executable->modules.size()) {
        const auto module_load_start = SteadyClock::now();
        executable->modules.push_back(load_module(context, source.hsaco, source.hsaco_size));
        executable->module_load_ns += elapsed_ns(module_load_start);
      }
      Module* module = &executable->modules[module_index];
      const auto kernel_resolve_start = SteadyClock::now();
      KernelInfo kernel = load_kernel(context, module, source.symbol);
      executable->kernel_resolve_ns += elapsed_ns(kernel_resolve_start);
      const bool loader_alignment_valid =
          kernel.kernarg_align != 0 &&
          (kernel.kernarg_align & (kernel.kernarg_align - 1)) == 0;
      const bool alignment_compatible =
          loader_alignment_valid && source.kernarg_align != 0 &&
          kernel.kernarg_align >= source.kernarg_align &&
          kernel.kernarg_align % source.kernarg_align == 0;
      if (kernel.kernarg_size != source.kernarg_size || !alignment_compatible ||
          kernel.group_size != source.expected_group_segment_size ||
          kernel.private_size != source.expected_private_segment_size ||
          kernel.dynamic_stack != (source.expected_dynamic_stack != 0)) {
        std::ostringstream mismatch;
        mismatch << "public HSA kernel metadata disagrees with inspected HIP metadata"
                 << " (kernarg_size=" << kernel.kernarg_size << "/" << source.kernarg_size
                 << ", align=" << kernel.kernarg_align << "/" << source.kernarg_align
                 << ", group=" << kernel.group_size << "/"
                 << source.expected_group_segment_size << ", private=" << kernel.private_size
                 << "/" << source.expected_private_segment_size << ", dynamic_stack="
                 << kernel.dynamic_stack << "/" << (source.expected_dynamic_stack != 0) << ")";
        throw Error(mismatch.str());
      }
      if (source.expected_wavefront_size != 32 || (kernel.properties & kEnableWave32) == 0)
        throw Error("graph node is not an admitted wave32 kernel");
      if (source.kernarg_size != 0 && source.kernarg == nullptr)
        throw Error("graph node kernarg bytes are null");
      Dispatch dispatch;
      dispatch.kernel = kernel;
      dispatch.symbol = source.symbol;
      dispatch.module_index = module_index;
      const auto kernarg_allocate_start = SteadyClock::now();
      dispatch.kernarg = allocate(
          context->pool, context->pool_granule, context->pool_alignment,
          context->gpu, source.kernarg_size,
          std::max<size_t>({16, source.kernarg_align, kernel.kernarg_align}),
          HSA_AMD_MEMORY_POOL_STANDARD_FLAG);
      if (source.kernarg_size != 0)
        std::memcpy(dispatch.kernarg.pointer, source.kernarg, source.kernarg_size);
      executable->kernarg_allocate_ns += elapsed_ns(kernarg_allocate_start);
      for (size_t axis = 0; axis < 3; ++axis) {
        if (source.grid[axis] == 0 || source.block[axis] == 0 || source.block[axis] > 0xffff)
          throw Error("graph node has invalid launch geometry");
        dispatch.grid[axis] = source.grid[axis];
        dispatch.block[axis] = source.block[axis];
      }
      dispatch.dynamic_lds = source.dynamic_lds;
      executable->dispatches.push_back(std::move(dispatch));
    }

    const auto aql_packet_build_start = SteadyClock::now();
    executable->aql_packets.resize(count);
    const uint16_t header = kernel_header();
    for (size_t index = 0; index < count; ++index) {
      const Dispatch& dispatch = executable->dispatches[index];
      hsa_kernel_dispatch_packet_t packet{};
      packet.header = header;
      const uint16_t dimensions = dispatch.grid[2] > 1 ? 3 : (dispatch.grid[1] > 1 ? 2 : 1);
      packet.setup = dimensions << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
      packet.workgroup_size_x = static_cast<uint16_t>(dispatch.block[0]);
      packet.workgroup_size_y = static_cast<uint16_t>(dispatch.block[1]);
      packet.workgroup_size_z = static_cast<uint16_t>(dispatch.block[2]);
      packet.grid_size_x = dispatch.grid[0];
      packet.grid_size_y = dispatch.grid[1];
      packet.grid_size_z = dispatch.grid[2];
      packet.private_segment_size = dispatch.kernel.private_size;
      packet.group_segment_size = dispatch.kernel.group_size + dispatch.dynamic_lds;
      packet.kernel_object = dispatch.kernel.kernel_object;
      packet.kernarg_address = dispatch.kernarg.pointer;
      executable->aql_packets[index] = packet;
    }
    executable->aql_packet_build_ns = elapsed_ns(aql_packet_build_start);

    const auto pm4_encode_start = SteadyClock::now();
    append_acquire_system(&executable->pm4_words);
    RegisterState register_state;
    RegisterState* state = executable->stateful_registers ? &register_state : nullptr;
    for (size_t index = 0; index < executable->dispatches.size(); ++index) {
      if (index != 0) append_dependency_global(&executable->pm4_words);
      append_dispatch(&executable->pm4_words, executable->dispatches[index], state);
    }
    append_wait_compute_idle(&executable->pm4_words);
    if ((flags & kExecutableFlagTimestamps) != 0) {
      executable->timestamps = allocate(
          context->pool, context->pool_granule, context->pool_alignment, context->gpu,
          16, 16, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG);
      const uint64_t address = reinterpret_cast<uintptr_t>(executable->timestamps.pointer);
      std::vector<uint32_t> timed;
      timed.reserve(executable->pm4_words.size() + 14);
      append_copy_timestamp(&timed, address);
      timed.insert(timed.end(), executable->pm4_words.begin(), executable->pm4_words.end());
      append_release_timestamp(&timed, address + 8);
      executable->pm4_words.swap(timed);
    }
    executable->pm4_encode_ns = elapsed_ns(pm4_encode_start);
    if (executable->pm4_words.size() > 0x000fffff)
      throw Error("retained PM4 tape exceeds vendor packet size");
    const auto ib_allocate_start = SteadyClock::now();
    executable->indirect = allocate(
        context->pool, context->pool_granule, context->pool_alignment, context->gpu,
        executable->pm4_words.size() * sizeof(uint32_t), 16,
        HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG);
    std::memcpy(executable->indirect.pointer, executable->pm4_words.data(),
                executable->indirect.length);
    executable->pm4_packet = vendor_packet(
        executable->indirect.pointer, static_cast<uint32_t>(executable->pm4_words.size()),
        context->completion);
    executable->ib_allocate_ns = elapsed_ns(ib_allocate_start);
    ++context->children;
    *output = owner.release();
  });
}

int he_pm4_executable_create(he_pm4_context* context_opaque, const he_pm4_node* input,
                             size_t count, he_pm4_executable** output, char* error,
                             size_t error_size) {
  return he_pm4_executable_create_ex(context_opaque, input, count, 0, output, error,
                                     error_size);
}

int he_pm4_executable_destroy(he_pm4_executable* opaque, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) return;
    Executable* executable = &opaque->value;
    Context* context = executable->context;
    std::lock_guard<std::mutex> lock(context->mutex);
    if (executable->in_flight || !executable->retired)
      throw Error("cannot free PM4 packet pointees without proven retirement");
    if (context->children == 0) throw Error("PM4 executable/context child ledger underflow");
    executable->indirect.release_checked();
    executable->timestamps.release_checked();
    for (auto& dispatch : executable->dispatches) dispatch.kernarg.release_checked();
    for (auto& module : executable->modules) module.release_checked();
    --context->children;
    delete opaque;
  });
}

int he_pm4_launch_aql(he_pm4_executable* opaque, uint64_t timeout_ns, char* error,
                      size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) throw Error("AQL executable is null");
    launch_aql(&opaque->value, timeout_ns);
  });
}

int he_pm4_launch_pm4(he_pm4_executable* opaque, uint64_t timeout_ns, char* error,
                      size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) throw Error("PM4 executable is null");
    launch_pm4(&opaque->value, timeout_ns);
  });
}

int he_pm4_context_json(he_pm4_context* opaque, char* output, size_t output_size,
                        size_t* required, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) throw Error("PM4 context is null");
    copy_json(context_json(&opaque->value), output, output_size, required);
  });
}

int he_pm4_executable_json(he_pm4_executable* opaque, char* output, size_t output_size,
                           size_t* required, char* error, size_t error_size) {
  return guarded(error, error_size, [&] {
    if (opaque == nullptr) throw Error("PM4 executable is null");
    copy_json(executable_json(&opaque->value), output, output_size, required);
  });
}

}  // extern "C"
