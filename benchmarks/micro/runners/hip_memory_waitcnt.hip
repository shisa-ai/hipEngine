#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HIPENGINE_MEM_MODE
#define HIPENGINE_MEM_MODE 0
#endif

#ifndef HIPENGINE_MEM_PARAM
#define HIPENGINE_MEM_PARAM 1
#endif

#ifndef HIPENGINE_MEM_FIXED_BLOCK
#define HIPENGINE_MEM_FIXED_BLOCK 0
#endif

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kModeCoalesced = 0;
constexpr uint32_t kModeStrided = 1;
constexpr uint32_t kModeGather = 2;
constexpr uint32_t kModeInterleave = 3;

#if HIPENGINE_MEM_FIXED_BLOCK
#define HIPENGINE_MEM_LAUNCH_BOUNDS __launch_bounds__(kBlockSize)
#else
#define HIPENGINE_MEM_LAUNCH_BOUNDS
#endif

struct Args {
  std::string json_path;
  uint32_t n = 32768;
  uint32_t body_iters = 128;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  int device_index = 0;
};

struct Row {
  std::string mode;
  uint32_t param;
  uint32_t n;
  uint32_t body_iters;
  uint32_t block_size;
  uint32_t data_elems;
  double bytes_per_dispatch;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double bandwidth_gbps;
  double gops;
  float max_abs;
  float max_rel;
  bool correctness_pass;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(hipError_t result, const char* what) {
  if (result != hipSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << hipGetErrorString(result);
    fail(oss.str());
  }
}

std::string require_value(int& index, int argc, char** argv, const std::string& flag) {
  if (index + 1 >= argc) {
    fail(flag + " requires a value");
  }
  ++index;
  return argv[index];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--n") {
      args.n = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--body-iters") {
      args.body_iters = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--samples") {
      args.samples = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = std::stoi(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.n == 0 || args.body_iters == 0 || args.reps == 0 || args.samples == 0) {
    fail("--n, --body-iters, --reps, and --samples must be positive");
  }
  return args;
}

const char* mode_name() {
#if HIPENGINE_MEM_MODE == 0
  return "coalesced";
#elif HIPENGINE_MEM_MODE == 1
  return "strided";
#elif HIPENGINE_MEM_MODE == 2
  return "gather";
#elif HIPENGINE_MEM_MODE == 3
  return "interleave";
#else
  return "unknown";
#endif
}

uint32_t next_power_of_two(uint64_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  ++value;
  if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    fail("required data size exceeds uint32 range");
  }
  return static_cast<uint32_t>(value);
}

uint32_t required_data_elems(uint32_t n, uint32_t body_iters) {
  uint64_t base = static_cast<uint64_t>(n) * body_iters;
#if HIPENGINE_MEM_MODE == 2
  uint64_t elems = base;
#else
  uint64_t elems = base * HIPENGINE_MEM_PARAM;
#endif
  return next_power_of_two(std::max<uint64_t>(1024, elems));
}

double bytes_per_dispatch(uint32_t n, uint32_t body_iters) {
  double iters = static_cast<double>(n) * body_iters;
#if HIPENGINE_MEM_MODE == 0 || HIPENGINE_MEM_MODE == 3
  return iters * HIPENGINE_MEM_PARAM * sizeof(float);
#elif HIPENGINE_MEM_MODE == 1
  return iters * sizeof(float);
#elif HIPENGINE_MEM_MODE == 2
  return iters * (sizeof(uint32_t) + sizeof(float));
#else
  return iters * sizeof(float);
#endif
}

double ops_per_dispatch(uint32_t n, uint32_t body_iters) {
  double iters = static_cast<double>(n) * body_iters;
#if HIPENGINE_MEM_MODE == 3
  return iters * HIPENGINE_MEM_PARAM * 4.0;
#elif HIPENGINE_MEM_MODE == 0
  return iters * HIPENGINE_MEM_PARAM * 2.0;
#else
  return iters * 2.0;
#endif
}

__host__ __device__ uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__host__ __device__ float data_value(uint32_t i) {
  uint32_t bits = hash_u32(i * 1664525u + 1013904223u);
  int value = static_cast<int>(bits & 0x3ffu) - 512;
  return static_cast<float>(value) * 0.0009765625f;
}

void fill_inputs(std::vector<float>& x, std::vector<uint32_t>& ids, uint32_t n, uint32_t body_iters) {
  uint32_t data_elems = required_data_elems(n, body_iters);
  x.resize(data_elems);
  for (uint32_t i = 0; i < data_elems; ++i) {
    x[i] = data_value(i);
  }
  ids.resize(static_cast<size_t>(n) * body_iters);
  uint32_t mask = data_elems - 1;
  for (uint32_t iter = 0; iter < body_iters; ++iter) {
    for (uint32_t idx = 0; idx < n; ++idx) {
      ids[static_cast<size_t>(iter) * n + idx] =
          hash_u32(idx * 747796405u + iter * 2891336453u) & mask;
    }
  }
}

__host__ __device__ float run_value(
    const float* __restrict__ x,
    const uint32_t* __restrict__ ids,
    uint32_t idx,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask) {
  float sum = 0.0f;
  uint32_t state = idx * 747796405u + 2891336453u;
  for (uint32_t iter = 0; iter < body_iters; ++iter) {
#if HIPENGINE_MEM_MODE == 0
    uint32_t base = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
#pragma unroll
    for (uint32_t u = 0; u < HIPENGINE_MEM_PARAM; ++u) {
      float value = x[(base + u) & data_mask];
      sum = fmaf(value, 1.000001f + static_cast<float>(u) * 0.000001f, sum);
    }
#elif HIPENGINE_MEM_MODE == 1
    uint32_t addr = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
    sum = fmaf(x[addr], 1.000001f, sum);
#elif HIPENGINE_MEM_MODE == 2
    uint32_t id = ids[static_cast<size_t>(iter) * n + idx] & data_mask;
    sum = fmaf(x[id], 1.000001f, sum);
#elif HIPENGINE_MEM_MODE == 3
    uint32_t base = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
#pragma unroll
    for (uint32_t u = 0; u < HIPENGINE_MEM_PARAM; ++u) {
      state = hash_u32(state + iter + u);
      float value = x[(base + u) & data_mask];
      float bias = static_cast<float>(state & 0xffu) * 0.0000001f;
      sum = fmaf(value + bias, 0.999999f + static_cast<float>(u) * 0.000001f, sum);
    }
#endif
  }
  return sum;
}

HIPENGINE_MEM_LAUNCH_BOUNDS
__global__ void memory_waitcnt_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ ids,
    float* __restrict__ out,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask) {
  uint32_t idx =
#if HIPENGINE_MEM_FIXED_BLOCK
      blockIdx.x * kBlockSize + threadIdx.x;
#else
      blockIdx.x * blockDim.x + threadIdx.x;
#endif
  if (idx >= n) {
    return;
  }
  out[idx] = run_value(x, ids, idx, n, body_iters, data_mask);
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    fail("cannot compute percentile of empty vector");
  }
  std::sort(values.begin(), values.end());
  double pos = q * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(pos));
  size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return values[lo];
  }
  double t = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - t) + values[hi] * t;
}

Row run_config(const Args& args) {
  std::vector<float> x;
  std::vector<uint32_t> ids;
  fill_inputs(x, ids, args.n, args.body_iters);
  std::vector<float> actual(args.n, 0.0f);
  uint32_t data_mask = static_cast<uint32_t>(x.size() - 1);

  float* d_x = nullptr;
  uint32_t* d_ids = nullptr;
  float* d_out = nullptr;
  size_t x_bytes = sizeof(float) * x.size();
  size_t ids_bytes = sizeof(uint32_t) * ids.size();
  size_t out_bytes = sizeof(float) * actual.size();

  check(hipMalloc(&d_x, x_bytes), "hipMalloc x");
  check(hipMalloc(&d_ids, ids_bytes), "hipMalloc ids");
  check(hipMalloc(&d_out, out_bytes), "hipMalloc out");
  check(hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice), "hipMemcpy x");
  check(hipMemcpy(d_ids, ids.data(), ids_bytes, hipMemcpyHostToDevice), "hipMemcpy ids");
  check(hipMemset(d_out, 0, out_bytes), "hipMemset out");

  dim3 block(kBlockSize);
  dim3 grid((args.n + kBlockSize - 1) / kBlockSize);
  memory_waitcnt_kernel<<<grid, block>>>(d_x, d_ids, d_out, args.n, args.body_iters, data_mask);
  check(hipGetLastError(), "correctness launch");
  check(hipDeviceSynchronize(), "correctness sync");
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost), "copy out");

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  uint32_t checked = std::min<uint32_t>(args.n, 64);
  for (uint32_t i = 0; i < checked; ++i) {
    float expected = run_value(x.data(), ids.data(), i, args.n, args.body_iters, data_mask);
    float diff = std::abs(actual[i] - expected);
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, diff / std::max(1.0e-6f, std::abs(expected)));
  }
  bool pass = max_abs <= 5.0e-3f || max_rel <= 5.0e-4f;

  for (uint32_t i = 0; i < args.warmup; ++i) {
    memory_waitcnt_kernel<<<grid, block>>>(d_x, d_ids, d_out, args.n, args.body_iters, data_mask);
  }
  check(hipDeviceSynchronize(), "warmup sync");

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipEventCreate(&start), "hipEventCreate start");
  check(hipEventCreate(&stop), "hipEventCreate stop");
  std::vector<double> samples;
  samples.reserve(args.samples);
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    check(hipEventRecord(start, 0), "hipEventRecord start");
    for (uint32_t rep = 0; rep < args.reps; ++rep) {
      memory_waitcnt_kernel<<<grid, block>>>(d_x, d_ids, d_out, args.n, args.body_iters, data_mask);
    }
    check(hipEventRecord(stop, 0), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    samples.push_back(static_cast<double>(elapsed_ms) * 1000.0 / args.reps);
  }

  check(hipEventDestroy(start), "hipEventDestroy start");
  check(hipEventDestroy(stop), "hipEventDestroy stop");
  check(hipFree(d_x), "hipFree x");
  check(hipFree(d_ids), "hipFree ids");
  check(hipFree(d_out), "hipFree out");

  double median_us = percentile(samples, 0.5);
  double bytes = bytes_per_dispatch(args.n, args.body_iters);
  double ops = ops_per_dispatch(args.n, args.body_iters);
  return Row{
      mode_name(),
      HIPENGINE_MEM_PARAM,
      args.n,
      args.body_iters,
      kBlockSize,
      static_cast<uint32_t>(x.size()),
      bytes,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      bytes / median_us / 1000.0,
      ops / median_us / 1000.0,
      max_abs,
      max_rel,
      pass,
  };
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void write_json(const Args& args, const hipDeviceProp_t& prop, const Row& row) {
  std::ostream* out = &std::cout;
  std::ofstream file;
  if (!args.json_path.empty()) {
    file.open(args.json_path);
    if (!file) {
      fail("could not open JSON path: " + args.json_path);
    }
    out = &file;
  }
  *out << std::setprecision(10);
  *out << "{\n";
  *out << "  \"run_tag\": \"hip-memory-waitcnt\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << json_escape(prop.name) << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << json_escape(prop.gcnArchName) << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "    \"mode_id\": " << HIPENGINE_MEM_MODE << ",\n";
  *out << "    \"param\": " << row.param << ",\n";
  *out << "    \"n\": " << row.n << ",\n";
  *out << "    \"body_iters\": " << row.body_iters << ",\n";
  *out << "    \"block_size\": " << row.block_size << ",\n";
  *out << "    \"fixed_block_indexing\": "
       << (HIPENGINE_MEM_FIXED_BLOCK ? "true" : "false") << ",\n";
  *out << "    \"data_elems\": " << row.data_elems << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"method\": \"global-memory load plus accumulate diagnostic; sampled CPU oracle\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  *out << "    {\n";
  *out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  *out << "      \"param\": " << row.param << ",\n";
  *out << "      \"n\": " << row.n << ",\n";
  *out << "      \"body_iters\": " << row.body_iters << ",\n";
  *out << "      \"block_size\": " << row.block_size << ",\n";
  *out << "      \"data_elems\": " << row.data_elems << ",\n";
  *out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  *out << "      \"median_us\": " << row.median_us << ",\n";
  *out << "      \"p05_us\": " << row.p05_us << ",\n";
  *out << "      \"p95_us\": " << row.p95_us << ",\n";
  *out << "      \"min_us\": " << row.min_us << ",\n";
  *out << "      \"max_us\": " << row.max_us << ",\n";
  *out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  *out << "      \"gops\": " << row.gops << ",\n";
  *out << "      \"max_abs\": " << row.max_abs << ",\n";
  *out << "      \"max_rel\": " << row.max_rel << ",\n";
  *out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << "\n";
  *out << "    }\n";
  *out << "  ]\n";
  *out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    check(hipSetDevice(args.device_index), "hipSetDevice");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, args.device_index), "hipGetDeviceProperties");
    Row row = run_config(args);
    std::cerr << "[hip] mode=" << row.mode << " param=" << row.param
              << " median=" << row.median_us
              << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
              << "\n";
    write_json(args, prop, row);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
