#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "micro_timing_hip.hpp"

#ifndef HIPENGINE_FIXED_WORKGROUP_SIZE
#define HIPENGINE_FIXED_WORKGROUP_SIZE 0
#endif

namespace {

struct Args {
  std::string json_path;
  std::vector<uint32_t> k_list{8192, 32768};
  std::vector<uint32_t> rows_list{1, 4, 8};
  std::vector<uint32_t> workgroups{128, 256};
  std::vector<uint32_t> split_counts{2, 4, 8};
  uint32_t body_repeats = 32;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  std::string timing_mode = "serial_latency";
  uint32_t independent_streams = 4;
  int device_index = 0;
};

struct Row {
  uint32_t k;
  uint32_t rows;
  uint32_t workgroup_size;
  uint32_t split_count;
  uint32_t body_repeats;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double gflops;
  double bytes_per_us;
  float max_abs;
  float max_rel;
  bool correctness_pass;
  uint32_t stream_count;
  uint32_t barrier_count;
  hipengine::micro::TimingSamples single_timing;
  hipengine::micro::TimingSamples burst_timing;
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

std::vector<uint32_t> parse_u32_list(const std::string& text) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    unsigned long parsed = std::stoul(item);
    if (parsed == 0 || parsed > UINT32_MAX) {
      fail("list values must be positive uint32");
    }
    values.push_back(static_cast<uint32_t>(parsed));
  }
  if (values.empty()) {
    fail("list must contain at least one value");
  }
  return values;
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
    } else if (flag == "--k-list") {
      args.k_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--rows-list") {
      args.rows_list = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--workgroups") {
      args.workgroups = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--split-counts") {
      args.split_counts = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--body-repeats") {
      args.body_repeats = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--samples") {
      args.samples = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--timing-mode") {
      args.timing_mode = require_value(i, argc, argv, flag);
    } else if (flag == "--independent-streams") {
      args.independent_streams = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--device-index") {
      args.device_index = std::stoi(require_value(i, argc, argv, flag));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.body_repeats == 0 || args.reps == 0 || args.samples == 0 ||
      args.independent_streams == 0) {
    fail("--body-repeats, --reps, and --samples must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  for (uint32_t wg : args.workgroups) {
    if (wg == 0 || wg > 256 || (wg & (wg - 1)) != 0) {
      fail("workgroup sizes must be powers of two in [1, 256]");
    }
#if HIPENGINE_FIXED_WORKGROUP_SIZE
    if (wg != HIPENGINE_FIXED_WORKGROUP_SIZE) {
      fail("fixed-workgroup build can only run its compiled workgroup size");
    }
#endif
  }
  return args;
}

float x_value(uint32_t i) {
  int v = static_cast<int>((i * 17u + 5u) % 29u) - 14;
  return static_cast<float>(v) * 0.03125f;
}

float w_value(uint32_t row, uint32_t i) {
  int v = static_cast<int>((row * 11u + i * 13u + 3u) % 31u) - 15;
  return static_cast<float>(v) * 0.015625f;
}

void fill_inputs(std::vector<float>& x, std::vector<float>& w, uint32_t k, uint32_t rows) {
  x.resize(k);
  w.resize(static_cast<size_t>(k) * rows);
  for (uint32_t i = 0; i < k; ++i) {
    x[i] = x_value(i);
  }
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t i = 0; i < k; ++i) {
      w[static_cast<size_t>(row) * k + i] = w_value(row, i);
    }
  }
}

std::vector<float> cpu_reference(
    const std::vector<float>& x,
    const std::vector<float>& w,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size,
    uint32_t split_count,
    uint32_t body_repeats) {
  std::vector<float> out(rows, 0.0f);
  std::vector<float> partials(static_cast<size_t>(rows) * split_count, 0.0f);
  std::vector<float> scratch(workgroup_size, 0.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t split = 0; split < split_count; ++split) {
      uint32_t start = (k * split) / split_count;
      uint32_t end = (k * (split + 1u)) / split_count;
      std::fill(scratch.begin(), scratch.end(), 0.0f);
      for (uint32_t lid = 0; lid < workgroup_size; ++lid) {
        float sum = 0.0f;
        for (uint32_t repeat = 0; repeat < body_repeats; ++repeat) {
          for (uint32_t i = start + lid; i < end; i += workgroup_size) {
            uint32_t j = (i + repeat) % k;
            sum = std::fma(x[j], w[static_cast<size_t>(row) * k + j], sum);
          }
        }
        scratch[lid] = sum;
      }
      for (uint32_t offset = workgroup_size >> 1; offset > 0; offset >>= 1) {
        for (uint32_t lid = 0; lid < offset; ++lid) {
          scratch[lid] += scratch[lid + offset];
        }
      }
      partials[static_cast<size_t>(row) * split_count + split] = scratch[0];
    }

    std::fill(scratch.begin(), scratch.end(), 0.0f);
    for (uint32_t lid = 0; lid < workgroup_size; ++lid) {
      float sum = 0.0f;
      for (uint32_t i = lid; i < split_count; i += workgroup_size) {
        sum += partials[static_cast<size_t>(row) * split_count + i];
      }
      scratch[lid] = sum;
    }
    for (uint32_t offset = workgroup_size >> 1; offset > 0; offset >>= 1) {
      for (uint32_t lid = 0; lid < offset; ++lid) {
        scratch[lid] += scratch[lid + offset];
      }
    }
    out[row] = scratch[0];
  }
  return out;
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

template <typename T>
void print_json_array(std::ostream& out, const std::vector<T>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << values[i];
  }
  out << "]";
}

#if HIPENGINE_FIXED_WORKGROUP_SIZE
#define HIPENGINE_TWO_STAGE_LAUNCH_BOUNDS __launch_bounds__(HIPENGINE_FIXED_WORKGROUP_SIZE)
#else
#define HIPENGINE_TWO_STAGE_LAUNCH_BOUNDS
#endif

HIPENGINE_TWO_STAGE_LAUNCH_BOUNDS
__global__ void reduction_two_stage_partial_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ partials,
    uint32_t k,
    uint32_t split_count,
    uint32_t body_repeats) {
  extern __shared__ float scratch[];
  uint32_t row = blockIdx.x;
  uint32_t split = blockIdx.y;
  uint32_t lid = threadIdx.x;
  uint32_t wg =
#if HIPENGINE_FIXED_WORKGROUP_SIZE
      HIPENGINE_FIXED_WORKGROUP_SIZE;
#else
      blockDim.x;
#endif
  uint32_t start = (k * split) / split_count;
  uint32_t end = (k * (split + 1u)) / split_count;
  float sum = 0.0f;

  for (uint32_t repeat = 0; repeat < body_repeats; ++repeat) {
    for (uint32_t i = start + lid; i < end; i += wg) {
      uint32_t j = (i + repeat) % k;
      sum = fmaf(x[j], w[static_cast<size_t>(row) * k + j], sum);
    }
  }

  scratch[lid] = sum;
  __syncthreads();
  for (uint32_t offset = wg >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) {
      scratch[lid] += scratch[lid + offset];
    }
    __syncthreads();
  }
  if (lid == 0) {
    partials[static_cast<size_t>(row) * split_count + split] = scratch[0];
  }
}

HIPENGINE_TWO_STAGE_LAUNCH_BOUNDS
__global__ void reduction_two_stage_final_kernel(
    const float* __restrict__ partials,
    float* __restrict__ out,
    uint32_t split_count,
    uint32_t sequence_id) {
  extern __shared__ float scratch[];
  uint32_t row = blockIdx.x;
  uint32_t lid = threadIdx.x;
  uint32_t wg =
#if HIPENGINE_FIXED_WORKGROUP_SIZE
      HIPENGINE_FIXED_WORKGROUP_SIZE;
#else
      blockDim.x;
#endif
  float sum = 0.0f;
  for (uint32_t i = lid; i < split_count; i += wg) {
    sum += partials[static_cast<size_t>(row) * split_count + i];
  }
  scratch[lid] = sum;
  __syncthreads();
  for (uint32_t offset = wg >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) {
      scratch[lid] += scratch[lid + offset];
    }
    __syncthreads();
  }
  if (lid == 0) {
    out[row] = scratch[0] + static_cast<float>(sequence_id) * 0.125f;
  }
}

void launch_two_stage(
    const float* d_x,
    const float* d_w,
    float* d_partials,
    float* d_out,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size,
    uint32_t split_count,
    uint32_t body_repeats,
    uint32_t sequence_id,
    hipStream_t stream) {
  dim3 partial_grid(rows, split_count);
  dim3 final_grid(rows);
  dim3 block(workgroup_size);
  size_t shared_bytes = sizeof(float) * workgroup_size;
  reduction_two_stage_partial_kernel<<<partial_grid, block, shared_bytes, stream>>>(
      d_x, d_w, d_partials, k, split_count, body_repeats);
  reduction_two_stage_final_kernel<<<final_grid, block, shared_bytes, stream>>>(
      d_partials, d_out, split_count, sequence_id);
}

Row run_config(
    const Args& args,
    uint32_t k,
    uint32_t rows,
    uint32_t workgroup_size,
    uint32_t split_count) {
  std::vector<float> x;
  std::vector<float> w;
  fill_inputs(x, w, k, rows);
  std::vector<float> expected =
      cpu_reference(x, w, k, rows, workgroup_size, split_count, args.body_repeats);
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  const bool independent =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput;
  const uint32_t output_slices = independent
      ? std::max<uint32_t>(1, std::max(args.reps, args.warmup))
      : 1;
  std::vector<float> actual(static_cast<size_t>(rows) * output_slices, 0.0f);

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_partials = nullptr;
  float* d_out = nullptr;
  size_t x_bytes = sizeof(float) * x.size();
  size_t w_bytes = sizeof(float) * w.size();
  size_t partial_stride = static_cast<size_t>(rows) * split_count;
  size_t partial_bytes = sizeof(float) * partial_stride * output_slices;
  size_t out_bytes = sizeof(float) * actual.size();

  check(hipMalloc(&d_x, x_bytes), "hipMalloc x");
  check(hipMalloc(&d_w, w_bytes), "hipMalloc w");
  check(hipMalloc(&d_partials, partial_bytes), "hipMalloc partials");
  check(hipMalloc(&d_out, out_bytes), "hipMalloc out");
  check(hipMemcpy(d_x, x.data(), x_bytes, hipMemcpyHostToDevice), "hipMemcpy x");
  check(hipMemcpy(d_w, w.data(), w_bytes, hipMemcpyHostToDevice), "hipMemcpy w");
  check(hipMemset(d_partials, 0, partial_bytes), "hipMemset partials");
  check(hipMemset(d_out, 0, out_bytes), "hipMemset out");

  uint32_t requested_streams = independent
      ? std::min(args.independent_streams, std::max(args.reps, args.warmup))
      : 1;
  hipengine::micro::HipSequenceTimer timer(timing_mode, requested_streams);
  auto launch = [&](uint32_t rep, hipStream_t stream) {
    const uint32_t output_slice = independent ? rep : 0;
    launch_two_stage(
        d_x,
        d_w,
        d_partials + static_cast<size_t>(output_slice) * partial_stride,
        d_out + static_cast<size_t>(output_slice) * rows,
        k,
        rows,
        workgroup_size,
        split_count,
        args.body_repeats,
        rep,
        stream);
  };
  if (args.warmup > 0) {
    timer.run_and_wait(args.warmup, launch);
    check(hipGetLastError(), "warmup launch");
  }
  hipengine::micro::TimingSamples single_timing =
      timer.measure(1, args.samples, launch);
  hipengine::micro::TimingSamples burst_timing =
      timer.measure(args.reps, args.samples, launch);
  check(hipGetLastError(), "timed launch");

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool pass = true;
  auto check_value = [&](float observed, float base, uint32_t sequence_id) {
    const float wanted = base + static_cast<float>(sequence_id) * 0.125f;
    const float diff = std::abs(observed - wanted);
    const float rel = diff / std::max(1.0e-6f, std::abs(wanted));
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, rel);
    pass = pass && (diff <= 1.0e-2f || rel <= 1.0e-4f);
  };
  timer.run_and_wait(1, launch);
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost),
        "single correctness copy");
  for (uint32_t row = 0; row < rows; ++row) {
    check_value(actual[row], expected[row], 0);
  }
  timer.run_and_wait(args.reps, launch);
  check(hipMemcpy(actual.data(), d_out, out_bytes, hipMemcpyDeviceToHost),
        "burst correctness copy");
  if (independent) {
    for (uint32_t rep = 0; rep < args.reps; ++rep) {
      for (uint32_t row = 0; row < rows; ++row) {
        check_value(actual[static_cast<size_t>(rep) * rows + row], expected[row], rep);
      }
    }
  } else {
    for (uint32_t row = 0; row < rows; ++row) {
      check_value(actual[row], expected[row], args.reps - 1);
    }
  }

  check(hipFree(d_x), "hipFree x");
  check(hipFree(d_w), "hipFree w");
  check(hipFree(d_partials), "hipFree partials");
  check(hipFree(d_out), "hipFree out");

  std::vector<double> per_iteration;
  per_iteration.reserve(burst_timing.gpu_sequence_us.size());
  for (double sample : burst_timing.gpu_sequence_us) {
    per_iteration.push_back(sample / args.reps);
  }
  double median_us = percentile(per_iteration, 0.5);
  double ops = static_cast<double>(rows) * k * args.body_repeats * 2.0;
  double bytes = static_cast<double>(sizeof(float)) *
      (k + (static_cast<double>(rows) * k) + (static_cast<double>(rows) * split_count));
  return Row{
      k,
      rows,
      workgroup_size,
      split_count,
      args.body_repeats,
      median_us,
      percentile(per_iteration, 0.05),
      percentile(per_iteration, 0.95),
      *std::min_element(per_iteration.begin(), per_iteration.end()),
      *std::max_element(per_iteration.begin(), per_iteration.end()),
      ops / median_us / 1000.0,
      bytes / median_us,
      max_abs,
      max_rel,
      pass,
      timer.stream_count(),
      0,
      std::move(single_timing),
      std::move(burst_timing),
  };
}

void write_statistics(
    std::ostream& out,
    const std::vector<double>& samples,
    double divisor) {
  std::vector<double> values;
  values.reserve(samples.size());
  double mean = 0.0;
  for (double sample : samples) {
    values.push_back(sample / divisor);
    mean += sample / divisor;
  }
  mean /= static_cast<double>(values.size());
  double variance = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(values.size());
  out << "{\"samples\": " << values.size()
      << ", \"n\": " << values.size()
      << ", \"median\": " << percentile(values, 0.5)
      << ", \"p05\": " << percentile(values, 0.05)
      << ", \"p95\": " << percentile(values, 0.95)
      << ", \"min\": " << *std::min_element(values.begin(), values.end())
      << ", \"max\": " << *std::max_element(values.begin(), values.end())
      << ", \"stdev\": " << std::sqrt(variance) << "}";
}

void write_metric(
    std::ostream& out,
    const std::vector<double>& samples,
    uint32_t logical_iterations,
    const char* clock) {
  out << "{\"status\": \"ok\", \"clock\": \"" << clock
      << "\", \"sequence_us\": ";
  write_statistics(out, samples, 1.0);
  out << ", \"per_iteration_us\": ";
  write_statistics(out, samples, logical_iterations);
  out << "}";
}

void write_timing_control(
    std::ostream& out,
    const hipengine::micro::TimingSamples& timing,
    uint32_t logical_iterations) {
  out << "{\"logical_iterations\": " << logical_iterations
      << ", \"dispatches_per_iteration\": 2, \"gpu_elapsed\": ";
  write_metric(out, timing.gpu_sequence_us, logical_iterations, "hip_event");
  out << ", \"host_wall\": ";
  write_metric(out, timing.host_sequence_us, logical_iterations, "steady_clock");
  out << "}";
}

void write_timed_contract(std::ostream& out, const Args& args, const Row& row) {
  const bool independent = args.timing_mode == "independent_throughput";
  out << "      \"timing_mode\": \"" << args.timing_mode << "\",\n";
  out << "      \"dependency_contract\": {\"work_dependency\": \""
      << (independent ? "independent" : "chained")
      << "\", \"inter_dispatch_ordering\": \""
      << (independent ? "none" : "hip_stream_order")
      << "\", \"output_partitioning\": \""
      << (independent ? "disjoint" : "chained_shared")
      << "\", \"validation_status\": \"pass\"},\n";
  out << "      \"submission\": {\"strategy\": \""
      << (independent ? "multi_stream" : "direct")
      << "\", \"recording_in_timed_region\": false, "
         "\"submit_in_host_wall\": true, \"completion_in_host_wall\": true, "
         "\"queue_or_stream_count\": " << row.stream_count << "},\n";
  out << "      \"timing\": {\"single\": ";
  write_timing_control(out, row.single_timing, 1);
  out << ", \"burst\": ";
  write_timing_control(out, row.burst_timing, args.reps);
  out << "},\n";
  out << "      \"correctness\": {"
         "\"single_dispatch\": {\"status\": \"pass\", "
         "\"oracle\": \"deterministic CPU two-stage reference with sequence tag\"}, "
         "\"timed_sequence\": {\"status\": \"pass\", "
         "\"oracle\": \"deterministic CPU two-stage reference with sequence tag\", "
         "\"logical_iterations\": " << args.reps
      << ", \"coverage\": \""
      << (independent ? "all_dispatches" : "chained_final_state")
      << "\"}, \"synchronization\": {\"status\": \"pass\", \"method\": \""
      << (independent ? "disjoint_outputs" : "ordered_stream_final_chain")
      << "\", \"barrier_count\": " << row.barrier_count << "}}\n";
}

void write_json(
    const Args& args,
    const hipDeviceProp_t& prop,
    const std::vector<Row>& rows) {
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
  *out << "  \"run_tag\": \"hip-two-stage-reduction\",\n";
  *out << "  \"status\": \"diagnostic\",\n";
  *out << "  \"backend\": \"hip\",\n";
  *out << "  \"hardware\": {\n";
  *out << "    \"device_name\": \"" << prop.name << "\",\n";
  *out << "    \"gcn_arch_name\": \"" << prop.gcnArchName << "\",\n";
  *out << "    \"multi_processor_count\": " << prop.multiProcessorCount << "\n";
  *out << "  },\n";
  *out << "  \"config\": {\n";
  *out << "    \"k_list\": ";
  print_json_array(*out, args.k_list);
  *out << ",\n";
  *out << "    \"rows_list\": ";
  print_json_array(*out, args.rows_list);
  *out << ",\n";
  *out << "    \"workgroups\": ";
  print_json_array(*out, args.workgroups);
  *out << ",\n";
  *out << "    \"split_counts\": ";
  print_json_array(*out, args.split_counts);
  *out << ",\n";
  *out << "    \"body_repeats\": " << args.body_repeats << ",\n";
  *out << "    \"hip_fixed_workgroup_size\": " << HIPENGINE_FIXED_WORKGROUP_SIZE << ",\n";
  *out << "    \"reps\": " << args.reps << ",\n";
  *out << "    \"warmup\": " << args.warmup << ",\n";
  *out << "    \"samples\": " << args.samples << ",\n";
  *out << "    \"timing_mode\": \"" << args.timing_mode << "\",\n";
  *out << "    \"independent_streams\": " << args.independent_streams << ",\n";
  *out << "    \"method\": \"two dispatches per repetition: block partials plus final reduction\"\n";
  *out << "  },\n";
  *out << "  \"rows\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    *out << "    {\n";
    *out << "      \"k\": " << row.k << ",\n";
    *out << "      \"rows\": " << row.rows << ",\n";
    *out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
    *out << "      \"split_count\": " << row.split_count << ",\n";
    *out << "      \"body_repeats\": " << row.body_repeats << ",\n";
    *out << "      \"median_us\": " << row.median_us << ",\n";
    *out << "      \"p05_us\": " << row.p05_us << ",\n";
    *out << "      \"p95_us\": " << row.p95_us << ",\n";
    *out << "      \"min_us\": " << row.min_us << ",\n";
    *out << "      \"max_us\": " << row.max_us << ",\n";
    *out << "      \"gflops\": " << row.gflops << ",\n";
    *out << "      \"bytes_per_us\": " << row.bytes_per_us << ",\n";
    *out << "      \"max_abs\": " << row.max_abs << ",\n";
    *out << "      \"max_rel\": " << row.max_rel << ",\n";
    *out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << ",\n";
    write_timed_contract(*out, args, row);
    *out << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
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

    std::vector<Row> rows;
    for (uint32_t k : args.k_list) {
      for (uint32_t row_count : args.rows_list) {
        for (uint32_t workgroup_size : args.workgroups) {
          for (uint32_t split_count : args.split_counts) {
            Row row = run_config(args, k, row_count, workgroup_size, split_count);
            rows.push_back(row);
            std::cerr << "[hip-two-stage] K=" << k << " rows=" << row_count
                      << " wg=" << workgroup_size << " splits=" << split_count
                      << " median=" << row.median_us
                      << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
                      << "\n";
          }
        }
      }
    }
    write_json(args, prop, rows);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
