#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
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
#include <utility>
#include <vector>

#include "micro_timing_vulkan.hpp"

#ifndef HIPENGINE_VOPD_MODE
#define HIPENGINE_VOPD_MODE 0
#endif

#ifndef HIPENGINE_VOPD_ACCUMS
#define HIPENGINE_VOPD_ACCUMS 4
#endif

#ifndef HIPENGINE_BLOCK_SIZE
#define HIPENGINE_BLOCK_SIZE 256
#endif

namespace {

constexpr uint32_t kBlockSize = HIPENGINE_BLOCK_SIZE;

struct Args {
  std::string spirv_path;
  std::string json_path;
  uint32_t n = 65536;
  uint32_t body_iters = 2048;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  std::string timing_mode = "serial_latency";
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t n;
  uint32_t body_iters;
  uint32_t output_offset;
};

struct Row {
  std::string mode;
  uint32_t accums;
  uint32_t n;
  uint32_t body_iters;
  uint32_t block_size;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double gops;
  float max_abs;
  float max_rel;
  bool correctness_pass;
  bool timed_sequence_correctness_pass;
  bool gpu_timestamps_supported;
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    std::ostringstream oss;
    oss << what << " failed with VkResult " << static_cast<int>(result);
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
    if (flag == "--spirv") {
      args.spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
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
    } else if (flag == "--timing-mode") {
      args.timing_mode = require_value(i, argc, argv, flag);
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.spirv_path.empty()) {
    fail("--spirv is required");
  }
  if (args.n == 0 || args.body_iters == 0 || args.reps == 0 || args.samples == 0) {
    fail("--n, --body-iters, --reps, and --samples must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  return args;
}

const char* mode_name() {
#if HIPENGINE_VOPD_MODE == 0
  return "independent_fma";
#elif HIPENGINE_VOPD_MODE == 1
  return "dependent_fma";
#elif HIPENGINE_VOPD_MODE == 2
  return "mixed_int_float";
#elif HIPENGINE_VOPD_MODE == 3
  return "dequant_like";
#else
  return "unknown";
#endif
}

float init_accum(uint32_t idx, uint32_t lane) {
  uint32_t bits = (idx * 747796405u) ^ (lane * 2891336453u) ^ 0x9e3779b9u;
  return 0.25f + static_cast<float>(bits & 0xffu) * 0.0009765625f;
}

float lane_bias(uint32_t idx, uint32_t iter, uint32_t lane) {
  uint32_t bits = idx * 1664525u + iter * 1013904223u + lane * 2246822519u;
  return static_cast<float>((bits >> 24) & 0x1fu) * 0.000001f + 0.000003f;
}

uint32_t hash_step(uint32_t value, uint32_t iter, uint32_t lane) {
  value ^= iter * 0x9e3779b9u + lane * 0x85ebca6bu;
  value = value * 1664525u + 1013904223u;
  value ^= value >> 16;
  return value;
}

float run_value(uint32_t idx, uint32_t body_iters) {
  float a0 = init_accum(idx, 0);
  float a1 = init_accum(idx, 1);
  float a2 = init_accum(idx, 2);
  float a3 = init_accum(idx, 3);
  float a4 = init_accum(idx, 4);
  float a5 = init_accum(idx, 5);
  float a6 = init_accum(idx, 6);
  float a7 = init_accum(idx, 7);
  uint32_t u0 = idx * 747796405u + 2891336453u;
  uint32_t u1 = idx ^ 0xa5a5a5a5u;

  for (uint32_t iter = 0; iter < body_iters; ++iter) {
#if HIPENGINE_VOPD_MODE == 0
    a0 = std::fma(a0, 1.000001f, lane_bias(idx, iter, 0));
#if HIPENGINE_VOPD_ACCUMS >= 2
    a1 = std::fma(a1, 0.999999f, lane_bias(idx, iter, 1));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    a2 = std::fma(a2, 1.000002f, lane_bias(idx, iter, 2));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    a3 = std::fma(a3, 0.999998f, lane_bias(idx, iter, 3));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
    a4 = std::fma(a4, 1.000003f, lane_bias(idx, iter, 4));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
    a5 = std::fma(a5, 0.999997f, lane_bias(idx, iter, 5));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
    a6 = std::fma(a6, 1.000004f, lane_bias(idx, iter, 6));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
    a7 = std::fma(a7, 0.999996f, lane_bias(idx, iter, 7));
#endif
#elif HIPENGINE_VOPD_MODE == 1
    a0 = std::fma(a0, 1.000001f, lane_bias(idx, iter, 0));
#if HIPENGINE_VOPD_ACCUMS >= 2
    a0 = std::fma(a0, 0.999999f, lane_bias(idx, iter, 1));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    a0 = std::fma(a0, 1.000002f, lane_bias(idx, iter, 2));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    a0 = std::fma(a0, 0.999998f, lane_bias(idx, iter, 3));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
    a0 = std::fma(a0, 1.000003f, lane_bias(idx, iter, 4));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
    a0 = std::fma(a0, 0.999997f, lane_bias(idx, iter, 5));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
    a0 = std::fma(a0, 1.000004f, lane_bias(idx, iter, 6));
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
    a0 = std::fma(a0, 0.999996f, lane_bias(idx, iter, 7));
#endif
#elif HIPENGINE_VOPD_MODE == 2
    u0 = hash_step(u0, iter, 0);
    a0 = std::fma(a0, 1.000001f, static_cast<float>(u0 & 0x1fu) * 0.000001f);
#if HIPENGINE_VOPD_ACCUMS >= 2
    u1 = hash_step(u1, iter, 1);
    a1 = std::fma(a1, 0.999999f, static_cast<float>((u1 >> 3) & 0x1fu) * 0.000001f);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    u0 = hash_step(u0, iter, 2);
    a2 = std::fma(a2, 1.000002f, static_cast<float>((u0 >> 7) & 0x1fu) * 0.000001f);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    u1 = hash_step(u1, iter, 3);
    a3 = std::fma(a3, 0.999998f, static_cast<float>((u1 >> 11) & 0x1fu) * 0.000001f);
#endif
#elif HIPENGINE_VOPD_MODE == 3
    u0 = hash_step(u0, iter, 0);
    int q0 = static_cast<int>((u0 >> ((iter + 0u) & 15u)) & 0xffu) - 128;
    a0 = std::fma(static_cast<float>(q0) * 0.0078125f, 0.03125f, a0);
#if HIPENGINE_VOPD_ACCUMS >= 2
    u1 = hash_step(u1, iter, 1);
    int q1 = static_cast<int>((u1 >> ((iter + 3u) & 15u)) & 0xffu) - 128;
    a1 = std::fma(static_cast<float>(q1) * 0.0078125f, 0.03125f, a1);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
    u0 = hash_step(u0, iter, 2);
    int q2 = static_cast<int>((u0 >> ((iter + 5u) & 15u)) & 0xffu) - 128;
    a2 = std::fma(static_cast<float>(q2) * 0.0078125f, 0.03125f, a2);
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
    u1 = hash_step(u1, iter, 3);
    int q3 = static_cast<int>((u1 >> ((iter + 7u) & 15u)) & 0xffu) - 128;
    a3 = std::fma(static_cast<float>(q3) * 0.0078125f, 0.03125f, a3);
#endif
#endif
  }

  float out = a0;
#if HIPENGINE_VOPD_ACCUMS >= 2
  out += a1;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 3
  out += a2;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 4
  out += a3;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 5
  out += a4;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 6
  out += a5;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 7
  out += a6;
#endif
#if HIPENGINE_VOPD_ACCUMS >= 8
  out += a7;
#endif
  return out;
}

std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fail("could not open SPIR-V file: " + path);
  }
  std::streamsize size = file.tellg();
  if (size <= 0 || (size % 4) != 0) {
    fail("SPIR-V file size must be a positive multiple of 4");
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
    fail("could not read SPIR-V file: " + path);
  }
  return words;
}

uint32_t find_queue_family(VkPhysicalDevice physical_device) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  if (count == 0) {
    fail("physical device has no queue families");
  }
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      return i;
    }
  }
  fail("physical device has no compute queue family");
}

uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_bits,
    VkMemoryPropertyFlags required) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    bool type_matches = (type_bits & (1u << i)) != 0;
    bool flags_match = (properties.memoryTypes[i].propertyFlags & required) == required;
    if (type_matches && flags_match) {
      return i;
    }
  }
  fail("no compatible memory type found");
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    fail("cannot compute percentile of empty values");
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

double submit_once(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer, VkFence fence) {
  check(vkResetFences(device, 1, &fence), "vkResetFences");
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  auto t0 = std::chrono::steady_clock::now();
  check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
  check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

uint32_t grid_blocks(uint32_t n) {
  return std::max<uint32_t>(1, (n + kBlockSize - 1) / kBlockSize);
}

VkCommandBuffer record_command_buffer(
    VkDevice device,
    VkCommandPool command_pool,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    VkBuffer output_buffer,
    const Args& args,
    uint32_t logical_iterations,
    hipengine::micro::TimingMode timing_mode,
    const hipengine::micro::VulkanSequenceTimer* timer) {
  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = command_pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
        "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  VkBufferMemoryBarrier begin_barrier{};
  begin_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  begin_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  begin_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  begin_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  begin_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  begin_barrier.buffer = output_buffer;
  begin_barrier.offset = 0;
  begin_barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      1,
      &begin_barrier,
      0,
      nullptr);
  if (timer != nullptr) {
    timer->record_begin(command_buffer);
  }
  for (uint32_t rep = 0; rep < logical_iterations; ++rep) {
    const uint32_t output_offset =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? rep * args.n
            : 0u;
    PushConstants push{args.n, args.body_iters, output_offset};
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, grid_blocks(args.n), 1, 1);
    if (timing_mode == hipengine::micro::TimingMode::SerialLatency &&
        rep + 1 < logical_iterations) {
      hipengine::micro::compute_buffer_barrier(
          command_buffer,
          {hipengine::micro::make_compute_buffer_barrier(
              output_buffer,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)});
    }
  }
  if (timer != nullptr) {
    timer->record_end(command_buffer);
  }
  VkBufferMemoryBarrier end_barrier{};
  end_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  end_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  end_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  end_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  end_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  end_barrier.buffer = output_buffer;
  end_barrier.offset = 0;
  end_barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0,
      nullptr,
      1,
      &end_barrier,
      0,
      nullptr);
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  return command_buffer;
}

double ops_per_element_iter() {
  return static_cast<double>(HIPENGINE_VOPD_ACCUMS);
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

void write_samples(std::ostream& out, const std::vector<double>& samples) {
  out << "[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << samples[i];
  }
  out << "]";
}

void write_timing_raw(
    std::ostream& out,
    const char* name,
    uint32_t logical_iterations,
    const std::vector<double>& gpu_samples,
    const std::vector<double>& host_samples,
    bool trailing_comma) {
  out << "      \"" << name << "\": {\n";
  out << "        \"logical_iterations\": " << logical_iterations << ",\n";
  out << "        \"dispatches_per_iteration\": 1,\n";
  out << "        \"gpu_samples_us\": ";
  write_samples(out, gpu_samples);
  out << ",\n";
  out << "        \"host_samples_us\": ";
  write_samples(out, host_samples);
  out << "\n      }" << (trailing_comma ? "," : "") << "\n";
}

std::string version_string(uint32_t version) {
  std::ostringstream out;
  out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
      << VK_VERSION_PATCH(version);
  return out.str();
}

Row make_row(
    const Args& args,
    const std::vector<double>& samples,
    float max_abs,
    float max_rel,
    bool single_pass,
    bool burst_pass,
    bool gpu_timestamps_supported,
    std::vector<double> single_gpu_samples,
    std::vector<double> single_host_samples,
    std::vector<double> burst_gpu_samples,
    std::vector<double> burst_host_samples) {
  double median_us = percentile(samples, 0.5);
  double ops = static_cast<double>(args.n) * args.body_iters * ops_per_element_iter();
  return Row{
      mode_name(),
      HIPENGINE_VOPD_ACCUMS,
      args.n,
      args.body_iters,
      kBlockSize,
      median_us,
      percentile(samples, 0.05),
      percentile(samples, 0.95),
      *std::min_element(samples.begin(), samples.end()),
      *std::max_element(samples.begin(), samples.end()),
      ops / median_us / 1000.0,
      max_abs,
      max_rel,
      single_pass && burst_pass,
      burst_pass,
      gpu_timestamps_supported,
      std::move(single_gpu_samples),
      std::move(single_host_samples),
      std::move(burst_gpu_samples),
      std::move(burst_host_samples),
  };
}

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const Row& row,
    std::ostream& out) {
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"run_tag\": \"vulkan-vopd-sweep\",\n";
  out << "  \"status\": \"diagnostic\",\n";
  out << "  \"backend\": \"vulkan\",\n";
  out << "  \"hardware\": {\n";
  out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  out << "    \"device_id\": " << properties.deviceID << ",\n";
  out << "    \"device_type\": " << properties.deviceType << ",\n";
  out << "    \"api_version\": \"" << version_string(properties.apiVersion) << "\",\n";
  out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  out << "    \"queue_family\": " << queue_family << "\n";
  out << "  },\n";
  out << "  \"config\": {\n";
  out << "    \"mode\": \"" << json_escape(row.mode) << "\",\n";
  out << "    \"mode_id\": " << HIPENGINE_VOPD_MODE << ",\n";
  out << "    \"accums\": " << row.accums << ",\n";
  out << "    \"n\": " << row.n << ",\n";
  out << "    \"body_iters\": " << row.body_iters << ",\n";
  out << "    \"block_size\": " << row.block_size << ",\n";
  out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"samples\": " << args.samples << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffer; pure VALU VOPD diagnostic; sampled CPU oracle\"\n";
  out << "  },\n";
  out << "  \"rows\": [\n";
  out << "    {\n";
  out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  out << "      \"accums\": " << row.accums << ",\n";
  out << "      \"n\": " << row.n << ",\n";
  out << "      \"body_iters\": " << row.body_iters << ",\n";
  out << "      \"block_size\": " << row.block_size << ",\n";
  out << "      \"median_us\": " << row.median_us << ",\n";
  out << "      \"p05_us\": " << row.p05_us << ",\n";
  out << "      \"p95_us\": " << row.p95_us << ",\n";
  out << "      \"min_us\": " << row.min_us << ",\n";
  out << "      \"max_us\": " << row.max_us << ",\n";
  out << "      \"gops\": " << row.gops << ",\n";
  out << "      \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "      \"queue_or_stream_count\": 1,\n";
  out << "      \"gpu_timestamps_supported\": "
      << (row.gpu_timestamps_supported ? "true" : "false") << ",\n";
  out << "      \"timed_sequence_correctness_pass\": "
      << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  out << "      \"synchronization_pass\": "
      << (row.timed_sequence_correctness_pass ? "true" : "false") << ",\n";
  out << "      \"barrier_count\": "
      << (args.timing_mode == "serial_latency" ? args.reps - 1 : 0) << ",\n";
  out << "      \"timing_raw\": {\n";
  write_timing_raw(
      out,
      "single",
      1,
      row.single_gpu_samples_us,
      row.single_host_samples_us,
      true);
  write_timing_raw(
      out,
      "burst",
      args.reps,
      row.burst_gpu_samples_us,
      row.burst_host_samples_us,
      false);
  out << "      },\n";
  out << "      \"max_abs\": " << row.max_abs << ",\n";
  out << "      \"max_rel\": " << row.max_rel << ",\n";
  out << "      \"correctness_pass\": " << (row.correctness_pass ? "true" : "false") << "\n";
  out << "    }\n";
  out << "  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<uint32_t> spirv = read_spirv(args.spirv_path);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan VOPD sweep";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "hipEngine microbench";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t physical_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    if (physical_count == 0) {
      fail("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    check(vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data()),
          "vkEnumeratePhysicalDevices(list)");
    if (args.device_index >= physical_count) {
      fail("--device-index is outside the physical-device list");
    }
    VkPhysicalDevice physical_device = physical_devices[args.device_index];
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    uint32_t queue_family = find_queue_family(physical_device);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module),
          "vkCreateShaderModule");

    const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
    const uint32_t output_slots =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? std::max(args.reps, args.warmup)
            : 1u;
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size =
        static_cast<VkDeviceSize>(args.n) * output_slots * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memory_requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
    uint32_t memory_type = find_memory_type(
        physical_device,
        memory_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    check(vkAllocateMemory(device, &allocate_info, nullptr, &memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
    void* mapped = nullptr;
    check(vkMapMemory(device, memory, 0, memory_requirements.size, 0, &mapped), "vkMapMemory");
    std::memset(mapped, 0, static_cast<size_t>(memory_requirements.size));

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = 1;
    descriptor_layout_info.pBindings = &binding;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr, &descriptor_layout),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
          "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
          "vkCreateComputePipelines");

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
          "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo descriptor_allocate_info{};
    descriptor_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate_info.descriptorPool = descriptor_pool;
    descriptor_allocate_info.descriptorSetCount = 1;
    descriptor_allocate_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check(vkAllocateDescriptorSets(device, &descriptor_allocate_info, &descriptor_set),
          "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo descriptor_buffer_info{};
    descriptor_buffer_info.buffer = buffer;
    descriptor_buffer_info.offset = 0;
    descriptor_buffer_info.range = buffer_info.size;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor_buffer_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = queue_family;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool),
          "vkCreateCommandPool");
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    {
      hipengine::micro::VulkanSequenceTimer timer(physical_device, device, queue_family);
      VkCommandBuffer single_command_buffer = record_command_buffer(
        device,
        command_pool,
        pipeline,
        pipeline_layout,
        descriptor_set,
        buffer,
        args,
        1,
        timing_mode,
        &timer);
      VkCommandBuffer burst_command_buffer = record_command_buffer(
        device,
        command_pool,
        pipeline,
        pipeline_layout,
        descriptor_set,
        buffer,
        args,
        args.reps,
        timing_mode,
        &timer);

      auto reset_outputs = [&]() {
        std::memset(mapped, 0, static_cast<size_t>(buffer_info.size));
      };
      auto check_outputs = [&](uint32_t slots,
                               uint32_t logical_iterations,
                               float& max_abs,
                               float& max_rel) {
        const float* values = static_cast<const float*>(mapped);
        const uint32_t checked = std::min<uint32_t>(args.n, 64);
        for (uint32_t slot = 0; slot < slots; ++slot) {
          for (uint32_t i = 0; i < checked; ++i) {
            const float dispatch_value = run_value(i, args.body_iters);
            float expected = 0.0f;
            for (uint32_t iteration = 0; iteration < logical_iterations; ++iteration) {
              expected += dispatch_value;
            }
            float observed = values[static_cast<size_t>(slot) * args.n + i];
            float diff = std::abs(observed - expected);
            max_abs = std::max(max_abs, diff);
            max_rel =
                std::max(max_rel, diff / std::max(1.0e-6f, std::abs(expected)));
          }
        }
      };

      reset_outputs();
      (void)timer.submit_and_wait(queue, single_command_buffer, fence);
      float max_abs = 0.0f;
      float max_rel = 0.0f;
      check_outputs(1, 1, max_abs, max_rel);
      const bool single_pass = max_abs <= 2.5e-3f || max_rel <= 2.5e-4f;

      reset_outputs();
      for (uint32_t i = 0; i < args.warmup; ++i) {
        (void)timer.submit_and_wait(queue, single_command_buffer, fence);
      }
      reset_outputs();
      std::vector<double> single_gpu_samples;
      std::vector<double> single_host_samples;
      for (uint32_t sample = 0; sample < args.samples; ++sample) {
        auto timing = timer.submit_and_wait(queue, single_command_buffer, fence);
        single_gpu_samples.push_back(timing.gpu_sequence_us);
        single_host_samples.push_back(timing.host_sequence_us);
      }
      reset_outputs();
      std::vector<double> burst_gpu_samples;
      std::vector<double> burst_host_samples;
      for (uint32_t sample = 0; sample < args.samples; ++sample) {
        auto timing = timer.submit_and_wait(queue, burst_command_buffer, fence);
        burst_gpu_samples.push_back(timing.gpu_sequence_us);
        burst_host_samples.push_back(timing.host_sequence_us);
      }

      reset_outputs();
      (void)timer.submit_and_wait(queue, burst_command_buffer, fence);
      check_outputs(
          timing_mode == hipengine::micro::TimingMode::IndependentThroughput
              ? args.reps
              : 1u,
          timing_mode == hipengine::micro::TimingMode::IndependentThroughput
              ? 1u
              : args.reps,
          max_abs,
          max_rel);
      const bool burst_pass = max_abs <= 2.5e-3f || max_rel <= 2.5e-4f;
      std::vector<double> samples;
      const std::vector<double>& burst_source =
          timer.gpu_timestamps_supported() ? burst_gpu_samples : burst_host_samples;
      for (double sample : burst_source) {
        samples.push_back(sample / args.reps);
      }
      Row row = make_row(
          args,
          samples,
          max_abs,
          max_rel,
          single_pass,
          burst_pass,
          timer.gpu_timestamps_supported(),
          std::move(single_gpu_samples),
          std::move(single_host_samples),
          std::move(burst_gpu_samples),
          std::move(burst_host_samples));
      std::cout << "[vulkan] mode=" << row.mode << " accums=" << row.accums
                << " median=" << row.median_us
                << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
                << "\n";

      if (args.json_path.empty()) {
        write_json(args, properties, queue_family, row, std::cout);
      } else {
        std::ofstream output(args.json_path);
        if (!output) {
          fail("could not open JSON path: " + args.json_path);
        }
        write_json(args, properties, queue_family, row, output);
      }

      vkFreeCommandBuffers(device, command_pool, 1, &single_command_buffer);
      vkFreeCommandBuffers(device, command_pool, 1, &burst_command_buffer);
    }
    vkUnmapMemory(device, memory);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
