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
#include <vector>

#include "micro_timing_vulkan.hpp"

#ifndef HIPENGINE_ARGMAX_WG
#define HIPENGINE_ARGMAX_WG 256
#endif

#ifndef HIPENGINE_ARGMAX_TOPK
#define HIPENGINE_ARGMAX_TOPK 1
#endif

namespace {

constexpr uint32_t kWorkgroupSize = HIPENGINE_ARGMAX_WG;
constexpr uint32_t kTopK = HIPENGINE_ARGMAX_TOPK;
static_assert(kTopK > 0, "HIPENGINE_ARGMAX_TOPK must be positive");
static_assert(kTopK <= kWorkgroupSize, "HIPENGINE_ARGMAX_TOPK must be <= HIPENGINE_ARGMAX_WG");

struct Args {
  std::string spirv_path;
  std::string json_path;
  uint32_t rows = 4;
  uint32_t vocab = 32768;
  uint32_t reps = 50;
  uint32_t warmup = 10;
  uint32_t samples = 9;
  std::string timing_mode = "serial_latency";
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t rows;
  uint32_t vocab;
  uint32_t output_base;
  uint32_t reserved1;
};

struct Row {
  uint32_t rows;
  uint32_t vocab;
  uint32_t workgroup_size;
  uint32_t top_k;
  double bytes_per_dispatch;
  double comparisons_per_dispatch;
  double median_us;
  double p05_us;
  double p95_us;
  double min_us;
  double max_us;
  double bandwidth_gbps;
  double gcomparisons_per_s;
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
  uint32_t single_mismatches;
  uint32_t burst_mismatches;
  float max_abs;
  bool correctness_pass;
  bool gpu_timestamps_supported;
  double timestamp_period_ns;
  uint32_t timestamp_valid_bits;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
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
    } else if (flag == "--rows") {
      args.rows = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--vocab") {
      args.vocab = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
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
  if (args.rows == 0 || args.vocab == 0 || args.reps == 0 || args.samples == 0) {
    fail("--rows, --vocab, --reps, and --samples must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  return args;
}

uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

uint32_t peak_index(uint32_t row, uint32_t vocab) {
  return hash_u32(row * 747796405u + 2891336453u) % vocab;
}

float logit_value(uint32_t row, uint32_t col, uint32_t vocab) {
  if (col == peak_index(row, vocab)) {
    return 64.0f + static_cast<float>(row) * 0.25f;
  }
  uint32_t bits = hash_u32(row * 1664525u + col * 1013904223u);
  int value = static_cast<int>(bits & 0xffffu) - 32768;
  return static_cast<float>(value) * 0.000030517578125f;
}

struct Pair {
  float value;
  uint32_t index;
};

bool better_pair(Pair a, Pair b) {
  return (a.value > b.value) || (a.value == b.value && a.index < b.index);
}

void insert_topk(std::vector<Pair>& top, Pair candidate) {
  for (uint32_t i = 0; i < kTopK; ++i) {
    if (better_pair(candidate, top[i])) {
      for (uint32_t j = kTopK - 1; j > i; --j) {
        top[j] = top[j - 1];
      }
      top[i] = candidate;
      return;
    }
  }
}

void cpu_topk(const std::vector<float>& logits, uint32_t row, uint32_t vocab, std::vector<Pair>& out) {
  out.assign(kTopK, Pair{-std::numeric_limits<float>::infinity(), 0xffffffffu});
  size_t base = static_cast<size_t>(row) * vocab;
  for (uint32_t col = 0; col < vocab; ++col) {
    insert_topk(out, Pair{logits[base + col], col});
  }
}

void fill_logits(std::vector<float>& logits, uint32_t rows, uint32_t vocab) {
  logits.resize(static_cast<size_t>(rows) * vocab);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < vocab; ++col) {
      logits[static_cast<size_t>(row) * vocab + col] = logit_value(row, col, vocab);
    }
  }
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

Buffer create_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    bool map) {
  Buffer buffer{};
  buffer.size = size;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer.buffer), "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex =
      find_memory_type(physical_device, requirements.memoryTypeBits, properties);
  check(vkAllocateMemory(device, &allocate_info, nullptr, &buffer.memory), "vkAllocateMemory");
  check(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");
  if (map) {
    check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped), "vkMapMemory");
  }
  return buffer;
}

void destroy_buffer(VkDevice device, Buffer& buffer) {
  if (buffer.mapped != nullptr) {
    vkUnmapMemory(device, buffer.memory);
    buffer.mapped = nullptr;
  }
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, buffer.buffer, nullptr);
    buffer.buffer = VK_NULL_HANDLE;
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, buffer.memory, nullptr);
    buffer.memory = VK_NULL_HANDLE;
  }
}

VkCommandBuffer begin_one_time(VkDevice device, VkCommandPool command_pool) {
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
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
  return command_buffer;
}

void submit_and_free(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    VkCommandBuffer command_buffer) {
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
  check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(3);
  for (uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.bindingCount = static_cast<uint32_t>(bindings.size());
  create_info.pBindings = bindings.data();
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  check(vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout),
        "vkCreateDescriptorSetLayout");
  return layout;
}

VkPipelineLayout create_pipeline_layout(VkDevice device, VkDescriptorSetLayout descriptor_set_layout) {
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(PushConstants);
  VkPipelineLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.setLayoutCount = 1;
  create_info.pSetLayouts = &descriptor_set_layout;
  create_info.pushConstantRangeCount = 1;
  create_info.pPushConstantRanges = &push_range;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  check(vkCreatePipelineLayout(device, &create_info, nullptr, &layout),
        "vkCreatePipelineLayout");
  return layout;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& spirv) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(uint32_t);
  create_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &create_info, nullptr, &module), "vkCreateShaderModule");
  return module;
}

VkPipeline create_pipeline(VkDevice device, VkPipelineLayout pipeline_layout, VkShaderModule shader_module) {
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader_module;
  stage.pName = "main";
  VkComputePipelineCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  create_info.stage = stage;
  create_info.layout = pipeline_layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pipeline),
        "vkCreateComputePipelines");
  return pipeline;
}

VkDescriptorSet create_descriptor_set(
    VkDevice device,
    VkDescriptorSetLayout descriptor_set_layout,
    const Buffer& logits_device,
    const Buffer& indices_device,
    const Buffer& values_device,
    VkDescriptorPool& descriptor_pool) {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 3;
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
        "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.descriptorPool = descriptor_pool;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_set_layout;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  check(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set),
        "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo infos[3] = {
      {logits_device.buffer, 0, logits_device.size},
      {indices_device.buffer, 0, indices_device.size},
      {values_device.buffer, 0, values_device.size},
  };
  std::vector<VkWriteDescriptorSet> writes(3);
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return descriptor_set;
}

void copy_inputs_to_device(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const Buffer& logits_stage,
    const Buffer& logits_device,
    VkDeviceSize logits_bytes) {
  VkCommandBuffer command_buffer = begin_one_time(device, command_pool);
  VkBufferCopy copy{};
  copy.size = logits_bytes;
  vkCmdCopyBuffer(command_buffer, logits_stage.buffer, logits_device.buffer, 1, &copy);
  submit_and_free(device, queue, command_pool, command_buffer);
}

void record_dispatches(
    VkCommandBuffer command_buffer,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    const Args& args,
    hipengine::micro::TimingMode timing_mode,
    uint32_t reps,
    bool copy_out,
    const Buffer& indices_device,
    const Buffer& indices_stage,
    const Buffer& values_device,
    const Buffer& values_stage,
    VkDeviceSize copy_index_bytes,
    VkDeviceSize copy_value_bytes,
    const hipengine::micro::VulkanSequenceTimer* timer = nullptr) {
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
  if (timer != nullptr) {
    timer->record_begin(command_buffer);
  }
  const uint32_t output_elems = args.rows * kTopK;
  for (uint32_t rep = 0; rep < reps; ++rep) {
    PushConstants push{
        args.rows,
        args.vocab,
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? rep * output_elems
            : 0u,
        rep};
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, args.rows, 1, 1);
    if (timing_mode == hipengine::micro::TimingMode::SerialLatency && rep + 1 < reps) {
      std::vector<VkBufferMemoryBarrier> barriers{
          hipengine::micro::make_compute_buffer_barrier(
              indices_device.buffer,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
              0,
              copy_index_bytes),
          hipengine::micro::make_compute_buffer_barrier(
              values_device.buffer,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_WRITE_BIT,
              0,
              copy_value_bytes),
      };
      hipengine::micro::compute_buffer_barrier(command_buffer, barriers);
    }
  }
  if (timer != nullptr) {
    timer->record_end(command_buffer);
  }
  if (copy_out) {
    VkBufferMemoryBarrier barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = indices_device.buffer;
    barriers[0].offset = 0;
    barriers[0].size = copy_index_bytes;
    barriers[1] = barriers[0];
    barriers[1].buffer = values_device.buffer;
    barriers[1].size = copy_value_bytes;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        2,
        barriers,
        0,
        nullptr);
    VkBufferCopy index_copy{};
    index_copy.size = copy_index_bytes;
    vkCmdCopyBuffer(command_buffer, indices_device.buffer, indices_stage.buffer, 1, &index_copy);
    VkBufferCopy value_copy{};
    value_copy.size = copy_value_bytes;
    vkCmdCopyBuffer(command_buffer, values_device.buffer, values_stage.buffer, 1, &value_copy);
  }
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

struct CorrectnessResult {
  uint32_t mismatches = 0;
  float max_abs = 0.0f;
};

CorrectnessResult validate_outputs(
    const std::vector<float>& logits,
    const std::vector<uint32_t>& actual_indices,
    const std::vector<float>& actual_values,
    uint32_t rows,
    uint32_t vocab,
    uint32_t copies,
    bool independent,
    uint32_t final_sequence_id) {
  CorrectnessResult result;
  std::vector<Pair> expected;
  const size_t output_elems = static_cast<size_t>(rows) * kTopK;
  for (uint32_t copy = 0; copy < copies; ++copy) {
    for (uint32_t row = 0; row < rows; ++row) {
      cpu_topk(logits, row, vocab, expected);
      for (uint32_t i = 0; i < kTopK; ++i) {
        size_t offset = static_cast<size_t>(copy) * output_elems +
                        static_cast<size_t>(row) * kTopK + i;
        uint32_t sequence_id = independent ? copy : final_sequence_id;
        float tagged_expected =
            expected[i].value + static_cast<float>(sequence_id) * 0.125f;
        float diff = std::abs(actual_values[offset] - tagged_expected);
        result.max_abs = std::max(result.max_abs, diff);
        if (actual_indices[offset] != expected[i].index || diff > 0.0f) {
          ++result.mismatches;
        }
      }
    }
  }
  return result;
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

std::string version_string(uint32_t version) {
  std::ostringstream out;
  out << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
      << VK_VERSION_PATCH(version);
  return out.str();
}

void write_samples(std::ostream& out, const std::vector<double>& samples) {
  out << "[";
  for (size_t i = 0; i < samples.size(); ++i) {
    out << (i == 0 ? "" : ", ") << samples[i];
  }
  out << "]";
}

Row run_config(
    const Args& args,
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipelineLayout pipeline_layout,
    VkShaderModule shader_module,
    VkFence fence) {
  if (kTopK > args.vocab) {
    fail("top-k must be <= vocab");
  }
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  const bool independent =
      timing_mode == hipengine::micro::TimingMode::IndependentThroughput;
  std::vector<float> logits;
  fill_logits(logits, args.rows, args.vocab);
  const size_t output_elems = static_cast<size_t>(args.rows) * kTopK;
  const uint32_t output_copies = independent ? args.reps : 1u;
  const uint32_t allocation_copies =
      independent ? std::max(args.reps, args.warmup) : 1u;
  const size_t allocation_elems = output_elems * allocation_copies;
  std::vector<uint32_t> actual_indices(allocation_elems, 0);
  std::vector<float> actual_values(allocation_elems, 0.0f);
  VkDeviceSize logits_bytes = sizeof(float) * logits.size();
  VkDeviceSize index_bytes = sizeof(uint32_t) * allocation_elems;
  VkDeviceSize value_bytes = sizeof(float) * allocation_elems;

  Buffer logits_stage = create_buffer(
      physical_device,
      device,
      logits_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer indices_stage = create_buffer(
      physical_device,
      device,
      index_bytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer values_stage = create_buffer(
      physical_device,
      device,
      value_bytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer logits_device = create_buffer(
      physical_device,
      device,
      logits_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer indices_device = create_buffer(
      physical_device,
      device,
      index_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer values_device = create_buffer(
      physical_device,
      device,
      value_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);

  std::memcpy(logits_stage.mapped, logits.data(), static_cast<size_t>(logits_bytes));
  copy_inputs_to_device(device, queue, command_pool, logits_stage, logits_device, logits_bytes);

  VkPipeline pipeline = create_pipeline(device, pipeline_layout, shader_module);
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = create_descriptor_set(
      device, descriptor_set_layout, logits_device, indices_device, values_device, descriptor_pool);

  VkCommandBuffer single_correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      single_correctness_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      timing_mode,
      1,
      true,
      indices_device,
      indices_stage,
      values_device,
      values_stage,
      sizeof(uint32_t) * output_elems,
      sizeof(float) * output_elems);
  submit_and_free(device, queue, command_pool, single_correctness_cmd);
  std::memcpy(
      actual_indices.data(), indices_stage.mapped, sizeof(uint32_t) * output_elems);
  std::memcpy(actual_values.data(), values_stage.mapped, sizeof(float) * output_elems);
  CorrectnessResult single_correctness =
      validate_outputs(logits, actual_indices, actual_values, args.rows, args.vocab, 1, independent, 0);

  VkCommandBuffer burst_correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      burst_correctness_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      timing_mode,
      args.reps,
      true,
      indices_device,
      indices_stage,
      values_device,
      values_stage,
      index_bytes,
      value_bytes);
  submit_and_free(device, queue, command_pool, burst_correctness_cmd);
  std::memcpy(actual_indices.data(), indices_stage.mapped, static_cast<size_t>(index_bytes));
  std::memcpy(actual_values.data(), values_stage.mapped, static_cast<size_t>(value_bytes));
  CorrectnessResult burst_correctness = validate_outputs(
      logits,
      actual_indices,
      actual_values,
      args.rows,
      args.vocab,
      output_copies,
      independent,
      args.reps - 1);

  hipengine::micro::VulkanSequenceTimer timer(physical_device, device, find_queue_family(physical_device));
  auto make_timing_command = [&](uint32_t iterations) {
    VkCommandBuffer command = begin_one_time(device, command_pool);
    record_dispatches(
        command,
        pipeline,
        pipeline_layout,
        descriptor_set,
        args,
        timing_mode,
        iterations,
        false,
        indices_device,
        indices_stage,
        values_device,
        values_stage,
        index_bytes,
        value_bytes,
        &timer);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer timing");
    return command;
  };
  VkCommandBuffer single_timing_cmd = make_timing_command(1);
  VkCommandBuffer burst_timing_cmd = make_timing_command(args.reps);
  if (args.warmup > 0) {
    VkCommandBuffer warmup_cmd = begin_one_time(device, command_pool);
    record_dispatches(
        warmup_cmd,
        pipeline,
        pipeline_layout,
        descriptor_set,
        args,
        timing_mode,
        args.warmup,
        false,
        indices_device,
        indices_stage,
        values_device,
        values_stage,
        index_bytes,
        value_bytes);
    submit_and_free(device, queue, command_pool, warmup_cmd);
  }
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    auto single = timer.submit_and_wait(queue, single_timing_cmd, fence);
    if (timer.gpu_timestamps_supported()) {
      single_gpu_samples_us.push_back(single.gpu_sequence_us);
    }
    single_host_samples_us.push_back(single.host_sequence_us);
    auto burst = timer.submit_and_wait(queue, burst_timing_cmd, fence);
    if (timer.gpu_timestamps_supported()) {
      burst_gpu_samples_us.push_back(burst.gpu_sequence_us);
    }
    burst_host_samples_us.push_back(burst.host_sequence_us);
  }

  vkFreeCommandBuffers(device, command_pool, 1, &single_timing_cmd);
  vkFreeCommandBuffers(device, command_pool, 1, &burst_timing_cmd);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  destroy_buffer(device, logits_stage);
  destroy_buffer(device, indices_stage);
  destroy_buffer(device, values_stage);
  destroy_buffer(device, logits_device);
  destroy_buffer(device, indices_device);
  destroy_buffer(device, values_device);

  std::vector<double> burst_per_iteration_us;
  const std::vector<double>& sequence_samples =
      timer.gpu_timestamps_supported() ? burst_gpu_samples_us : burst_host_samples_us;
  burst_per_iteration_us.reserve(sequence_samples.size());
  for (double value : sequence_samples) {
    burst_per_iteration_us.push_back(value / args.reps);
  }
  double median_us = percentile(burst_per_iteration_us, 0.5);
  double scanned_values =
      static_cast<double>(args.rows) *
      (static_cast<double>(kTopK) * args.vocab - static_cast<double>(kTopK * (kTopK - 1)) / 2.0);
  double bytes = scanned_values * sizeof(float);
  double comparisons = scanned_values;
  bool pass = single_correctness.mismatches == 0 && burst_correctness.mismatches == 0;
  return Row{
      args.rows,
      args.vocab,
      kWorkgroupSize,
      kTopK,
      bytes,
      comparisons,
      median_us,
      percentile(burst_per_iteration_us, 0.05),
      percentile(burst_per_iteration_us, 0.95),
      *std::min_element(burst_per_iteration_us.begin(), burst_per_iteration_us.end()),
      *std::max_element(burst_per_iteration_us.begin(), burst_per_iteration_us.end()),
      bytes / median_us / 1000.0,
      comparisons / median_us / 1000.0,
      std::move(single_gpu_samples_us),
      std::move(single_host_samples_us),
      std::move(burst_gpu_samples_us),
      std::move(burst_host_samples_us),
      single_correctness.mismatches,
      burst_correctness.mismatches,
      std::max(single_correctness.max_abs, burst_correctness.max_abs),
      pass,
      timer.gpu_timestamps_supported(),
      timer.timestamp_period_ns(),
      timer.timestamp_valid_bits(),
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
  out << "  \"run_tag\": \"vulkan-sampler-argmax\",\n";
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
  out << "    \"rows\": " << row.rows << ",\n";
  out << "    \"vocab\": " << row.vocab << ",\n";
  out << "    \"workgroup_size\": " << row.workgroup_size << ",\n";
  out << "    \"top_k\": " << row.top_k << ",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"samples\": " << args.samples << ",\n";
  out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "    \"gpu_timestamps_supported\": "
      << (row.gpu_timestamps_supported ? "true" : "false") << ",\n";
  out << "    \"timestamp_period_ns\": " << row.timestamp_period_ns << ",\n";
  out << "    \"timestamp_valid_bits\": " << row.timestamp_valid_bits << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffers with GPU timestamps and submit/fence host wall; exact single and burst CPU oracles\"\n";
  out << "  },\n";
  out << "  \"rows\": [\n";
  out << "    {\n";
  out << "      \"rows\": " << row.rows << ",\n";
  out << "      \"vocab\": " << row.vocab << ",\n";
  out << "      \"workgroup_size\": " << row.workgroup_size << ",\n";
  out << "      \"top_k\": " << row.top_k << ",\n";
  out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  out << "      \"comparisons_per_dispatch\": " << row.comparisons_per_dispatch << ",\n";
  out << "      \"median_us\": " << row.median_us << ",\n";
  out << "      \"p05_us\": " << row.p05_us << ",\n";
  out << "      \"p95_us\": " << row.p95_us << ",\n";
  out << "      \"min_us\": " << row.min_us << ",\n";
  out << "      \"max_us\": " << row.max_us << ",\n";
  out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  out << "      \"gcomparisons_per_s\": " << row.gcomparisons_per_s << ",\n";
  out << "      \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "      \"single_gpu_samples_us\": ";
  write_samples(out, row.single_gpu_samples_us);
  out << ",\n      \"single_host_samples_us\": ";
  write_samples(out, row.single_host_samples_us);
  out << ",\n      \"burst_gpu_samples_us\": ";
  write_samples(out, row.burst_gpu_samples_us);
  out << ",\n      \"burst_host_samples_us\": ";
  write_samples(out, row.burst_host_samples_us);
  out << ",\n";
  out << "      \"gpu_timestamps_supported\": "
      << (row.gpu_timestamps_supported ? "true" : "false") << ",\n";
  out << "      \"single_mismatches\": " << row.single_mismatches << ",\n";
  out << "      \"burst_mismatches\": " << row.burst_mismatches << ",\n";
  out << "      \"mismatches\": " << (row.single_mismatches + row.burst_mismatches) << ",\n";
  out << "      \"max_abs\": " << row.max_abs << ",\n";
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
    app_info.pApplicationName = "hipEngine Vulkan sampler argmax";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "hipEngine microbench";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
          "vkEnumeratePhysicalDevices count");
    if (device_count == 0 || args.device_index >= device_count) {
      fail("requested Vulkan physical device is unavailable");
    }
    std::vector<VkPhysicalDevice> physical_devices(device_count);
    check(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()),
          "vkEnumeratePhysicalDevices");
    VkPhysicalDevice physical_device = physical_devices[args.device_index];
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    uint32_t queue_family = find_queue_family(physical_device);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool), "vkCreateCommandPool");

    VkDescriptorSetLayout descriptor_set_layout = create_descriptor_set_layout(device);
    VkPipelineLayout pipeline_layout = create_pipeline_layout(device, descriptor_set_layout);
    VkShaderModule shader_module = create_shader_module(device, spirv);
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    Row row = run_config(
        args,
        physical_device,
        device,
        queue,
        command_pool,
        descriptor_set_layout,
        pipeline_layout,
        shader_module,
        fence);
    std::cerr << "[vulkan] rows=" << row.rows << " vocab=" << row.vocab
              << " wg=" << row.workgroup_size << " median=" << row.median_us
              << " us correctness=" << (row.correctness_pass ? "pass" : "fail")
              << "\n";

    if (!args.json_path.empty()) {
      std::ofstream file(args.json_path);
      if (!file) {
        fail("could not open JSON path: " + args.json_path);
      }
      write_json(args, properties, queue_family, row, file);
    } else {
      write_json(args, properties, queue_family, row, std::cout);
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return row.correctness_pass ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
