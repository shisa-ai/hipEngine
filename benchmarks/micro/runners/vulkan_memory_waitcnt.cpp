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

#ifndef HIPENGINE_MEM_MODE
#define HIPENGINE_MEM_MODE 0
#endif

#ifndef HIPENGINE_MEM_PARAM
#define HIPENGINE_MEM_PARAM 1
#endif

namespace {

constexpr uint32_t kBlockSize = 256;

struct Args {
  std::string spirv_path;
  std::string json_path;
  uint32_t n = 32768;
  uint32_t body_iters = 128;
  uint32_t reps = 20;
  uint32_t warmup = 5;
  uint32_t samples = 7;
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t n;
  uint32_t body_iters;
  uint32_t data_mask;
  uint32_t reserved;
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

uint32_t hash_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

float data_value(uint32_t i) {
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

float run_value(
    const float* x,
    const uint32_t* ids,
    uint32_t idx,
    uint32_t n,
    uint32_t body_iters,
    uint32_t data_mask) {
  float sum = 0.0f;
  uint32_t state = idx * 747796405u + 2891336453u;
  for (uint32_t iter = 0; iter < body_iters; ++iter) {
#if HIPENGINE_MEM_MODE == 0
    uint32_t base = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
    for (uint32_t u = 0; u < HIPENGINE_MEM_PARAM; ++u) {
      float value = x[(base + u) & data_mask];
      sum = std::fma(value, 1.000001f + static_cast<float>(u) * 0.000001f, sum);
    }
#elif HIPENGINE_MEM_MODE == 1
    uint32_t addr = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
    sum = std::fma(x[addr], 1.000001f, sum);
#elif HIPENGINE_MEM_MODE == 2
    uint32_t id = ids[static_cast<size_t>(iter) * n + idx] & data_mask;
    sum = std::fma(x[id], 1.000001f, sum);
#elif HIPENGINE_MEM_MODE == 3
    uint32_t base = (iter * n * HIPENGINE_MEM_PARAM + idx * HIPENGINE_MEM_PARAM) & data_mask;
    for (uint32_t u = 0; u < HIPENGINE_MEM_PARAM; ++u) {
      state = hash_u32(state + iter + u);
      float value = x[(base + u) & data_mask];
      float bias = static_cast<float>(state & 0xffu) * 0.0000001f;
      sum = std::fma(value + bias, 0.999999f + static_cast<float>(u) * 0.000001f, sum);
    }
#endif
  }
  return sum;
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

void copy_inputs_to_device(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const Buffer& x_stage,
    const Buffer& ids_stage,
    const Buffer& x_device,
    const Buffer& ids_device,
    VkDeviceSize x_bytes,
    VkDeviceSize ids_bytes) {
  VkCommandBuffer command_buffer = begin_one_time(device, command_pool);
  VkBufferCopy x_copy{};
  x_copy.size = x_bytes;
  vkCmdCopyBuffer(command_buffer, x_stage.buffer, x_device.buffer, 1, &x_copy);
  VkBufferCopy ids_copy{};
  ids_copy.size = ids_bytes;
  vkCmdCopyBuffer(command_buffer, ids_stage.buffer, ids_device.buffer, 1, &ids_copy);
  submit_and_free(device, queue, command_pool, command_buffer);
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
    const Buffer& x_device,
    const Buffer& ids_device,
    const Buffer& out_device,
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
      {x_device.buffer, 0, x_device.size},
      {ids_device.buffer, 0, ids_device.size},
      {out_device.buffer, 0, out_device.size},
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

uint32_t grid_blocks(uint32_t n) {
  return std::max<uint32_t>(1, (n + kBlockSize - 1) / kBlockSize);
}

void record_dispatches(
    VkCommandBuffer command_buffer,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    const Args& args,
    uint32_t data_mask,
    uint32_t reps,
    bool copy_out,
    const Buffer& out_device,
    const Buffer& out_stage,
    VkDeviceSize out_bytes) {
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
  PushConstants push{args.n, args.body_iters, data_mask, 0};
  for (uint32_t rep = 0; rep < reps; ++rep) {
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, grid_blocks(args.n), 1, 1);
  }
  if (copy_out) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = out_device.buffer;
    barrier.offset = 0;
    barrier.size = out_bytes;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &barrier,
        0,
        nullptr);
    VkBufferCopy copy{};
    copy.size = out_bytes;
    vkCmdCopyBuffer(command_buffer, out_device.buffer, out_stage.buffer, 1, &copy);
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
  std::vector<float> x;
  std::vector<uint32_t> ids;
  fill_inputs(x, ids, args.n, args.body_iters);
  std::vector<float> actual(args.n, 0.0f);
  uint32_t data_mask = static_cast<uint32_t>(x.size() - 1);
  VkDeviceSize x_bytes = sizeof(float) * x.size();
  VkDeviceSize ids_bytes = sizeof(uint32_t) * ids.size();
  VkDeviceSize out_bytes = sizeof(float) * actual.size();

  Buffer x_stage = create_buffer(
      physical_device,
      device,
      x_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer ids_stage = create_buffer(
      physical_device,
      device,
      ids_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer out_stage = create_buffer(
      physical_device,
      device,
      out_bytes,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
  Buffer x_device = create_buffer(
      physical_device,
      device,
      x_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer ids_device = create_buffer(
      physical_device,
      device,
      ids_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);
  Buffer out_device = create_buffer(
      physical_device,
      device,
      out_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      false);

  std::memcpy(x_stage.mapped, x.data(), static_cast<size_t>(x_bytes));
  std::memcpy(ids_stage.mapped, ids.data(), static_cast<size_t>(ids_bytes));
  copy_inputs_to_device(
      device, queue, command_pool, x_stage, ids_stage, x_device, ids_device, x_bytes, ids_bytes);

  VkPipeline pipeline = create_pipeline(device, pipeline_layout, shader_module);
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = create_descriptor_set(
      device, descriptor_set_layout, x_device, ids_device, out_device, descriptor_pool);

  VkCommandBuffer correctness_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      correctness_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      1,
      true,
      out_device,
      out_stage,
      out_bytes);
  submit_and_free(device, queue, command_pool, correctness_cmd);
  std::memcpy(actual.data(), out_stage.mapped, static_cast<size_t>(out_bytes));

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

  VkCommandBuffer timing_cmd = begin_one_time(device, command_pool);
  record_dispatches(
      timing_cmd,
      pipeline,
      pipeline_layout,
      descriptor_set,
      args,
      data_mask,
      args.reps,
      false,
      out_device,
      out_stage,
      out_bytes);
  check(vkEndCommandBuffer(timing_cmd), "vkEndCommandBuffer timing");

  for (uint32_t i = 0; i < args.warmup; ++i) {
    (void)submit_once(device, queue, timing_cmd, fence);
  }
  std::vector<double> samples;
  samples.reserve(args.samples);
  for (uint32_t sample = 0; sample < args.samples; ++sample) {
    samples.push_back(submit_once(device, queue, timing_cmd, fence) / args.reps);
  }

  vkFreeCommandBuffers(device, command_pool, 1, &timing_cmd);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  destroy_buffer(device, x_stage);
  destroy_buffer(device, ids_stage);
  destroy_buffer(device, out_stage);
  destroy_buffer(device, x_device);
  destroy_buffer(device, ids_device);
  destroy_buffer(device, out_device);

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

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const Row& row,
    std::ostream& out) {
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"run_tag\": \"vulkan-memory-waitcnt\",\n";
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
  out << "    \"mode_id\": " << HIPENGINE_MEM_MODE << ",\n";
  out << "    \"param\": " << row.param << ",\n";
  out << "    \"n\": " << row.n << ",\n";
  out << "    \"body_iters\": " << row.body_iters << ",\n";
  out << "    \"block_size\": " << row.block_size << ",\n";
  out << "    \"data_elems\": " << row.data_elems << ",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"samples\": " << args.samples << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffer; device-local memory load plus accumulate diagnostic; sampled CPU oracle\"\n";
  out << "  },\n";
  out << "  \"rows\": [\n";
  out << "    {\n";
  out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
  out << "      \"param\": " << row.param << ",\n";
  out << "      \"n\": " << row.n << ",\n";
  out << "      \"body_iters\": " << row.body_iters << ",\n";
  out << "      \"block_size\": " << row.block_size << ",\n";
  out << "      \"data_elems\": " << row.data_elems << ",\n";
  out << "      \"bytes_per_dispatch\": " << row.bytes_per_dispatch << ",\n";
  out << "      \"median_us\": " << row.median_us << ",\n";
  out << "      \"p05_us\": " << row.p05_us << ",\n";
  out << "      \"p95_us\": " << row.p95_us << ",\n";
  out << "      \"min_us\": " << row.min_us << ",\n";
  out << "      \"max_us\": " << row.max_us << ",\n";
  out << "      \"bandwidth_gbps\": " << row.bandwidth_gbps << ",\n";
  out << "      \"gops\": " << row.gops << ",\n";
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
    app_info.pApplicationName = "hipEngine Vulkan memory/waitcnt";
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
    std::vector<uint32_t> shader_words = read_spirv(args.spirv_path);
    VkShaderModule shader_module = create_shader_module(device, shader_words);
    VkDescriptorSetLayout descriptor_layout = create_descriptor_set_layout(device);
    VkPipelineLayout pipeline_layout = create_pipeline_layout(device, descriptor_layout);

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

    Row row = run_config(
        args,
        physical_device,
        device,
        queue,
        command_pool,
        descriptor_layout,
        pipeline_layout,
        shader_module,
        fence);
    std::cout << "[vulkan] mode=" << row.mode << " param=" << row.param
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

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
