#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "micro_timing_vulkan.hpp"

namespace {

constexpr uint32_t kLocalSizeX = 256;

struct Args {
  std::string spirv_path;
  std::string json_path;
  std::vector<uint32_t> counts{1, 50, 200, 941};
  std::vector<uint32_t> grid_sweep;
  uint32_t grid_sweep_count = 941;
  uint32_t n = 256;
  uint32_t reps = 50;
  uint32_t warmup = 10;
  std::string timing_mode = "serial_latency";
  uint32_t device_index = 0;
};

struct PushConstants {
  uint32_t n;
  uint32_t output_base;
};

struct Row {
  uint32_t dispatch_count;
  uint32_t grid_blocks;
  double burst_us_median;
  double us_per_dispatch;
  double burst_us_min;
  uint32_t reps;
  std::vector<double> single_gpu_samples_us;
  std::vector<double> single_host_samples_us;
  std::vector<double> burst_gpu_samples_us;
  std::vector<double> burst_host_samples_us;
  bool single_correctness_pass;
  bool burst_correctness_pass;
  uint32_t correctness_mismatches;
  bool gpu_timestamps_supported;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
  uint32_t memory_type = 0;
  VkMemoryPropertyFlags memory_properties = 0;
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

std::vector<uint32_t> parse_u32_list(const std::string& text) {
  std::vector<uint32_t> values;
  if (text.empty()) {
    return values;
  }
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
    if (flag == "--spirv") {
      args.spirv_path = require_value(i, argc, argv, flag);
    } else if (flag == "--json") {
      args.json_path = require_value(i, argc, argv, flag);
    } else if (flag == "--counts") {
      args.counts = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--grid-sweep") {
      args.grid_sweep = parse_u32_list(require_value(i, argc, argv, flag));
    } else if (flag == "--grid-sweep-count") {
      args.grid_sweep_count = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--n") {
      args.n = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--reps") {
      args.reps = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--warmup") {
      args.warmup = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv, flag)));
    } else if (flag == "--timing-mode") {
      args.timing_mode = require_value(i, argc, argv, flag);
    } else if (flag == "--device-index") {
      args.device_index = static_cast<uint32_t>(
          std::stoul(require_value(i, argc, argv, flag)));
    } else {
      fail("unknown argument: " + flag);
    }
  }
  if (args.spirv_path.empty()) {
    fail("--spirv is required");
  }
  if (args.counts.empty()) {
    fail("--counts must not be empty");
  }
  if (args.n == 0 || args.reps == 0) {
    fail("--n and --reps must be positive");
  }
  (void)hipengine::micro::parse_timing_mode(args.timing_mode);
  return args;
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
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
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
    VkMemoryPropertyFlags required,
    bool map) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  Buffer result{};
  result.size = size;
  check(vkCreateBuffer(device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
  result.memory_type = find_memory_type(
      physical_device, requirements.memoryTypeBits, required);
  VkPhysicalDeviceMemoryProperties memory_properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
  result.memory_properties =
      memory_properties.memoryTypes[result.memory_type].propertyFlags;

  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = result.memory_type;
  check(vkAllocateMemory(device, &allocate_info, nullptr, &result.memory),
        "vkAllocateMemory");
  check(vkBindBufferMemory(device, result.buffer, result.memory, 0),
        "vkBindBufferMemory");
  if (map) {
    check(vkMapMemory(device, result.memory, 0, requirements.size, 0, &result.mapped),
          "vkMapMemory");
  }
  return result;
}

void destroy_buffer(VkDevice device, Buffer& buffer) {
  if (buffer.mapped != nullptr) {
    vkUnmapMemory(device, buffer.memory);
  }
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, buffer.buffer, nullptr);
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, buffer.memory, nullptr);
  }
  buffer = {};
}

uint32_t auto_grid(uint32_t n) {
  return std::max<uint32_t>(1, (n + kLocalSizeX - 1) / kLocalSizeX);
}

double median(std::vector<double> values) {
  if (values.empty()) {
    fail("cannot compute median of empty values");
  }
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if ((values.size() % 2) == 1) {
    return values[mid];
  }
  return (values[mid - 1] + values[mid]) * 0.5;
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

VkCommandBuffer record_command_buffer(
    VkDevice device,
    VkCommandPool command_pool,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    const Buffer& output,
    hipengine::micro::TimingMode timing_mode,
    uint32_t dispatch_count,
    uint32_t grid_blocks,
    uint32_t n,
    const hipengine::micro::VulkanSequenceTimer* timer = nullptr,
    const Buffer* readback = nullptr) {
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
  if (readback != nullptr) {
    vkCmdFillBuffer(command_buffer, output.buffer, 0, output.size, 0);
    VkBufferMemoryBarrier clear_barrier{};
    clear_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clear_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.buffer = output.buffer;
    clear_barrier.offset = 0;
    clear_barrier.size = output.size;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &clear_barrier,
        0,
        nullptr);
  }
  if (timer != nullptr) {
    timer->record_begin(command_buffer);
  }
  for (uint32_t i = 0; i < dispatch_count; ++i) {
    PushConstants push{
        n,
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? i * n
            : 0u};
    vkCmdPushConstants(
        command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PushConstants),
        &push);
    vkCmdDispatch(command_buffer, grid_blocks, 1, 1);
    if (timing_mode == hipengine::micro::TimingMode::SerialLatency &&
        i + 1 < dispatch_count) {
      hipengine::micro::compute_buffer_barrier(
          command_buffer,
          {hipengine::micro::make_compute_buffer_barrier(
              output.buffer,
              VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
              0,
              static_cast<VkDeviceSize>(n) * sizeof(float))});
    }
  }
  if (timer != nullptr) {
    timer->record_end(command_buffer);
  }
  if (readback != nullptr) {
    VkBufferMemoryBarrier copy_barrier{};
    copy_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copy_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier.buffer = output.buffer;
    copy_barrier.offset = 0;
    copy_barrier.size = output.size;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &copy_barrier,
        0,
        nullptr);
    VkBufferCopy copy{0, 0, output.size};
    vkCmdCopyBuffer(command_buffer, output.buffer, readback->buffer, 1, &copy);

    VkBufferMemoryBarrier host_barrier{};
    host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.buffer = readback->buffer;
    host_barrier.offset = 0;
    host_barrier.size = readback->size;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &host_barrier,
        0,
        nullptr);
  }
  check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  return command_buffer;
}

Row measure_row(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkDescriptorSet descriptor_set,
    const Buffer& output,
    const Buffer& readback,
    VkFence fence,
    const Args& args,
    uint32_t dispatch_count,
    uint32_t grid_blocks) {
  const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
  auto validate = [&](uint32_t count) {
    const uint32_t copies =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput ? count : 1u;
    const float expected =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? 1.0f
            : static_cast<float>(count);
    uint32_t mismatches = 0;
    const float* values = static_cast<const float*>(readback.mapped);
    for (uint32_t copy = 0; copy < copies; ++copy) {
      for (uint32_t i = 0; i < args.n; ++i) {
        if (values[static_cast<size_t>(copy) * args.n + i] != expected) {
          ++mismatches;
        }
      }
    }
    return mismatches;
  };
  auto run_correctness = [&](uint32_t count) {
    VkCommandBuffer command = record_command_buffer(
        device,
        command_pool,
        pipeline,
        pipeline_layout,
        descriptor_set,
        output,
        timing_mode,
        count,
        grid_blocks,
        args.n,
        nullptr,
        &readback);
    (void)submit_once(device, queue, command, fence);
    vkFreeCommandBuffers(device, command_pool, 1, &command);
    return validate(count);
  };
  uint32_t single_mismatches = run_correctness(1);
  uint32_t burst_mismatches = run_correctness(dispatch_count);

  hipengine::micro::VulkanSequenceTimer timer(
      physical_device, device, find_queue_family(physical_device));
  VkCommandBuffer single_command_buffer = record_command_buffer(
      device,
      command_pool,
      pipeline,
      pipeline_layout,
      descriptor_set,
      output,
      timing_mode,
      1,
      grid_blocks,
      args.n,
      &timer);
  VkCommandBuffer burst_command_buffer = record_command_buffer(
      device,
      command_pool,
      pipeline,
      pipeline_layout,
      descriptor_set,
      output,
      timing_mode,
      dispatch_count,
      grid_blocks,
      args.n,
      &timer);

  if (args.warmup > 0) {
    VkCommandBuffer warmup_command_buffer = record_command_buffer(
        device,
        command_pool,
        pipeline,
        pipeline_layout,
        descriptor_set,
        output,
        timing_mode,
        args.warmup,
        grid_blocks,
        args.n);
    (void)submit_once(device, queue, warmup_command_buffer, fence);
    vkFreeCommandBuffers(device, command_pool, 1, &warmup_command_buffer);
  }
  std::vector<double> single_gpu_samples;
  std::vector<double> single_host_samples;
  std::vector<double> burst_gpu_samples;
  std::vector<double> burst_host_samples;
  single_host_samples.reserve(args.reps);
  burst_host_samples.reserve(args.reps);
  for (uint32_t i = 0; i < args.reps; ++i) {
    auto single = timer.submit_and_wait(queue, single_command_buffer, fence);
    if (timer.gpu_timestamps_supported()) {
      single_gpu_samples.push_back(single.gpu_sequence_us);
    }
    single_host_samples.push_back(single.host_sequence_us);
  }
  for (uint32_t i = 0; i < args.reps; ++i) {
    auto burst = timer.submit_and_wait(queue, burst_command_buffer, fence);
    if (timer.gpu_timestamps_supported()) {
      burst_gpu_samples.push_back(burst.gpu_sequence_us);
    }
    burst_host_samples.push_back(burst.host_sequence_us);
  }
  vkFreeCommandBuffers(device, command_pool, 1, &single_command_buffer);
  vkFreeCommandBuffers(device, command_pool, 1, &burst_command_buffer);

  const auto& burst_samples =
      timer.gpu_timestamps_supported() ? burst_gpu_samples : burst_host_samples;
  double burst_median = median(burst_samples);
  double burst_min = *std::min_element(burst_samples.begin(), burst_samples.end());
  return Row{
      dispatch_count,
      grid_blocks,
      burst_median,
      burst_median / static_cast<double>(dispatch_count),
      burst_min,
      args.reps,
      std::move(single_gpu_samples),
      std::move(single_host_samples),
      std::move(burst_gpu_samples),
      std::move(burst_host_samples),
      single_mismatches == 0,
      burst_mismatches == 0,
      single_mismatches + burst_mismatches,
      timer.gpu_timestamps_supported()};
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

void write_rows(std::ostream& out, const std::vector<Row>& rows, int indent) {
  std::string pad(static_cast<size_t>(indent), ' ');
  out << "[\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    out << pad << "  {\n";
    out << pad << "    \"dispatch_count\": " << row.dispatch_count << ",\n";
    out << pad << "    \"grid_blocks\": " << row.grid_blocks << ",\n";
    out << pad << "    \"burst_us_median\": " << row.burst_us_median << ",\n";
    out << pad << "    \"us_per_dispatch\": " << row.us_per_dispatch << ",\n";
    out << pad << "    \"burst_us_min\": " << row.burst_us_min << ",\n";
    out << pad << "    \"reps\": " << row.reps << ",\n";
    out << pad << "    \"single_gpu_samples_us\": ";
    write_samples(out, row.single_gpu_samples_us);
    out << ",\n";
    out << pad << "    \"single_host_samples_us\": ";
    write_samples(out, row.single_host_samples_us);
    out << ",\n";
    out << pad << "    \"burst_gpu_samples_us\": ";
    write_samples(out, row.burst_gpu_samples_us);
    out << ",\n";
    out << pad << "    \"burst_host_samples_us\": ";
    write_samples(out, row.burst_host_samples_us);
    out << ",\n";
    out << pad << "    \"single_correctness_pass\": "
        << (row.single_correctness_pass ? "true" : "false") << ",\n";
    out << pad << "    \"burst_correctness_pass\": "
        << (row.burst_correctness_pass ? "true" : "false") << ",\n";
    out << pad << "    \"correctness_mismatches\": " << row.correctness_mismatches << ",\n";
    out << pad << "    \"gpu_timestamps_supported\": "
        << (row.gpu_timestamps_supported ? "true" : "false") << "\n";
    out << pad << "  }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
  out << pad << "]";
}

void write_json(
    const Args& args,
    const VkPhysicalDeviceProperties& properties,
    uint32_t queue_family,
    const Buffer& output_buffer,
    const Buffer& readback_buffer,
    const std::vector<Row>& rows,
    const std::vector<Row>& grid_rows,
    std::ostream& out) {
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"run_tag\": \"vulkan-dispatch-floor\",\n";
  out << "  \"status\": \"diagnostic\",\n";
  out << "  \"hardware\": {\n";
  out << "    \"device_name\": \"" << json_escape(properties.deviceName) << "\",\n";
  out << "    \"vendor_id\": " << properties.vendorID << ",\n";
  out << "    \"device_id\": " << properties.deviceID << ",\n";
  out << "    \"device_type\": " << properties.deviceType << ",\n";
  out << "    \"api_version\": \"" << version_string(properties.apiVersion) << "\",\n";
  out << "    \"driver_version_raw\": " << properties.driverVersion << ",\n";
  out << "    \"queue_family\": " << queue_family << ",\n";
  out << "    \"output_memory_type\": " << output_buffer.memory_type << ",\n";
  out << "    \"output_memory_property_flags\": "
      << output_buffer.memory_properties << ",\n";
  out << "    \"output_device_local\": "
      << ((output_buffer.memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
              ? "true"
              : "false")
      << ",\n";
  out << "    \"readback_memory_type\": " << readback_buffer.memory_type << ",\n";
  out << "    \"readback_memory_property_flags\": "
      << readback_buffer.memory_properties << "\n";
  out << "  },\n";
  out << "  \"config\": {\n";
  out << "    \"counts\": [";
  for (size_t i = 0; i < args.counts.size(); ++i) {
    out << (i == 0 ? "" : ", ") << args.counts[i];
  }
  out << "],\n";
  out << "    \"grid_sweep\": [";
  for (size_t i = 0; i < args.grid_sweep.size(); ++i) {
    out << (i == 0 ? "" : ", ") << args.grid_sweep[i];
  }
  out << "],\n";
  out << "    \"grid_sweep_count\": " << args.grid_sweep_count << ",\n";
  out << "    \"n_elements\": " << args.n << ",\n";
  out << "    \"reps\": " << args.reps << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"timing_mode\": \"" << json_escape(args.timing_mode) << "\",\n";
  out << "    \"local_size_x\": " << kLocalSizeX << ",\n";
  out << "    \"method\": \"pre-recorded Vulkan command buffers with timestamp and host-wall single/burst controls; exact burst correctness\"\n";
  out << "  },\n";
  out << "  \"rows\": ";
  write_rows(out, rows, 2);
  out << ",\n";
  out << "  \"grid_sweep_rows\": ";
  write_rows(out, grid_rows, 2);
  out << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<uint32_t> spirv = read_spirv(args.spirv_path);

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "hipEngine Vulkan dispatch floor";
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

    uint32_t max_dispatch_count = args.grid_sweep_count;
    for (uint32_t count : args.counts) {
      max_dispatch_count = std::max(max_dispatch_count, count);
    }
    max_dispatch_count = std::max(max_dispatch_count, std::max(args.warmup, 1u));
    const auto timing_mode = hipengine::micro::parse_timing_mode(args.timing_mode);
    uint32_t output_copies =
        timing_mode == hipengine::micro::TimingMode::IndependentThroughput
            ? max_dispatch_count
            : 1u;
    VkDeviceSize output_size =
        static_cast<VkDeviceSize>(args.n) * output_copies * sizeof(float);
    Buffer output_buffer = create_buffer(
        physical_device,
        device,
        output_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);
    Buffer readback_buffer = create_buffer(
        physical_device,
        device,
        output_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);

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
    check(vkCreateDescriptorSetLayout(
              device, &descriptor_layout_info, nullptr, &descriptor_layout),
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
    check(vkCreateComputePipelines(
              device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
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
    descriptor_buffer_info.buffer = output_buffer.buffer;
    descriptor_buffer_info.offset = 0;
    descriptor_buffer_info.range = output_buffer.size;
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

    std::vector<Row> rows;
    rows.reserve(args.counts.size());
    for (uint32_t count : args.counts) {
      rows.push_back(measure_row(
          physical_device,
          device,
          queue,
          command_pool,
          pipeline,
          pipeline_layout,
          descriptor_set,
          output_buffer,
          readback_buffer,
          fence,
          args,
          count,
          auto_grid(args.n)));
      const Row& row = rows.back();
      std::cout << "[vulkan] N=" << row.dispatch_count
                << " grid=" << row.grid_blocks
                << " submit=" << row.us_per_dispatch << " us/dispatch\n";
    }

    std::vector<Row> grid_rows;
    grid_rows.reserve(args.grid_sweep.size());
    for (uint32_t grid_blocks : args.grid_sweep) {
      grid_rows.push_back(measure_row(
          physical_device,
          device,
          queue,
          command_pool,
          pipeline,
          pipeline_layout,
          descriptor_set,
          output_buffer,
          readback_buffer,
          fence,
          args,
          args.grid_sweep_count,
          grid_blocks));
      const Row& row = grid_rows.back();
      std::cout << "[grid] blocks=" << row.grid_blocks
                << " submit=" << row.us_per_dispatch << " us/dispatch\n";
    }

    if (args.json_path.empty()) {
      write_json(
          args,
          properties,
          queue_family,
          output_buffer,
          readback_buffer,
          rows,
          grid_rows,
          std::cout);
    } else {
      std::ofstream output(args.json_path);
      if (!output) {
        fail("could not open JSON output: " + args.json_path);
      }
      write_json(
          args,
          properties,
          queue_family,
          output_buffer,
          readback_buffer,
          rows,
          grid_rows,
          output);
      std::cout << "wrote " << args.json_path << "\n";
    }

    check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    destroy_buffer(device, readback_buffer);
    destroy_buffer(device, output_buffer);
    vkDestroyShaderModule(device, shader_module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
