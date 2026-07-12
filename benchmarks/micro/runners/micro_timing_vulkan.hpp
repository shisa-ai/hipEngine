#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipengine::micro {

enum class TimingMode { SerialLatency, IndependentThroughput };

inline TimingMode parse_timing_mode(const std::string& value) {
  if (value == "serial_latency") {
    return TimingMode::SerialLatency;
  }
  if (value == "independent_throughput") {
    return TimingMode::IndependentThroughput;
  }
  throw std::runtime_error(
      "timing mode must be serial_latency or independent_throughput");
}

inline const char* timing_mode_name(TimingMode mode) {
  return mode == TimingMode::SerialLatency ? "serial_latency" : "independent_throughput";
}

inline void timing_vk_check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        std::string(what) + " failed with VkResult " +
        std::to_string(static_cast<int>(result)));
  }
}

inline void compute_buffer_barrier(
    VkCommandBuffer command_buffer,
    const std::vector<VkBufferMemoryBarrier>& barriers) {
  if (barriers.empty()) {
    return;
  }
  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(barriers.size()),
      barriers.data(),
      0,
      nullptr);
}

inline VkBufferMemoryBarrier make_compute_buffer_barrier(
    VkBuffer buffer,
    VkAccessFlags src_access,
    VkAccessFlags dst_access,
    VkDeviceSize offset = 0,
    VkDeviceSize size = VK_WHOLE_SIZE) {
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer;
  barrier.offset = offset;
  barrier.size = size;
  return barrier;
}

struct VulkanTimingSample {
  double gpu_sequence_us;
  double host_sequence_us;
};

struct VulkanMultiQueueTimingSample {
  double gpu_sequence_us;
  double host_sequence_us;
  std::vector<double> lane_gpu_us;
  bool calibrated_timestamp_domain;
};

struct VulkanQueueFamilySelection {
  uint32_t index;
  uint32_t queue_count;
  VkQueueFlags flags;
  uint32_t timestamp_valid_bits;
};

inline VulkanQueueFamilySelection select_compute_queue_family(
    VkPhysicalDevice physical_device,
    uint32_t requested_queues = 1) {
  if (requested_queues == 0) {
    throw std::runtime_error("requested Vulkan queue count must be positive");
  }
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
  if (family_count == 0) {
    throw std::runtime_error("physical device has no Vulkan queue families");
  }
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &family_count, families.data());

  uint32_t best_index = UINT32_MAX;
  uint32_t best_usable = 0;
  bool best_compute_only = false;
  for (uint32_t i = 0; i < family_count; ++i) {
    const auto& family = families[i];
    if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0 ||
        family.queueCount == 0 || family.timestampValidBits == 0) {
      continue;
    }
    const uint32_t usable = std::min(requested_queues, family.queueCount);
    const bool compute_only = (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
    if (best_index == UINT32_MAX || usable > best_usable ||
        (usable == best_usable && compute_only && !best_compute_only)) {
      best_index = i;
      best_usable = usable;
      best_compute_only = compute_only;
    }
  }
  if (best_index == UINT32_MAX) {
    throw std::runtime_error(
        "physical device has no timestamp-capable Vulkan compute queue family");
  }
  const auto& best = families[best_index];
  return {best_index, best_usable, best.queueFlags, best.timestampValidBits};
}

inline bool timeline_semaphore_supported(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
  timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  VkPhysicalDeviceFeatures2 features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features.pNext = &timeline;
  vkGetPhysicalDeviceFeatures2(physical_device, &features);
  return timeline.timelineSemaphore == VK_TRUE;
}

inline bool device_extension_supported(
    VkPhysicalDevice physical_device,
    const char* extension_name) {
  uint32_t count = 0;
  timing_vk_check(
      vkEnumerateDeviceExtensionProperties(
          physical_device, nullptr, &count, nullptr),
      "vkEnumerateDeviceExtensionProperties count");
  std::vector<VkExtensionProperties> extensions(count);
  timing_vk_check(
      vkEnumerateDeviceExtensionProperties(
          physical_device, nullptr, &count, extensions.data()),
      "vkEnumerateDeviceExtensionProperties");
  return std::any_of(
      extensions.begin(),
      extensions.end(),
      [extension_name](const VkExtensionProperties& extension) {
        return std::string(extension.extensionName) == extension_name;
      });
}

inline const char* calibrated_timestamps_extension(
    VkPhysicalDevice physical_device) {
#ifdef VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
  if (device_extension_supported(
          physical_device, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    return VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  }
#endif
#ifdef VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
  if (device_extension_supported(
          physical_device, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    return VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  }
#endif
  return nullptr;
}

class VulkanSequenceTimer {
 public:
  VulkanSequenceTimer(
      VkPhysicalDevice physical_device,
      VkDevice device,
      uint32_t queue_family_index)
      : device_(device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    timestamp_period_ns_ = static_cast<double>(properties.limits.timestampPeriod);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, families.data());
    if (queue_family_index >= family_count) {
      throw std::runtime_error("invalid Vulkan queue family index");
    }
    timestamp_valid_bits_ = families[queue_family_index].timestampValidBits;
    if (timestamp_valid_bits_ == 0) {
      return;
    }
    VkQueryPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = 2;
    timing_vk_check(
        vkCreateQueryPool(device_, &create_info, nullptr, &query_pool_),
        "vkCreateQueryPool");
  }

  VulkanSequenceTimer(const VulkanSequenceTimer&) = delete;
  VulkanSequenceTimer& operator=(const VulkanSequenceTimer&) = delete;

  ~VulkanSequenceTimer() {
    if (query_pool_ != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device_, query_pool_, nullptr);
    }
  }

  bool gpu_timestamps_supported() const {
    return query_pool_ != VK_NULL_HANDLE;
  }

  double timestamp_period_ns() const {
    return timestamp_period_ns_;
  }

  uint32_t timestamp_valid_bits() const {
    return timestamp_valid_bits_;
  }

  void record_begin(VkCommandBuffer command_buffer) const {
    if (!gpu_timestamps_supported()) {
      return;
    }
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, 2);
    vkCmdWriteTimestamp(
        command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
  }

  void record_end(VkCommandBuffer command_buffer) const {
    if (!gpu_timestamps_supported()) {
      return;
    }
    vkCmdWriteTimestamp(
        command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
  }

  VulkanTimingSample submit_and_wait(
      VkQueue queue,
      VkCommandBuffer command_buffer,
      VkFence fence) const {
    timing_vk_check(vkResetFences(device_, 1, &fence), "vkResetFences");
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    const auto host_start = std::chrono::steady_clock::now();
    timing_vk_check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
    timing_vk_check(
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
    const auto host_stop = std::chrono::steady_clock::now();

    double gpu_us = -1.0;
    if (gpu_timestamps_supported()) {
      uint64_t timestamps[2] = {};
      timing_vk_check(
          vkGetQueryPoolResults(
              device_,
              query_pool_,
              0,
              2,
              sizeof(timestamps),
              timestamps,
              sizeof(uint64_t),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
      uint64_t delta = 0;
      if (timestamp_valid_bits_ >= 64) {
        delta = timestamps[1] - timestamps[0];
      } else {
        const uint64_t mask = (uint64_t{1} << timestamp_valid_bits_) - 1;
        delta = (timestamps[1] - timestamps[0]) & mask;
      }
      gpu_us = static_cast<double>(delta) * timestamp_period_ns_ / 1000.0;
    }
    return {
        gpu_us,
        std::chrono::duration<double, std::micro>(host_stop - host_start).count(),
    };
  }

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  double timestamp_period_ns_ = 0.0;
  uint32_t timestamp_valid_bits_ = 0;
};

class VulkanMultiQueueTimer {
 public:
  VulkanMultiQueueTimer(
      VkPhysicalDevice physical_device,
      VkDevice device,
      uint32_t queue_family_index,
      uint32_t lane_count,
      const char* calibrated_timestamps_extension_name)
      : device_(device), lane_count_(lane_count) {
    if (lane_count_ < 2) {
      throw std::runtime_error("Vulkan multi-queue timing requires at least two lanes");
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    timestamp_period_ns_ = static_cast<double>(properties.limits.timestampPeriod);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, families.data());
    if (queue_family_index >= family_count) {
      throw std::runtime_error("invalid Vulkan queue family index");
    }
    const auto& family = families[queue_family_index];
    if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0 ||
        family.queueCount < lane_count_) {
      throw std::runtime_error(
          "Vulkan queue family cannot provide the requested compute lanes");
    }
    timestamp_valid_bits_ = family.timestampValidBits;
    if (timestamp_valid_bits_ == 0) {
      throw std::runtime_error(
          "Vulkan multi-queue timing requires timestamp-capable queues");
    }
    if (!timeline_semaphore_supported(physical_device)) {
      throw std::runtime_error(
          "Vulkan multi-queue timing requires timeline semaphore support");
    }
    if (calibrated_timestamps_extension_name == nullptr ||
        !device_extension_supported(
            physical_device, calibrated_timestamps_extension_name)) {
      throw std::runtime_error(
          "Vulkan cross-queue GPU timing requires calibrated timestamps");
    }
    const bool calibrated_command_enabled =
        vkGetDeviceProcAddr(device_, "vkGetCalibratedTimestampsKHR") != nullptr ||
        vkGetDeviceProcAddr(device_, "vkGetCalibratedTimestampsEXT") != nullptr;
    if (!calibrated_command_enabled) {
      throw std::runtime_error(
          "calibrated timestamps extension was not enabled on the Vulkan device");
    }
    calibrated_timestamps_extension_name_ = calibrated_timestamps_extension_name;

    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2 * lane_count_;
    timing_vk_check(
        vkCreateQueryPool(device_, &query_info, nullptr, &query_pool_),
        "vkCreateQueryPool multi-queue");

    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0;
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &type_info;
    timing_vk_check(
        vkCreateSemaphore(device_, &semaphore_info, nullptr, &start_semaphore_),
        "vkCreateSemaphore multi-queue start");
  }

  VulkanMultiQueueTimer(const VulkanMultiQueueTimer&) = delete;
  VulkanMultiQueueTimer& operator=(const VulkanMultiQueueTimer&) = delete;

  ~VulkanMultiQueueTimer() {
    if (start_semaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, start_semaphore_, nullptr);
    }
    if (query_pool_ != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device_, query_pool_, nullptr);
    }
  }

  uint32_t lane_count() const {
    return lane_count_;
  }

  double timestamp_period_ns() const {
    return timestamp_period_ns_;
  }

  uint32_t timestamp_valid_bits() const {
    return timestamp_valid_bits_;
  }

  const std::string& calibrated_timestamps_extension_name() const {
    return calibrated_timestamps_extension_name_;
  }

  void record_begin(VkCommandBuffer command_buffer, uint32_t lane) const {
    validate_lane(lane);
    const uint32_t first_query = 2 * lane;
    vkCmdResetQueryPool(command_buffer, query_pool_, first_query, 2);
    vkCmdWriteTimestamp(
        command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        query_pool_,
        first_query);
  }

  void record_end(VkCommandBuffer command_buffer, uint32_t lane) const {
    validate_lane(lane);
    vkCmdWriteTimestamp(
        command_buffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        query_pool_,
        2 * lane + 1);
  }

  VulkanMultiQueueTimingSample submit_and_wait(
      const std::vector<VkQueue>& queues,
      const std::vector<VkCommandBuffer>& command_buffers,
      const std::vector<VkFence>& fences) {
    if (queues.size() != lane_count_ || command_buffers.size() != lane_count_ ||
        fences.size() != lane_count_) {
      throw std::runtime_error(
          "Vulkan multi-queue submit vectors must match the lane count");
    }
    timing_vk_check(
        vkResetFences(
            device_, lane_count_, fences.data()),
        "vkResetFences multi-queue");

    const uint64_t wait_value = ++timeline_value_;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const auto host_start = std::chrono::steady_clock::now();
    for (uint32_t lane = 0; lane < lane_count_; ++lane) {
      VkTimelineSemaphoreSubmitInfo timeline_info{};
      timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
      timeline_info.waitSemaphoreValueCount = 1;
      timeline_info.pWaitSemaphoreValues = &wait_value;
      VkSubmitInfo submit_info{};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.pNext = &timeline_info;
      submit_info.waitSemaphoreCount = 1;
      submit_info.pWaitSemaphores = &start_semaphore_;
      submit_info.pWaitDstStageMask = &wait_stage;
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &command_buffers[lane];
      timing_vk_check(
          vkQueueSubmit(queues[lane], 1, &submit_info, fences[lane]),
          "vkQueueSubmit multi-queue lane");
    }
    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = start_semaphore_;
    signal_info.value = wait_value;
    timing_vk_check(
        vkSignalSemaphore(device_, &signal_info),
        "vkSignalSemaphore multi-queue start");
    timing_vk_check(
        vkWaitForFences(
            device_, lane_count_, fences.data(), VK_TRUE, UINT64_MAX),
        "vkWaitForFences multi-queue");
    const auto host_stop = std::chrono::steady_clock::now();

    std::vector<uint64_t> timestamps(2 * lane_count_, 0);
    timing_vk_check(
        vkGetQueryPoolResults(
            device_,
            query_pool_,
            0,
            2 * lane_count_,
            timestamps.size() * sizeof(uint64_t),
            timestamps.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults multi-queue");

    const uint64_t base = timestamps[0];
    int64_t earliest = 0;
    int64_t latest = 0;
    std::vector<double> lane_gpu_us;
    lane_gpu_us.reserve(lane_count_);
    for (uint32_t lane = 0; lane < lane_count_; ++lane) {
      const int64_t start = modular_offset(timestamps[2 * lane], base);
      const int64_t end = modular_offset(timestamps[2 * lane + 1], base);
      if (end < start) {
        throw std::runtime_error("invalid Vulkan per-lane timestamp interval");
      }
      lane_gpu_us.push_back(
          static_cast<double>(end - start) * timestamp_period_ns_ / 1000.0);
      earliest = std::min(
          earliest, start);
      latest = std::max(
          latest, end);
    }
    if (latest < earliest) {
      throw std::runtime_error("invalid Vulkan multi-queue timestamp span");
    }
    return {
        static_cast<double>(latest - earliest) * timestamp_period_ns_ / 1000.0,
        std::chrono::duration<double, std::micro>(host_stop - host_start).count(),
        std::move(lane_gpu_us),
        true,
    };
  }

 private:
  void validate_lane(uint32_t lane) const {
    if (lane >= lane_count_) {
      throw std::runtime_error("Vulkan multi-queue lane index is out of range");
    }
  }

  int64_t modular_offset(uint64_t value, uint64_t base) const {
    const uint64_t delta = value - base;
    if (timestamp_valid_bits_ >= 64) {
      if (delta <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return static_cast<int64_t>(delta);
      }
      return -static_cast<int64_t>(base - value);
    }
    const uint64_t modulus = uint64_t{1} << timestamp_valid_bits_;
    const uint64_t mask = modulus - 1;
    const uint64_t wrapped = delta & mask;
    if (wrapped < modulus / 2) {
      return static_cast<int64_t>(wrapped);
    }
    return -static_cast<int64_t>((base - value) & mask);
  }

  VkDevice device_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  VkSemaphore start_semaphore_ = VK_NULL_HANDLE;
  uint32_t lane_count_ = 0;
  double timestamp_period_ns_ = 0.0;
  uint32_t timestamp_valid_bits_ = 0;
  uint64_t timeline_value_ = 0;
  std::string calibrated_timestamps_extension_name_;
};

}  // namespace hipengine::micro
