#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
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

inline void timing_hip_check(hipError_t result, const char* what) {
  if (result != hipSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + hipGetErrorString(result));
  }
}

struct TimingSamples {
  std::vector<double> gpu_sequence_us;
  std::vector<double> host_sequence_us;
};

class HipSequenceTimer {
 public:
  HipSequenceTimer(TimingMode mode, uint32_t independent_streams)
      : mode_(mode) {
    uint32_t worker_count =
        mode == TimingMode::SerialLatency ? 1u : std::max(1u, independent_streams);
    timing_hip_check(
        hipStreamCreateWithFlags(&coordinator_, hipStreamNonBlocking),
        "hipStreamCreateWithFlags coordinator");
    workers_.resize(worker_count, nullptr);
    done_.resize(worker_count, nullptr);
    for (uint32_t i = 0; i < worker_count; ++i) {
      if (mode == TimingMode::SerialLatency && i == 0) {
        workers_[i] = coordinator_;
      } else {
        timing_hip_check(
            hipStreamCreateWithFlags(&workers_[i], hipStreamNonBlocking),
            "hipStreamCreateWithFlags worker");
      }
      timing_hip_check(hipEventCreate(&done_[i]), "hipEventCreate done");
    }
    timing_hip_check(hipEventCreate(&start_), "hipEventCreate start");
    timing_hip_check(hipEventCreate(&stop_), "hipEventCreate stop");
  }

  HipSequenceTimer(const HipSequenceTimer&) = delete;
  HipSequenceTimer& operator=(const HipSequenceTimer&) = delete;

  ~HipSequenceTimer() {
    for (hipEvent_t event : done_) {
      if (event != nullptr) {
        (void)hipEventDestroy(event);
      }
    }
    if (start_ != nullptr) {
      (void)hipEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      (void)hipEventDestroy(stop_);
    }
    for (hipStream_t stream : workers_) {
      if (stream != nullptr && stream != coordinator_) {
        (void)hipStreamDestroy(stream);
      }
    }
    if (coordinator_ != nullptr) {
      (void)hipStreamDestroy(coordinator_);
    }
  }

  uint32_t stream_count() const {
    return static_cast<uint32_t>(workers_.size());
  }

  template <typename Launch>
  TimingSamples measure(uint32_t logical_iterations, uint32_t samples, Launch&& launch) {
    if (logical_iterations == 0 || samples == 0) {
      throw std::runtime_error("logical_iterations and samples must be positive");
    }
    TimingSamples result;
    result.gpu_sequence_us.reserve(samples);
    result.host_sequence_us.reserve(samples);
    for (uint32_t sample = 0; sample < samples; ++sample) {
      timing_hip_check(hipStreamSynchronize(coordinator_), "pre-sample coordinator sync");
      for (hipStream_t worker : workers_) {
        timing_hip_check(hipStreamSynchronize(worker), "pre-sample worker sync");
      }

      const auto host_start = std::chrono::steady_clock::now();
      timing_hip_check(hipEventRecord(start_, coordinator_), "hipEventRecord start");
      if (mode_ == TimingMode::IndependentThroughput) {
        for (hipStream_t worker : workers_) {
          timing_hip_check(
              hipStreamWaitEvent(worker, start_, 0), "hipStreamWaitEvent start");
        }
      }
      for (uint32_t rep = 0; rep < logical_iterations; ++rep) {
        hipStream_t stream = workers_[rep % workers_.size()];
        launch(rep, stream);
      }
      if (mode_ == TimingMode::IndependentThroughput) {
        for (size_t i = 0; i < workers_.size(); ++i) {
          timing_hip_check(hipEventRecord(done_[i], workers_[i]), "hipEventRecord done");
          timing_hip_check(
              hipStreamWaitEvent(coordinator_, done_[i], 0), "hipStreamWaitEvent done");
        }
      }
      timing_hip_check(hipEventRecord(stop_, coordinator_), "hipEventRecord stop");
      timing_hip_check(hipEventSynchronize(stop_), "hipEventSynchronize stop");
      const auto host_stop = std::chrono::steady_clock::now();

      float elapsed_ms = 0.0f;
      timing_hip_check(
          hipEventElapsedTime(&elapsed_ms, start_, stop_), "hipEventElapsedTime");
      result.gpu_sequence_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
      result.host_sequence_us.push_back(
          std::chrono::duration<double, std::micro>(host_stop - host_start).count());
    }
    return result;
  }

  template <typename Launch>
  void run_and_wait(uint32_t logical_iterations, Launch&& launch) {
    for (uint32_t rep = 0; rep < logical_iterations; ++rep) {
      hipStream_t stream = workers_[rep % workers_.size()];
      launch(rep, stream);
    }
    for (hipStream_t worker : workers_) {
      timing_hip_check(hipStreamSynchronize(worker), "run_and_wait worker sync");
    }
  }

 private:
  TimingMode mode_;
  hipStream_t coordinator_ = nullptr;
  std::vector<hipStream_t> workers_;
  std::vector<hipEvent_t> done_;
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

}  // namespace hipengine::micro
