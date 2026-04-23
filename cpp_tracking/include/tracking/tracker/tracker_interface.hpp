#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "tracking/core/config.hpp"
#include "tracking/core/types.hpp"

namespace tracking {

class ITracker {
 public:
  virtual ~ITracker() = default;
  virtual Tracks Update(const Detections& detections, const cv::Mat& frame) = 0;
};

class PassthroughTracker final : public ITracker {
 public:
  Tracks Update(const Detections& detections, const cv::Mat& frame) override;

 private:
  std::int64_t next_id_ = 1;
};

std::unique_ptr<ITracker> CreateTracker(const TrackerRuntimeConfig& cfg);

}  // namespace tracking
