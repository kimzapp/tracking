#include "tracking/tracker/tracker_interface.hpp"

#include <memory>

#include "tracking/tracker/botsort_tracker.hpp"

namespace tracking {

Tracks PassthroughTracker::Update(const Detections& detections, const cv::Mat& /*frame*/) {
  Tracks tracks;
  tracks.reserve(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    const Detection& det = detections[i];
    Track t;
    t.x1 = det.x1;
    t.y1 = det.y1;
    t.x2 = det.x2;
    t.y2 = det.y2;
    t.id = next_id_++;
    t.conf = det.conf;
    t.cls = det.cls;
    t.det_index = static_cast<int>(i);
    tracks.push_back(t);
  }
  return tracks;
}

std::unique_ptr<ITracker> CreateTracker(const TrackerRuntimeConfig& cfg) {
  if (cfg.tracker_type == "passthrough") {
    return std::make_unique<PassthroughTracker>();
  }
  return std::make_unique<BoTSORTTracker>(cfg);
}

}  // namespace tracking
