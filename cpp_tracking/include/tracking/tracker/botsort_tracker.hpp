#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "tracking/core/config.hpp"
#include "tracking/core/types.hpp"
#include "tracking/tracker/tracker_interface.hpp"

namespace tracking {

enum class TrackState : std::uint8_t {
  kNew = 0,
  kTracked = 1,
  kLost = 2,
  kRemoved = 3,
};

class BoTSORTTracker final : public ITracker {
 public:
  explicit BoTSORTTracker(TrackerRuntimeConfig cfg);
  Tracks Update(const Detections& detections, const cv::Mat& frame) override;

 private:
  struct DetectionEx {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float conf = 0.0F;
    float cls = 0.0F;
    int det_index = -1;
  };

  struct KalmanState {
    std::array<float, 8> mean{};
    std::array<float, 64> covariance{};
  };

  struct TrackInternal {
    std::int64_t id = -1;
    TrackState state = TrackState::kNew;
    bool is_activated = false;
    std::array<float, 4> xywh{};
    KalmanState kf_state{};
    bool has_kf_state = false;
    float conf = 0.0F;
    float cls = 0.0F;
    int det_index = -1;
    int frame_id = 0;
    int start_frame = 0;
    int tracklet_len = 0;
    std::vector<float> curr_feat;
    std::vector<float> smooth_feat;
    std::vector<std::pair<float, float>> cls_hist;
  };

  class ReIdExtractor {
   public:
    explicit ReIdExtractor(const TrackerRuntimeConfig& cfg);
    bool IsReady() const { return ready_; }
    std::vector<std::vector<float>> Extract(const cv::Mat& frame,
                                            const std::vector<DetectionEx>& detections) const;

   private:
    std::vector<float> PreprocessCrop(const cv::Mat& crop) const;
    std::vector<std::vector<float>> RunBatch(const std::vector<cv::Mat>& crops) const;

    bool ready_ = false;
    int input_width_ = 128;
    int input_height_ = 256;
    int max_detections_ = 48;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "tracking_reid"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
  };

  class CameraMotionCompensator {
   public:
    explicit CameraMotionCompensator(const TrackerRuntimeConfig& cfg);
    cv::Mat Estimate(const cv::Mat& frame, const std::vector<DetectionEx>& detections);

   private:
    bool enabled_ = false;
    int interval_ = 3;
    int max_side_ = 960;
    int frame_counter_ = 0;
    cv::Mat last_warp_;
    cv::Mat prev_gray_;
  };

  static std::array<float, 4> XYXYToXYWH(const DetectionEx& det);
  static std::array<float, 4> XYWHToXYXY(const std::array<float, 4>& xywh);
  static float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b);
  static float CosineDistance(const std::vector<float>& a, const std::vector<float>& b);
  static float NormCenterDistance(const std::array<float, 4>& a, const std::array<float, 4>& b,
                                  float frame_diag);
  static void L2Normalize(std::vector<float>* values);
  static std::vector<std::pair<int, int>> LinearAssignment(const std::vector<std::vector<float>>& cost,
                                                           float threshold,
                                                           std::vector<int>* unmatched_rows,
                                                           std::vector<int>* unmatched_cols);

  void PredictTracks(const cv::Mat& frame, const std::vector<DetectionEx>& detections,
                     std::vector<TrackInternal*>* strack_pool,
                     std::vector<TrackInternal*>* unconfirmed);
  void ApplyWarp(TrackInternal* track, const cv::Mat& warp) const;
  std::vector<std::vector<float>> IoUDistance(const std::vector<TrackInternal*>& tracks,
                                              const std::vector<DetectionEx>& detections) const;
  std::vector<std::vector<float>> EmbeddingDistance(const std::vector<TrackInternal*>& tracks,
                                                    const std::vector<DetectionEx>& detections,
                                                    const std::vector<std::vector<float>>& features) const;
  static void FuseScore(std::vector<std::vector<float>>* cost,
                        const std::vector<DetectionEx>& detections);
  static std::vector<DetectionEx> SelectByIndices(const std::vector<DetectionEx>& detections,
                                                  const std::vector<int>& indices);

  TrackInternal BuildTrackFromDet(const DetectionEx& det, const std::vector<float>* feat) const;
  void ActivateTrack(TrackInternal* track, int frame_id);
  void UpdateTrack(TrackInternal* track, const DetectionEx& det, const std::vector<float>* feat,
                   int frame_id);
  void ReActivateTrack(TrackInternal* track, const DetectionEx& det, const std::vector<float>* feat,
                       int frame_id, bool new_id);
  void MarkLost(TrackInternal* track);
  void MarkRemoved(TrackInternal* track);
  static void UpdateClassHistory(TrackInternal* track, float cls, float conf);

  void KalmanInitiate(TrackInternal* track);
  void KalmanPredict(TrackInternal* track);
  void KalmanUpdate(TrackInternal* track, const std::array<float, 4>& measurement);

  std::vector<TrackInternal*> JoinTracks(const std::vector<TrackInternal*>& first,
                                         const std::vector<TrackInternal*>& second) const;
  std::vector<TrackInternal*> SubTracks(const std::vector<TrackInternal*>& source,
                                        const std::vector<TrackInternal*>& to_remove) const;
  void RemoveDuplicateTracks(std::vector<TrackInternal*>* a, std::vector<TrackInternal*>* b) const;

  void RefreshViews();
  std::array<float, 4> CurrentXYWH(const TrackInternal& track) const;
  std::array<float, 4> CurrentXYXY(const TrackInternal& track) const;
  float FrameDiag() const;

  TrackerRuntimeConfig cfg_;
  int frame_count_ = 0;
  std::int64_t next_id_ = 1;
  int max_time_lost_ = 30;
  int frame_width_ = 1;
  int frame_height_ = 1;

  std::deque<TrackInternal> tracks_;
  std::vector<TrackInternal*> active_tracks_;
  std::vector<TrackInternal*> lost_tracks_;
  std::vector<TrackInternal*> removed_tracks_;

  std::unique_ptr<ReIdExtractor> reid_;
  std::unique_ptr<CameraMotionCompensator> cmc_;
};

std::unique_ptr<ITracker> CreateTracker(const TrackerRuntimeConfig& cfg);

}  // namespace tracking

