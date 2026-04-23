#include "tracking/tracker/botsort_tracker.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "tracking/utils/logging.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace tracking {

namespace {

constexpr float kStdWeightPosition = 1.0F / 20.0F;
constexpr float kStdWeightVelocity = 1.0F / 160.0F;
constexpr float kHugeCost = 1e6F;

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& input) {
  if (input.empty()) {
    return {};
  }
  const int required = MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
  if (required <= 0) {
    throw std::runtime_error("Failed to convert UTF-8 path to wide string");
  }
  std::wstring output(static_cast<size_t>(required), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, output.data(), required);
  if (!output.empty() && output.back() == L'\0') {
    output.pop_back();
  }
  return output;
}
#endif

float Clamp(float value, float lo, float hi) {
  return std::max(lo, std::min(value, hi));
}

std::array<float, 4> DiagToArray4(const std::array<float, 4>& diag) {
  return {diag[0] * diag[0], diag[1] * diag[1], diag[2] * diag[2], diag[3] * diag[3]};
}

bool Invert4x4(const std::array<float, 16>& m, std::array<float, 16>* inv) {
  std::array<float, 32> aug{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      aug[r * 8 + c] = m[r * 4 + c];
      aug[r * 8 + (c + 4)] = (r == c) ? 1.0F : 0.0F;
    }
  }
  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    float best = std::fabs(aug[pivot * 8 + col]);
    for (int r = col + 1; r < 4; ++r) {
      const float candidate = std::fabs(aug[r * 8 + col]);
      if (candidate > best) {
        best = candidate;
        pivot = r;
      }
    }
    if (best < 1e-9F) {
      return false;
    }
    if (pivot != col) {
      for (int c = 0; c < 8; ++c) {
        std::swap(aug[col * 8 + c], aug[pivot * 8 + c]);
      }
    }
    const float denom = aug[col * 8 + col];
    for (int c = 0; c < 8; ++c) {
      aug[col * 8 + c] /= denom;
    }
    for (int r = 0; r < 4; ++r) {
      if (r == col) {
        continue;
      }
      const float factor = aug[r * 8 + col];
      for (int c = 0; c < 8; ++c) {
        aug[r * 8 + c] -= factor * aug[col * 8 + c];
      }
    }
  }
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      (*inv)[r * 4 + c] = aug[r * 8 + (c + 4)];
    }
  }
  return true;
}

}  // namespace

BoTSORTTracker::BoTSORTTracker(TrackerRuntimeConfig cfg) : cfg_(std::move(cfg)) {
  max_time_lost_ = static_cast<int>(static_cast<float>(cfg_.frame_rate) / 30.0F *
                                    static_cast<float>(cfg_.track_buffer));
  if (max_time_lost_ <= 0) {
    max_time_lost_ = cfg_.track_buffer;
  }

  if (cfg_.with_reid) {
    reid_ = std::make_unique<ReIdExtractor>(cfg_);
    if (!reid_->IsReady()) {
      LogWarn("ReID extractor failed to initialize from model: " + cfg_.reid_model_path +
              ". ReID is disabled for this run.");
      cfg_.with_reid = false;
    } else {
      LogInfo("ReID extractor initialized: " + cfg_.reid_model_path);
    }
  }

  cmc_ = std::make_unique<CameraMotionCompensator>(cfg_);
}

std::array<float, 4> BoTSORTTracker::XYXYToXYWH(const DetectionEx& det) {
  const float w = std::max(1e-4F, det.x2 - det.x1);
  const float h = std::max(1e-4F, det.y2 - det.y1);
  return {(det.x1 + det.x2) * 0.5F, (det.y1 + det.y2) * 0.5F, w, h};
}

std::array<float, 4> BoTSORTTracker::XYWHToXYXY(const std::array<float, 4>& xywh) {
  const float half_w = xywh[2] * 0.5F;
  const float half_h = xywh[3] * 0.5F;
  return {xywh[0] - half_w, xywh[1] - half_h, xywh[0] + half_w, xywh[1] + half_h};
}

float BoTSORTTracker::IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float w = std::max(0.0F, x2 - x1);
  const float h = std::max(0.0F, y2 - y1);
  const float inter = w * h;
  const float area_a = std::max(0.0F, (a[2] - a[0])) * std::max(0.0F, (a[3] - a[1]));
  const float area_b = std::max(0.0F, (b[2] - b[0])) * std::max(0.0F, (b[3] - b[1]));
  const float denom = area_a + area_b - inter;
  if (denom <= 1e-6F) {
    return 0.0F;
  }
  return inter / denom;
}

void BoTSORTTracker::L2Normalize(std::vector<float>* values) {
  float sum = 0.0F;
  for (float v : *values) {
    sum += v * v;
  }
  const float norm = std::sqrt(std::max(sum, 1e-12F));
  for (float& v : *values) {
    v /= norm;
  }
}

float BoTSORTTracker::CosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return 1.0F;
  }
  float dot = 0.0F;
  float na = 0.0F;
  float nb = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const float denom = std::sqrt(std::max(na * nb, 1e-12F));
  const float sim = dot / denom;
  return std::max(0.0F, 1.0F - sim);
}

float BoTSORTTracker::NormCenterDistance(const std::array<float, 4>& a, const std::array<float, 4>& b,
                                         float frame_diag) {
  const float ax = (a[0] + a[2]) * 0.5F;
  const float ay = (a[1] + a[3]) * 0.5F;
  const float bx = (b[0] + b[2]) * 0.5F;
  const float by = (b[1] + b[3]) * 0.5F;
  const float d = std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
  return Clamp(d / std::max(frame_diag, 1.0F), 0.0F, 1.0F);
}

std::vector<std::pair<int, int>> BoTSORTTracker::LinearAssignment(
    const std::vector<std::vector<float>>& cost, float threshold, std::vector<int>* unmatched_rows,
    std::vector<int>* unmatched_cols) {
  unmatched_rows->clear();
  unmatched_cols->clear();
  const int n = static_cast<int>(cost.size());
  const int m = n > 0 ? static_cast<int>(cost[0].size()) : 0;
  if (n == 0 || m == 0) {
    for (int i = 0; i < n; ++i) {
      unmatched_rows->push_back(i);
    }
    for (int j = 0; j < m; ++j) {
      unmatched_cols->push_back(j);
    }
    return {};
  }

  bool transposed = false;
  std::vector<std::vector<float>> a = cost;
  int rows = n;
  int cols = m;
  if (rows > cols) {
    transposed = true;
    std::vector<std::vector<float>> t(static_cast<size_t>(cols),
                                      std::vector<float>(static_cast<size_t>(rows), 0.0F));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        t[j][i] = a[i][j];
      }
    }
    a = std::move(t);
    std::swap(rows, cols);
  }

  std::vector<float> u(static_cast<size_t>(rows + 1), 0.0F);
  std::vector<float> v(static_cast<size_t>(cols + 1), 0.0F);
  std::vector<int> p(static_cast<size_t>(cols + 1), 0);
  std::vector<int> way(static_cast<size_t>(cols + 1), 0);

  for (int i = 1; i <= rows; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(static_cast<size_t>(cols + 1), kHugeCost);
    std::vector<char> used(static_cast<size_t>(cols + 1), false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      float delta = kHugeCost;
      int j1 = 0;
      for (int j = 1; j <= cols; ++j) {
        if (used[j]) {
          continue;
        }
        const float raw = a[i0 - 1][j - 1];
        const float c = std::isfinite(raw) ? raw : kHugeCost;
        const float cur = c - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= cols; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(static_cast<size_t>(rows), -1);
  for (int j = 1; j <= cols; ++j) {
    if (p[j] > 0) {
      assignment[static_cast<size_t>(p[j] - 1)] = j - 1;
    }
  }

  std::vector<std::pair<int, int>> matches;
  std::vector<char> row_matched(static_cast<size_t>(n), false);
  std::vector<char> col_matched(static_cast<size_t>(m), false);

  for (int r = 0; r < rows; ++r) {
    const int c = assignment[static_cast<size_t>(r)];
    if (c < 0) {
      continue;
    }
    int row = r;
    int col = c;
    if (transposed) {
      row = c;
      col = r;
    }
    if (row >= n || col >= m) {
      continue;
    }
    const float candidate = cost[static_cast<size_t>(row)][static_cast<size_t>(col)];
    if (std::isfinite(candidate) && candidate <= threshold) {
      matches.emplace_back(row, col);
      row_matched[static_cast<size_t>(row)] = true;
      col_matched[static_cast<size_t>(col)] = true;
    }
  }

  for (int i = 0; i < n; ++i) {
    if (!row_matched[static_cast<size_t>(i)]) {
      unmatched_rows->push_back(i);
    }
  }
  for (int j = 0; j < m; ++j) {
    if (!col_matched[static_cast<size_t>(j)]) {
      unmatched_cols->push_back(j);
    }
  }
  return matches;
}

void BoTSORTTracker::KalmanInitiate(TrackInternal* track) {
  const auto xywh = track->xywh;
  track->kf_state.mean = {xywh[0], xywh[1], xywh[2], xywh[3], 0.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 8> stdv = {
      2.0F * kStdWeightPosition * xywh[2], 2.0F * kStdWeightPosition * xywh[3],
      2.0F * kStdWeightPosition * xywh[2], 2.0F * kStdWeightPosition * xywh[3],
      10.0F * kStdWeightVelocity * xywh[2], 10.0F * kStdWeightVelocity * xywh[3],
      10.0F * kStdWeightVelocity * xywh[2], 10.0F * kStdWeightVelocity * xywh[3]};
  track->kf_state.covariance.fill(0.0F);
  for (int i = 0; i < 8; ++i) {
    track->kf_state.covariance[static_cast<size_t>(i * 8 + i)] = stdv[static_cast<size_t>(i)] *
                                                                 stdv[static_cast<size_t>(i)];
  }
  track->has_kf_state = true;
}

void BoTSORTTracker::KalmanPredict(TrackInternal* track) {
  if (!track->has_kf_state) {
    KalmanInitiate(track);
  }

  std::array<float, 8> mean = track->kf_state.mean;
  if (track->state != TrackState::kTracked) {
    mean[6] = 0.0F;
    mean[7] = 0.0F;
  }

  std::array<float, 64> motion{};
  for (int i = 0; i < 8; ++i) {
    motion[static_cast<size_t>(i * 8 + i)] = 1.0F;
  }
  motion[4] = 1.0F;
  motion[13] = 1.0F;
  motion[22] = 1.0F;
  motion[31] = 1.0F;

  std::array<float, 8> new_mean{};
  for (int r = 0; r < 8; ++r) {
    float sum = 0.0F;
    for (int c = 0; c < 8; ++c) {
      sum += motion[static_cast<size_t>(r * 8 + c)] * mean[static_cast<size_t>(c)];
    }
    new_mean[static_cast<size_t>(r)] = sum;
  }
  new_mean[2] = std::max(new_mean[2], 1e-4F);
  new_mean[3] = std::max(new_mean[3], 1e-4F);

  const std::array<float, 8> stdv = {
      kStdWeightPosition * mean[2], kStdWeightPosition * mean[3], kStdWeightPosition * mean[2],
      kStdWeightPosition * mean[3], kStdWeightVelocity * mean[2], kStdWeightVelocity * mean[3],
      kStdWeightVelocity * mean[2], kStdWeightVelocity * mean[3]};

  std::array<float, 64> q{};
  for (int i = 0; i < 8; ++i) {
    q[static_cast<size_t>(i * 8 + i)] = stdv[static_cast<size_t>(i)] * stdv[static_cast<size_t>(i)];
  }

  std::array<float, 64> tmp{};
  std::array<float, 64> new_cov{};
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      float sum = 0.0F;
      for (int k = 0; k < 8; ++k) {
        sum += motion[static_cast<size_t>(r * 8 + k)] *
               track->kf_state.covariance[static_cast<size_t>(k * 8 + c)];
      }
      tmp[static_cast<size_t>(r * 8 + c)] = sum;
    }
  }
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      float sum = 0.0F;
      for (int k = 0; k < 8; ++k) {
        sum += tmp[static_cast<size_t>(r * 8 + k)] * motion[static_cast<size_t>(c * 8 + k)];
      }
      new_cov[static_cast<size_t>(r * 8 + c)] = sum + q[static_cast<size_t>(r * 8 + c)];
    }
  }

  track->kf_state.mean = new_mean;
  track->kf_state.covariance = new_cov;
}

void BoTSORTTracker::KalmanUpdate(TrackInternal* track, const std::array<float, 4>& measurement) {
  if (!track->has_kf_state) {
    KalmanInitiate(track);
  }

  const auto& mean = track->kf_state.mean;
  const auto& cov = track->kf_state.covariance;

  std::array<float, 4> projected_mean = {mean[0], mean[1], mean[2], mean[3]};
  const std::array<float, 4> std_noise = {kStdWeightPosition * mean[2], kStdWeightPosition * mean[3],
                                          kStdWeightPosition * mean[2], kStdWeightPosition * mean[3]};
  const std::array<float, 4> r_diag = DiagToArray4(std_noise);

  std::array<float, 16> projected_cov{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      projected_cov[static_cast<size_t>(r * 4 + c)] =
          cov[static_cast<size_t>(r * 8 + c)] + (r == c ? r_diag[static_cast<size_t>(r)] : 0.0F);
    }
  }

  std::array<float, 16> inv_s{};
  if (!Invert4x4(projected_cov, &inv_s)) {
    return;
  }

  std::array<float, 32> pht{};
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 4; ++c) {
      pht[static_cast<size_t>(r * 4 + c)] = cov[static_cast<size_t>(r * 8 + c)];
    }
  }

  std::array<float, 32> k{};
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 4; ++c) {
      float sum = 0.0F;
      for (int x = 0; x < 4; ++x) {
        sum += pht[static_cast<size_t>(r * 4 + x)] * inv_s[static_cast<size_t>(x * 4 + c)];
      }
      k[static_cast<size_t>(r * 4 + c)] = sum;
    }
  }

  std::array<float, 4> innovation{};
  for (int i = 0; i < 4; ++i) {
    innovation[static_cast<size_t>(i)] = measurement[static_cast<size_t>(i)] - projected_mean[static_cast<size_t>(i)];
  }

  std::array<float, 8> new_mean = mean;
  for (int r = 0; r < 8; ++r) {
    float delta = 0.0F;
    for (int c = 0; c < 4; ++c) {
      delta += k[static_cast<size_t>(r * 4 + c)] * innovation[static_cast<size_t>(c)];
    }
    new_mean[static_cast<size_t>(r)] += delta;
  }
  new_mean[2] = std::max(new_mean[2], 1e-4F);
  new_mean[3] = std::max(new_mean[3], 1e-4F);

  std::array<float, 64> kh{};
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      kh[static_cast<size_t>(r * 8 + c)] = (c < 4) ? k[static_cast<size_t>(r * 4 + c)] : 0.0F;
    }
  }
  std::array<float, 64> i_kh{};
  for (int i = 0; i < 8; ++i) {
    i_kh[static_cast<size_t>(i * 8 + i)] = 1.0F;
  }
  for (size_t i = 0; i < i_kh.size(); ++i) {
    i_kh[i] -= kh[i];
  }
  std::array<float, 64> new_cov{};
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      float sum = 0.0F;
      for (int x = 0; x < 8; ++x) {
        sum += i_kh[static_cast<size_t>(r * 8 + x)] * cov[static_cast<size_t>(x * 8 + c)];
      }
      new_cov[static_cast<size_t>(r * 8 + c)] = sum;
    }
  }

  track->kf_state.mean = new_mean;
  track->kf_state.covariance = new_cov;
}

BoTSORTTracker::TrackInternal BoTSORTTracker::BuildTrackFromDet(const DetectionEx& det,
                                                                const std::vector<float>* feat) const {
  TrackInternal t;
  t.xywh = XYXYToXYWH(det);
  t.conf = det.conf;
  t.cls = det.cls;
  t.det_index = det.det_index;
  UpdateClassHistory(&t, det.cls, det.conf);
  if (feat != nullptr && !feat->empty()) {
    t.curr_feat = *feat;
    t.smooth_feat = *feat;
  }
  return t;
}

void BoTSORTTracker::ActivateTrack(TrackInternal* track, int frame_id) {
  track->id = next_id_++;
  KalmanInitiate(track);
  track->state = TrackState::kTracked;
  track->tracklet_len = 0;
  track->frame_id = frame_id;
  track->start_frame = frame_id;
  track->is_activated = (frame_id == 1);
}

void BoTSORTTracker::UpdateTrack(TrackInternal* track, const DetectionEx& det,
                                 const std::vector<float>* feat, int frame_id) {
  track->frame_id = frame_id;
  track->tracklet_len += 1;
  KalmanUpdate(track, XYXYToXYWH(det));
  track->xywh = XYXYToXYWH(det);
  track->state = TrackState::kTracked;
  track->is_activated = true;
  track->conf = det.conf;
  UpdateClassHistory(track, det.cls, det.conf);
  track->det_index = det.det_index;
  if (feat != nullptr && !feat->empty()) {
    track->curr_feat = *feat;
    if (track->smooth_feat.empty()) {
      track->smooth_feat = *feat;
    } else {
      for (size_t i = 0; i < feat->size() && i < track->smooth_feat.size(); ++i) {
        track->smooth_feat[i] = 0.95F * track->smooth_feat[i] + 0.05F * (*feat)[i];
      }
      L2Normalize(&track->smooth_feat);
    }
  }
}

void BoTSORTTracker::ReActivateTrack(TrackInternal* track, const DetectionEx& det,
                                     const std::vector<float>* feat, int frame_id, bool new_id) {
  KalmanUpdate(track, XYXYToXYWH(det));
  track->xywh = XYXYToXYWH(det);
  if (feat != nullptr && !feat->empty()) {
    track->curr_feat = *feat;
    if (track->smooth_feat.empty()) {
      track->smooth_feat = *feat;
    } else {
      for (size_t i = 0; i < feat->size() && i < track->smooth_feat.size(); ++i) {
        track->smooth_feat[i] = 0.95F * track->smooth_feat[i] + 0.05F * (*feat)[i];
      }
      L2Normalize(&track->smooth_feat);
    }
  }
  track->tracklet_len = 0;
  track->state = TrackState::kTracked;
  track->is_activated = true;
  track->frame_id = frame_id;
  track->conf = det.conf;
  UpdateClassHistory(track, det.cls, det.conf);
  track->det_index = det.det_index;
  if (new_id) {
    track->id = next_id_++;
  }
}

void BoTSORTTracker::MarkLost(TrackInternal* track) {
  track->state = TrackState::kLost;
}

void BoTSORTTracker::MarkRemoved(TrackInternal* track) {
  track->state = TrackState::kRemoved;
}

void BoTSORTTracker::UpdateClassHistory(TrackInternal* track, float cls, float conf) {
  float max_freq = 0.0F;
  bool found = false;
  for (auto& entry : track->cls_hist) {
    if (entry.first == cls) {
      entry.second += conf;
      found = true;
    }
    if (entry.second > max_freq) {
      max_freq = entry.second;
      track->cls = entry.first;
    }
  }

  if (!found) {
    track->cls_hist.emplace_back(cls, conf);
    track->cls = cls;
  }
}

void BoTSORTTracker::ApplyWarp(TrackInternal* track, const cv::Mat& warp) const {
  if (!track->has_kf_state || warp.empty() || warp.rows != 2 || warp.cols != 3) {
    return;
  }
  const float a00 = warp.at<float>(0, 0);
  const float a01 = warp.at<float>(0, 1);
  const float a02 = warp.at<float>(0, 2);
  const float a10 = warp.at<float>(1, 0);
  const float a11 = warp.at<float>(1, 1);
  const float a12 = warp.at<float>(1, 2);
  auto& mean = track->kf_state.mean;
  const float x = mean[0];
  const float y = mean[1];
  mean[0] = a00 * x + a01 * y + a02;
  mean[1] = a10 * x + a11 * y + a12;
  const float sx = std::max(1e-3F, std::sqrt(a00 * a00 + a10 * a10));
  const float sy = std::max(1e-3F, std::sqrt(a01 * a01 + a11 * a11));
  mean[2] *= sx;
  mean[3] *= sy;
  const float vx = mean[4];
  const float vy = mean[5];
  mean[4] = a00 * vx + a01 * vy;
  mean[5] = a10 * vx + a11 * vy;
  mean[6] *= sx;
  mean[7] *= sy;
}

std::vector<std::vector<float>> BoTSORTTracker::IoUDistance(
    const std::vector<TrackInternal*>& tracks, const std::vector<DetectionEx>& detections) const {
  std::vector<std::vector<float>> cost(
      tracks.size(), std::vector<float>(detections.size(), 1.0F));
  for (size_t i = 0; i < tracks.size(); ++i) {
    const auto tbox = CurrentXYXY(*tracks[i]);
    for (size_t j = 0; j < detections.size(); ++j) {
      const std::array<float, 4> dbox = {detections[j].x1, detections[j].y1, detections[j].x2,
                                         detections[j].y2};
      cost[i][j] = 1.0F - IoU(tbox, dbox);
    }
  }
  return cost;
}

std::vector<std::vector<float>> BoTSORTTracker::EmbeddingDistance(
    const std::vector<TrackInternal*>& tracks, const std::vector<DetectionEx>& detections,
    const std::vector<std::vector<float>>& features) const {
  std::vector<std::vector<float>> cost(
      tracks.size(), std::vector<float>(detections.size(), 1.0F));
  for (size_t i = 0; i < tracks.size(); ++i) {
    for (size_t j = 0; j < detections.size(); ++j) {
      if (!tracks[i]->smooth_feat.empty() && j < features.size() && !features[j].empty()) {
        cost[i][j] = CosineDistance(tracks[i]->smooth_feat, features[j]);
      }
    }
  }
  return cost;
}

void BoTSORTTracker::FuseScore(std::vector<std::vector<float>>* cost,
                               const std::vector<DetectionEx>& detections) {
  for (size_t i = 0; i < cost->size(); ++i) {
    for (size_t j = 0; j < (*cost)[i].size(); ++j) {
      const float iou_sim = 1.0F - (*cost)[i][j];
      (*cost)[i][j] = 1.0F - (iou_sim * detections[j].conf);
    }
  }
}

std::vector<BoTSORTTracker::DetectionEx> BoTSORTTracker::SelectByIndices(
    const std::vector<DetectionEx>& detections, const std::vector<int>& indices) {
  std::vector<DetectionEx> out;
  out.reserve(indices.size());
  for (int idx : indices) {
    out.push_back(detections[static_cast<size_t>(idx)]);
  }
  return out;
}

std::vector<BoTSORTTracker::TrackInternal*> BoTSORTTracker::JoinTracks(
    const std::vector<TrackInternal*>& first, const std::vector<TrackInternal*>& second) const {
  std::unordered_map<std::int64_t, TrackInternal*> map;
  std::vector<TrackInternal*> out;
  out.reserve(first.size() + second.size());
  for (TrackInternal* t : first) {
    map[t->id] = t;
    out.push_back(t);
  }
  for (TrackInternal* t : second) {
    if (map.find(t->id) == map.end()) {
      map[t->id] = t;
      out.push_back(t);
    }
  }
  return out;
}

std::vector<BoTSORTTracker::TrackInternal*> BoTSORTTracker::SubTracks(
    const std::vector<TrackInternal*>& source, const std::vector<TrackInternal*>& to_remove) const {
  std::unordered_map<std::int64_t, bool> rm;
  for (TrackInternal* t : to_remove) {
    rm[t->id] = true;
  }
  std::vector<TrackInternal*> out;
  out.reserve(source.size());
  for (TrackInternal* t : source) {
    if (rm.find(t->id) == rm.end()) {
      out.push_back(t);
    }
  }
  return out;
}

void BoTSORTTracker::RemoveDuplicateTracks(std::vector<TrackInternal*>* a,
                                           std::vector<TrackInternal*>* b) const {
  std::vector<int> drop_a;
  std::vector<int> drop_b;
  for (size_t i = 0; i < a->size(); ++i) {
    for (size_t j = 0; j < b->size(); ++j) {
      const float dist = 1.0F - IoU(CurrentXYXY(*(*a)[i]), CurrentXYXY(*(*b)[j]));
      if (dist < 0.15F) {
        const int time_a = (*a)[i]->frame_id - (*a)[i]->start_frame;
        const int time_b = (*b)[j]->frame_id - (*b)[j]->start_frame;
        if (time_a > time_b) {
          drop_b.push_back(static_cast<int>(j));
        } else {
          drop_a.push_back(static_cast<int>(i));
        }
      }
    }
  }
  std::sort(drop_a.begin(), drop_a.end());
  drop_a.erase(std::unique(drop_a.begin(), drop_a.end()), drop_a.end());
  std::sort(drop_b.begin(), drop_b.end());
  drop_b.erase(std::unique(drop_b.begin(), drop_b.end()), drop_b.end());

  std::vector<TrackInternal*> kept_a;
  for (size_t i = 0; i < a->size(); ++i) {
    if (!std::binary_search(drop_a.begin(), drop_a.end(), static_cast<int>(i))) {
      kept_a.push_back((*a)[i]);
    }
  }
  std::vector<TrackInternal*> kept_b;
  for (size_t i = 0; i < b->size(); ++i) {
    if (!std::binary_search(drop_b.begin(), drop_b.end(), static_cast<int>(i))) {
      kept_b.push_back((*b)[i]);
    }
  }
  *a = std::move(kept_a);
  *b = std::move(kept_b);
}

void BoTSORTTracker::RefreshViews() {
  active_tracks_.clear();
  lost_tracks_.clear();
  removed_tracks_.clear();
  for (TrackInternal& t : tracks_) {
    if (t.state == TrackState::kTracked || t.state == TrackState::kNew) {
      active_tracks_.push_back(&t);
    } else if (t.state == TrackState::kLost) {
      lost_tracks_.push_back(&t);
    } else if (t.state == TrackState::kRemoved) {
      removed_tracks_.push_back(&t);
    }
  }
}

std::array<float, 4> BoTSORTTracker::CurrentXYWH(const TrackInternal& track) const {
  if (track.has_kf_state) {
    return {track.kf_state.mean[0], track.kf_state.mean[1], std::max(track.kf_state.mean[2], 1e-4F),
            std::max(track.kf_state.mean[3], 1e-4F)};
  }
  return track.xywh;
}

std::array<float, 4> BoTSORTTracker::CurrentXYXY(const TrackInternal& track) const {
  return XYWHToXYXY(CurrentXYWH(track));
}

float BoTSORTTracker::FrameDiag() const {
  return std::sqrt(static_cast<float>(frame_width_ * frame_width_ + frame_height_ * frame_height_));
}

void BoTSORTTracker::PredictTracks(const cv::Mat& frame, const std::vector<DetectionEx>& detections,
                                   std::vector<TrackInternal*>* strack_pool,
                                   std::vector<TrackInternal*>* unconfirmed) {
  for (TrackInternal* t : *strack_pool) {
    KalmanPredict(t);
  }
  if (!cmc_) {
    return;
  }
  const cv::Mat warp = cmc_->Estimate(frame, detections);
  if (warp.empty()) {
    return;
  }
  for (TrackInternal* t : *strack_pool) {
    ApplyWarp(t, warp);
  }
  for (TrackInternal* t : *unconfirmed) {
    ApplyWarp(t, warp);
  }
}

Tracks BoTSORTTracker::Update(const Detections& detections, const cv::Mat& frame) {
  frame_count_ += 1;
  if (!frame.empty()) {
    frame_width_ = frame.cols;
    frame_height_ = frame.rows;
  }

  RefreshViews();

  std::vector<DetectionEx> dets;
  dets.reserve(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    DetectionEx ex{d.x1, d.y1, d.x2, d.y2, d.conf, d.cls, static_cast<int>(i)};
    dets.push_back(ex);
  }

  std::vector<DetectionEx> dets_first;
  std::vector<DetectionEx> dets_second;
  for (size_t i = 0; i < dets.size(); ++i) {
    if (dets[i].conf > cfg_.track_high_thresh) {
      dets_first.push_back(dets[i]);
    } else if (dets[i].conf > cfg_.track_low_thresh && dets[i].conf < cfg_.track_high_thresh) {
      dets_second.push_back(dets[i]);
    }
  }

  std::vector<std::vector<float>> features_first(dets_first.size());
  if (cfg_.with_reid && reid_ && reid_->IsReady() && !frame.empty() && !dets_first.empty()) {
    features_first = reid_->Extract(frame, dets_first);
  }

  std::vector<TrackInternal*> unconfirmed;
  std::vector<TrackInternal*> tracked;
  for (TrackInternal* t : active_tracks_) {
    if (!t->is_activated) {
      unconfirmed.push_back(t);
    } else {
      tracked.push_back(t);
    }
  }

  std::vector<TrackInternal*> strack_pool = JoinTracks(tracked, lost_tracks_);
  PredictTracks(frame, dets, &strack_pool, &unconfirmed);

  std::vector<TrackInternal*> activated_stracks;
  std::vector<TrackInternal*> refind_stracks;
  std::vector<TrackInternal*> lost_stracks;
  std::vector<TrackInternal*> removed_stracks;

  auto iou_first = IoUDistance(strack_pool, dets_first);
  std::vector<std::vector<float>> first_cost = iou_first;
  if (cfg_.fuse_first_associate) {
    FuseScore(&first_cost, dets_first);
  }
  if (cfg_.with_reid && !features_first.empty()) {
    auto emb = EmbeddingDistance(strack_pool, dets_first, features_first);
    for (size_t i = 0; i < first_cost.size(); ++i) {
      for (size_t j = 0; j < first_cost[i].size(); ++j) {
        if (emb[i][j] > cfg_.appearance_thresh || iou_first[i][j] > cfg_.proximity_thresh) {
          emb[i][j] = 1.0F;
        }
        first_cost[i][j] = std::min(first_cost[i][j], emb[i][j]);
      }
    }
  }

  std::vector<int> u_track_first;
  std::vector<int> u_det_first;
  std::vector<std::pair<int, int>> matches_first;
  if (strack_pool.empty()) {
    u_det_first.resize(dets_first.size());
    std::iota(u_det_first.begin(), u_det_first.end(), 0);
  } else {
    matches_first = LinearAssignment(first_cost, cfg_.match_thresh, &u_track_first, &u_det_first);
  }

  for (const auto& m : matches_first) {
    TrackInternal* track = strack_pool[static_cast<size_t>(m.first)];
    const DetectionEx& det = dets_first[static_cast<size_t>(m.second)];
    const std::vector<float>* feat =
        (m.second < static_cast<int>(features_first.size())) ? &features_first[static_cast<size_t>(m.second)]
                                                             : nullptr;
    if (track->state == TrackState::kTracked) {
      UpdateTrack(track, det, feat, frame_count_);
      activated_stracks.push_back(track);
    } else {
      ReActivateTrack(track, det, feat, frame_count_, false);
      refind_stracks.push_back(track);
    }
  }

  std::vector<TrackInternal*> r_tracked;
  for (int idx : u_track_first) {
    TrackInternal* t = strack_pool[static_cast<size_t>(idx)];
    if (t->state == TrackState::kTracked) {
      r_tracked.push_back(t);
    }
  }
  std::vector<int> u_track_second;
  std::vector<int> u_det_second;
  std::vector<std::pair<int, int>> matches_second;
  if (r_tracked.empty()) {
    u_det_second.resize(dets_second.size());
    std::iota(u_det_second.begin(), u_det_second.end(), 0);
  } else {
    auto second_cost = IoUDistance(r_tracked, dets_second);
    matches_second = LinearAssignment(second_cost, cfg_.second_association_thresh, &u_track_second,
                                      &u_det_second);
  }
  for (const auto& m : matches_second) {
    TrackInternal* track = r_tracked[static_cast<size_t>(m.first)];
    UpdateTrack(track, dets_second[static_cast<size_t>(m.second)], nullptr, frame_count_);
    activated_stracks.push_back(track);
  }
  for (int idx : u_track_second) {
    TrackInternal* t = r_tracked[static_cast<size_t>(idx)];
    if (t->state != TrackState::kLost) {
      MarkLost(t);
      lost_stracks.push_back(t);
    }
  }

  std::vector<DetectionEx> unc_detections = SelectByIndices(dets_first, u_det_first);
  std::vector<std::vector<float>> unc_features;
  unc_features.reserve(u_det_first.size());
  for (int idx : u_det_first) {
    if (idx >= 0 && idx < static_cast<int>(features_first.size())) {
      unc_features.push_back(features_first[static_cast<size_t>(idx)]);
    } else {
      unc_features.push_back({});
    }
  }

  auto iou_unc = IoUDistance(unconfirmed, unc_detections);
  auto unc_cost = iou_unc;
  FuseScore(&unc_cost, unc_detections);
  if (cfg_.with_reid) {
    auto emb_unc = EmbeddingDistance(unconfirmed, unc_detections, unc_features);
    for (size_t i = 0; i < unc_cost.size(); ++i) {
      for (size_t j = 0; j < unc_cost[i].size(); ++j) {
        emb_unc[i][j] *= 0.5F;
        if (emb_unc[i][j] > cfg_.appearance_thresh || iou_unc[i][j] > cfg_.proximity_thresh) {
          emb_unc[i][j] = 1.0F;
        }
        unc_cost[i][j] = std::min(unc_cost[i][j], emb_unc[i][j]);
      }
    }
  }

  std::vector<int> u_unconfirmed;
  std::vector<int> u_det_unc;
  std::vector<std::pair<int, int>> matches_unc;
  if (unconfirmed.empty()) {
    u_det_unc.resize(unc_detections.size());
    std::iota(u_det_unc.begin(), u_det_unc.end(), 0);
  } else {
    matches_unc = LinearAssignment(unc_cost, cfg_.unconfirmed_association_thresh, &u_unconfirmed,
                                   &u_det_unc);
  }
  for (const auto& m : matches_unc) {
    TrackInternal* t = unconfirmed[static_cast<size_t>(m.first)];
    const std::vector<float>* feat =
        (m.second < static_cast<int>(unc_features.size())) ? &unc_features[static_cast<size_t>(m.second)]
                                                           : nullptr;
    UpdateTrack(t, unc_detections[static_cast<size_t>(m.second)], feat, frame_count_);
    activated_stracks.push_back(t);
  }
  for (int idx : u_unconfirmed) {
    TrackInternal* t = unconfirmed[static_cast<size_t>(idx)];
    MarkRemoved(t);
    removed_stracks.push_back(t);
  }

  std::vector<DetectionEx> remaining_for_new;
  std::vector<std::vector<float>> remaining_feat;
  if (cfg_.with_reid && cfg_.reid_recovery_enabled && !u_det_unc.empty()) {
    std::vector<TrackInternal*> lost_candidates;
    for (int idx : u_track_first) {
      TrackInternal* t = strack_pool[static_cast<size_t>(idx)];
      if (t->state == TrackState::kLost) {
        lost_candidates.push_back(t);
      }
    }
    std::vector<DetectionEx> recovery_dets;
    std::vector<std::vector<float>> recovery_feats;
    for (int idx : u_det_unc) {
      recovery_dets.push_back(unc_detections[static_cast<size_t>(idx)]);
      recovery_feats.push_back(unc_features[static_cast<size_t>(idx)]);
    }
    if (!lost_candidates.empty() && !recovery_dets.empty()) {
      std::vector<std::vector<float>> recovery_cost(
          lost_candidates.size(), std::vector<float>(recovery_dets.size(), 1.0F));
      const float diag = FrameDiag();
      for (size_t i = 0; i < lost_candidates.size(); ++i) {
        const auto tbox = CurrentXYXY(*lost_candidates[i]);
        for (size_t j = 0; j < recovery_dets.size(); ++j) {
          const std::array<float, 4> dbox = {recovery_dets[j].x1, recovery_dets[j].y1,
                                             recovery_dets[j].x2, recovery_dets[j].y2};
          float emb = 1.0F;
          if (!lost_candidates[i]->smooth_feat.empty() && !recovery_feats[j].empty()) {
            emb = CosineDistance(lost_candidates[i]->smooth_feat, recovery_feats[j]);
          }
          if (emb > cfg_.reid_recovery_appearance_thresh) {
            emb = 1.0F;
          }
          const float center = NormCenterDistance(tbox, dbox, diag);
          if (center > cfg_.reid_recovery_proximity_thresh) {
            recovery_cost[i][j] = 1.0F;
            continue;
          }
          recovery_cost[i][j] = 1.2F * center + 0.2F * emb;
        }
      }

      std::vector<int> u_lost;
      std::vector<int> u_det_recovery;
      const auto rec_matches = LinearAssignment(recovery_cost, cfg_.reid_recovery_thresh, &u_lost,
                                                &u_det_recovery);
      for (const auto& m : rec_matches) {
        const std::vector<float>* feat =
            (m.second < static_cast<int>(recovery_feats.size()))
                ? &recovery_feats[static_cast<size_t>(m.second)]
                : nullptr;
        ReActivateTrack(lost_candidates[static_cast<size_t>(m.first)],
                        recovery_dets[static_cast<size_t>(m.second)], feat, frame_count_, false);
        refind_stracks.push_back(lost_candidates[static_cast<size_t>(m.first)]);
      }
      for (int idx : u_det_recovery) {
        remaining_for_new.push_back(recovery_dets[static_cast<size_t>(idx)]);
        remaining_feat.push_back(recovery_feats[static_cast<size_t>(idx)]);
      }
    } else {
      for (int idx : u_det_unc) {
        remaining_for_new.push_back(unc_detections[static_cast<size_t>(idx)]);
        remaining_feat.push_back(unc_features[static_cast<size_t>(idx)]);
      }
    }
  } else {
    for (int idx : u_det_unc) {
      remaining_for_new.push_back(unc_detections[static_cast<size_t>(idx)]);
      remaining_feat.push_back(unc_features[static_cast<size_t>(idx)]);
    }
  }

  for (size_t i = 0; i < remaining_for_new.size(); ++i) {
    if (remaining_for_new[i].conf < cfg_.new_track_thresh) {
      continue;
    }
    tracks_.push_back(BuildTrackFromDet(remaining_for_new[i], &remaining_feat[i]));
    TrackInternal* t = &tracks_.back();
    ActivateTrack(t, frame_count_);
    activated_stracks.push_back(t);
  }

  for (TrackInternal* t : lost_tracks_) {
    if (frame_count_ - t->frame_id > max_time_lost_) {
      MarkRemoved(t);
      removed_stracks.push_back(t);
    }
  }

  RefreshViews();
  std::vector<TrackInternal*> tracked_now;
  for (TrackInternal* t : active_tracks_) {
    if (t->state == TrackState::kTracked) {
      tracked_now.push_back(t);
    }
  }
  tracked_now = JoinTracks(tracked_now, activated_stracks);
  tracked_now = JoinTracks(tracked_now, refind_stracks);
  auto lost_now = SubTracks(lost_tracks_, tracked_now);
  for (TrackInternal* t : lost_stracks) {
    lost_now.push_back(t);
  }
  lost_now = SubTracks(lost_now, removed_tracks_);
  std::vector<TrackInternal*> removed_now = removed_tracks_;
  removed_now.insert(removed_now.end(), removed_stracks.begin(), removed_stracks.end());
  RemoveDuplicateTracks(&tracked_now, &lost_now);

  std::unordered_map<std::int64_t, TrackState> next_state;
  for (TrackInternal* t : tracked_now) {
    next_state[t->id] = TrackState::kTracked;
  }
  for (TrackInternal* t : lost_now) {
    if (next_state.find(t->id) == next_state.end()) {
      next_state[t->id] = TrackState::kLost;
    }
  }
  for (TrackInternal* t : removed_now) {
    if (next_state.find(t->id) == next_state.end()) {
      next_state[t->id] = TrackState::kRemoved;
    }
  }
  for (TrackInternal& t : tracks_) {
    auto it = next_state.find(t.id);
    if (it != next_state.end()) {
      t.state = it->second;
    }
  }
  RefreshViews();

  Tracks output;
  for (TrackInternal* t : active_tracks_) {
    const bool visible_this_frame = t->is_activated || (t->frame_id == frame_count_);
    if (t->state != TrackState::kTracked || !visible_this_frame) {
      continue;
    }
    const auto xyxy = CurrentXYXY(*t);
    Track track;
    track.x1 = xyxy[0];
    track.y1 = xyxy[1];
    track.x2 = xyxy[2];
    track.y2 = xyxy[3];
    track.id = t->id;
    track.conf = t->conf;
    track.cls = t->cls;
    track.det_index = t->det_index;
    output.push_back(track);
  }
  return output;
}

BoTSORTTracker::ReIdExtractor::ReIdExtractor(const TrackerRuntimeConfig& cfg)
    : input_width_(cfg.reid_input_width),
      input_height_(cfg.reid_input_height),
      max_detections_(cfg.reid_max_detections) {
  try {
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options_.SetIntraOpNumThreads(1);
#ifdef _WIN32
    const std::wstring path = Utf8ToWide(cfg.reid_model_path);
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, cfg.reid_model_path.c_str(), options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    {
      auto name = session_->GetInputNameAllocated(0, allocator);
      input_name_ = name.get();
    }
    {
      auto name = session_->GetOutputNameAllocated(0, allocator);
      output_name_ = name.get();
    }
    const auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto shape = input_info.GetShape();
    if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
      input_height_ = static_cast<int>(shape[2]);
      input_width_ = static_cast<int>(shape[3]);
    }
    ready_ = true;
  } catch (...) {
    ready_ = false;
  }
}

std::vector<float> BoTSORTTracker::ReIdExtractor::PreprocessCrop(const cv::Mat& crop) const {
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(input_width_, input_height_), 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat f32;
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels;
  cv::split(f32, channels);
  const std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
  const std::array<float, 3> stdv = {0.229F, 0.224F, 0.225F};
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - mean[c]) / stdv[c];
  }

  const size_t plane = static_cast<size_t>(input_width_ * input_height_);
  std::vector<float> tensor(plane * 3);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(tensor.data() + static_cast<size_t>(c) * plane, channels[c].ptr<float>(0),
                plane * sizeof(float));
  }
  return tensor;
}

std::vector<std::vector<float>> BoTSORTTracker::ReIdExtractor::RunBatch(
    const std::vector<cv::Mat>& crops) const {
  if (crops.empty()) {
    return {};
  }
  const size_t batch = crops.size();
  const size_t plane = static_cast<size_t>(input_width_ * input_height_);
  const size_t single = plane * 3;
  std::vector<float> input(single * batch);
  for (size_t i = 0; i < batch; ++i) {
    std::vector<float> pre = PreprocessCrop(crops[i]);
    std::memcpy(input.data() + i * single, pre.data(), single * sizeof(float));
  }
  const std::array<std::int64_t, 4> input_shape = {static_cast<std::int64_t>(batch), 3, input_height_,
                                                    input_width_};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                       input_shape.data(), input_shape.size());
  std::array<const char*, 1> in_names = {input_name_.c_str()};
  std::array<const char*, 1> out_names = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &tensor, 1,
                               out_names.data(), 1);
  if (outputs.empty() || !outputs[0].IsTensor()) {
    return std::vector<std::vector<float>>(batch);
  }
  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  const float* data = outputs[0].GetTensorData<float>();
  int64_t dim = 1;
  for (int64_t s : shape) {
    if (s > 0) {
      dim *= s;
    }
  }
  if (dim <= 0 || static_cast<size_t>(dim) < batch) {
    return std::vector<std::vector<float>>(batch);
  }
  const size_t feature_dim = static_cast<size_t>(dim / static_cast<int64_t>(batch));
  std::vector<std::vector<float>> out(batch);
  for (size_t i = 0; i < batch; ++i) {
    if (feature_dim == 0) {
      continue;
    }
    out[i].resize(feature_dim);
    std::memcpy(out[i].data(), data + (i * feature_dim), feature_dim * sizeof(float));
    BoTSORTTracker::L2Normalize(&out[i]);
  }
  return out;
}

std::vector<std::vector<float>> BoTSORTTracker::ReIdExtractor::Extract(
    const cv::Mat& frame, const std::vector<DetectionEx>& detections) const {
  std::vector<std::vector<float>> feats;
  feats.reserve(detections.size());
  if (detections.empty()) {
    return feats;
  }
  struct Candidate {
    int index = -1;
    cv::Mat crop;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(std::min(static_cast<int>(detections.size()), max_detections_));

  std::vector<int> order(detections.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return detections[static_cast<size_t>(a)].conf > detections[static_cast<size_t>(b)].conf;
  });

  for (int det_idx : order) {
    if (static_cast<int>(candidates.size()) >= max_detections_) {
      break;
    }
    const auto& det = detections[static_cast<size_t>(det_idx)];
    const int x1 = std::max(0, static_cast<int>(std::floor(det.x1)));
    const int y1 = std::max(0, static_cast<int>(std::floor(det.y1)));
    const int x2 = std::min(frame.cols, static_cast<int>(std::ceil(det.x2)));
    const int y2 = std::min(frame.rows, static_cast<int>(std::ceil(det.y2)));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    candidates.push_back({det_idx, frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone()});
  }

  feats.assign(detections.size(), {});
  std::vector<cv::Mat> crops;
  crops.reserve(candidates.size());
  for (const auto& c : candidates) {
    crops.push_back(c.crop);
  }
  const auto batch_feats = RunBatch(crops);
  for (size_t i = 0; i < candidates.size() && i < batch_feats.size(); ++i) {
    feats[static_cast<size_t>(candidates[i].index)] = batch_feats[i];
  }
  return feats;
}

BoTSORTTracker::CameraMotionCompensator::CameraMotionCompensator(const TrackerRuntimeConfig& cfg)
    : interval_(cfg.cmc_interval), max_side_(cfg.cmc_max_side) {
  std::string lowered = cfg.cmc_method;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  enabled_ = (lowered == "ecc");
  last_warp_ = cv::Mat::eye(2, 3, CV_32F);
}

cv::Mat BoTSORTTracker::CameraMotionCompensator::Estimate(
    const cv::Mat& frame, const std::vector<DetectionEx>& detections) {
  if (!enabled_ || frame.empty()) {
    return cv::Mat();
  }
  frame_counter_ += 1;
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  if (prev_gray_.empty()) {
    prev_gray_ = gray;
    return cv::Mat::eye(2, 3, CV_32F);
  }
  if ((frame_counter_ % interval_) != 0) {
    prev_gray_ = gray;
    return last_warp_;
  }

  const int longest_side = std::max(gray.cols, gray.rows);
  float scale = 1.0F;
  if (longest_side > max_side_) {
    scale = static_cast<float>(max_side_) / static_cast<float>(longest_side);
  }

  cv::Mat gray_small = gray;
  cv::Mat prev_small = prev_gray_;
  if (scale < 1.0F) {
    cv::resize(gray, gray_small, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::resize(prev_gray_, prev_small, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
  cv::Mat mask(gray_small.size(), CV_8UC1, cv::Scalar(255));
  for (const auto& det : detections) {
    const int x1 = std::max(0, static_cast<int>(det.x1 * scale));
    const int y1 = std::max(0, static_cast<int>(det.y1 * scale));
    const int x2 = std::min(gray_small.cols, static_cast<int>(det.x2 * scale));
    const int y2 = std::min(gray_small.rows, static_cast<int>(det.y2 * scale));
    if (x2 > x1 && y2 > y1) {
      cv::rectangle(mask, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0), cv::FILLED);
    }
  }

  try {
    const auto criteria =
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 1e-3);
    cv::findTransformECC(prev_small, gray_small, warp, cv::MOTION_AFFINE, criteria, mask);
    if (scale < 1.0F) {
      warp.at<float>(0, 2) /= scale;
      warp.at<float>(1, 2) /= scale;
    }
  } catch (...) {
    warp = cv::Mat::eye(2, 3, CV_32F);
  }
  prev_gray_ = gray;
  last_warp_ = warp;
  return last_warp_;
}

}  // namespace tracking

