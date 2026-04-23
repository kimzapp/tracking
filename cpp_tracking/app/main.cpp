#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "tracking/core/config.hpp"
#include "tracking/detector/yolo_onnx_detector.hpp"
#include "tracking/tracker/tracker_interface.hpp"
#include "tracking/utils/logging.hpp"

namespace tracking {
namespace {

int DynamicThickness(const cv::Mat& frame, float scale, int min_value) {
  const int base = static_cast<int>(std::round(std::min(frame.cols, frame.rows) * scale));
  return std::max(min_value, base);
}

cv::Scalar ColorFromId(std::int64_t id) {
  const int wrapped_hue = static_cast<int>((std::llabs(id) * 37LL) % 360LL);
  // OpenCV HSV hue uses [0, 179], satur/value use [0, 255].
  const unsigned char hue = static_cast<unsigned char>(wrapped_hue / 2);
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 199, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

cv::Scalar DetectionColor(float cls) {
  static const std::array<cv::Scalar, 4> kDetectionPalette = {
      cv::Scalar(40, 220, 255),   // yellow-cyan
      cv::Scalar(40, 255, 180),   // green-cyan
      cv::Scalar(140, 255, 80),   // green
      cv::Scalar(255, 210, 90),   // orange
  };
  const int cls_id = static_cast<int>(std::round(cls));
  const size_t idx = static_cast<size_t>(std::abs(cls_id)) % kDetectionPalette.size();
  return kDetectionPalette[idx];
}

std::string FrameProgressLabel(int frame_index, int total_frames) {
  if (total_frames > 0) {
    return std::to_string(frame_index) + "/" + std::to_string(total_frames);
  }
  return std::to_string(frame_index) + "/?";
}

void LogFrameProgress(int frame_index, int total_frames, size_t detection_count, size_t active_track_count,
                      size_t unique_seen_ids) {
  LogInfo("Frame " + FrameProgressLabel(frame_index, total_frames) + " | dets=" +
          std::to_string(detection_count) + " | active_tracks=" +
          std::to_string(active_track_count) + " | unique_ids_seen=" +
          std::to_string(unique_seen_ids));
}

void DrawTrack(cv::Mat* frame, const Track& track) {
  cv::Rect box(static_cast<int>(track.x1), static_cast<int>(track.y1),
               static_cast<int>(track.x2 - track.x1), static_cast<int>(track.y2 - track.y1));
  box &= cv::Rect(0, 0, frame->cols, frame->rows);
  if (box.width <= 0 || box.height <= 0) {
    return;
  }

  const cv::Scalar color = ColorFromId(track.id);
  const int thickness = DynamicThickness(*frame, 0.0022F, 2);
  const double font_scale = std::max(0.55, std::min(0.8, std::min(frame->cols, frame->rows) / 1200.0));
  cv::Mat overlay = frame->clone();
  cv::rectangle(overlay, box, color, cv::FILLED, cv::LINE_AA);
  cv::addWeighted(overlay, 0.08, *frame, 0.92, 0.0, *frame);
  cv::rectangle(*frame, box, color, thickness, cv::LINE_AA);

  const std::string label = "ID " + std::to_string(track.id);
  int baseline = 0;
  const cv::Size text_size =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 2, &baseline);
  const int pad = 6;
  int label_x1 = box.x;
  int label_y2 = box.y - 6;
  const int label_height = text_size.height + 2 * pad;
  if (label_y2 - label_height < 0) {
    label_y2 = box.y + label_height + 6;
  }
  label_y2 = std::min(frame->rows - 1, label_y2);
  const int label_y1 = std::max(0, label_y2 - label_height);
  const int label_x2 = std::min(frame->cols - 1, label_x1 + text_size.width + 2 * pad);
  cv::Rect bg(label_x1, label_y1, std::max(0, label_x2 - label_x1), std::max(0, label_y2 - label_y1));
  if (bg.width > 0 && bg.height > 0) {
    cv::rectangle(*frame, bg, color, cv::FILLED, cv::LINE_AA);
  }
  cv::putText(*frame, label, cv::Point(bg.x + pad, bg.y + bg.height - pad - baseline),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
}

void DrawDetection(cv::Mat* frame, const Detection& det) {
  cv::Rect box(static_cast<int>(det.x1), static_cast<int>(det.y1),
               static_cast<int>(det.x2 - det.x1), static_cast<int>(det.y2 - det.y1));
  box &= cv::Rect(0, 0, frame->cols, frame->rows);
  if (box.width <= 0 || box.height <= 0) {
    return;
  }
  const cv::Scalar color = DetectionColor(det.cls);
  const int stroke_outer = DynamicThickness(*frame, 0.0028F, 3);
  const int stroke_inner = DynamicThickness(*frame, 0.0018F, 2);
  const double font_scale = std::max(0.5, std::min(0.75, std::min(frame->cols, frame->rows) / 1300.0));
  cv::rectangle(*frame, box, cv::Scalar(0, 0, 0), stroke_outer, cv::LINE_AA);
  cv::rectangle(*frame, box, color, stroke_inner, cv::LINE_AA);
  const std::string label = "det " + std::to_string(static_cast<int>(det.cls)) + " " +
                            std::to_string(static_cast<int>(det.conf * 100.0F)) + "%";
  cv::putText(*frame, label, cv::Point(box.x, std::max(0, box.y - 6)), cv::FONT_HERSHEY_SIMPLEX,
              font_scale, color, 2, cv::LINE_AA);
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char c : text) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

void WriteBaselineJson(const tracking::AppConfig& cfg,
                       const std::vector<tracking::Detections>& frame_dets,
                       const std::vector<tracking::Tracks>& frame_tracks) {
  if (cfg.baseline_output_path.empty()) {
    return;
  }
  const std::filesystem::path out_path(cfg.baseline_output_path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }

  std::ofstream ofs(cfg.baseline_output_path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Unable to write baseline json: " + cfg.baseline_output_path);
  }
  ofs << std::fixed << std::setprecision(6);
  ofs << "{\n";
  ofs << "  \"video\": \"" << JsonEscape(cfg.video_path) << "\",\n";
  ofs << "  \"frames_exported\": " << frame_tracks.size() << ",\n";
  ofs << "  \"records\": [\n";
  for (size_t i = 0; i < frame_tracks.size(); ++i) {
    ofs << "    {\n";
    ofs << "      \"frame_index\": " << i << ",\n";
    ofs << "      \"dets\": [";
    for (size_t d = 0; d < frame_dets[i].size(); ++d) {
      const auto& det = frame_dets[i][d];
      if (d > 0) {
        ofs << ",";
      }
      ofs << "[" << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << "," << det.conf
          << "," << det.cls << "]";
    }
    ofs << "],\n";
    ofs << "      \"tracks\": [";
    for (size_t t = 0; t < frame_tracks[i].size(); ++t) {
      const auto& track = frame_tracks[i][t];
      if (t > 0) {
        ofs << ",";
      }
      ofs << "[" << track.x1 << "," << track.y1 << "," << track.x2 << "," << track.y2 << ","
          << static_cast<double>(track.id) << "," << track.conf << "," << track.cls << ","
          << track.det_index << "]";
    }
    ofs << "],\n";
    ofs << "      \"det_count\": " << frame_dets[i].size() << ",\n";
    ofs << "      \"track_count\": " << frame_tracks[i].size() << "\n";
    ofs << "    }";
    if (i + 1 < frame_tracks.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "  ]\n";
  ofs << "}\n";
}

}  // namespace
}  // namespace tracking

int main(int argc, char** argv) {
  try {
    std::string config_path = "config/app.yaml";
    if (argc > 1) {
      config_path = argv[1];
    }

    const tracking::AppConfig cfg = tracking::LoadAppConfig(config_path);

    tracking::LogInfo("Loading ONNX detector: " + cfg.model_path);
    tracking::LogInfo("Inference device preference: " + cfg.execution_device +
                      " (gpu_device_id=" + std::to_string(cfg.gpu_device_id) + ")");
    tracking::YoloOnnxDetector detector(cfg.model_path, cfg.input_width, cfg.input_height, cfg.det_conf,
                                        cfg.det_iou, cfg.execution_device, cfg.gpu_device_id);

    std::unique_ptr<tracking::ITracker> tracker = tracking::CreateTracker(cfg.tracker);
    tracking::LogInfo("Tracker type: " + cfg.tracker.tracker_type);

    cv::VideoCapture cap(cfg.video_path);
    if (!cap.isOpened()) {
      throw std::runtime_error("Unable to open video: " + cfg.video_path);
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
      fps = 30.0;
    }
    int total_frames = static_cast<int>(std::llround(cap.get(cv::CAP_PROP_FRAME_COUNT)));
    if (total_frames <= 0) {
      total_frames = -1;
    }
    const int frame_log_interval = cfg.show_window ? 30 : 1;

    const std::filesystem::path output_path(cfg.output_path);
    if (!output_path.parent_path().empty()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    cv::VideoWriter writer(cfg.output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
      throw std::runtime_error("Unable to create output writer: " + cfg.output_path);
    }

    std::set<std::int64_t> seen_ids;
    std::vector<tracking::Detections> all_dets;
    std::vector<tracking::Tracks> all_tracks;
    cv::Mat frame;
    int frame_index = 0;

    while (cap.read(frame)) {
      frame_index += 1;
      tracking::Detections detections = detector.Predict(frame);
      tracking::Tracks tracks = tracker->Update(detections, frame);
      for (const tracking::Track& t : tracks) {
        seen_ids.insert(t.id);
      }

      if (cfg.draw_detections) {
        for (const tracking::Detection& d : detections) {
          tracking::DrawDetection(&frame, d);
        }
      }
      if (cfg.draw_tracks) {
        for (const tracking::Track& t : tracks) {
          tracking::DrawTrack(&frame, t);
        }
      }

      if (!cfg.baseline_output_path.empty()) {
        all_dets.push_back(detections);
        all_tracks.push_back(tracks);
      }

      writer.write(frame);

      if (frame_index == 1 || frame_index % frame_log_interval == 0 ||
          (total_frames > 0 && frame_index >= total_frames)) {
        tracking::LogFrameProgress(frame_index, total_frames, detections.size(), tracks.size(),
                                   seen_ids.size());
      }

      if (cfg.show_window) {
        cv::imshow(cfg.window_name, frame);
        if ((cv::waitKey(1) & 0xFF) == 'q') {
          break;
        }
      }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    if (!cfg.baseline_output_path.empty()) {
      tracking::WriteBaselineJson(cfg, all_dets, all_tracks);
      tracking::LogInfo("Saved baseline json to: " + cfg.baseline_output_path);
    }

    tracking::LogInfo("Processed frames: " + std::to_string(frame_index));
    tracking::LogInfo("Done. Seen IDs: " + std::to_string(seen_ids.size()));
    tracking::LogInfo("Saved output to: " + cfg.output_path);
    return 0;
  } catch (const std::exception& ex) {
    tracking::LogError(ex.what());
    return 1;
  }
}
