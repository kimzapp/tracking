#include "tracking/core/config.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace tracking {

namespace {

template <typename T>
void ReadScalarIfPresent(const YAML::Node& node, const char* key, T* out) {
  if (!node[key]) {
    return;
  }
  *out = node[key].as<T>();
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

YAML::Node ExtractCfgNode(const YAML::Node& node) {
  if (!node || !node.IsMap()) {
    return node;
  }
  if (node["value"]) {
    return node["value"];
  }
  if (node["default"]) {
    return node["default"];
  }
  if (node["current"]) {
    return node["current"];
  }
  return node;
}

template <typename T>
void ReadCfgValueIfPresent(const YAML::Node& root, const char* key, T* out) {
  if (!root[key]) {
    return;
  }
  *out = ExtractCfgNode(root[key]).as<T>();
}

bool ParseBoolNode(const YAML::Node& node, bool fallback) {
  if (!node) {
    return fallback;
  }
  const YAML::Node extracted = ExtractCfgNode(node);
  if (extracted.IsScalar()) {
    const std::string text = ToLower(extracted.as<std::string>());
    if (text == "1" || text == "true" || text == "yes" || text == "on") {
      return true;
    }
    if (text == "0" || text == "false" || text == "no" || text == "off") {
      return false;
    }
  }
  return extracted.as<bool>();
}

void ValidateZeroToOne(float value, const char* name) {
  if (value < 0.0F || value > 1.0F) {
    throw std::runtime_error(std::string(name) + " must be in [0, 1]");
  }
}

}  // namespace

AppConfig LoadAppConfig(const std::string& path) {
  AppConfig cfg;
  const YAML::Node root = YAML::LoadFile(path);

  ReadScalarIfPresent(root, "model_path", &cfg.model_path);
  ReadScalarIfPresent(root, "video_path", &cfg.video_path);
  ReadScalarIfPresent(root, "output_path", &cfg.output_path);
  ReadScalarIfPresent(root, "execution_device", &cfg.execution_device);
  ReadScalarIfPresent(root, "gpu_device_id", &cfg.gpu_device_id);

  ReadScalarIfPresent(root, "det_conf", &cfg.det_conf);
  ReadScalarIfPresent(root, "det_iou", &cfg.det_iou);
  ReadScalarIfPresent(root, "input_width", &cfg.input_width);
  ReadScalarIfPresent(root, "input_height", &cfg.input_height);

  ReadScalarIfPresent(root, "show_window", &cfg.show_window);
  ReadScalarIfPresent(root, "window_name", &cfg.window_name);
  ReadScalarIfPresent(root, "draw_detections", &cfg.draw_detections);
  ReadScalarIfPresent(root, "draw_tracks", &cfg.draw_tracks);
  ReadScalarIfPresent(root, "baseline_output_path", &cfg.baseline_output_path);
  ReadCfgValueIfPresent(root, "tracker_type", &cfg.tracker.tracker_type);
  cfg.tracker.execution_device = cfg.execution_device;
  cfg.tracker.gpu_device_id = cfg.gpu_device_id;
  ReadCfgValueIfPresent(root, "reid_execution_device", &cfg.tracker.execution_device);
  ReadCfgValueIfPresent(root, "reid_gpu_device_id", &cfg.tracker.gpu_device_id);

  ReadCfgValueIfPresent(root, "track_high_thresh", &cfg.tracker.track_high_thresh);
  ReadCfgValueIfPresent(root, "track_low_thresh", &cfg.tracker.track_low_thresh);
  ReadCfgValueIfPresent(root, "new_track_thresh", &cfg.tracker.new_track_thresh);
  ReadCfgValueIfPresent(root, "track_buffer", &cfg.tracker.track_buffer);
  ReadCfgValueIfPresent(root, "match_thresh", &cfg.tracker.match_thresh);
  ReadCfgValueIfPresent(root, "proximity_thresh", &cfg.tracker.proximity_thresh);
  ReadCfgValueIfPresent(root, "appearance_thresh", &cfg.tracker.appearance_thresh);
  ReadCfgValueIfPresent(root, "reid_recovery_proximity_thresh",
                        &cfg.tracker.reid_recovery_proximity_thresh);
  ReadCfgValueIfPresent(root, "reid_recovery_appearance_thresh",
                        &cfg.tracker.reid_recovery_appearance_thresh);
  ReadCfgValueIfPresent(root, "reid_recovery_thresh", &cfg.tracker.reid_recovery_thresh);
  ReadCfgValueIfPresent(root, "second_association_thresh",
                        &cfg.tracker.second_association_thresh);
  ReadCfgValueIfPresent(root, "unconfirmed_association_thresh",
                        &cfg.tracker.unconfirmed_association_thresh);
  ReadCfgValueIfPresent(root, "cmc_method", &cfg.tracker.cmc_method);
  ReadCfgValueIfPresent(root, "reid_model_path", &cfg.tracker.reid_model_path);
  ReadCfgValueIfPresent(root, "reid_input_width", &cfg.tracker.reid_input_width);
  ReadCfgValueIfPresent(root, "reid_input_height", &cfg.tracker.reid_input_height);
  ReadCfgValueIfPresent(root, "reid_max_detections", &cfg.tracker.reid_max_detections);
  ReadCfgValueIfPresent(root, "frame_rate", &cfg.tracker.frame_rate);
  ReadCfgValueIfPresent(root, "cmc_interval", &cfg.tracker.cmc_interval);
  ReadCfgValueIfPresent(root, "cmc_max_side", &cfg.tracker.cmc_max_side);

  if (root["reid_recovery_enabled"]) {
    cfg.tracker.reid_recovery_enabled = ParseBoolNode(root["reid_recovery_enabled"], true);
  }
  if (root["with_reid"]) {
    cfg.tracker.with_reid = ParseBoolNode(root["with_reid"], true);
  }
  if (root["fuse_first_associate"]) {
    cfg.tracker.fuse_first_associate = ParseBoolNode(root["fuse_first_associate"], false);
  }

  if (cfg.input_width <= 0 || cfg.input_height <= 0) {
    throw std::runtime_error("input_width/input_height must be > 0");
  }
  ValidateZeroToOne(cfg.det_conf, "det_conf");
  ValidateZeroToOne(cfg.det_iou, "det_iou");
  ValidateZeroToOne(cfg.tracker.track_high_thresh, "track_high_thresh");
  ValidateZeroToOne(cfg.tracker.track_low_thresh, "track_low_thresh");
  ValidateZeroToOne(cfg.tracker.new_track_thresh, "new_track_thresh");
  ValidateZeroToOne(cfg.tracker.match_thresh, "match_thresh");
  ValidateZeroToOne(cfg.tracker.proximity_thresh, "proximity_thresh");
  ValidateZeroToOne(cfg.tracker.appearance_thresh, "appearance_thresh");
  ValidateZeroToOne(cfg.tracker.reid_recovery_proximity_thresh,
                    "reid_recovery_proximity_thresh");
  ValidateZeroToOne(cfg.tracker.reid_recovery_appearance_thresh,
                    "reid_recovery_appearance_thresh");
  ValidateZeroToOne(cfg.tracker.reid_recovery_thresh, "reid_recovery_thresh");
  ValidateZeroToOne(cfg.tracker.second_association_thresh, "second_association_thresh");
  ValidateZeroToOne(cfg.tracker.unconfirmed_association_thresh,
                    "unconfirmed_association_thresh");

  if (cfg.tracker.track_buffer <= 0) {
    throw std::runtime_error("track_buffer must be > 0");
  }
  if (cfg.tracker.reid_input_width <= 0 || cfg.tracker.reid_input_height <= 0) {
    throw std::runtime_error("reid_input_width/reid_input_height must be > 0");
  }
  if (cfg.tracker.reid_max_detections <= 0) {
    throw std::runtime_error("reid_max_detections must be > 0");
  }
  if (cfg.tracker.frame_rate <= 0) {
    throw std::runtime_error("frame_rate must be > 0");
  }
  if (cfg.tracker.cmc_interval <= 0) {
    throw std::runtime_error("cmc_interval must be > 0");
  }
  if (cfg.tracker.cmc_max_side <= 0) {
    throw std::runtime_error("cmc_max_side must be > 0");
  }

  cfg.tracker.tracker_type = ToLower(cfg.tracker.tracker_type);
  if (cfg.tracker.tracker_type != "botsort" && cfg.tracker.tracker_type != "passthrough") {
    throw std::runtime_error("tracker_type must be one of: botsort, passthrough");
  }

  cfg.execution_device = ToLower(cfg.execution_device);
  if (cfg.execution_device != "auto" && cfg.execution_device != "cpu" &&
      cfg.execution_device != "gpu") {
    throw std::runtime_error("execution_device must be one of: auto, cpu, gpu");
  }
  if (cfg.gpu_device_id < 0) {
    throw std::runtime_error("gpu_device_id must be >= 0");
  }

  cfg.tracker.execution_device = ToLower(cfg.tracker.execution_device);
  if (cfg.tracker.execution_device != "auto" && cfg.tracker.execution_device != "cpu" &&
      cfg.tracker.execution_device != "gpu") {
    throw std::runtime_error("reid_execution_device must be one of: auto, cpu, gpu");
  }
  if (cfg.tracker.gpu_device_id < 0) {
    throw std::runtime_error("reid_gpu_device_id must be >= 0");
  }

  cfg.tracker.cmc_method = ToLower(cfg.tracker.cmc_method);
  if (cfg.tracker.cmc_method == "none" || cfg.tracker.cmc_method == "null") {
    cfg.tracker.cmc_method.clear();
  }

  return cfg;
}

}  // namespace tracking
