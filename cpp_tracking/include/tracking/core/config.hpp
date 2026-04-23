#pragma once

#include <cstdint>
#include <string>

namespace tracking {

struct TrackerRuntimeConfig {
  std::string tracker_type = "botsort";
  std::string execution_device = "auto";
  int gpu_device_id = 0;

  float track_high_thresh = 0.25F;
  float track_low_thresh = 0.10F;
  float new_track_thresh = 0.25F;
  int track_buffer = 30;
  float match_thresh = 0.80F;
  float proximity_thresh = 0.50F;
  float appearance_thresh = 0.25F;
  bool reid_recovery_enabled = true;
  float reid_recovery_proximity_thresh = 0.50F;
  float reid_recovery_appearance_thresh = 0.25F;
  float reid_recovery_thresh = 0.35F;
  float second_association_thresh = 0.50F;
  float unconfirmed_association_thresh = 0.70F;
  std::string cmc_method = "ecc";
  bool with_reid = true;
  bool fuse_first_associate = false;
  std::string reid_model_path = "../models/osnet_x0_5_msmt17.onnx";
  int reid_input_width = 128;
  int reid_input_height = 256;
  int reid_max_detections = 48;
  int frame_rate = 30;
  int cmc_interval = 3;
  int cmc_max_side = 960;
};

struct AppConfig {
  std::string model_path = "../best.onnx";
  std::string video_path = "../video/longchau.mp4";
  std::string output_path = "../video/longchau_cpp_output.mp4";
  std::string execution_device = "auto";
  int gpu_device_id = 0;

  float det_conf = 0.65F;
  float det_iou = 0.50F;
  int input_width = 640;
  int input_height = 640;

  bool show_window = true;
  std::string window_name = "YOLO ONNX + Tracker (C++)";
  bool draw_detections = true;
  bool draw_tracks = true;
  std::string baseline_output_path;
  TrackerRuntimeConfig tracker;
};

AppConfig LoadAppConfig(const std::string& path);

}  // namespace tracking
