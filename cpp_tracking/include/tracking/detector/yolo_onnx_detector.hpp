#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "tracking/core/types.hpp"
#include "tracking/core/ort_runtime.hpp"

namespace tracking {

class YoloOnnxDetector {
 public:
  YoloOnnxDetector(const std::string& model_path, int input_width, int input_height,
                   float conf_thresh, float iou_thresh, const std::string& execution_device,
                   int gpu_device_id);

  Detections Predict(const cv::Mat& frame);

 private:
  struct LetterboxMeta {
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
  };

  std::vector<float> Preprocess(const cv::Mat& frame, LetterboxMeta* meta) const;
  Detections Postprocess(const float* output_data, const std::vector<std::int64_t>& shape,
                         const LetterboxMeta& meta, int frame_w, int frame_h) const;

  std::string input_name_;
  std::string output_name_;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;

  int input_width_;
  int input_height_;
  float conf_thresh_;
  float iou_thresh_;
  OrtRuntimeSelection runtime_selection_;
};

}  // namespace tracking
