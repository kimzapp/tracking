#include "tracking/detector/yolo_onnx_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace tracking {

namespace {

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& input) {
  if (input.empty()) {
    return {};
  }

  const int required = MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
  if (required <= 0) {
    throw std::runtime_error("Failed to convert model path to wide string");
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

float Sigmoid(float x) {
  if (x >= 0.0F) {
    const float z = std::exp(-x);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(x);
  return z / (1.0F + z);
}

}  // namespace

YoloOnnxDetector::YoloOnnxDetector(const std::string& model_path, int input_width,
                                   int input_height, float conf_thresh, float iou_thresh)
    : env_(ORT_LOGGING_LEVEL_WARNING, "tracking_yolo"),
      input_width_(input_width),
      input_height_(input_height),
      conf_thresh_(conf_thresh),
      iou_thresh_(iou_thresh) {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetIntraOpNumThreads(1);

#ifdef _WIN32
  const std::wstring wide_model_path = Utf8ToWide(model_path);
  session_ = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options_);
#else
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  {
    Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = name.get();
  }
  {
    Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = name.get();
  }
}

std::vector<float> YoloOnnxDetector::Preprocess(const cv::Mat& frame, LetterboxMeta* meta) const {
  if (frame.empty()) {
    throw std::runtime_error("Received empty frame");
  }

  const float scale = std::min(static_cast<float>(input_width_) / static_cast<float>(frame.cols),
                               static_cast<float>(input_height_) / static_cast<float>(frame.rows));
  const int resized_w = static_cast<int>(std::round(static_cast<float>(frame.cols) * scale));
  const int resized_h = static_cast<int>(std::round(static_cast<float>(frame.rows) * scale));

  const int pad_x = (input_width_ - resized_w) / 2;
  const int pad_y = (input_height_ - resized_h) / 2;

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat letterboxed(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

  cv::Mat float_img;
  rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);

  std::vector<float> chw(static_cast<size_t>(3 * input_height_ * input_width_));
  const size_t plane_size = static_cast<size_t>(input_height_ * input_width_);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(chw.data() + static_cast<size_t>(c) * plane_size, channels[c].data,
                plane_size * sizeof(float));
  }

  meta->scale = scale;
  meta->pad_x = static_cast<float>(pad_x);
  meta->pad_y = static_cast<float>(pad_y);

  return chw;
}

Detections YoloOnnxDetector::Postprocess(const float* output_data,
                                         const std::vector<std::int64_t>& shape,
                                         const LetterboxMeta& meta, int frame_w,
                                         int frame_h) const {
  if (shape.size() != 3) {
    std::ostringstream oss;
    oss << "Unexpected output rank, expected 3 dimensions. Got shape [";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << shape[i];
    }
    oss << "]";
    throw std::runtime_error(oss.str());
  }

  const int64_t dim1 = shape[1];
  const int64_t dim2 = shape[2];

  bool channels_first = false;
  int64_t num_preds = 0;
  int64_t num_attrs = 0;

  if (dim1 <= 256 && dim2 > dim1) {
    channels_first = true;
    num_attrs = dim1;
    num_preds = dim2;
  } else {
    channels_first = false;
    num_preds = dim1;
    num_attrs = dim2;
  }

  if (num_attrs < 5) {
    std::ostringstream oss;
    oss << "Unexpected YOLO output attributes count: " << num_attrs << " (shape [";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << shape[i];
    }
    oss << "])";
    throw std::runtime_error(oss.str());
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  boxes.reserve(static_cast<size_t>(num_preds));
  scores.reserve(static_cast<size_t>(num_preds));
  class_ids.reserve(static_cast<size_t>(num_preds));

  // YOLO exports may be either:
  // - [x, y, w, h, obj, cls...] (objectness present)
  // - [x, y, w, h, cls...]      (no objectness; common for single-class models)
  const bool has_objectness = num_attrs >= 6;
  const int class_start = has_objectness ? 5 : 4;
  const int class_count = static_cast<int>(num_attrs - class_start);
  if (class_count <= 0) {
    throw std::runtime_error("YOLO output has no class scores");
  }

  auto at = [&](int64_t pred_idx, int64_t attr_idx) -> float {
    if (channels_first) {
      return output_data[attr_idx * num_preds + pred_idx];
    }
    return output_data[pred_idx * num_attrs + attr_idx];
  };

  float max_coord_magnitude = 0.0F;
  bool score_out_of_range = false;
  const int64_t probe_count = std::min<int64_t>(num_preds, 64);
  for (int64_t i = 0; i < probe_count; ++i) {
    max_coord_magnitude = std::max(max_coord_magnitude, std::fabs(at(i, 0)));
    max_coord_magnitude = std::max(max_coord_magnitude, std::fabs(at(i, 1)));
    max_coord_magnitude = std::max(max_coord_magnitude, std::fabs(at(i, 2)));
    max_coord_magnitude = std::max(max_coord_magnitude, std::fabs(at(i, 3)));
    if (has_objectness) {
      const float obj = at(i, 4);
      if (obj < 0.0F || obj > 1.0F) {
        score_out_of_range = true;
      }
    }
    for (int c = 0; c < class_count; ++c) {
      const float cls_score = at(i, class_start + c);
      if (cls_score < 0.0F || cls_score > 1.0F) {
        score_out_of_range = true;
        break;
      }
    }
  }
  const bool normalized_coords = max_coord_magnitude <= 2.5F;

  for (int64_t i = 0; i < num_preds; ++i) {
    float cx = at(i, 0);
    float cy = at(i, 1);
    float w = at(i, 2);
    float h = at(i, 3);
    if (normalized_coords) {
      cx *= static_cast<float>(input_width_);
      cy *= static_cast<float>(input_height_);
      w *= static_cast<float>(input_width_);
      h *= static_cast<float>(input_height_);
    }

    float objectness = 1.0F;
    if (has_objectness) {
      objectness = at(i, 4);
    }
    if (score_out_of_range) {
      objectness = Sigmoid(objectness);
    }

    int best_class = -1;
    float best_class_score = 0.0F;
    for (int c = 0; c < class_count; ++c) {
      float cls_score = at(i, class_start + c);
      if (score_out_of_range) {
        cls_score = Sigmoid(cls_score);
      }
      if (cls_score > best_class_score) {
        best_class_score = cls_score;
        best_class = c;
      }
    }

    const float conf = objectness * best_class_score;
    if (conf < conf_thresh_) {
      continue;
    }

    const float x1_l = cx - (w * 0.5F);
    const float y1_l = cy - (h * 0.5F);
    const float x2_l = cx + (w * 0.5F);
    const float y2_l = cy + (h * 0.5F);

    const float x1 = (x1_l - meta.pad_x) / meta.scale;
    const float y1 = (y1_l - meta.pad_y) / meta.scale;
    const float x2 = (x2_l - meta.pad_x) / meta.scale;
    const float y2 = (y2_l - meta.pad_y) / meta.scale;

    const float x1_c = Clamp(x1, 0.0F, static_cast<float>(frame_w - 1));
    const float y1_c = Clamp(y1, 0.0F, static_cast<float>(frame_h - 1));
    const float x2_c = Clamp(x2, 0.0F, static_cast<float>(frame_w - 1));
    const float y2_c = Clamp(y2, 0.0F, static_cast<float>(frame_h - 1));

    const int bw = static_cast<int>(std::max(0.0F, x2_c - x1_c));
    const int bh = static_cast<int>(std::max(0.0F, y2_c - y1_c));
    if (bw <= 0 || bh <= 0) {
      continue;
    }

    boxes.emplace_back(static_cast<int>(x1_c), static_cast<int>(y1_c), bw, bh);
    scores.push_back(conf);
    class_ids.push_back(best_class);
  }

  std::vector<int> keep_indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, iou_thresh_, keep_indices);

  Detections detections;
  detections.reserve(keep_indices.size());
  for (const int idx : keep_indices) {
    const cv::Rect& rect = boxes.at(static_cast<size_t>(idx));

    Detection det;
    det.x1 = static_cast<float>(rect.x);
    det.y1 = static_cast<float>(rect.y);
    det.x2 = static_cast<float>(rect.x + rect.width);
    det.y2 = static_cast<float>(rect.y + rect.height);
    det.conf = scores.at(static_cast<size_t>(idx));
    det.cls = static_cast<float>(class_ids.at(static_cast<size_t>(idx)));
    detections.push_back(det);
  }

  return detections;
}

Detections YoloOnnxDetector::Predict(const cv::Mat& frame) {
  LetterboxMeta meta;
  std::vector<float> input_tensor = Preprocess(frame, &meta);

  const std::array<std::int64_t, 4> input_shape = {1, 3, input_height_, input_width_};
  const std::array<const char*, 1> input_names = {input_name_.c_str()};
  const std::array<const char*, 1> output_names = {output_name_.c_str()};

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_value = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

  std::vector<Ort::Value> outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_names.data(), &input_value, 1, output_names.data(), 1);

  if (outputs.empty() || !outputs[0].IsTensor()) {
    throw std::runtime_error("ONNX Runtime returned empty or invalid output tensor");
  }

  const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<std::int64_t> shape = output_info.GetShape();
  const float* output_data = outputs[0].GetTensorData<float>();

  return Postprocess(output_data, shape, meta, frame.cols, frame.rows);
}

}  // namespace tracking
