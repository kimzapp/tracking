#include "tracking/core/ort_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tracking {
namespace {

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

bool ContainsProvider(const std::unordered_set<std::string>& available, const std::string& name) {
  return available.find(name) != available.end();
}

bool AppendProvider(Ort::SessionOptions* options, const std::string& provider,
                    const std::vector<std::pair<std::string, std::string>>& provider_options,
                    std::string* error_message) {
#if ORT_API_VERSION >= 16
  std::vector<const char*> keys;
  std::vector<const char*> values;
  keys.reserve(provider_options.size());
  values.reserve(provider_options.size());
  for (const auto& entry : provider_options) {
    keys.push_back(entry.first.c_str());
    values.push_back(entry.second.c_str());
  }

  OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider(
      *options, provider.c_str(), keys.empty() ? nullptr : keys.data(),
      values.empty() ? nullptr : values.data(), keys.size());
  if (status == nullptr) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = Ort::GetApi().GetErrorMessage(status);
  }
  Ort::GetApi().ReleaseStatus(status);
  return false;
#else
  (void)options;
  (void)provider;
  (void)provider_options;
  if (error_message != nullptr) {
    *error_message = "SessionOptionsAppendExecutionProvider is unavailable in this ONNX Runtime version";
  }
  return false;
#endif
}

}  // namespace

std::vector<std::string> GetAvailableOrtProviders() {
  char** providers = nullptr;
  int provider_count = 0;
  Ort::ThrowOnError(Ort::GetApi().GetAvailableProviders(&providers, &provider_count));

  std::vector<std::string> result;
  result.reserve(static_cast<size_t>(provider_count));
  for (int i = 0; i < provider_count; ++i) {
    if (providers[i] != nullptr) {
      result.emplace_back(providers[i]);
    }
  }

  Ort::ThrowOnError(Ort::GetApi().ReleaseAvailableProviders(providers, provider_count));
  return result;
}

OrtRuntimeSelection ConfigureOrtExecutionProvider(Ort::SessionOptions* options,
                                                  const std::string& requested_device,
                                                  int gpu_device_id) {
  if (options == nullptr) {
    throw std::runtime_error("ConfigureOrtExecutionProvider: session options is null");
  }

  OrtRuntimeSelection selection;
  selection.requested_device = ToLower(requested_device);
  if (selection.requested_device.empty()) {
    selection.requested_device = "auto";
  }

  if (selection.requested_device == "cpu") {
    return selection;
  }

  if (selection.requested_device != "auto" && selection.requested_device != "gpu") {
    throw std::runtime_error("execution_device must be one of: auto, cpu, gpu");
  }

  const auto available = GetAvailableOrtProviders();
  selection.available_providers = available;
  const std::unordered_set<std::string> available_set(available.begin(), available.end());

  std::vector<std::pair<std::string, std::string>> gpu_options = {
      {"device_id", std::to_string(std::max(0, gpu_device_id))}};

  const std::array<std::string, 5> gpu_provider_priority = {
      "CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider",
      "ROCMExecutionProvider", "CoreMLExecutionProvider"};

  std::string last_error;
  for (const std::string& provider : gpu_provider_priority) {
    if (!ContainsProvider(available_set, provider)) {
      continue;
    }
    if (AppendProvider(options, provider, gpu_options, &last_error) ||
        AppendProvider(options, provider, {}, &last_error)) {
      selection.effective_device = "gpu";
      selection.provider = provider;
      return selection;
    }
  }

  if (selection.requested_device == "gpu") {
    std::string message = "GPU mode requested, but no supported GPU provider could be enabled";
    if (!last_error.empty()) {
      message += ". Last provider error: " + last_error;
    }
    throw std::runtime_error(message);
  }

  if (available.empty()) {
    selection.fallback_reason = "No ONNX Runtime providers reported";
  } else {
    selection.fallback_reason = "No GPU provider available or provider init failed";
  }

  return selection;
}

}  // namespace tracking
