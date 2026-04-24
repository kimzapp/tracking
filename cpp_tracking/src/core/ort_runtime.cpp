#include "tracking/core/ort_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <sstream>
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

bool ContainsAnyProviderAlias(const std::unordered_set<std::string>& available,
                              const std::vector<std::string>& aliases) {
  for (const std::string& alias : aliases) {
    if (ContainsProvider(available, alias)) {
      return true;
    }
  }
  return false;
}

std::string JoinProviders(const std::vector<std::string>& providers) {
  if (providers.empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  for (size_t i = 0; i < providers.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << providers[i];
  }
  return oss.str();
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

bool AppendCudaProvider(Ort::SessionOptions* options, int gpu_device_id, std::string* error_message) {
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = std::max(0, gpu_device_id);
  cuda_options.arena_extend_strategy = 0;
  cuda_options.gpu_mem_limit = SIZE_MAX;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.do_copy_in_default_stream = 1;
  cuda_options.has_user_compute_stream = 0;
  cuda_options.user_compute_stream = nullptr;
  cuda_options.default_memory_arena_cfg = nullptr;

  OrtStatus* status =
      Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(*options, &cuda_options);
  if (status == nullptr) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = Ort::GetApi().GetErrorMessage(status);
  }
  Ort::GetApi().ReleaseStatus(status);
  return false;
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

  struct GpuProviderCandidate {
    std::string provider;
    std::vector<std::string> availability_aliases;
    std::vector<std::string> append_aliases;
    bool try_cuda_specialized_api = false;
  };

  const std::array<GpuProviderCandidate, 5> gpu_provider_priority = {{
      {"CUDAExecutionProvider",
       {"CUDAExecutionProvider", "CUDA"},
       {"CUDAExecutionProvider", "CUDA"},
       true},
      {"DmlExecutionProvider", {"DmlExecutionProvider", "DMLExecutionProvider", "DML"},
       {"DmlExecutionProvider", "DMLExecutionProvider", "DML"}, false},
      {"TensorrtExecutionProvider",
       {"TensorrtExecutionProvider", "TensorRTExecutionProvider", "TENSORRT"},
       {"TensorrtExecutionProvider", "TensorRTExecutionProvider", "TENSORRT"}, false},
      {"ROCMExecutionProvider", {"ROCMExecutionProvider", "ROCM"},
       {"ROCMExecutionProvider", "ROCM"}, false},
      {"CoreMLExecutionProvider", {"CoreMLExecutionProvider", "COREML"},
       {"CoreMLExecutionProvider", "COREML"}, false},
  }};

  std::string last_error;
  for (const GpuProviderCandidate& candidate : gpu_provider_priority) {
    if (!ContainsAnyProviderAlias(available_set, candidate.availability_aliases)) {
      continue;
    }

    if (candidate.try_cuda_specialized_api &&
        AppendCudaProvider(options, gpu_device_id, &last_error)) {
      selection.effective_device = "gpu";
      selection.provider = candidate.provider;
      return selection;
    }

    for (const std::string& provider_alias : candidate.append_aliases) {
      if (AppendProvider(options, provider_alias, gpu_options, &last_error) ||
          AppendProvider(options, provider_alias, {}, &last_error)) {
        selection.effective_device = "gpu";
        selection.provider = candidate.provider;
        return selection;
      }
    }
  }

  if (selection.requested_device == "gpu") {
    std::string message =
        "GPU mode requested, but no supported GPU provider could be enabled. "
        "Available ONNX Runtime providers: " +
        JoinProviders(available);
    if (!last_error.empty()) {
      message += ". Last provider error: " + last_error;
    }
    message +=
        ". Ensure the loaded ONNX Runtime build includes CUDA EP and that CUDA/cuDNN runtime "
        "libraries are visible to the process (LD_LIBRARY_PATH/PATH).";
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
