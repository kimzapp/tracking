#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace tracking {

struct OrtRuntimeSelection {
  std::string requested_device = "auto";
  std::string effective_device = "cpu";
  std::string provider;
  std::string fallback_reason;
  std::vector<std::string> available_providers;
};

std::vector<std::string> GetAvailableOrtProviders();

OrtRuntimeSelection ConfigureOrtExecutionProvider(Ort::SessionOptions* options,
                                                  const std::string& requested_device,
                                                  int gpu_device_id);

}  // namespace tracking
