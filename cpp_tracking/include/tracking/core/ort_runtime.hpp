#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace tracking {

struct OrtRuntimeSelection {
  std::string requested_device = "auto";
  std::string effective_device = "cpu";
  std::string provider;
};

std::vector<std::string> GetAvailableOrtProviders();

OrtRuntimeSelection ConfigureOrtExecutionProvider(Ort::SessionOptions* options,
                                                  const std::string& requested_device,
                                                  int gpu_device_id);

}  // namespace tracking
