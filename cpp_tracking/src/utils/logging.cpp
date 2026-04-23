#include "tracking/utils/logging.hpp"

#include <iostream>

namespace tracking {

void LogInfo(const std::string& message) { std::cout << "[INFO] " << message << '\n'; }

void LogWarn(const std::string& message) { std::cout << "[WARN] " << message << '\n'; }

void LogError(const std::string& message) { std::cerr << "[ERROR] " << message << '\n'; }

}  // namespace tracking
