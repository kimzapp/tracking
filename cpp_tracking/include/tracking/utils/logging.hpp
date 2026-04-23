#pragma once

#include <string>

namespace tracking {

void LogInfo(const std::string& message);
void LogWarn(const std::string& message);
void LogError(const std::string& message);

}  // namespace tracking
