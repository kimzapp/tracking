#pragma once

#include <cstdint>
#include <vector>

namespace tracking {

struct Detection {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float conf = 0.0F;
  float cls = 0.0F;
};

struct Track {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  std::int64_t id = -1;
  float conf = 0.0F;
  float cls = 0.0F;
  int det_index = -1;
};

using Detections = std::vector<Detection>;
using Tracks = std::vector<Track>;

}  // namespace tracking
