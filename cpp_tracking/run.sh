#!/usr/bin/env bash
set -euo pipefail

# ================================
# Resolve script directory
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================================
# FUNCTIONS
# ================================
prepend_path_if_exists() {
  local dir="${1:-}"
  [[ -z "$dir" || ! -d "$dir" ]] && return 0

  local current="${LD_LIBRARY_PATH:-}"
  case ":$current:" in
    *":$dir:"*) return 0 ;;
  esac

  if [[ -n "$current" ]]; then
    export LD_LIBRARY_PATH="$dir:$current"
  else
    export LD_LIBRARY_PATH="$dir"
  fi
}

find_app_exe() {
  local p
  for p in \
    "${SCRIPT_DIR}/build/bin/Release/tracking_app" \
    "${SCRIPT_DIR}/build/bin/tracking_app" \
    "${SCRIPT_DIR}/build/Release/tracking_app" \
    "${SCRIPT_DIR}/build/tracking_app"; do
    if [[ -x "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

# ================================
# Resolve APP_EXE safely
# ================================
APP_EXE="${TRACKING_APP_EXE:-}"
if [[ -z "$APP_EXE" ]]; then
  APP_EXE="$(find_app_exe || true)"
fi

if [[ -z "$APP_EXE" || ! -x "$APP_EXE" ]]; then
  echo "[ERROR] tracking_app executable not found."
  echo "[HINT] Build first:"
  echo "       cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
  echo "       cmake --build build --config Release"
  echo "[HINT] Or set TRACKING_APP_EXE to absolute executable path."
  exit 1
fi

# ================================
# Resolve config file
# ================================
CONFIG_FILE="${1:-${SCRIPT_DIR}/config/app.yaml}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE"
  echo "[HINT] Usage: ./run.sh [path-to-app.yaml]"
  exit 1
fi

# ================================
# Get APP_DIR safely
# ================================
APP_DIR="$(cd "$(dirname "$APP_EXE")" && pwd)"

# ================================
# Update LD_LIBRARY_PATH safely
# ================================
prepend_path_if_exists "$APP_DIR"

if [[ -n "${ONNXRUNTIME_LIB_DIR:-}" ]]; then
  prepend_path_if_exists "$ONNXRUNTIME_LIB_DIR"
elif [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
  prepend_path_if_exists "${ONNXRUNTIME_ROOT}/lib"
  prepend_path_if_exists "${ONNXRUNTIME_ROOT}/lib64"
fi

if [[ -n "${OPENCV_LIB_DIR:-}" ]]; then
  prepend_path_if_exists "$OPENCV_LIB_DIR"
else
  if [[ -n "${OpenCV_DIR:-}" ]]; then
    prepend_path_if_exists "${OpenCV_DIR}/../lib"
  fi
  if [[ -n "${OPENCV_ROOT:-}" ]]; then
    prepend_path_if_exists "${OPENCV_ROOT}/lib"
    prepend_path_if_exists "${OPENCV_ROOT}/lib64"
  fi
  prepend_path_if_exists "/usr/local/lib"
  prepend_path_if_exists "/usr/lib/x86_64-linux-gnu"
fi

echo "[INFO] Executable: \"$APP_EXE\""
echo "[INFO] Config: \"$CONFIG_FILE\""
echo "[INFO] ONNXRUNTIME_ROOT: \"${ONNXRUNTIME_ROOT:-}\""
echo "[INFO] ONNXRUNTIME_LIB_DIR: \"${ONNXRUNTIME_LIB_DIR:-}\""
echo "[INFO] LD_LIBRARY_PATH: \"${LD_LIBRARY_PATH:-}\""
if command -v ldd >/dev/null 2>&1; then
  echo "[INFO] ldd (onnxruntime/cuda/cudnn related):"
  ldd "$APP_EXE" 2>/dev/null | awk '/onnxruntime|cuda|cudnn|cublas|cufft|curand/ { print "  " $0 }'
fi

# ================================
# Run app
# ================================
pushd "$SCRIPT_DIR" >/dev/null
set +e
"$APP_EXE" "$CONFIG_FILE"
APP_EXIT=$?
set -e
popd >/dev/null

if [[ "$APP_EXIT" -ne 0 ]]; then
  echo "[ERROR] tracking_app exited with code $APP_EXIT"
fi

exit "$APP_EXIT"
