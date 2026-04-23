# C++ Tracking Runtime (WIP)

This folder is the production-oriented C++ migration scaffold.

Current milestone:
- CMake build layout
- ONNX Runtime YOLO detector implementation
- Runtime app loop for video input/output
- BoTSORT C++ tracker parity port (2-stage association, unconfirmed handling, ReID recovery, CMC ECC)

## Prerequisites

- CMake 3.20+
- C++17 compiler (MSVC 2019+ recommended)
- OpenCV (core,imgproc,videoio,highgui,dnn)
- yaml-cpp
- ONNX Runtime C/C++ package

Set ONNX Runtime path:

```powershell
$env:ONNXRUNTIME_ROOT="C:/onnxruntime-win-x64-1.22.0"
```

## Configure and build

```powershell
cd cpp_tracking
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build build --config Release
```

If OpenCV is not auto-detected, set either `OpenCV_DIR` or `CMAKE_PREFIX_PATH`.

Recommended Windows OpenCV package path (on this repo setup):

```powershell
-DOpenCV_DIR="D:/opencv/build/x64/vc16/lib"
```

## Run

```powershell
cd cpp_tracking
./run.bat
```

Run with custom config:

```powershell
./run.bat .\config\app.yaml
```

Linux:

```bash
cd cpp_tracking
chmod +x run.sh
./run.sh
```

Run with custom config:

```bash
./run.sh ./config/app.yaml
```

## Portable runtime pipeline

`run.bat` and `run.sh` are designed to run on different machines with minimal edits:

- Auto-detect `tracking_app` executable in common CMake output locations.
- Accept optional config path argument (`app.yaml`).
- Support environment overrides:
	- `TRACKING_APP_EXE`: absolute path to app binary.
	- `ONNXRUNTIME_ROOT` or `ONNXRUNTIME_LIB_DIR`: ONNX Runtime libs.
	- `OpenCV_DIR`, `OPENCV_ROOT`, `OPENCV_BIN` (Windows), `OPENCV_LIB_DIR` (Linux): OpenCV runtime libs.

Example overrides (Windows PowerShell):

```powershell
$env:TRACKING_APP_EXE="E:/tracking/cpp_tracking/build/bin/Release/tracking_app.exe"
$env:ONNXRUNTIME_ROOT="D:/onnxruntime-win-x64-1.25.0"
$env:OPENCV_BIN="D:/opencv/build/x64/vc16/bin"
./run.bat
```

Example overrides (Linux bash):

```bash
export TRACKING_APP_EXE=/opt/tracking/build/bin/tracking_app
export ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.25.0
export OPENCV_LIB_DIR=/usr/local/lib
./run.sh
```

## Notes

- Runtime mặc định dùng `tracker_type: botsort`. `passthrough` chỉ giữ làm fallback debug.
- Cấu hình tracker/ReID/CMC được load trực tiếp từ YAML (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh`, `second_association_thresh`, `unconfirmed_association_thresh`, `reid_recovery_*`, `cmc_method`, `with_reid`, `fuse_first_associate`...).
- Khi set `baseline_output_path`, app C++ sẽ export JSON baseline để đối chiếu với Python.

## Migration helpers

Export baseline from Python runtime:

```powershell
cd ..
D:/miniconda3/envs/tracking/python.exe tools/export_python_baseline.py --max-frames 300 --output video/python_baseline.json
```

Export detector + ReID ONNX:

```powershell
D:/miniconda3/envs/tracking/python.exe tools/export_onnx_models.py --detector-pt best.pt --reid-pt models/osnet_x0_5_msmt17.pt --out-dir models
```

Compare Python and C++ baselines:

```powershell
D:/miniconda3/envs/tracking/python.exe tools/compare_tracking_baselines.py --python video/python_baseline.json --cpp video/cpp_baseline.json --max-frames 300
```
