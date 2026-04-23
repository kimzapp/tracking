# Tracking Project

Repo hiện có 2 luồng chạy chính:

- Python runtime: `main.py`
- C++ runtime: `cpp_tracking/` (ONNX Runtime + BoT-SORT)

## Cấu trúc chính

- `tracking_engine/`: thư viện tracking Python (pipeline, tracker, ReID, utils).
- `main.py`: demo/runtime Python.
- `cpp_tracking/`: bản C++ gồm app loop, detector ONNX, tracker BoT-SORT.
- `cpp_tracking/config/app.yaml`: config mặc định khi chạy C++.

## Chạy nhanh bản Python

```bash
python main.py
```

## Setup môi trường và chạy bản C++

### 1) Yêu cầu

- CMake >= 3.20
- Compiler hỗ trợ C++17 (Windows: MSVC 2019+)
- OpenCV (core, imgproc, videoio, highgui, dnn)
- yaml-cpp
- ONNX Runtime C/C++

### 2) Chuẩn bị model/data (mặc định theo config)

- Detector ONNX: `models/best.onnx`
- ReID ONNX: `models/osnet_x0_5_msmt17.onnx`
- Video input: `video/longchau.mp4`

Nếu dùng đúng đường dẫn mặc định trong `cpp_tracking/config/app.yaml` thì không cần sửa thêm.

### 3) Build (Windows PowerShell)

```powershell
cd cpp_tracking
cmake -S . -B build -DOpenCV_DIR="D:/opencv/build/x64/vc16/lib" -DONNXRUNTIME_INCLUDE_DIR="D:/onnxruntime-win-x64-1.25.0/include" -DONNXRUNTIME_LIBRARY="D:/onnxruntime-win-x64-1.25.0/lib/onnxruntime.lib"
cmake --build build --config Release
```

Nếu máy bạn khác đường dẫn, thay `OpenCV_DIR` và `ONNXRUNTIME_*` tương ứng.

### 4) Chạy (Windows)

```powershell
cd cpp_tracking
.\run.bat
```

Chạy với config tùy chỉnh:

```powershell
.\run.bat .\config\app.yaml
```

### 5) Chạy (Linux)

```bash
cd cpp_tracking
chmod +x run.sh
./run.sh
```

Với config tùy chỉnh:

```bash
./run.sh ./config/app.yaml
```

### 6) Kết quả đầu ra

- Video output mặc định: `video/longchau_cpp_output.mp4`
- Baseline JSON (nếu cấu hình `baseline_output_path`): `video/longchau_cpp_baseline.json`

## Ghi chú cho C++ runtime

- Mặc định tracker là `botsort` (đặt trong `cpp_tracking/config/app.yaml`).
- Có thể đổi sang `passthrough` để debug, nhưng không dùng cho luồng tracking chuẩn.
- Nếu muốn nhìn rõ ID hơn, giữ `draw_tracks: true` và `show_window: true` trong config.
- Có thể bật tự động dùng GPU bằng `execution_device: auto` (mặc định). Đặt `execution_device: gpu` để ép GPU (sẽ báo lỗi nếu không có provider GPU), hoặc `cpu` để chỉ chạy CPU. `gpu_device_id` chọn GPU index.
