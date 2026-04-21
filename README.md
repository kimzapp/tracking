# Tracking Project

Dự án được tổ chức theo hướng gọn, dễ chạy và dễ bảo trì:

- `tracking_engine/`: thư viện tracking (pipeline, tracker, ReID, utils).
- `main.py`: script chạy chính, gồm config runtime, vòng lặp video và vẽ annotation.

## Cấu trúc chính

- `main.py`: điểm vào chính để chạy ứng dụng theo cấu hình mặc định.
- `botsort.yaml`: tham số tracker để tinh chỉnh runtime.
- `track2.py`: script baseline dùng trực tiếp Ultralytics `model.track(...)`.

## Cách chạy

```bash
python main.py
```

## Nguyên tắc tổ chức module

- Logic tracking lõi nằm trong `tracking_engine`, không chỉnh lẫn với script chạy.
- `main.py` là lớp orchestration cho bài toán demo/runtime của project.
- Không tách thêm package ứng dụng để tránh phức tạp hóa cấu trúc khi chưa cần.
