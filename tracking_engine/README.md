# Tracking Engine

Module này kết hợp:

- Detection: Ultralytics YOLO (`yolo11_best.pt`)
- Tracking: BoTSORT + ReID OSNet_AIN từ BoxMOT local

Mục tiêu là giữ nguyên thuật toán tracking gốc của BoxMOT, nhưng tách pipeline theo frame để phù hợp với project.
