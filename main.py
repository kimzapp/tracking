from __future__ import annotations
import numpy as np
import colorsys
from dataclasses import dataclass
from pathlib import Path

import cv2


from ultralytics import YOLO
from tracking_engine.tracker_lib import Tracker


@dataclass(slots=True)
class TrackingConfig:
	"""Cấu hình runtime cho bài toán detect + track trên video."""
	detector_path: str = "./best_openvino_model"
	detector_weights: str = "openvino"
	tracker_config_path: str = "botsort.yaml"
	reid_model: str = "models/osnet_x0_5_msmt17.pt"
	video_path: str = "video/longchau.mp4"
	output_video_path: str = video_path.replace(".mp4", "_output.mp4")
	det_conf: float = 0.65
	det_iou: float = 0.5
	window_name: str = "YOLO + BoTSORT Tracking"

	def ensure_output_parent(self) -> None:
		"""Đảm bảo thư mục chứa video đầu ra luôn tồn tại."""
		Path(self.output_video_path).parent.mkdir(parents=True, exist_ok=True)


def color_from_id(track_id: int) -> tuple[int, int, int]:
	# Sinh màu ổn định theo ID để quan sát trực quan theo thời gian.
	h = ((track_id * 37) % 360) / 360.0
	r, g, b = colorsys.hsv_to_rgb(h, 0.78, 1.0)
	return int(b * 255), int(g * 255), int(r * 255)


def draw_annotation(frame, x1: int, y1: int, x2: int, y2: int, track_id: int) -> None:
	"""Vẽ bbox và nhãn ID lên frame hiện tại."""
	h, w = frame.shape[:2]
	x1 = max(0, min(x1, w - 1))
	y1 = max(0, min(y1, h - 1))
	x2 = max(0, min(x2, w - 1))
	y2 = max(0, min(y2, h - 1))
	if x2 <= x1 or y2 <= y1:
		return

	color = color_from_id(track_id)
	thickness = max(2, int(min(h, w) * 0.0022))

	overlay = frame.copy()
	cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
	cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
	cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

	label = f"ID {track_id}"
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = max(0.55, min(0.8, min(h, w) / 1200))
	(txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, 2)
	pad = 6

	label_x1 = x1
	label_y2 = y1 - 6
	if label_y2 - (txt_h + 2 * pad) < 0:
		label_y2 = y1 + txt_h + 2 * pad + 6

	label_y1 = label_y2 - (txt_h + 2 * pad)
	label_x2 = min(w - 1, label_x1 + txt_w + 2 * pad)

	cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
	cv2.putText(
		frame,
		label,
		(label_x1 + pad, label_y2 - pad - baseline),
		font,
		font_scale,
		(20, 20, 20),
		2,
	)


def annotate_tracks(frame, tracks, seen_ids: set[int]) -> None:
	"""Duyệt kết quả track và vẽ annotation cho từng đối tượng."""
	if tracks is None or len(tracks) == 0:
		return

	for row in tracks:
		x1, y1, x2, y2 = row[:4].astype(int)
		track_id = int(row[4])
		seen_ids.add(track_id)
		draw_annotation(frame, x1, y1, x2, y2, track_id)


def create_video_writer(cap: cv2.VideoCapture, cfg: TrackingConfig) -> cv2.VideoWriter:
	"""Tạo writer cho video đầu ra dựa trên thông số từ video input."""
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	cfg.ensure_output_parent()
	writer = cv2.VideoWriter(
		cfg.output_video_path,
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)
	if not writer.isOpened():
		raise RuntimeError(f"Không mở được video output: {cfg.output_video_path}")
	return writer


def run_tracking_video(cfg: TrackingConfig) -> set[int]:
	"""Chạy tracking theo từng frame và trả về tập ID đã xuất hiện."""
	# Khởi tạo detector YOLO và tracker
	detector = YOLO(cfg.detector_path, task="detect")
	tracker = Tracker(
		tracker_config_path=cfg.tracker_config_path,
		reid_model=cfg.reid_model,
	)
	cap = cv2.VideoCapture(cfg.video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Không mở được video: {cfg.video_path}")
	writer = create_video_writer(cap, cfg)

	seen_ids: set[int] = set()
	try:
		while True:
			success, frame = cap.read()
			if not success:
				print("Video frame rỗng hoặc đã xử lý xong.")
				break

			# Detect
			pred = detector.predict(
				source=frame,
				conf=cfg.det_conf,
				iou=cfg.det_iou,
				verbose=False,
			)[0]
			boxes = pred.boxes
			if boxes is None or len(boxes) == 0:
				dets = np.empty((0, 6), dtype=np.float32)
			else:
				xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
				conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
				cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
				dets = np.hstack((xyxy, conf, cls))

			# Track
			tracks = tracker.update(dets, frame)
			annotate_tracks(frame, tracks, seen_ids)

			writer.write(frame)
			cv2.imshow(cfg.window_name, frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		cap.release()
		writer.release()
		cv2.destroyAllWindows()

	return seen_ids


def main() -> None:
	"""Điểm vào của chương trình demo tracking."""
	cfg = TrackingConfig()
	seen_ids = run_tracking_video(cfg)
	print(f"Done. Tổng số ID đã theo dõi: {len(seen_ids)}")
	print(f"Saved output video to: {cfg.output_video_path}")


if __name__ == "__main__":
	main()