from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from tracking_engine.trackers.botsort.botsort import BotSort
from tracking_engine.config import (
    DEFAULT_REID_MODEL,
    TrackerRuntimeConfig,
    load_tracker_runtime_config,
)


@dataclass(slots=True)
class FrameTrackingResult:
    detections: np.ndarray
    tracks: np.ndarray


class YoloBotSortPipeline:
    def __init__(
        self,
        detector_weights: str | Path,
        tracker_config_path: str | Path,
        reid_model: str | Path | None = None,
        device: str | None = None,
        half: bool | None = None,
        with_reid: bool | None = None,
        fuse_first_associate: bool = False,
        det_conf: float = 0.25,
        det_iou: float = 0.55,
    ) -> None:
        # Detection vẫn dùng Ultralytics YOLO như yêu cầu.
        detector_source = self._resolve_detector_weights(detector_weights)
        self.detector = YOLO(detector_source, task="detect")
        self.det_conf = float(det_conf)
        self.det_iou = float(det_iou)

        self.tracker_cfg: TrackerRuntimeConfig = load_tracker_runtime_config(
            tracker_config_path
        )

        # ReID model và runtime được ưu tiên truyền trực tiếp qua pipeline.
        self.reid_model = Path(reid_model) if reid_model is not None else Path(DEFAULT_REID_MODEL)
        self.with_reid = True if with_reid is None else bool(with_reid)
        self.fuse_first_associate = bool(fuse_first_associate)
        if device is not None:
            self.tracker_cfg.device = device
        if half is not None:
            self.tracker_cfg.half = bool(half)

        self.tracker = self._build_tracker(self.tracker_cfg)

    def _resolve_device(self, device_text: str) -> torch.device:
        if device_text == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device_text)

    def _build_tracker(self, cfg: TrackerRuntimeConfig) -> BotSort:
        device = self._resolve_device(cfg.device)

        tracker = BotSort(
            reid_weights=self.reid_model,
            device=device,
            half=cfg.half,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            reid_recovery_enabled=cfg.reid_recovery_enabled,
            reid_recovery_proximity_thresh=cfg.reid_recovery_proximity_thresh,
            reid_recovery_appearance_thresh=cfg.reid_recovery_appearance_thresh,
            reid_recovery_thresh=cfg.reid_recovery_thresh,
            second_association_thresh=cfg.second_association_thresh,
            unconfirmed_association_thresh=cfg.unconfirmed_association_thresh,
            cmc_method=cfg.cmc_method,
            fuse_first_associate=self.fuse_first_associate,
            with_reid=self.with_reid,
            det_thresh=cfg.track_low_thresh,
            per_class=False,
            is_obb=False,
        )

        if self.with_reid and hasattr(tracker, "model"):
            tracker.model.warmup()

        return tracker

    @staticmethod
    def _resolve_detector_weights(detector_weights: str | Path) -> str:
        """Chuẩn hóa đường dẫn model detector, hỗ trợ cả thư mục OpenVINO."""
        path = Path(detector_weights)
        if path.is_dir():
            xml_files = sorted(path.rglob("*.xml"))
            if len(xml_files) == 1:
                return str(xml_files[0])
            if len(xml_files) > 1:
                raise ValueError(
                    "Phát hiện nhiều file .xml trong thư mục detector '{}'. "
                    "Hãy chỉ định trực tiếp file .xml cần dùng.".format(path)
                )
            raise ValueError(
                "Thư mục detector '{}' không chứa file .xml OpenVINO hợp lệ. "
                "Hãy dùng file .pt hoặc chỉ định đúng model.xml.".format(path)
            )
        return str(path)

    @staticmethod
    def _to_dets_array(pred) -> np.ndarray:
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.hstack((xyxy, conf, cls))

    def track_frame(self, frame: np.ndarray) -> FrameTrackingResult:
        # Luồng xử lý: detect trước, sau đó đưa detections vào BoTSORT.
        pred = self.detector.predict(
            source=frame,
            conf=self.det_conf,
            iou=self.det_iou,
            verbose=False,
        )[0]
        dets = self._to_dets_array(pred)
        tracks = self.tracker.update(dets, frame)
        return FrameTrackingResult(detections=dets, tracks=tracks)

