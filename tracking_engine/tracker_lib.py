from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tracking_engine.trackers.botsort.botsort import BotSort
from tracking_engine.config import TrackerRuntimeConfig, load_tracker_runtime_config

class Tracker:
    def __init__(
        self,
        tracker_config_path: str | Path,
        reid_model: str | Path,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        with_reid: Optional[bool] = None,
        fuse_first_associate: bool = False,
    ):
        self.tracker_cfg: TrackerRuntimeConfig = load_tracker_runtime_config(tracker_config_path)
        self.reid_model = Path(reid_model)
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

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Cập nhật tracker với bbox detect và frame hiện tại.
        dets: ndarray [N, 6] (x1, y1, x2, y2, conf, cls)
        frame: ảnh gốc
        Trả về: ndarray [M, 7] (x1, y1, x2, y2, id, conf, cls, ...)
        """
        return self.tracker.update(dets, frame)
