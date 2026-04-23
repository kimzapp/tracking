from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from tracking_engine.tracker_lib import Tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Python baseline tracking outputs to JSON")
    parser.add_argument("--video", default="video/longchau.mp4", help="Input video path")
    parser.add_argument("--detector", default="best_openvino_model", help="Detector path for Ultralytics")
    parser.add_argument("--tracker-config", default="botsort.yaml", help="Tracker config YAML")
    parser.add_argument("--reid-model", default="models/osnet_x0_5_msmt17.pt", help="ReID model")
    parser.add_argument("--det-conf", type=float, default=0.65)
    parser.add_argument("--det-iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", default="video/python_baseline.json", help="Output JSON path")
    return parser.parse_args()


def dets_from_pred(pred) -> np.ndarray:
    boxes = pred.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.hstack((xyxy, conf, cls))


def main() -> None:
    args = parse_args()

    detector = YOLO(args.detector, task="detect")
    tracker = Tracker(
        tracker_config_path=args.tracker_config,
        reid_model=args.reid_model,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    records: list[dict] = []
    frame_idx = 0

    while frame_idx < args.max_frames:
        success, frame = cap.read()
        if not success:
            break

        pred = detector.predict(source=frame, conf=args.det_conf, iou=args.det_iou, verbose=False)[0]
        dets = dets_from_pred(pred)
        tracks = tracker.update(dets, frame)

        records.append(
            {
                "frame_index": frame_idx,
                "dets": dets.astype(float).tolist(),
                "tracks": np.asarray(tracks, dtype=np.float32).astype(float).tolist(),
                "det_count": int(dets.shape[0]),
                "track_count": int(len(tracks)),
            }
        )

        frame_idx += 1

    cap.release()

    payload = {
        "video": args.video,
        "detector": args.detector,
        "tracker_config": args.tracker_config,
        "reid_model": args.reid_model,
        "det_conf": args.det_conf,
        "det_iou": args.det_iou,
        "frames_exported": frame_idx,
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Exported baseline: {output_path} (frames={frame_idx})")


if __name__ == "__main__":
    main()
