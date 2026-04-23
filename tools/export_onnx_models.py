from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# Allow running as: python tools/export_onnx_models.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking_engine.reid.core.reid import ReID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detector and ReID models to ONNX")
    parser.add_argument("--detector-pt", default="best.pt", help="Path to YOLO pt model")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--detector-opset", type=int, default=12)
    parser.add_argument("--reid-pt", default="models/osnet_x0_5_msmt17.pt", help="Path to ReID pt model")
    parser.add_argument("--reid-opset", type=int, default=12)
    parser.add_argument("--out-dir", default="models", help="Output directory")
    return parser.parse_args()


def export_detector(detector_pt: str, imgsz: int, opset: int, out_dir: Path) -> Path:
    model = YOLO(detector_pt, task="detect")
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False)

    exported_path = Path(exported)
    target_path = out_dir / "best.onnx"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != target_path.resolve():
        target_path.write_bytes(exported_path.read_bytes())
    return target_path


def export_reid(reid_pt: str, opset: int, out_dir: Path) -> Path:
    reid = ReID(weights=reid_pt, device="cpu", half=False)
    backend = reid.model
    torch_model = backend.model
    torch_model.eval()

    h, w = backend.input_shape
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    stem = Path(reid_pt).stem
    onnx_path = out_dir / f"{stem}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        torch_model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )

    return onnx_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    detector_onnx = export_detector(args.detector_pt, args.detector_imgsz, args.detector_opset, out_dir)
    print(f"Detector ONNX exported: {detector_onnx}")

    reid_onnx = export_reid(args.reid_pt, args.reid_opset, out_dir)
    print(f"ReID ONNX exported: {reid_onnx}")


if __name__ == "__main__":
    main()
