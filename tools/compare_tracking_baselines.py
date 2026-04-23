from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python and C++ tracking baselines")
    parser.add_argument("--python", required=True, help="Path to python_baseline.json")
    parser.add_argument("--cpp", required=True, help="Path to cpp_baseline.json")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--box-atol", type=float, default=8.0, help="Absolute tolerance in pixels for boxes")
    parser.add_argument("--id-mismatch-ratio-thresh", type=float, default=0.25)
    parser.add_argument("--count-mismatch-ratio-thresh", type=float, default=0.10)
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("records", [])


def to_tracks_array(frame: dict) -> np.ndarray:
    tracks = np.asarray(frame.get("tracks", []), dtype=np.float32)
    if tracks.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)
    if tracks.shape[1] < 8:
        pad = np.full((tracks.shape[0], 8 - tracks.shape[1]), -1.0, dtype=np.float32)
        tracks = np.concatenate([tracks, pad], axis=1)
    return tracks[:, :8]


def main() -> None:
    args = parse_args()
    py_records = load_records(args.python)
    cpp_records = load_records(args.cpp)
    n = min(len(py_records), len(cpp_records), args.max_frames)
    if n == 0:
        raise RuntimeError("No overlapping frames found in both baseline files.")

    count_mismatches = 0
    id_mismatches = 0
    box_mismatches = 0
    compared_tracks = 0

    for i in range(n):
        py_t = to_tracks_array(py_records[i])
        cpp_t = to_tracks_array(cpp_records[i])

        if py_t.shape[0] != cpp_t.shape[0]:
            count_mismatches += 1
            continue

        if py_t.shape[0] == 0:
            continue

        py_order = np.argsort(py_t[:, 7])
        cpp_order = np.argsort(cpp_t[:, 7])
        py_t = py_t[py_order]
        cpp_t = cpp_t[cpp_order]

        compared_tracks += py_t.shape[0]

        id_diff = py_t[:, 4] != cpp_t[:, 4]
        id_mismatches += int(id_diff.sum())

        box_diff = np.abs(py_t[:, :4] - cpp_t[:, :4])
        box_fail = np.any(box_diff > args.box_atol, axis=1)
        box_mismatches += int(box_fail.sum())

    count_ratio = count_mismatches / float(n)
    id_ratio = id_mismatches / float(max(1, compared_tracks))
    box_ratio = box_mismatches / float(max(1, compared_tracks))

    print(f"frames_compared={n}")
    print(f"count_mismatch_frames={count_mismatches} ({count_ratio:.3f})")
    print(f"id_mismatches={id_mismatches}/{max(1,compared_tracks)} ({id_ratio:.3f})")
    print(f"box_mismatches={box_mismatches}/{max(1,compared_tracks)} ({box_ratio:.3f})")

    if count_ratio > args.count_mismatch_ratio_thresh:
        raise SystemExit(
            f"Count mismatch ratio {count_ratio:.3f} > threshold {args.count_mismatch_ratio_thresh:.3f}"
        )
    if id_ratio > args.id_mismatch_ratio_thresh:
        raise SystemExit(
            f"ID mismatch ratio {id_ratio:.3f} > threshold {args.id_mismatch_ratio_thresh:.3f}"
        )


if __name__ == "__main__":
    main()

