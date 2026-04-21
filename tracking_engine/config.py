from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TrackerRuntimeConfig:
    # Các ngưỡng gốc của BoTSORT
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    reid_recovery_enabled: bool = True
    reid_recovery_proximity_thresh: float = 0.5
    reid_recovery_appearance_thresh: float = 0.25
    reid_recovery_thresh: float = 0.35
    second_association_thresh: float = 0.5
    unconfirmed_association_thresh: float = 0.7

    # Tùy chọn CMC từ config chính thống BoTSORT
    cmc_method: str | None = "ecc"

    # Thiết lập runtime cho pipeline
    device: str = "auto"
    half: bool = False


DEFAULT_REID_MODEL = "osnet_x0_25_msmt17.pt"


OFFICIAL_BOTSORT_KEYS = (
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_buffer",
    "match_thresh",
    "proximity_thresh",
    "appearance_thresh",
    "reid_recovery_enabled",
    "reid_recovery_proximity_thresh",
    "reid_recovery_appearance_thresh",
    "reid_recovery_thresh",
    "second_association_thresh",
    "unconfirmed_association_thresh",
    "cmc_method",
)


def _normalize_cmc_method(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _extract_cfg_value(value: Any) -> Any:
    """Lấy giá trị runtime từ nhiều kiểu schema config khác nhau.

    Hỗ trợ:
    - Schema runtime trực tiếp: `key: 0.3`
    - Schema tuning: `key: {type: ..., default: 0.3, ...}`
    """
    if isinstance(value, dict):
        for key in ("value", "default", "current"):
            if key in value:
                return value[key]
    return value


def _coerce_like_default(raw_value: Any, default_value: Any) -> Any:
    """Ép kiểu theo kiểu của giá trị mặc định trong dataclass config."""
    if raw_value is None:
        return default_value

    if isinstance(default_value, bool):
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw_value)

    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)

    if isinstance(default_value, float):
        return float(raw_value)

    if isinstance(default_value, str):
        return str(raw_value)

    return raw_value


def load_tracker_runtime_config(config_path: str | Path) -> TrackerRuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tracker: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = TrackerRuntimeConfig()

    for key in OFFICIAL_BOTSORT_KEYS:
        if key not in data:
            continue

        parsed_value = _extract_cfg_value(data[key])
        default_value = getattr(cfg, key)
        setattr(cfg, key, _coerce_like_default(parsed_value, default_value))

    cfg.cmc_method = _normalize_cmc_method(cfg.cmc_method)
    return cfg
