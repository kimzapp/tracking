from __future__ import annotations

from pathlib import Path

import torch

from tracking_engine.reid.backends.pytorch_backend import PyTorchBackend
from tracking_engine.utils import WEIGHTS
from tracking_engine.utils.torch_utils import select_device


class ReID:
    """Runtime ReID tối giản cho tracking: chỉ dùng backend PyTorch."""

    def __init__(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        *,
        weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        device: str | torch.device = "cpu",
        half: bool = False,
    ) -> None:
        model_ref = path if path is not None else weights
        if model_ref is None:
            model_ref = WEIGHTS / "osnet_ain_x1_0_msmt17.pt"

        self.weights = model_ref
        self.device = device if isinstance(device, torch.device) else select_device(device)
        self.half = bool(half)
        self.model = PyTorchBackend(self.weights, self.device, self.half)


__all__ = ["ReID"]
