"""Device selection for Apple Silicon (MPS) and CPU fallback."""

from __future__ import annotations

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
