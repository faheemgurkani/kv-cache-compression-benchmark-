"""TurboQuant compressor: WHT -> Lloyd-Max -> QJL Residual."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor


class TurboQuantCompressor(KVCompressor):
    """TurboQuant pipeline placeholder."""

    name = "turboquant"

    def __init__(self, bitwidth: int = 4) -> None:
        self.bitwidth = bitwidth

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> CompressedKV:
        raise NotImplementedError(
            "TurboQuant compressor pending Phase 1 implementation (WHT + Lloyd-Max + QJL)."
        )

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TurboQuant decompress pending Phase 1 implementation.")
