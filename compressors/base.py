"""Generic KV-cache compressor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CompressedKV:
    """Compressed representation of one layer's key/value tensors."""

    keys: object
    values: object
    original_shape: tuple[int, ...]
    nbytes: int
    bitwidth: int | None = None


class KVCompressor(ABC):
    """Paper-independent interface for KV-cache compression methods."""

    name: str = "base"
    bitwidth: int = 16

    @abstractmethod
    def compress(self, key: torch.Tensor, value: torch.Tensor) -> CompressedKV:
        """Compress key/value tensors for one transformer layer."""

    @abstractmethod
    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore key/value tensors from a compressed representation."""

    def compression_ratio(self, key: torch.Tensor, value: torch.Tensor) -> float:
        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        compressed = self.compress(key, value)
        if compressed.nbytes == 0:
            return 1.0
        return original_bytes / compressed.nbytes
