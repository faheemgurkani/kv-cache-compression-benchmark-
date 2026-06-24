"""Identity (no-compression) baseline compressor."""

from __future__ import annotations

import torch

from compressors.base import CompressedKV, KVCompressor


class IdentityCompressor(KVCompressor):
    """Passthrough compressor used to validate the evaluation pipeline."""

    name = "identity"
    bitwidth = 16

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> CompressedKV:
        key_copy = key.detach().clone()
        value_copy = value.detach().clone()
        nbytes = key_copy.numel() * key_copy.element_size() + value_copy.numel() * value_copy.element_size()
        return CompressedKV(
            keys=key_copy,
            values=value_copy,
            original_shape=tuple(key.shape),
            nbytes=nbytes,
            bitwidth=self.bitwidth,
        )

    def decompress(self, compressed: CompressedKV) -> tuple[torch.Tensor, torch.Tensor]:
        return compressed.keys, compressed.values  # type: ignore[return-value]
