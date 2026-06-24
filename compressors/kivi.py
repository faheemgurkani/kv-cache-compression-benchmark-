"""KIVI compressor: asymmetric INT2 quantization."""

from __future__ import annotations

from compressors.base import CompressedKV, KVCompressor


class KIVICompressor(KVCompressor):
    """KIVI baseline placeholder."""

    name = "kivi"

    def __init__(self, bitwidth: int = 2) -> None:
        self.bitwidth = bitwidth

    def compress(self, key, value) -> CompressedKV:
        raise NotImplementedError("KIVI compressor pending implementation.")

    def decompress(self, compressed: CompressedKV):
        raise NotImplementedError("KIVI decompress pending implementation.")
