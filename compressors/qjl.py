"""QJL compressor: random projection + 1-bit quantization."""

from __future__ import annotations

from compressors.base import CompressedKV, KVCompressor


class QJLCompressor(KVCompressor):
    """QJL baseline placeholder."""

    name = "qjl"

    def __init__(self, bitwidth: int = 1) -> None:
        self.bitwidth = bitwidth

    def compress(self, key, value) -> CompressedKV:
        raise NotImplementedError("QJL compressor pending implementation.")

    def decompress(self, compressed: CompressedKV):
        raise NotImplementedError("QJL decompress pending implementation.")
