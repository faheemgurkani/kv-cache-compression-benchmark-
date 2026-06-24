"""RocketKV compressor: token selection + eviction."""

from __future__ import annotations

from compressors.base import CompressedKV, KVCompressor


class RocketKVCompressor(KVCompressor):
    """RocketKV baseline placeholder."""

    name = "rocketkv"

    def __init__(self, bitwidth: int = 16) -> None:
        self.bitwidth = bitwidth

    def compress(self, key, value) -> CompressedKV:
        raise NotImplementedError("RocketKV compressor pending implementation.")

    def decompress(self, compressed: CompressedKV):
        raise NotImplementedError("RocketKV decompress pending implementation.")
