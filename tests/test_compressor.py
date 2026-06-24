"""Tests for compressor interface."""

import torch

from compressors.identity import IdentityCompressor


def test_identity_compressor_roundtrip():
    compressor = IdentityCompressor()
    key = torch.randn(1, 8, 4, 128)
    value = torch.randn(1, 8, 4, 128)

    compressed = compressor.compress(key, value)
    restored_key, restored_value = compressor.decompress(compressed)

    assert torch.allclose(key, restored_key)
    assert torch.allclose(value, restored_value)
    assert compressed.nbytes == key.numel() * key.element_size() + value.numel() * value.element_size()
    assert compressor.compression_ratio(key, value) == 1.0
