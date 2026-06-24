"""KV-cache memory evaluation (paper-independent)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from compressors.base import KVCompressor
from framework.kv_cache import apply_compressor, compressed_size_bytes, get_cache_size_bytes
from framework.model import ModelLayer


@dataclass
class MemoryMetrics:
    context_length: int
    uncompressed_bytes: int
    compressed_bytes: int
    compression_ratio: float
    process_memory_mb: float


def process_memory_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def kv_cache_bytes(num_layers: int, seq_len: int, num_kv_heads: int, head_dim: int, bitwidth: int = 16) -> int:
    """Analytical KV-cache size estimate in bytes."""
    elements = 2 * num_layers * seq_len * num_kv_heads * head_dim
    return int(elements * (bitwidth / 8))


@torch.no_grad()
def evaluate_memory(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    compressor: KVCompressor,
) -> MemoryMetrics:
    outputs = model_layer.forward_with_cache(input_ids)
    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values.")

    uncompressed_bytes = get_cache_size_bytes(past_key_values)
    compressed_layers = apply_compressor(past_key_values, compressor)
    compressed_bytes = compressed_size_bytes(compressed_layers)
    ratio = uncompressed_bytes / compressed_bytes if compressed_bytes > 0 else 1.0

    return MemoryMetrics(
        context_length=input_ids.size(1),
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        process_memory_mb=process_memory_mb(),
    )
