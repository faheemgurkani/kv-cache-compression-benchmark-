"""Throughput and latency evaluation (paper-independent)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from framework.model import ModelLayer


@dataclass
class ThroughputMetrics:
    context_length: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    latency_ms_per_token: float


@torch.no_grad()
def evaluate_throughput(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    num_new_tokens: int = 128,
) -> ThroughputMetrics:
    start = time.perf_counter()
    model_layer.generate(input_ids, max_new_tokens=num_new_tokens)
    elapsed = time.perf_counter() - start

    tokens_per_second = num_new_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms = (elapsed / num_new_tokens) * 1000 if num_new_tokens > 0 else 0.0

    return ThroughputMetrics(
        context_length=input_ids.size(1),
        generated_tokens=num_new_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens_per_second,
        latency_ms_per_token=latency_ms,
    )


def measure_tokens_per_second(generate_fn, num_tokens: int = 128) -> float:
    start = time.perf_counter()
    generate_fn(num_tokens)
    elapsed = time.perf_counter() - start
    return num_tokens / elapsed if elapsed > 0 else 0.0
