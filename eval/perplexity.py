"""Perplexity evaluation (paper-independent)."""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

from framework.model import ModelLayer


@torch.no_grad()
def evaluate_perplexity(
    model_layer: ModelLayer,
    input_ids: torch.Tensor,
    max_length: int | None = None,
    stride: int = 512,
) -> float:
    """Sliding-window perplexity following HuggingFace Transformers guidance."""
    model = model_layer.model
    device = model_layer.device
    max_length = max_length or getattr(model.config, "max_position_embeddings", input_ids.size(1))
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="perplexity", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = window.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(window, labels=target_ids)
        neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if n_tokens == 0:
        raise ValueError("No tokens available for perplexity evaluation.")

    avg_nll = nll_sum / n_tokens
    return math.exp(avg_nll)
