"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from framework.config import PROJECT_ROOT, load_eval_config


def load_wikitext2(
    split: str | None = None,
    cache_dir: Path | str | None = None,
):
    from datasets import load_dataset

    config = load_eval_config()
    wikitext_cfg = config["wikitext"]
    cache = Path(cache_dir) if cache_dir else PROJECT_ROOT / ".cache" / "huggingface" / "datasets"
    cache.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        wikitext_cfg["name"],
        wikitext_cfg["config"],
        split=split or wikitext_cfg["split"],
        cache_dir=str(cache),
    )


def iter_nonempty_texts(dataset) -> list[str]:
    texts: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            texts.append(text)
    return texts


def build_long_context_ids(
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    target_length: int,
    separator: str = "\n\n",
) -> torch.LongTensor:
    """Concatenate WikiText-2 samples until reaching target_length tokens."""
    pieces: list[str] = []
    for text in iter_nonempty_texts(dataset):
        pieces.append(text)
        candidate = separator.join(pieces)
        token_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids
        if token_ids.size(1) >= target_length:
            return token_ids[:, :target_length]

    if not pieces:
        raise ValueError("WikiText-2 split contains no non-empty text samples.")

    token_ids = tokenizer(separator.join(pieces), return_tensors="pt", add_special_tokens=False).input_ids
    if token_ids.size(1) < target_length:
        repeats = (target_length // token_ids.size(1)) + 1
        token_ids = token_ids.repeat(1, repeats)
    return token_ids[:, :target_length]
