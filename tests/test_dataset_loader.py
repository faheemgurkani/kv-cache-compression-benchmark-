"""Tests for WikiText-2 loading and long-context construction."""

from pathlib import Path

import pytest

from data.loader import build_long_context_ids, iter_nonempty_texts, load_wikitext2
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.fixture(scope="module")
def wikitext_dataset():
    return load_wikitext2()


@pytest.fixture(scope="module")
def tokenizer():
    if not MODEL_DIR.exists():
        pytest.skip("Local model not downloaded")
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def test_wikitext_has_text(wikitext_dataset):
    texts = iter_nonempty_texts(wikitext_dataset)
    assert len(texts) > 0


def test_build_long_context_ids(wikitext_dataset, tokenizer):
    token_ids = build_long_context_ids(tokenizer, wikitext_dataset, target_length=256)
    assert token_ids.shape == (1, 256)
