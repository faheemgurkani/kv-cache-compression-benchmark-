"""Phase 1 tests: model load, generation, KV cache."""

from pathlib import Path

import pytest
import torch

from framework.kv_cache import iter_layer_kv
from framework.model import ModelLayer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_1.7b"


@pytest.fixture(scope="module")
def model_layer():
    if not MODEL_DIR.exists():
        pytest.skip("Model not downloaded")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return ModelLayer(device=device)


def test_model_loads_with_eager_and_cache(model_layer):
    assert model_layer.attn_implementation == "eager"
    assert model_layer.model.config.use_cache is True
    assert model_layer.config.num_hidden_layers == 28


def test_generates_text(model_layer):
    input_ids = model_layer.tokenize("The capital of France is")
    output_ids = model_layer.generate(input_ids, max_new_tokens=8)
    assert output_ids.shape[1] > input_ids.shape[1]
    text = model_layer.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert len(text) > 0


def test_kv_cache_exists(model_layer):
    input_ids = model_layer.tokenize("Hello world")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)

    assert outputs.past_key_values is not None
    layers = list(iter_layer_kv(outputs.past_key_values))
    assert len(layers) == model_layer.config.num_hidden_layers

    key, value = layers[0]
    assert key.shape == value.shape
    assert key.shape[2] == input_ids.shape[1]
