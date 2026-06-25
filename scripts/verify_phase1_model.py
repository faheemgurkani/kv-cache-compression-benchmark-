"""Phase 1 verification: model load, text generation, KV cache exists."""

from __future__ import annotations

import setup_path  # noqa: F401
import torch

from framework.kv_cache import iter_layer_kv
from framework.model import ModelLayer


def verify_phase1() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[1/3] Loading model (eager attention, fp16) on {device} ...")
    model_layer = ModelLayer(device=torch.device(device))

    assert model_layer.model.config.use_cache is True
    assert model_layer.attn_implementation == "eager"
    print("      Model loaded.")
    print(f"      attn_implementation: {model_layer.attn_implementation}")
    print(f"      use_cache: {model_layer.model.config.use_cache}")

    prompt = "The capital of France is"
    input_ids = model_layer.tokenize(prompt)
    print(f"[2/3] Generating text from prompt: {prompt!r}")
    output_ids = model_layer.generate(input_ids, max_new_tokens=16)
    text = model_layer.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert output_ids.shape[1] > input_ids.shape[1], "Generation did not produce new tokens"
    print(f"      Generated: {text!r}")

    print("[3/3] Confirming KV cache exists ...")
    with torch.no_grad():
        outputs = model_layer.forward_with_cache(input_ids, use_cache=True)

    assert outputs.past_key_values is not None, "past_key_values is None"

    layers = list(iter_layer_kv(outputs.past_key_values))
    num_layers = len(layers)
    expected_layers = model_layer.config.num_hidden_layers
    assert num_layers == expected_layers, f"Expected {expected_layers} layers, got {num_layers}"

    key, value = layers[0]
    assert key.ndim == 4 and value.ndim == 4
    assert key.shape == value.shape
    seq_len = key.shape[2]
    assert seq_len == input_ids.shape[1]

    print(f"      Layers with KV cache: {num_layers}")
    print(f"      Layer 0 key shape:   {tuple(key.shape)}")
    print(f"      Layer 0 value shape: {tuple(value.shape)}")
    print("Phase 1 verified: model load, generation, and KV cache all OK.")


def main() -> None:
    verify_phase1()


if __name__ == "__main__":
    main()
