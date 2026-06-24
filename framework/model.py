"""Model layer: load Qwen3 and expose KV-cache forward passes."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from framework.config import PROJECT_ROOT, load_model_config
from framework.device import get_device


class ModelLayer:
    """Wrapper around a causal LM with consistent KV-cache access."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        config = load_model_config()
        self.model_path = Path(model_path or PROJECT_ROOT / config["local_path"])
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()

    @property
    def config(self):
        return self.model.config

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        return self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        past_key_values=None,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            do_sample=False,
        )
