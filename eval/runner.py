"""Paper-independent evaluation orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from compressors.base import KVCompressor
from compressors.registry import get_compressor
from data.loader import build_long_context_ids, load_wikitext2
from eval.memory import MemoryMetrics, evaluate_memory
from eval.perplexity import evaluate_perplexity
from eval.throughput import ThroughputMetrics, evaluate_throughput
from framework.config import load_eval_config, load_model_config
from framework.model import ModelLayer


@dataclass
class EvaluationResult:
    compressor: str
    bitwidth: int | None
    context_length: int
    perplexity: float | None
    memory: MemoryMetrics
    throughput: ThroughputMetrics

    def to_dict(self) -> dict:
        return {
            "compressor": self.compressor,
            "bitwidth": self.bitwidth,
            "context_length": self.context_length,
            "perplexity": self.perplexity,
            "memory": asdict(self.memory),
            "throughput": asdict(self.throughput),
        }


class EvaluationRunner:
    """Runs memory, throughput, and quality metrics for any compressor."""

    def __init__(
        self,
        model_layer: ModelLayer | None = None,
        compressor: KVCompressor | None = None,
        eval_config: dict | None = None,
        model_config: dict | None = None,
    ) -> None:
        self.model_config = model_config or load_model_config()
        self.eval_config = eval_config or load_eval_config()
        self.model_layer = model_layer or ModelLayer()
        self.compressor = compressor or get_compressor("identity")
        self.dataset = load_wikitext2()

    def build_context(self, context_length: int) -> torch.LongTensor:
        return build_long_context_ids(
            self.model_layer.tokenizer,
            self.dataset,
            target_length=context_length,
        ).to(self.model_layer.device)

    def run(
        self,
        context_length: int,
        run_perplexity: bool = True,
        run_memory: bool = True,
        run_throughput: bool = True,
        perplexity_stride: int | None = None,
        generated_tokens: int | None = None,
    ) -> EvaluationResult:
        input_ids = self.build_context(context_length)
        stride = perplexity_stride or self.eval_config.get("perplexity_stride", 512)
        num_new_tokens = generated_tokens or self.eval_config.get("generated_tokens", 64)

        ppl = evaluate_perplexity(self.model_layer, input_ids, stride=stride) if run_perplexity else None
        memory = evaluate_memory(self.model_layer, input_ids, self.compressor) if run_memory else None
        throughput = (
            evaluate_throughput(self.model_layer, input_ids, num_new_tokens=num_new_tokens)
            if run_throughput
            else None
        )

        if memory is None or throughput is None:
            raise ValueError("Memory and throughput metrics are required for a complete evaluation run.")

        return EvaluationResult(
            compressor=self.compressor.name,
            bitwidth=getattr(self.compressor, "bitwidth", None),
            context_length=context_length,
            perplexity=ppl,
            memory=memory,
            throughput=throughput,
        )

    def run_all_context_lengths(
        self,
        context_lengths: list[int] | None = None,
        **kwargs,
    ) -> list[EvaluationResult]:
        lengths = context_lengths or self.model_config.get("context_lengths", [4096])
        return [self.run(length, **kwargs) for length in lengths]
