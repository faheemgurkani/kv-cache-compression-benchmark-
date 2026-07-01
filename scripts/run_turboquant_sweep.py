#!/usr/bin/env python3
"""Run the full TurboQuant evaluation grid from docs/EVALUATION_PLAN.md."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import setup_path  # noqa: F401

from compressors.registry import get_compressor
from eval.runner import EvaluationRunner
from framework.config import load_eval_config, load_model_config
from reporting.reporter import ResultReporter


SWEEP_CONFIGS: list[tuple[str, dict]] = [
    ("identity_baseline", {"name": "identity"}),
    ("tq_full_b2", {"name": "turboquant", "stage": "full", "bitwidth": 2}),
    ("tq_full_b3", {"name": "turboquant", "stage": "full", "bitwidth": 3}),
    ("tq_full_b4", {"name": "turboquant", "stage": "full", "bitwidth": 4}),
    ("tq_mse_b4", {"name": "turboquant", "stage": "wht_quant", "bitwidth": 4}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboQuant evaluation sweep")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to evaluate (default: configs/model.yaml)",
    )
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--output-stem", default="phase5_turboquant_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    eval_config = load_eval_config()
    context_lengths = args.context_lengths or model_config["context_lengths"]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{args.output_stem}_{stamp}"
    reporter = ResultReporter()
    all_results = []

    print(f"Sweep started: ctx={context_lengths} configs={len(SWEEP_CONFIGS)}", flush=True)

    for ctx in context_lengths:
        print(f"\n=== context_length={ctx} ===", flush=True)
        runner = EvaluationRunner(eval_config=eval_config, model_config=model_config)

        for label, cfg in SWEEP_CONFIGS:
            params = dict(cfg)
            name = params.pop("name")
            compressor = get_compressor(name, **params)
            runner.compressor = compressor
            print(f"  [{label}] bitwidth={getattr(compressor, 'bitwidth', None)}", flush=True)
            result = runner.run(
                ctx,
                run_perplexity=not args.skip_perplexity,
                run_throughput=not args.skip_throughput,
                include_baselines=True,
            )
            all_results.append(result)
            reporter.save_json(all_results, stem)
            reporter.save_summary_csv(all_results, stem)
            reporter.print_summary([result])

    print(f"\nSweep complete. Wrote results/{stem}.json and .csv", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
