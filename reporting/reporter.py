"""Reporting layer for benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.runner import EvaluationResult
from framework.config import PROJECT_ROOT


class ResultReporter:
    """Persist evaluation outputs to results/ and plots/."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir or PROJECT_ROOT / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, results: list[EvaluationResult] | EvaluationResult, name: str) -> Path:
        payload = results if isinstance(results, list) else [results]
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "results": [item.to_dict() for item in payload],
        }
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(report, indent=2))
        return path

    def save_summary_csv(self, results: list[EvaluationResult], name: str) -> Path:
        import csv

        path = self.output_dir / f"{name}.csv"
        rows = [item.to_dict() for item in results]
        fieldnames = [
            "compressor",
            "bitwidth",
            "context_length",
            "perplexity",
            "uncompressed_bytes",
            "compressed_bytes",
            "compression_ratio",
            "tokens_per_second",
            "latency_ms_per_token",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "compressor": row["compressor"],
                        "bitwidth": row["bitwidth"],
                        "context_length": row["context_length"],
                        "perplexity": row["perplexity"],
                        "uncompressed_bytes": row["memory"]["uncompressed_bytes"],
                        "compressed_bytes": row["memory"]["compressed_bytes"],
                        "compression_ratio": row["memory"]["compression_ratio"],
                        "tokens_per_second": row["throughput"]["tokens_per_second"],
                        "latency_ms_per_token": row["throughput"]["latency_ms_per_token"],
                    }
                )
        return path

    @staticmethod
    def print_summary(results: list[EvaluationResult]) -> None:
        for result in results:
            print(
                f"[{result.compressor}] ctx={result.context_length} "
                f"ppl={result.perplexity:.4f} "
                f"ratio={result.memory.compression_ratio:.2f}x "
                f"tok/s={result.throughput.tokens_per_second:.2f}"
            )
