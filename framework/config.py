"""YAML configuration loaders."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "model.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def load_eval_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "eval.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
