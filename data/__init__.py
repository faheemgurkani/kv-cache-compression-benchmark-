"""Dataset loading utilities (avoids shadowing HuggingFace `datasets`)."""

from data.loader import build_long_context_ids, iter_nonempty_texts, load_wikitext2

__all__ = ["build_long_context_ids", "iter_nonempty_texts", "load_wikitext2"]
