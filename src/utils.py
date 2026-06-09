"""Small utilities for results I/O and plotting.

These are convenience helpers used by notebooks/scripts to persist metrics and
figures into results/. No training logic lives here.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .config import RESULTS_DIR


def save_metrics(metrics: Dict[str, Any], name: str) -> Path:
    """Write a metrics dict to results/<name>.json and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


def load_metrics(name: str) -> Dict[str, Any]:
    """Load a metrics dict previously saved with save_metrics."""
    with (RESULTS_DIR / f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to results/<name>.png and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
