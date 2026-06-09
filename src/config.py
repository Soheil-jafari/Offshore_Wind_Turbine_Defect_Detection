"""Shared paths and configuration for the project.

Centralizes directory locations so scripts and notebooks resolve data/results
the same way regardless of where they are launched from.
"""

from pathlib import Path

# Project layout
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# YOLO dataset config (standard Ultralytics layout under data/)
DATA_YAML = DATA_DIR / "wtbd.yaml"

# Pretrained checkpoint to fine-tune from (COCO-pretrained)
BASE_MODEL = "yolo11n.pt"

# Anomaly-detector crop size (square crops of healthy blade regions)
CROP_SIZE = 128
