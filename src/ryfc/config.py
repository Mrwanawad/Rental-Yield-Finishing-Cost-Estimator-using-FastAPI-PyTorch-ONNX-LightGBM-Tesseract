"""Paths and shared constants. Heavy artifacts can live outside the repo via RYFC_WORK_DIR."""

import os
from functools import lru_cache
from pathlib import Path

import yaml


def _root() -> Path:
    """Repo root: RYFC_ROOT, else the source checkout, else the working directory (installed package)."""
    if os.getenv("RYFC_ROOT"):
        return Path(os.environ["RYFC_ROOT"])
    src_root = Path(__file__).resolve().parents[2]
    return src_root if (src_root / "configs").is_dir() else Path.cwd()


ROOT = _root()
CONFIGS = ROOT / "configs"
MODELS = Path(os.getenv("RYFC_MODELS_DIR", ROOT / "models"))
REPORTS = ROOT / "reports"
SAMPLES = ROOT / "samples"
WORK = Path(os.getenv("RYFC_WORK_DIR", ROOT / ".work"))  # training data, mlruns, logs

# Segmentation classes. Door openings are labelled as wall so rooms come out as closed regions.
CLASSES = ["outside", "wall", "living", "bedroom", "kitchen", "bathroom", "balcony", "corridor"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
ROOM_CLASSES = CLASSES[2:]
SEG_SIZE = 512

PACKAGES = ["basic", "standard", "premium"]


@lru_cache
def load_yaml(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)
