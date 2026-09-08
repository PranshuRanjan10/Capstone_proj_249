"""Config loading. Single source of truth for every site/agronomy parameter."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def stage_names(cfg) -> list[str]:
    return [s["name"] for s in cfg["phenology"]["stages"]]
