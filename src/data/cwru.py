"""CWRU Bearing Dataset loading utilities.

Loads the official .mat files from the Case Western Reserve University
Bearing Data Center. Files are expected under `data/raw/cwru/` after
running `src/data/download_cwru.py`.

Each .mat file contains one or more channels named like:
  X097_DE_time, X097_FE_time, X097_BA_time, X097RPM
The prefix number is the file id; DE = drive end accelerometer,
FE = fan end accelerometer, BA = base accelerometer.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat

# Fault class labels used throughout the project.
CLASS_NAMES = [
    "normal",
    "ball_007",
    "ball_014",
    "ball_021",
    "inner_race_007",
    "inner_race_014",
    "inner_race_021",
    "outer_race_007",
    "outer_race_014",
    "outer_race_021",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class CWRUFile:
    """Metadata for a single CWRU .mat recording."""

    file_id: str
    label: str
    load_hp: int  # motor load in HP (0-3)
    filename: str


def load_mat_signal(path: Path, channel: str = "DE") -> np.ndarray:
    """Load the specified accelerometer channel from a CWRU .mat file.

    channel: one of "DE", "FE", "BA".
    """
    mat = loadmat(str(path))
    key = None
    for k in mat.keys():
        if k.endswith(f"{channel}_time"):
            key = k
            break
    if key is None:
        raise KeyError(f"No {channel}_time channel found in {path}")
    return mat[key].squeeze().astype(np.float64)


def list_available_files(raw_dir: Path) -> List[Path]:
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.glob("*.mat"))


def build_manifest(index: Dict[str, dict]) -> List[CWRUFile]:
    """Build a manifest of CWRUFile entries from an index dict.

    `index` maps filename -> {"label": str, "load_hp": int}.
    """
    manifest = []
    for filename, meta in index.items():
        manifest.append(
            CWRUFile(
                file_id=Path(filename).stem,
                label=meta["label"],
                load_hp=meta["load_hp"],
                filename=filename,
            )
        )
    return manifest
