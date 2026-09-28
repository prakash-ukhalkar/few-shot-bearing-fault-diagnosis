"""Build windowed train/test arrays from raw CWRU .mat files.

Usage:
    python -m src.data.build_dataset --raw data/raw/cwru --out data/processed \
        --window-size 1024 --stride 512 --channel DE

Saves data/processed/windows.npz containing:
  X: (N, window_size) float32
  y: (N,) int64 class labels
  load_hp: (N,) int64 motor load in HP
  class_names: list of class name strings
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.cwru import CLASS_NAMES, CLASS_TO_IDX, load_mat_signal
from src.data.cwru_manifest import build_index
from src.data.preprocessing import normalize_signal, windows_and_labels


def build(raw_dir: Path, window_size: int, stride: int, channel: str):
    index = build_index()
    all_X, all_y, all_load = [], [], []
    missing = []
    for filename, meta in index.items():
        path = raw_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        signal = load_mat_signal(path, channel=channel)
        signal = normalize_signal(signal)
        label_idx = CLASS_TO_IDX[meta["label"]]
        windows, labels = windows_and_labels(signal, label_idx, window_size, stride)
        all_X.append(windows)
        all_y.append(labels)
        all_load.append(np.full(len(windows), meta["load_hp"], dtype=np.int64))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw files missing (run download_cwru.py first): {missing[:5]}..."
        )

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0)
    load_hp = np.concatenate(all_load, axis=0)
    return X, y, load_hp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/cwru")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--channel", default="DE")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, load_hp = build(raw_dir, args.window_size, args.stride, args.channel)
    out_path = out_dir / "windows.npz"
    np.savez_compressed(
        out_path, X=X, y=y, load_hp=load_hp, class_names=np.array(CLASS_NAMES)
    )
    print(f"Saved {X.shape[0]} windows of length {X.shape[1]} to {out_path}")
    print(f"Class distribution: {np.bincount(y)}")


if __name__ == "__main__":
    main()
