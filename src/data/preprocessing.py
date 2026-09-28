"""Signal preprocessing: normalization and sliding-window segmentation."""
from typing import Tuple

import numpy as np


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization of a 1D signal."""
    signal = np.asarray(signal, dtype=np.float64)
    std = signal.std()
    if std < 1e-12:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def sliding_window(
    signal: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
    """Segment a 1D signal into overlapping windows.

    Returns an array of shape (n_windows, window_size). Any trailing samples
    that don't fill a full window are dropped.
    """
    signal = np.asarray(signal)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    n = len(signal)
    if n < window_size:
        return np.empty((0, window_size), dtype=signal.dtype)
    n_windows = (n - window_size) // stride + 1
    windows = np.empty((n_windows, window_size), dtype=signal.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = signal[start : start + window_size]
    return windows


def windows_and_labels(
    signal: np.ndarray, label: int, window_size: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Window a signal and produce a matching label array."""
    windows = sliding_window(signal, window_size, stride)
    labels = np.full(len(windows), label, dtype=np.int64)
    return windows, labels
