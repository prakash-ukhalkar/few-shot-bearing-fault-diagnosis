"""Train/test splitting utilities: label-fraction subsampling and
cross-load generalization splits."""
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_label_fraction_split(
    y: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Return indices of a stratified subsample of size `fraction` of y.

    Guarantees at least 1 sample per class when fraction > 0.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return np.arange(len(y))

    rng = np.random.RandomState(seed)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(round(len(cls_idx) * fraction)))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        indices.append(chosen)
    return np.sort(np.concatenate(indices))


def train_test_split_stratified(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


def cross_load_split(
    load_hp: np.ndarray, train_loads: List[int], test_loads: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for training on `train_loads` HP
    conditions and testing on held-out `test_loads` conditions."""
    train_idx = np.where(np.isin(load_hp, train_loads))[0]
    test_idx = np.where(np.isin(load_hp, test_loads))[0]
    return train_idx, test_idx
