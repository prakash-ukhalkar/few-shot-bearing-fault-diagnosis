import numpy as np

from src.data.splits import cross_load_split, stratified_label_fraction_split


def test_stratified_label_fraction_split_full():
    y = np.array([0, 0, 1, 1, 2, 2])
    idx = stratified_label_fraction_split(y, fraction=1.0, seed=0)
    assert len(idx) == len(y)


def test_stratified_label_fraction_split_partial_preserves_all_classes():
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    idx = stratified_label_fraction_split(y, fraction=0.1, seed=0)
    subset_labels = y[idx]
    assert set(np.unique(subset_labels)) == {0, 1, 2}
    assert len(idx) < len(y)


def test_stratified_label_fraction_split_deterministic():
    y = np.array([0] * 10 + [1] * 10)
    idx1 = stratified_label_fraction_split(y, fraction=0.3, seed=42)
    idx2 = stratified_label_fraction_split(y, fraction=0.3, seed=42)
    np.testing.assert_array_equal(idx1, idx2)


def test_cross_load_split_no_overlap():
    load_hp = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    train_idx, test_idx = cross_load_split(load_hp, train_loads=[0, 1, 2], test_loads=[3])
    assert set(train_idx).isdisjoint(set(test_idx))
    assert all(load_hp[train_idx] != 3)
    assert all(load_hp[test_idx] == 3)
