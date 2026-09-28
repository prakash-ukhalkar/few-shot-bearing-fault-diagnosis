import numpy as np
import pytest

from src.data.preprocessing import normalize_signal, sliding_window, windows_and_labels


def test_normalize_signal_zero_mean_unit_var():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalize_signal(signal)
    assert abs(normed.mean()) < 1e-8
    assert abs(normed.std() - 1.0) < 1e-8


def test_normalize_signal_constant_input():
    signal = np.full(10, 5.0)
    normed = normalize_signal(signal)
    assert np.allclose(normed, 0.0)


def test_sliding_window_basic():
    signal = np.arange(10)
    windows = sliding_window(signal, window_size=4, stride=2)
    assert windows.shape == (4, 4)
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[1], [2, 3, 4, 5])
    np.testing.assert_array_equal(windows[-1], [6, 7, 8, 9])


def test_sliding_window_no_overlap():
    signal = np.arange(12)
    windows = sliding_window(signal, window_size=4, stride=4)
    assert windows.shape == (3, 4)


def test_sliding_window_too_short_signal():
    signal = np.arange(3)
    windows = sliding_window(signal, window_size=4, stride=2)
    assert windows.shape == (0, 4)


def test_sliding_window_invalid_params():
    signal = np.arange(10)
    with pytest.raises(ValueError):
        sliding_window(signal, window_size=0, stride=2)
    with pytest.raises(ValueError):
        sliding_window(signal, window_size=4, stride=0)


def test_windows_and_labels_matches_shape():
    signal = np.arange(20)
    windows, labels = windows_and_labels(signal, label=3, window_size=5, stride=5)
    assert windows.shape[0] == labels.shape[0]
    assert np.all(labels == 3)
