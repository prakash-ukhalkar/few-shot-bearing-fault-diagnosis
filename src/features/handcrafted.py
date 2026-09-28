"""Hand-crafted time-domain and spectral features for classical baselines
(SVM / Random Forest)."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew


def time_domain_features(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    crest_factor = peak / rms if rms > 1e-12 else 0.0
    return np.array(
        [
            x.mean(),
            x.std(),
            rms,
            peak,
            crest_factor,
            skew(x),
            kurtosis(x),
            x.max() - x.min(),  # peak-to-peak
        ]
    )


def spectral_features(window: np.ndarray, fs: float = 12000.0) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    n = len(x)
    spectrum = np.abs(rfft(x))
    freqs = rfftfreq(n, d=1.0 / fs)
    total_energy = spectrum.sum()
    if total_energy < 1e-12:
        centroid = 0.0
    else:
        centroid = float(np.sum(freqs * spectrum) / total_energy)
    peak_freq = float(freqs[np.argmax(spectrum)])
    spectral_std = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (total_energy + 1e-12)))
    spectral_energy = float(np.sum(spectrum**2))
    return np.array([centroid, peak_freq, spectral_std, spectral_energy])


def extract_features(window: np.ndarray, fs: float = 12000.0) -> np.ndarray:
    """Concatenate time-domain and spectral feature vectors for one window."""
    return np.concatenate([time_domain_features(window), spectral_features(window, fs)])


def extract_features_batch(X: np.ndarray, fs: float = 12000.0) -> np.ndarray:
    """Apply extract_features to each row of X, shape (N, window_size)."""
    return np.stack([extract_features(w, fs) for w in X])


FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "peak",
    "crest_factor",
    "skewness",
    "kurtosis",
    "peak_to_peak",
    "spectral_centroid",
    "peak_frequency",
    "spectral_std",
    "spectral_energy",
]
