"""Statistical significance testing between methods across seeds."""
from typing import Dict, Tuple

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    """Paired t-test between two methods' per-seed scores. Returns
    (statistic, p_value)."""
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length (paired seeds)")
    stat, p = ttest_rel(scores_a, scores_b)
    return float(stat), float(p)


def wilcoxon_signed_rank(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length (paired seeds)")
    diffs = scores_a - scores_b
    if np.allclose(diffs, 0):
        return 0.0, 1.0
    stat, p = wilcoxon(scores_a, scores_b)
    return float(stat), float(p)


def summarize_across_seeds(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores)
    return {"mean": float(scores.mean()), "std": float(scores.std(ddof=1) if len(scores) > 1 else 0.0)}
