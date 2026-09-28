import numpy as np

from src.evaluation.metrics import compute_confusion_matrix, compute_metrics
from src.evaluation.significance import paired_ttest, summarize_across_seeds, wilcoxon_signed_rank


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0


def test_compute_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)


def test_summarize_across_seeds():
    scores = np.array([0.8, 0.9, 0.85])
    stats = summarize_across_seeds(scores)
    assert abs(stats["mean"] - scores.mean()) < 1e-8
    assert stats["std"] >= 0


def test_paired_ttest_identical_scores_gives_high_pvalue():
    scores = np.array([0.8, 0.85, 0.9, 0.75])
    stat, p = paired_ttest(scores, scores.copy())
    assert p == 1.0 or np.isnan(p)


def test_paired_ttest_mismatched_lengths_raises():
    import pytest

    with pytest.raises(ValueError):
        paired_ttest(np.array([1, 2, 3]), np.array([1, 2]))


def test_wilcoxon_identical_scores():
    scores = np.array([0.8, 0.85, 0.9, 0.75])
    stat, p = wilcoxon_signed_rank(scores, scores.copy())
    assert p == 1.0
