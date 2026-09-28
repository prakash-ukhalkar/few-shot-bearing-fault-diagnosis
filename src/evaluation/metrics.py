"""Evaluation metrics: accuracy, macro-F1, confusion matrix."""
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
