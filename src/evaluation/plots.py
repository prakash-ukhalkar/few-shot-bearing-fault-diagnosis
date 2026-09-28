"""Plotting utilities for results: accuracy vs. label fraction (central
figure), confusion matrices."""
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_accuracy_vs_label_fraction(
    results: Dict[str, Dict[float, Dict[str, float]]],
    metric: str = "accuracy",
    out_path: str | Path = "results/figures/accuracy_vs_label_fraction.png",
) -> None:
    """results: {method_name: {label_fraction: {"mean": .., "std": ..}}}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, per_fraction in results.items():
        fractions = sorted(per_fraction.keys())
        means = [per_fraction[f]["mean"] for f in fractions]
        stds = [per_fraction[f]["std"] for f in fractions]
        ax.errorbar(fractions, means, yerr=stds, marker="o", label=method, capsize=3)

    ax.set_xlabel("Label fraction")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs. Label Fraction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str | Path = "results/figures/confusion_matrix.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
