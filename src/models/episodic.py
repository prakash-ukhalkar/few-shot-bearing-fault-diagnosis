"""Episodic training and evaluation for Prototypical Networks."""
from typing import Tuple

import numpy as np
import torch

from src.models.prototypical import PrototypicalNetwork, episodic_loss


def sample_episode(
    X: np.ndarray, y: np.ndarray, n_support: int, n_query: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a support/query split across all available classes in y."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    support_idx, query_idx = [], []
    for c in classes:
        cls_idx = np.where(y == c)[0]
        n_needed = n_support + n_query
        replace = len(cls_idx) < n_needed
        chosen = rng.choice(cls_idx, size=n_needed, replace=replace)
        support_idx.extend(chosen[:n_support])
        query_idx.extend(chosen[n_support:])
    support_idx = np.array(support_idx)
    query_idx = np.array(query_idx)
    return X[support_idx], y[support_idx], X[query_idx], y[query_idx]


def remap_labels(y: np.ndarray) -> Tuple[np.ndarray, dict]:
    classes = sorted(np.unique(y).tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([mapping[v] for v in y], dtype=np.int64)
    return y_mapped, mapping


def train_protonet(
    model: PrototypicalNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_support: int,
    n_query: int,
    episodes_per_epoch: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str = "cpu",
) -> PrototypicalNetwork:
    y_mapped, _ = remap_labels(y_train)
    n_classes = len(np.unique(y_mapped))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for ep_i in range(episodes_per_epoch):
            ep_seed = seed * 100000 + epoch * 1000 + ep_i
            sx, sy, qx, qy = sample_episode(X_train, y_mapped, n_support, n_query, ep_seed)
            sx_t = torch.from_numpy(sx.astype(np.float32)).to(device)
            sy_t = torch.from_numpy(sy).to(device)
            qx_t = torch.from_numpy(qx.astype(np.float32)).to(device)
            qy_t = torch.from_numpy(qy).to(device)

            optimizer.zero_grad()
            logits = model(sx_t, sy_t, qx_t, n_classes)
            loss = episodic_loss(logits, qy_t)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_protonet(
    model: PrototypicalNetwork,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Classify X_query using prototypes computed from the full labeled
    (support) training set."""
    y_mapped, mapping = remap_labels(y_support)
    n_classes = len(mapping)
    inv_mapping = {v: k for k, v in mapping.items()}

    model.eval()
    sx_t = torch.from_numpy(X_support.astype(np.float32)).to(device)
    sy_t = torch.from_numpy(y_mapped).to(device)
    qx_t = torch.from_numpy(X_query.astype(np.float32)).to(device)

    preds, _ = model.classify_with_prototypes(sx_t, sy_t, qx_t, n_classes)
    preds_np = preds.cpu().numpy()
    return np.array([inv_mapping[p] for p in preds_np])
