"""Prototypical Networks (Snell et al., 2017) for few-shot fault diagnosis.

Class prototypes are the mean embedding of the support set for each class;
query samples are classified by nearest prototype (negative squared
Euclidean distance as logits).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn1d import CNN1DBackbone


def compute_prototypes(
    support_embeddings: torch.Tensor, support_labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Return (n_classes, embedding_dim) prototype tensor."""
    embedding_dim = support_embeddings.shape[1]
    prototypes = torch.zeros(n_classes, embedding_dim, device=support_embeddings.device)
    for c in range(n_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(dim=0)
    return prototypes


def prototypical_logits(
    query_embeddings: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    """Negative squared Euclidean distance from each query to each prototype."""
    dists = torch.cdist(query_embeddings, prototypes, p=2) ** 2
    return -dists


class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.backbone = CNN1DBackbone(in_channels, embedding_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        support_emb = self.embed(support_x)
        query_emb = self.embed(query_x)
        prototypes = compute_prototypes(support_emb, support_y, n_classes)
        return prototypical_logits(query_emb, prototypes)

    def classify_with_prototypes(
        self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor, n_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_labels, logits) for query_x given the support set."""
        logits = self.forward(support_x, support_y, query_x, n_classes)
        preds = logits.argmax(dim=1)
        return preds, logits


def episodic_loss(logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, query_labels)
