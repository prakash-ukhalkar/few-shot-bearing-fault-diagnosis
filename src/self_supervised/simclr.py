"""SimCLR-style contrastive pretraining for 1D vibration signals.

Uses the NT-Xent (normalized temperature-scaled cross entropy) loss over
two augmented views per sample, with the CNN1DBackbone as encoder plus a
small MLP projection head (discarded after pretraining).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn1d import CNN1DBackbone


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, embedding_dim: int = 64, projection_dim: int = 32):
        super().__init__()
        self.encoder = CNN1DBackbone(embedding_dim=embedding_dim)
        self.projector = ProjectionHead(embedding_dim, out_dim=projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss between two batches of L2-normalized
    embeddings, z1[i] and z2[i] being the positive pair."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    positive_idx = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(z.device)

    return F.cross_entropy(sim, positive_idx)
