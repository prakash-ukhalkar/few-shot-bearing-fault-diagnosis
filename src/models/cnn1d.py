"""Small 1D-CNN backbone/classifier for raw vibration windows.

Kept small (<500K params) to run comfortably on CPU: 3 conv blocks with
increasing channels, global average pooling, small FC head.
"""
import torch
import torch.nn as nn


class CNN1DBackbone(nn.Module):
    """Feature extractor: raw window -> embedding vector."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, embedding_dim, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window_size) -> (B, 1, window_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.net(x)
        z = self.pool(z).squeeze(-1)
        return z


class CNN1DClassifier(nn.Module):
    """Supervised classifier: backbone + linear head."""

    def __init__(self, n_classes: int, in_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.backbone = CNN1DBackbone(in_channels, embedding_dim)
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
