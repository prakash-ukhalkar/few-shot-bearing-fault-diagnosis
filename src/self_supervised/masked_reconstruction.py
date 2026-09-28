"""Masked-signal reconstruction pretext task (ablation vs. contrastive
pretraining). Randomly masks contiguous spans of the input signal and
trains an encoder-decoder to reconstruct the original values."""
import torch
import torch.nn as nn

from src.models.cnn1d import CNN1DBackbone


def mask_signal(x: torch.Tensor, mask_ratio: float = 0.15, span_len: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out random contiguous spans of the signal. Returns (masked_x, mask)
    where mask is 1 where values were masked (and should be reconstructed)."""
    batch_size, length = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    n_spans = max(1, int((length * mask_ratio) / span_len))
    for i in range(batch_size):
        for _ in range(n_spans):
            start = torch.randint(0, max(1, length - span_len), (1,)).item()
            mask[i, start : start + span_len] = True
    masked_x = x.clone()
    masked_x[mask] = 0.0
    return masked_x, mask


class MaskedReconstructionModel(nn.Module):
    """Encoder (shared CNN1DBackbone) + simple decoder that upsamples the
    pooled embedding back to the original window length."""

    def __init__(self, window_size: int, embedding_dim: int = 64):
        super().__init__()
        self.encoder = CNN1DBackbone(embedding_dim=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, window_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only over masked positions."""
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)
