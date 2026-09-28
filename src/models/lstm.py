"""Shallow LSTM baseline for raw vibration windows."""
import torch
import torch.nn as nn


class ShallowLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        subsample: int = 4,
    ):
        """`subsample` downsamples the raw window before feeding the LSTM,
        keeping sequence length (and thus compute) small on CPU."""
        super().__init__()
        self.subsample = subsample
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window_size) -> (B, seq_len, 1)
        if x.dim() == 2:
            x = x[:, :: self.subsample].unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)
