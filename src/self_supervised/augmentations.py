"""Signal-specific augmentations for contrastive pretraining on vibration
windows: jitter, scaling, and time-warping."""
import numpy as np
import torch


def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    factor = torch.randn(x.shape[0], 1, device=x.device) * sigma + 1.0
    return x * factor


def time_warp(x: torch.Tensor, n_knots: int = 4, sigma: float = 0.2) -> torch.Tensor:
    """Smoothly resample each signal along a randomly warped time axis
    using piecewise-linear interpolation on knot displacements."""
    batch_size, length = x.shape
    device = x.device
    out = torch.empty_like(x)
    orig_steps = np.linspace(0, length - 1, num=length)
    knot_positions = np.linspace(0, length - 1, num=n_knots + 2)
    for i in range(batch_size):
        warp = np.random.normal(loc=1.0, scale=sigma, size=n_knots + 2)
        warp[0] = warp[-1] = 1.0
        cum_warp = np.cumsum(warp)
        cum_warp = cum_warp / cum_warp[-1] * (length - 1)
        warped_steps = np.interp(orig_steps, knot_positions, cum_warp)
        signal_np = x[i].detach().cpu().numpy()
        out[i] = torch.from_numpy(np.interp(orig_steps, warped_steps, signal_np)).to(device)
    return out.float()


def augment_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce two augmented views of a batch for contrastive learning."""
    def view():
        v = jitter(x)
        v = scaling(v)
        v = time_warp(v)
        return v

    return view(), view()
