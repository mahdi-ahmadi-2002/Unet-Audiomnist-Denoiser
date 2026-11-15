"""Classical single-channel enhancement baselines (no training required)."""

from __future__ import annotations

import math

import torch

from .audio import EPS, STFT


def estimate_noise_psd(power: torch.Tensor, quantile: float = 0.3) -> torch.Tensor:
    """Per-frequency noise power from a low temporal quantile (Stahl et al., 2000).

    Speech is sparse in time-frequency, so a low quantile of each bin's power
    over time is dominated by noise. For Gaussian noise the bin power is
    exponentially distributed, whose q-quantile is ``-mean * ln(1 - q)``;
    dividing by that factor turns the quantile into an unbiased mean.
    """
    return torch.quantile(power, quantile, dim=-1, keepdim=True) / -math.log(1 - quantile)


def spectral_subtraction(
    noisy: torch.Tensor, stft: STFT, oversubtraction: float = 2.0, floor: float = 0.02
) -> torch.Tensor:
    """Power spectral subtraction with over-subtraction and a spectral floor (Berouti et al., 1979)."""
    spec = stft(noisy)
    power = spec.abs() ** 2
    noise = estimate_noise_psd(power)
    gain = torch.sqrt(torch.clamp(1 - oversubtraction * noise / (power + EPS), min=floor**2))
    return stft.inverse(spec * gain, length=noisy.shape[-1])


def wiener_filter(
    noisy: torch.Tensor, stft: STFT, smoothing: float = 0.98, min_gain: float = 0.05
) -> torch.Tensor:
    """Wiener filter with decision-directed a-priori SNR tracking (Ephraim & Malah, 1984).

    The a-priori SNR mixes the previous frame's enhanced power with the current
    a-posteriori estimate, which removes most of the "musical noise" that plain
    spectral subtraction produces.
    """
    spec = stft(noisy)
    power = spec.abs() ** 2
    noise = estimate_noise_psd(power) + EPS
    posterior = power / noise  # (B, F, T)
    gains = []
    previous = torch.ones_like(posterior[..., 0])  # |S_hat|^2 / noise of the previous frame
    for t in range(posterior.shape[-1]):
        prior = smoothing * previous + (1 - smoothing) * torch.clamp(posterior[..., t] - 1, min=0)
        gain = torch.clamp(prior / (1 + prior), min=min_gain)
        previous = gain**2 * posterior[..., t]
        gains.append(gain)
    return stft.inverse(spec * torch.stack(gains, dim=-1), length=noisy.shape[-1])


BASELINES = {"spectral_subtraction": spectral_subtraction, "wiener": wiener_filter}
