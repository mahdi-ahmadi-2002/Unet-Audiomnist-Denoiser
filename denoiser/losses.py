"""Training objectives and differentiable metrics: SNR, SI-SDR, multi-resolution STFT loss."""

from __future__ import annotations

import torch
from torch import nn

from .audio import EPS
from .config import LossConfig


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale-invariant signal-to-distortion ratio in dB, per item (Le Roux et al., 2019)."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    alpha = (estimate * reference).sum(-1, keepdim=True) / (reference.pow(2).sum(-1, keepdim=True) + EPS)
    target = alpha * reference
    distortion = estimate - target
    return 10 * torch.log10((target.pow(2).sum(-1) + EPS) / (distortion.pow(2).sum(-1) + EPS))


def snr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Plain waveform SNR in dB, per item."""
    return 10 * torch.log10(
        (reference.pow(2).sum(-1) + EPS) / ((reference - estimate).pow(2).sum(-1) + EPS)
    )


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 at several STFT resolutions (Yamamoto et al., 2020).

    Comparing spectra at more than one window length penalises both smeared
    transients and smeared harmonics, which a single resolution trades off.
    """

    def __init__(self, fft_sizes: tuple[int, ...] = (256, 512, 1024)) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        for n_fft in fft_sizes:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def _magnitude(self, wav: torch.Tensor, n_fft: int) -> torch.Tensor:
        spec = torch.stft(
            wav, n_fft, n_fft // 4, window=getattr(self, f"window_{n_fft}"), return_complex=True
        )
        return spec.abs().clamp_min(1e-5)

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        total = estimate.new_zeros(())
        for n_fft in self.fft_sizes:
            est, ref = self._magnitude(estimate, n_fft), self._magnitude(reference, n_fft)
            convergence = torch.linalg.norm(ref - est, dim=(1, 2)) / torch.linalg.norm(ref, dim=(1, 2))
            total = total + convergence.mean() + (est.log() - ref.log()).abs().mean()
        return total / len(self.fft_sizes)


class DenoiseLoss(nn.Module):
    """Weighted sum of the multi-resolution STFT loss and negative waveform SNR.

    The spectral term is phase-blind, so on its own a complex mask is free to
    flip polarity or drift in phase. Plain SNR (unlike SI-SDR, which ignores
    sign and scale) ties the output to the reference waveform.

    Signals are divided by the noisy input's RMS first so the loss is
    independent of recording level.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mrstft = MultiResolutionSTFTLoss()

    def forward(self, estimate: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        scale = noisy.std(dim=-1, keepdim=True).clamp_min(EPS)
        estimate, clean = estimate.float() / scale, clean.float() / scale
        loss = estimate.new_zeros(())
        if self.cfg.mrstft:
            loss = loss + self.cfg.mrstft * self.mrstft(estimate, clean)
        if self.cfg.snr:
            loss = loss - self.cfg.snr * snr(estimate, clean).mean()
        return loss
