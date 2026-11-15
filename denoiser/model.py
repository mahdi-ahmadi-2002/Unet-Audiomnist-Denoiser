"""Attention U-Net that predicts a time-frequency mask, wrapped as a waveform-to-waveform denoiser."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .audio import EPS, STFT
from .config import Config

COMPRESSION = 0.3  # power-law compression of the input spectrogram


class ResBlock(nn.Module):
    """Two 3x3 conv-BN layers with a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    """Additive attention on a skip connection (Oktay et al., 2018).

    The decoder feature decides which encoder activations are passed across,
    which suppresses noise-only regions that a plain skip would copy through.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        inter = max(channels // 2, 8)
        self.skip = nn.Conv2d(channels, inter, 1)
        self.gate = nn.Conv2d(channels, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return skip * torch.sigmoid(self.psi(F.relu(self.skip(skip) + self.gate(gate))))


class UNet(nn.Module):
    """Fully convolutional U-Net; accepts any (freq, time) size."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_channels: int = 32,
        depth: int = 4,
        attention: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        ch = [base_channels * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(ResBlock(in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(depth))
        self.bottleneck = ResBlock(ch[depth - 1], ch[depth], dropout)
        # Upsample + conv instead of transposed conv: no checkerboard artefacts.
        self.ups = nn.ModuleList(nn.Conv2d(ch[i + 1], ch[i], 3, padding=1) for i in reversed(range(depth)))
        self.gates = nn.ModuleList(
            AttentionGate(ch[i]) if attention else nn.Identity() for i in reversed(range(depth))
        )
        self.decoders = nn.ModuleList(ResBlock(2 * ch[i], ch[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(ch[0], out_ch, 1)
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        multiple = 2**self.depth
        x = F.pad(x, (0, -width % multiple, 0, -height % multiple))
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, gate, decoder in zip(self.ups, self.gates, self.decoders, strict=True):
            skip = skips.pop()
            x = up(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))
            if self.attention:
                skip = gate(skip, x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)[..., :height, :width]


class Denoiser(nn.Module):
    """Waveform in, waveform out.

    STFT -> U-Net -> bounded mask applied to the *noisy* STFT -> inverse STFT.

    * ``magnitude`` mask: real gain in [0, 1]; the noisy phase is reused.
    * ``complex`` mask: the same gain plus a phase rotation in [-pi, pi], i.e. a
      complex ratio mask in polar form (Williamson et al., 2016), so the
      network can also correct the phase. The rotation branch is initialised
      to zero, so training starts from a magnitude mask and learns the phase
      correction as a refinement.

    The input is normalised to unit RMS and the gain is restored afterwards, so
    the output does not depend on recording level.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.complex_mask = cfg.model.mask == "complex"
        self.stft = STFT(cfg.audio.n_fft, cfg.audio.hop_length)
        channels = 2 if self.complex_mask else 1
        self.net = UNet(
            channels,
            channels,
            cfg.model.base_channels,
            cfg.model.depth,
            cfg.model.attention,
            cfg.model.dropout,
        )
        if self.complex_mask:
            nn.init.zeros_(self.net.head.weight[1])
            nn.init.zeros_(self.net.head.bias[1])

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """``noisy``: (batch, samples). Returns the enhanced waveform, same shape."""
        noisy = noisy.float()
        scale = noisy.std(dim=-1, keepdim=True).clamp_min(EPS)
        spec = self.stft(noisy / scale)  # (B, F, T) complex
        mag = spec.abs().clamp_min(EPS)
        if self.complex_mask:
            # compressed complex spectrum: |X|^c * exp(j*phase), as (real, imag) channels
            feats = torch.view_as_real(spec).permute(0, 3, 1, 2) * (mag ** (COMPRESSION - 1)).unsqueeze(1)
            out = self.net(feats).float()
            gain = torch.sigmoid(out[:, 0])
            rotation = math.pi * torch.tanh(out[:, 1])
            mask = torch.polar(gain, rotation)
        else:
            mask = torch.sigmoid(self.net((mag**COMPRESSION).unsqueeze(1)).float()[:, 0])
        return self.stft.inverse(spec * mask, length=noisy.shape[-1]) * scale

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_checkpoint(path: str | Path, model: Denoiser, cfg: Config, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[Denoiser, Config, dict]:
    """Rebuild a trained model from a checkpoint. Returns (model, config, raw checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = Config.from_dict(ckpt["config"])
    model = Denoiser(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg, ckpt
