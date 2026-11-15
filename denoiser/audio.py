"""Audio I/O, STFT, noise synthesis and mixing utilities."""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn

EPS = 1e-8


# --------------------------------------------------------------------------- I/O


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    g = gcd(orig_sr, target_sr)
    return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load a file as mono float32 at ``sample_rate``."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(wav.mean(axis=1), sr, sample_rate)


def save_audio(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sample_rate, subtype="PCM_16")


# -------------------------------------------------------------------------- STFT


class STFT(nn.Module):
    """Hann-window STFT / inverse STFT pair operating on (batch, samples)."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            wav.float(), self.n_fft, self.hop_length, window=self.window, return_complex=True
        )

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(spec, self.n_fft, self.hop_length, window=self.window, length=length)


# ------------------------------------------------------------------------- noise


def colored_noise(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with a 1/f^exponent power spectrum (0 white, 1 pink, 2 brown)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / freqs ** (exponent / 2.0), n=n)


def hum_noise(n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Mains hum: a 50/60 Hz fundamental with decaying harmonics over a faint noise floor."""
    f0 = rng.choice([50.0, 60.0]) * rng.uniform(0.99, 1.01)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    for k in range(1, 8):
        if k * f0 >= sample_rate / 2:
            break
        out += rng.uniform(0.3, 1.0) / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return out + 0.05 * out.std() * rng.standard_normal(n)


def make_noise(kind: str, n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        return colored_noise(n, 0.0, rng)
    if kind == "pink":
        return colored_noise(n, 1.0, rng)
    if kind == "brown":
        return colored_noise(n, 2.0, rng)
    if kind == "hum":
        return hum_noise(n, sample_rate, rng)
    raise ValueError(f"unknown synthetic noise type: {kind!r}")


def mix_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float, clean_power: float | None = None
) -> np.ndarray:
    """Return ``clean + g * noise`` with g chosen so that P_clean / P_noise equals ``snr_db``.

    ``clean_power`` lets the caller pass the power of the un-padded utterance so
    that zero padding does not bias the effective SNR.
    """
    p_clean = float(np.mean(clean**2)) if clean_power is None else clean_power
    p_noise = float(np.mean(noise**2))
    gain = np.sqrt(max(p_clean, EPS**2) / (max(p_noise, EPS**2) * 10 ** (snr_db / 10)))
    return (clean + gain * noise).astype(np.float32)


# ----------------------------------------------------------------- long signals


def chunked_apply(
    fn: Callable[[np.ndarray], np.ndarray], wav: np.ndarray, chunk: int, overlap: int
) -> np.ndarray:
    """Apply ``fn`` to overlapping chunks and cross-fade the results.

    Keeps memory bounded on long recordings and matches the short segments the
    model saw during training. ``fn`` must preserve length.
    """
    n = len(wav)
    if n <= chunk:
        return fn(wav)
    if not 0 < overlap < chunk:
        raise ValueError("overlap must satisfy 0 < overlap < chunk")
    hop = chunk - overlap
    fade = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    for start in range(0, max(n - overlap, 1), hop):
        piece = wav[start : start + chunk]
        win = np.ones(len(piece), dtype=np.float32)
        if start > 0:
            win[:overlap] = fade[: len(piece)]
        if start + chunk < n:
            win[-overlap:] = 1.0 - fade
        out[start : start + len(piece)] += win * fn(piece)
        weight[start : start + len(piece)] += win
    return out / np.maximum(weight, EPS)
