import numpy as np
import pytest
import torch

from denoiser.audio import STFT
from denoiser.baselines import BASELINES, estimate_noise_psd
from denoiser.losses import si_sdr


def test_noise_psd_estimate_is_unbiased_for_white_noise():
    torch.manual_seed(0)
    stft = STFT(512, 128)
    power = stft(torch.randn(1, 160000) * 0.5).abs() ** 2
    ratio = estimate_noise_psd(power)[0, 5:-5, 0] / power[0, 5:-5].mean(-1)  # skip the real-valued edge bins
    assert ratio.mean().item() == pytest.approx(1.0, abs=0.03)  # no systematic bias
    assert 0.75 < ratio.min().item() and ratio.max().item() < 1.25  # per-bin scatter only


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_improves_a_tone_in_white_noise(name):
    torch.manual_seed(0)
    t = torch.arange(16000) / 16000
    envelope = (t > 0.3) & (t < 0.7)  # leave noise-only regions, as in a real utterance
    clean = (0.3 * sum(torch.sin(2 * np.pi * f * t) for f in (200, 400, 600)) * envelope)[None]
    noisy = clean + 0.1 * torch.randn_like(clean)
    out = BASELINES[name](noisy, STFT(512, 128))
    assert out.shape == noisy.shape
    assert (si_sdr(out, clean) - si_sdr(noisy, clean)).item() > 2.0
