import numpy as np
import pytest
import torch

from denoiser.audio import STFT, chunked_apply, load_audio, make_noise, mix_at_snr, resample, save_audio


def test_stft_round_trip_is_lossless():
    stft = STFT(512, 128)
    wav = torch.randn(3, 16000)
    assert torch.allclose(stft.inverse(stft(wav), length=16000), wav, atol=1e-4)


def test_stft_shape():
    assert STFT(512, 128)(torch.randn(2, 16000)).shape == (2, 257, 126)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 12.5])
def test_mix_at_snr_hits_the_requested_snr(snr_db):
    rng = np.random.default_rng(0)
    clean, noise = rng.standard_normal(16000), rng.standard_normal(16000) * 7
    noisy = mix_at_snr(clean, noise, snr_db)
    measured = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    assert measured == pytest.approx(snr_db, abs=1e-3)


def test_colored_noise_slopes():
    """Pink loses ~3 dB and brown ~6 dB of power per octave; white is flat."""
    rng = np.random.default_rng(0)

    def octave_drop(kind):
        power = np.abs(np.fft.rfft(make_noise(kind, 2**16, 16000, rng))) ** 2
        low, high = power[500:1000].mean(), power[4000:8000].mean()  # three octaves apart
        return 10 * np.log10(low / high) / 3

    assert octave_drop("white") == pytest.approx(0, abs=0.5)
    assert octave_drop("pink") == pytest.approx(3, abs=0.5)
    assert octave_drop("brown") == pytest.approx(6, abs=0.5)


def test_hum_energy_sits_at_low_frequencies():
    wav = make_noise("hum", 16000, 16000, np.random.default_rng(1))
    power = np.abs(np.fft.rfft(wav)) ** 2  # 1 Hz resolution
    assert power[:500].sum() / power.sum() > 0.9
    assert 45 <= int(power.argmax()) <= 65  # the 50/60 Hz fundamental dominates


def test_unknown_noise_type_is_rejected():
    with pytest.raises(ValueError):
        make_noise("purple", 100, 16000, np.random.default_rng(0))


def test_resample_preserves_a_tone():
    t = np.arange(48000) / 48000
    out = resample(np.sin(2 * np.pi * 440 * t).astype(np.float32), 48000, 16000)
    assert len(out) == 16000
    assert int(np.abs(np.fft.rfft(out)).argmax()) == 440


def test_save_and_load_round_trip(tmp_path):
    wav = (0.3 * np.sin(np.arange(16000) / 10)).astype(np.float32)
    save_audio(tmp_path / "a" / "x.wav", wav, 16000)
    assert np.allclose(load_audio(tmp_path / "a" / "x.wav", 16000), wav, atol=1e-4)


@pytest.mark.parametrize("n", [1000, 4000, 4001, 10_000, 12_345])
def test_chunked_apply_with_identity_reconstructs_input(n):
    wav = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    out = chunked_apply(lambda x: x, wav, chunk=4000, overlap=1000)
    assert out.shape == wav.shape
    assert np.allclose(out, wav, atol=1e-5)
