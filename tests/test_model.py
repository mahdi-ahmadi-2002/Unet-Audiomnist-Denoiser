import pytest
import torch

from denoiser.config import Config
from denoiser.losses import DenoiseLoss, MultiResolutionSTFTLoss, si_sdr, snr
from denoiser.model import Denoiser, UNet, load_checkpoint, save_checkpoint


def small(mask: str = "complex", attention: bool = True) -> Config:
    return Config.load(
        overrides=[f"model.mask={mask}", f"model.attention={attention}", "model.base_channels=8", "model.depth=3"]
    )


@pytest.mark.parametrize("size", [(257, 126), (64, 64), (33, 17)])
def test_unet_preserves_arbitrary_input_size(size):
    net = UNet(2, 2, base_channels=4, depth=3)
    assert net(torch.randn(1, 2, *size)).shape == (1, 2, *size)


@pytest.mark.parametrize("mask", ["complex", "magnitude"])
@pytest.mark.parametrize("attention", [True, False])
@pytest.mark.parametrize("samples", [16000, 12345])
def test_denoiser_output_matches_input_shape(mask, attention, samples):
    model = Denoiser(small(mask, attention)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, samples) * 0.1)
    assert out.shape == (2, samples)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mask", ["complex", "magnitude"])
def test_denoiser_is_level_invariant(mask):
    """Scaling the input by g scales the output by g: no dependence on recording level."""
    model = Denoiser(small(mask)).eval()
    wav = torch.randn(1, 8000) * 0.1
    with torch.no_grad():
        assert torch.allclose(model(wav * 0.01), model(wav) * 0.01, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("mask", ["complex", "magnitude"])
def test_mask_never_amplifies(mask):
    """Masks are bounded by 1, so no time-frequency bin can come out louder than it went in."""
    model = Denoiser(small(mask)).eval()
    wav = torch.randn(4, 16000)
    with torch.no_grad():
        spec_in, spec_out = model.stft(wav).abs(), model.stft(model(wav)).abs()
    assert spec_out.pow(2).sum() <= spec_in.pow(2).sum()


def test_gradients_reach_every_parameter():
    cfg = small()
    model, criterion = Denoiser(cfg), DenoiseLoss(cfg.loss)
    noisy, clean = torch.randn(2, 8000), torch.randn(2, 8000)
    criterion(model(noisy), clean, noisy).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)
    cfg = small()
    model, criterion = Denoiser(cfg), DenoiseLoss(cfg.loss)
    t = torch.arange(8000) / 16000
    clean = torch.stack([torch.sin(2 * torch.pi * f * t) for f in (220.0, 440.0)]) * 0.1
    noisy = clean + 0.05 * torch.randn_like(clean)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    before = si_sdr(noisy, clean).mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        criterion(model(noisy), clean, noisy).backward()
        optimizer.step()
    with torch.no_grad():  # train mode on purpose: this checks optimisation, not generalisation
        after = si_sdr(model(noisy), clean).mean().item()
    assert after > before + 3


def test_checkpoint_round_trip(tmp_path):
    cfg = small("magnitude")
    model = Denoiser(cfg).eval()
    save_checkpoint(tmp_path / "m.pt", model, cfg, epoch=3)
    loaded, loaded_cfg, raw = load_checkpoint(tmp_path / "m.pt")
    wav = torch.randn(1, 4000)
    with torch.no_grad():
        assert torch.allclose(loaded(wav), model(wav))
    assert loaded_cfg == cfg and raw["epoch"] == 3


def test_si_sdr_is_scale_invariant_and_snr_is_not():
    ref = torch.randn(2, 4000)
    est = ref + 0.1 * torch.randn(2, 4000)
    assert torch.allclose(si_sdr(est, ref), si_sdr(3 * est, ref), atol=1e-3)
    assert not torch.allclose(snr(est, ref), snr(3 * est, ref), atol=1.0)


def test_si_sdr_matches_known_value():
    """Orthogonal noise with 1/100 of the signal power gives exactly 20 dB."""
    t = torch.arange(16000) / 16000
    ref = torch.sin(2 * torch.pi * 100 * t)[None]
    noise = 0.1 * torch.sin(2 * torch.pi * 300 * t)[None]
    assert si_sdr(ref + noise, ref).item() == pytest.approx(20.0, abs=0.01)
    assert snr(ref + noise, ref).item() == pytest.approx(20.0, abs=0.01)


def test_mrstft_is_zero_for_identical_signals_and_positive_otherwise():
    loss = MultiResolutionSTFTLoss()
    wav = torch.randn(2, 8000)
    assert loss(wav, wav).item() == pytest.approx(0, abs=1e-5)
    assert loss(wav, torch.randn(2, 8000)).item() > 0.1


def test_loss_penalises_flipped_polarity():
    """The spectral term cannot see a sign flip; the SNR term must."""
    criterion = DenoiseLoss(small().loss)
    clean = torch.randn(2, 8000)
    noisy = clean + 0.3 * torch.randn(2, 8000)
    assert criterion(-clean, clean, noisy) > criterion(clean, clean, noisy) + 0.1
