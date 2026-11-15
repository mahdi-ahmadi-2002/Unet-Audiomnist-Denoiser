"""End-to-end: train a tiny model, evaluate it against the baselines, run inference."""

import json

import numpy as np
import pytest

from denoiser.audio import load_audio, save_audio
from denoiser.config import Config
from denoiser.evaluate import age_band, evaluate, snr_bucket
from denoiser.infer import Enhancer
from denoiser.train import train


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.load(overrides=["train.epoch=3"])
    with pytest.raises(ValueError, match="model.mask"):
        Config.load(overrides=["model.mask=mel"])
    with pytest.raises(ValueError, match="section.key=value"):
        Config.load(overrides=["epochs=3"])


def test_config_yaml_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epochs: 7\n  lr: 1\nmodel:\n  mask: magnitude\n")
    cfg = Config.load(path, ["train.batch_size=2"])
    assert (cfg.train.epochs, cfg.train.batch_size, cfg.model.mask) == (7, 2, "magnitude")
    assert isinstance(cfg.train.lr, float)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_bucketing_helpers():
    assert snr_bucket(-0.1) == "-5 to 0" and snr_bucket(0.0) == "0 to 5" and snr_bucket(19.9) == "15 to 20"
    assert [age_band(a) for a in (22, "30", 41, None)] == ["under 25", "25-34", "35 and over", "unknown"]


def test_train_evaluate_infer(cfg, tmp_path):
    best = train(cfg)
    history = json.loads((best.parent / "history.json").read_text())
    assert len(history) == 2 and all(np.isfinite(h["val_loss"]) for h in history)

    # resuming an already-finished run is a no-op that keeps the history
    train(cfg, resume=str(best.parent / "last.pt"))
    assert len(json.loads((best.parent / "history.json").read_text())) == 2

    out = tmp_path / "results"
    result = evaluate({"unet": str(best)}, None, [], out, num_samples=2)
    assert list(result["overall"]) == ["noisy", "spectral_subtraction", "wiener", "unet"]
    assert result["overall"]["noisy"]["si_sdr_i"] == pytest.approx(0, abs=1e-4)
    assert result["test_utterances"] == 8 and len(result["test_speakers"]) == 2
    assert result["overall"]["unet"]["parameters"] > 0 and result["overall"]["unet"]["rtf_cpu"] > 0
    assert set(result["demographics"]) == {"gender", "native speaker", "accent", "age"}
    for name in ("metrics.json", "results.md", "si_sdr_by_snr.png", "si_sdr_by_noise.png", "spectrograms.png"):
        assert (out / name).stat().st_size > 0
    assert len(list((out / "samples").glob("*.wav"))) == 2 * 5

    # inference on a file longer than one chunk, at a different sample rate
    rng = np.random.default_rng(0)
    save_audio(tmp_path / "long.wav", 0.05 * rng.standard_normal(5 * 22050).astype(np.float32), 22050)
    for enhancer in (Enhancer("unet", str(best)), Enhancer("wiener")):
        enhancer.file(tmp_path / "long.wav", tmp_path / "out.wav")
        wav = load_audio(tmp_path / "out.wav", 16000)
        assert len(wav) == 5 * 16000 and np.isfinite(wav).all()


def test_baselines_can_be_evaluated_without_a_checkpoint(cfg, tmp_path):
    result = evaluate({}, cfg, [], tmp_path / "r", with_stoi=False, num_samples=0)
    assert list(result["overall"]) == ["noisy", "spectral_subtraction", "wiener"]
    assert result["primary_system"] == "wiener"
