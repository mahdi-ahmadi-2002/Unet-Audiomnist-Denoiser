"""Shared fixtures: a tiny synthetic corpus laid out like AudioMNIST."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from denoiser.config import Config

SOURCE_SR = 48000


def fake_speech(rng: np.random.Generator, seconds: float = 0.5, sr: int = SOURCE_SR) -> np.ndarray:
    """A voiced-sounding harmonic tone with an amplitude envelope."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = rng.uniform(100, 220)
    wav = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) / k for k in range(1, 12))
    return (0.05 * wav * np.hanning(len(t))).astype(np.float32)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """8 speakers x 2 digits x 2 takes, with pre-mixed '*_noisy.wav' twins and speaker metadata."""
    root = tmp_path_factory.mktemp("audiomnist")
    rng = np.random.default_rng(0)
    meta = {}
    for s in range(1, 9):
        speaker = f"{s:02d}"
        (root / speaker).mkdir()
        meta[speaker] = {
            "gender": "female" if s % 2 else "male",
            "age": 20 + 3 * s,
            "accent": "German",
            "native speaker": "no",
        }
        for digit in range(2):
            for take in range(2):
                wav = fake_speech(rng)
                name = f"{digit}_{speaker}_{take}"
                sf.write(root / speaker / f"{name}.wav", wav, SOURCE_SR, subtype="PCM_16")
                noisy = wav + 0.01 * rng.standard_normal(len(wav)).astype(np.float32)
                sf.write(root / speaker / f"{name}_noisy.wav", noisy, SOURCE_SR, subtype="PCM_16")
    (root / "audioMNIST_meta.txt").write_text(json.dumps(meta))
    return root


@pytest.fixture
def cfg(corpus, tmp_path):
    """A small, CPU-friendly configuration pointing at the synthetic corpus."""
    return Config.load(
        overrides=[
            f"data.root={corpus.as_posix()}",
            "data.val_speakers=2",
            "data.test_speakers=2",
            "data.num_workers=0",
            "model.base_channels=8",
            "model.depth=2",
            "train.batch_size=4",
            "train.epochs=2",
            "train.amp=false",
            f"train.out_dir={(tmp_path / 'run').as_posix()}",
        ]
    )
