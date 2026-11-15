"""AudioMNIST discovery, speaker-independent splits and the noisy/clean dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import EPS, load_audio, make_noise, mix_at_snr
from .config import Config

NOISY_SUFFIX = "_noisy"


@dataclass(frozen=True)
class Utterance:
    path: str
    digit: int
    speaker: str
    noisy_path: str | None = None


def scan(root: str | Path) -> list[Utterance]:
    """Find ``<digit>_<speaker>_<take>.wav`` files (and optional ``*_noisy.wav`` twins)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset folder not found: {root}. See the README section 'Data' for how to get AudioMNIST."
        )
    wavs = sorted(root.rglob("*.wav"))
    noisy = {p for p in wavs if p.stem.endswith(NOISY_SUFFIX)}
    utterances = []
    for p in wavs:
        parts = p.stem.split("_")
        if p in noisy or len(parts) != 3 or not parts[0].isdigit():
            continue
        twin = p.with_name(p.stem + NOISY_SUFFIX + ".wav")
        utterances.append(Utterance(str(p), int(parts[0]), parts[1], str(twin) if twin in noisy else None))
    if not utterances:
        raise FileNotFoundError(f"no AudioMNIST-style wav files under {root}")
    return utterances


def split_by_speaker(
    utterances: list[Utterance], val_speakers: int, test_speakers: int, seed: int
) -> dict[str, list[Utterance]]:
    """Split so that no speaker appears in more than one of train / val / test."""
    speakers = sorted({u.speaker for u in utterances})
    if len(speakers) < val_speakers + test_speakers + 1:
        raise ValueError(f"need more than {val_speakers + test_speakers} speakers, found {len(speakers)}")
    order = np.random.default_rng(seed).permutation(len(speakers))
    shuffled = [speakers[i] for i in order]
    groups = {
        "test": set(shuffled[:test_speakers]),
        "val": set(shuffled[test_speakers : test_speakers + val_speakers]),
    }
    split: dict[str, list[Utterance]] = {"train": [], "val": [], "test": []}
    for u in utterances:
        name = next((k for k, members in groups.items() if u.speaker in members), "train")
        split[name].append(u)
    return split


def load_speaker_metadata(root: str | Path) -> dict[str, dict]:
    """Read ``audioMNIST_meta.txt`` (speaker id -> gender, age, accent, ...) if present."""
    for folder in (Path(root), Path(root).parent):
        meta = folder / "audioMNIST_meta.txt"
        if meta.is_file():
            return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def _placement(length: int, target: int, rng: np.random.Generator | None) -> tuple[int, int, int]:
    """Where to copy a ``length``-sample signal inside a ``target``-sample segment.

    Returns (source start, destination start, count). Random when ``rng`` is
    given (training), centred otherwise (evaluation).
    """
    slack = abs(length - target)
    offset = int(rng.integers(0, slack + 1)) if rng is not None else slack // 2
    return (offset, 0, target) if length >= target else (0, offset, length)


def _place(wav: np.ndarray, target: int, where: tuple[int, int, int]) -> np.ndarray:
    src, dst, count = where
    out = np.zeros(target, dtype=np.float32)
    piece = wav[src : src + count]
    out[dst : dst + len(piece)] = piece
    return out


class NoisyDigits(Dataset):
    """Fixed-length (noisy, clean) waveform pairs.

    Training draws a fresh noise type, SNR and time offset on every access.
    Evaluation derives them from ``(seed, index)`` so the val/test sets are
    identical across runs and across the systems being compared.
    """

    def __init__(self, utterances: list[Utterance], cfg: Config, train: bool, seed: int = 0) -> None:
        if cfg.data.noise_source == "paired":
            utterances = [u for u in utterances if u.noisy_path]
            if not utterances:
                raise ValueError("noise_source='paired' but no '*_noisy.wav' files were found")
        self.utterances = utterances
        self.cfg = cfg
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.utterances)

    def _babble(self, n: int, exclude: str, rng: np.random.Generator) -> np.ndarray:
        """Overlapping speech from several other speakers."""
        sr = self.cfg.audio.sample_rate
        out = np.zeros(n, dtype=np.float32)
        voices = int(rng.integers(3, 7))
        for _ in range(50):
            other = self.utterances[int(rng.integers(len(self.utterances)))]
            if other.speaker == exclude:
                continue
            wav = load_audio(other.path, sr)
            out += _place(wav / (wav.std() + EPS), n, _placement(len(wav), n, rng))
            voices -= 1
            if voices == 0:
                break
        return out if out.any() else make_noise("pink", n, sr, rng).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        cfg, utt = self.cfg, self.utterances[idx]
        sr, n = cfg.audio.sample_rate, cfg.audio.segment_samples
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        else:
            rng = np.random.default_rng([self.seed, idx])

        speech = load_audio(utt.path, sr)
        where = _placement(len(speech), n, rng if self.train else None)
        clean = _place(speech, n, where)

        if cfg.data.noise_source == "paired":
            noisy = _place(load_audio(utt.noisy_path, sr), n, where)
            kind = "paired"
            residual = float(np.sum((noisy - clean) ** 2))
            snr_db = 10 * np.log10((float(np.sum(clean**2)) + EPS) / (residual + EPS))
        else:
            kind = str(rng.choice(cfg.data.noise_types))
            snr_db = float(rng.uniform(cfg.data.snr_min, cfg.data.snr_max))
            noise = self._babble(n, utt.speaker, rng) if kind == "babble" else make_noise(kind, n, sr, rng)
            noisy = mix_at_snr(clean, noise, snr_db, clean_power=float(np.mean(speech**2)))

        peak = float(np.abs(noisy).max())
        if peak > 0.99:  # keep the mixture inside [-1, 1] without changing the SNR
            clean, noisy = clean * (0.99 / peak), noisy * (0.99 / peak)

        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "snr_db": float(snr_db),
            "noise_type": kind,
            "speaker": utt.speaker,
            "digit": utt.digit,
            "path": utt.path,
        }


def build_datasets(cfg: Config) -> dict[str, NoisyDigits]:
    split = split_by_speaker(
        scan(cfg.data.root), cfg.data.val_speakers, cfg.data.test_speakers, cfg.data.split_seed
    )
    limit = cfg.train.limit_train
    if limit > 0:
        split = {name: utts[:: max(len(utts) // limit, 1)][:limit] for name, utts in split.items()}
    return {
        "train": NoisyDigits(split["train"], cfg, train=True),
        "val": NoisyDigits(split["val"], cfg, train=False, seed=1),
        "test": NoisyDigits(split["test"], cfg, train=False, seed=2),
    }


def make_loader(dataset: NoisyDigits, cfg: Config, shuffle: bool) -> DataLoader:
    workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
