"""Typed experiment configuration, loadable from YAML and overridable from the CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

NOISE_TYPES = ("white", "pink", "brown", "hum", "babble")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    segment_seconds: float = 1.0

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate * self.segment_seconds))


@dataclass
class DataConfig:
    root: str = "data/AudioMNIST/data"
    # "synthetic": mix clean speech with generated noise on the fly.
    # "paired":    use pre-mixed "<name>_noisy.wav" files next to the clean ones.
    noise_source: str = "synthetic"
    noise_types: list[str] = field(default_factory=lambda: list(NOISE_TYPES))
    snr_min: float = -5.0
    snr_max: float = 20.0
    val_speakers: int = 6
    test_speakers: int = 6
    split_seed: int = 0
    num_workers: int = 4


@dataclass
class ModelConfig:
    mask: str = "complex"  # "complex" (magnitude + phase) or "magnitude"
    base_channels: int = 32
    depth: int = 4
    attention: bool = True
    dropout: float = 0.0


@dataclass
class LossConfig:
    mrstft: float = 1.0  # multi-resolution STFT loss weight
    snr: float = 0.05  # negative waveform SNR (dB) weight


@dataclass
class TrainConfig:
    out_dir: str = "runs/unet_complex"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    grad_clip: float = 5.0
    amp: bool = True
    patience: int = 8
    seed: int = 0
    limit_train: int = 0  # >0 caps the number of utterances per split (smoke runs)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.model.mask not in ("complex", "magnitude"):
            raise ValueError(f"model.mask must be 'complex' or 'magnitude', got {self.model.mask!r}")
        if self.data.noise_source not in ("synthetic", "paired"):
            raise ValueError(f"data.noise_source must be 'synthetic' or 'paired', got {self.data.noise_source!r}")
        unknown = set(self.data.noise_types) - set(NOISE_TYPES)
        if unknown or not self.data.noise_types:
            raise ValueError(f"data.noise_types must be a non-empty subset of {NOISE_TYPES}")
        if self.data.snr_min > self.data.snr_max:
            raise ValueError("data.snr_min must be <= data.snr_max")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return _build(cls, raw or {}, "")

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        """Read a YAML file (or defaults) and apply ``section.key=value`` overrides."""
        raw: dict[str, Any] = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw).override(overrides)

    def override(self, overrides: list[str] | None) -> Config:
        """Return a copy with ``section.key=value`` items applied."""
        raw = self.to_dict()
        for item in overrides or []:
            key, sep, value = item.partition("=")
            section, dot, name = key.partition(".")
            if not sep or not dot:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            raw.setdefault(section, {})[name] = yaml.safe_load(value)
        return Config.from_dict(raw)


def _build(cls: type, raw: dict[str, Any], prefix: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(prefix + k for k in unknown))}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else known[name].default
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value or {}, f"{prefix}{name}.")
        elif isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
