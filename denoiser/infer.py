"""Denoise audio files of any length and sample rate.

    python -m denoiser.infer noisy.wav -o clean.wav --checkpoint runs/unet_complex/best.pt
    python -m denoiser.infer recordings/ -o cleaned/ --method wiener
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .audio import STFT, chunked_apply, load_audio, save_audio
from .baselines import BASELINES
from .config import AudioConfig
from .model import load_checkpoint

METHODS = ("unet", *BASELINES)


class Enhancer:
    """Callable wrapper around a trained model or a classical baseline."""

    def __init__(self, method: str = "unet", checkpoint: str | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.method = method
        if method == "unet":
            if not checkpoint:
                raise ValueError("method 'unet' needs a checkpoint")
            self.model, cfg, _ = load_checkpoint(checkpoint, self.device)
            self.audio = cfg.audio
        elif method in BASELINES:
            self.audio = AudioConfig()
            self.stft = STFT(self.audio.n_fft, self.audio.hop_length).to(self.device)
        else:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @torch.no_grad()
    def _process(self, wav: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)
        y = self.model(x) if self.method == "unet" else BASELINES[self.method](x, self.stft)
        return y.squeeze(0).cpu().numpy()

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """Denoise a mono float waveform sampled at ``self.sample_rate``."""
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) < self.audio.n_fft:  # too short for one STFT frame
            padded = np.pad(wav, (0, self.audio.n_fft - len(wav)))
            return self._process(padded)[: len(wav)]
        # Chunks of four training segments, cross-faded, bound memory on long files.
        chunk = 4 * self.audio.segment_samples
        return chunked_apply(self._process, wav, chunk, overlap=chunk // 4)

    def file(self, source: str | Path, target: str | Path) -> None:
        save_audio(target, self(load_audio(source, self.sample_rate)), self.sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="audio file, or a folder of .wav files")
    parser.add_argument("-o", "--output", required=True, help="output file (or folder when the input is a folder)")
    parser.add_argument("--checkpoint", help="trained model checkpoint (required for --method unet)")
    parser.add_argument("--method", choices=METHODS, default="unet")
    parser.add_argument("--device", help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    enhancer = Enhancer(args.method, args.checkpoint, args.device)
    source, target = Path(args.input), Path(args.output)
    if source.is_dir():
        files = sorted(source.rglob("*.wav"))
        for path in files:
            enhancer.file(path, target / path.relative_to(source))
        print(f"denoised {len(files)} file(s) -> {target}")
    else:
        enhancer.file(source, target)
        print(f"wrote {target} ({enhancer.sample_rate} Hz)")


if __name__ == "__main__":
    main()
