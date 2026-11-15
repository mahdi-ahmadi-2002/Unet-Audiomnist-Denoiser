"""Interactive demo: upload or record speech, optionally add noise, and compare denoisers.

    pip install gradio
    python app.py --checkpoint runs/unet_complex/best.pt
"""

from __future__ import annotations

import argparse

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from denoiser.audio import make_noise, mix_at_snr, resample
from denoiser.infer import Enhancer

NOISES = ["none", "white", "pink", "brown", "hum"]


def spectrogram_figure(signals: dict[str, np.ndarray], sample_rate: int):
    fig, axes = plt.subplots(1, len(signals), figsize=(4.5 * len(signals), 3), squeeze=False)
    window = torch.hann_window(512)
    specs = {
        name: 20 * np.log10(
            torch.stft(torch.from_numpy(wav).float(), 512, 128, window=window, return_complex=True).abs().numpy() + 1e-8
        )
        for name, wav in signals.items()
    }
    top = max(s.max() for s in specs.values())
    for ax, (name, spec) in zip(axes[0], specs.items(), strict=True):
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=top - 70, vmax=top,
                  extent=(0, signals[name].size / sample_rate, 0, sample_rate / 2000))
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
    fig.tight_layout()
    return fig


def build(enhancers: dict[str, Enhancer]) -> gr.Blocks:
    def run(audio, method, noise, snr_db):
        if audio is None:
            raise gr.Error("Upload or record some audio first.")
        enhancer = enhancers[method]
        sr = enhancer.sample_rate
        in_sr, wav = audio
        wav = wav.astype(np.float32)
        if np.issubdtype(audio[1].dtype, np.integer):
            wav /= np.iinfo(audio[1].dtype).max
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = resample(wav, in_sr, sr)
        if noise != "none":
            wav = mix_at_snr(wav, make_noise(noise, len(wav), sr, np.random.default_rng(0)), snr_db)
        peak = max(float(np.abs(wav).max()), 1e-8)
        wav = wav * min(1.0, 0.99 / peak)
        out = enhancer(wav)
        return (sr, wav), (sr, np.clip(out, -1, 1)), spectrogram_figure({"Input": wav, "Denoised": out}, sr)

    with gr.Blocks(title="Speech denoiser") as demo:
        gr.Markdown("# Speech denoiser\nUpload or record speech. Add noise to a clean recording to test the model.")
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Input audio")
                method = gr.Dropdown(list(enhancers), value=next(iter(enhancers)), label="Method")
                noise = gr.Dropdown(NOISES, value="none", label="Add noise")
                snr = gr.Slider(-5, 20, value=5, step=1, label="SNR of the added noise (dB)")
                button = gr.Button("Denoise", variant="primary")
            with gr.Column():
                noisy_out = gr.Audio(label="Model input")
                clean_out = gr.Audio(label="Denoised")
        plot = gr.Plot(label="Spectrograms")
        button.click(run, [audio, method, noise, snr], [noisy_out, clean_out, plot])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="trained U-Net checkpoint (optional)")
    args = parser.parse_args()
    enhancers = {}
    if args.checkpoint:
        enhancers["U-Net"] = Enhancer("unet", args.checkpoint)
    enhancers["Wiener filter"] = Enhancer("wiener")
    enhancers["Spectral subtraction"] = Enhancer("spectral_subtraction")
    build(enhancers).launch()


if __name__ == "__main__":
    main()
