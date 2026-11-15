"""Result figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

# Categorical palette, assigned to systems in a fixed order (colour-blind validated).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

LABELS = {
    "spectral_subtraction": "Spectral subtraction",
    "wiener": "Wiener filter",
    "noisy": "Noisy input",
    "clean": "Clean target",
    "unet_complex": "U-Net (complex mask)",
    "unet_magnitude": "U-Net (magnitude mask)",
    "unet_no_attention": "U-Net (no attention)",
}


def label(system: str) -> str:
    return LABELS.get(system, system)


def system_colors(systems: list[str]) -> dict[str, str]:
    if len(systems) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} systems can be plotted together")
    return dict(zip(systems, PALETTE, strict=False))


def _style(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def _save(fig, path: str | Path, dpi: int = 160) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_by_snr(table: dict[str, dict[str, float]], systems: list[str], path: str | Path) -> None:
    """SI-SDR improvement of each system against the input SNR bucket.

    ``table[bucket][system]`` is the mean improvement in dB.
    """
    colors = system_colors(systems)
    buckets = list(table)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for system in systems:
        values = [table[b].get(system, np.nan) for b in buckets]
        ax.plot(buckets, values, color=colors[system], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label(system))
    ax.axhline(0, color=AXIS, linewidth=1)
    _style(ax, "SI-SDR improvement (dB)", "Improvement by input SNR")
    ax.set_xlabel("Input SNR (dB)", color=MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def plot_by_group(table: dict[str, dict[str, float]], systems: list[str], title: str, path: str | Path) -> None:
    """Grouped bars: one cluster per group (e.g. noise type), one bar per system."""
    colors = system_colors(systems)
    groups = list(table)
    width = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(groups) + 2), 4.2), facecolor=SURFACE)
    for i, system in enumerate(systems):
        x = np.arange(len(groups)) + (i - (len(systems) - 1) / 2) * width
        values = [table[g].get(system, np.nan) for g in groups]
        ax.bar(x, values, width * 0.9, color=colors[system], label=label(system))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(np.arange(len(groups)), groups)
    _style(ax, "SI-SDR improvement (dB)", title)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=38)
    ax.legend(frameon=False, labelcolor=INK, ncols=min(len(systems), 4), loc="lower left", bbox_to_anchor=(-0.01, 1.0))
    _save(fig, path)


def plot_history(history: list[dict], path: str | Path) -> None:
    """Training curves: loss and validation SI-SDR improvement on separate axes."""
    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    left.plot(epochs, [h["train_loss"] for h in history], color=PALETTE[0], linewidth=2, label="Train")
    left.plot(epochs, [h["val_loss"] for h in history], color=PALETTE[1], linewidth=2, label="Validation")
    _style(left, "Loss", "Loss")
    left.legend(frameon=False, labelcolor=INK)
    right.plot(epochs, [h["val_si_sdr_i"] for h in history], color=PALETTE[0], linewidth=2)
    _style(right, "SI-SDR improvement (dB)", "Validation SI-SDR improvement")
    for ax in (left, right):
        ax.set_xlabel("Epoch", color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, path)


def plot_spectrograms(
    examples: list[dict[str, np.ndarray]], captions: list[str], sample_rate: int, path: str | Path,
    n_fft: int = 512, hop_length: int = 128,
) -> None:
    """One row per example, one log-magnitude spectrogram per signal (same colour scale per row)."""
    columns = list(examples[0])
    fig, axes = plt.subplots(len(examples), len(columns), squeeze=False, facecolor=SURFACE,
                             figsize=(3.0 * len(columns), 2.4 * len(examples)))
    window = torch.hann_window(n_fft)
    for row, (example, caption) in enumerate(zip(examples, captions, strict=True)):
        specs = {
            name: 20 * np.log10(
                torch.stft(torch.from_numpy(wav).float(), n_fft, hop_length, window=window, return_complex=True)
                .abs().numpy() + 1e-8
            )
            for name, wav in example.items()
        }
        top = max(s.max() for s in specs.values())
        for col, name in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(specs[name], origin="lower", aspect="auto", cmap="magma", vmin=top - 70, vmax=top,
                      extent=(0, example[name].size / sample_rate, 0, sample_rate / 2000))
            ax.tick_params(colors=MUTED, labelsize=7, length=2)
            if row == 0:
                ax.set_title(label(name), fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"{caption}\nkHz", fontsize=8, color=MUTED)
            else:
                ax.set_yticklabels([])
            if row == len(examples) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=MUTED)
            else:
                ax.set_xticklabels([])
    fig.tight_layout()
    _save(fig, path, dpi=100)
