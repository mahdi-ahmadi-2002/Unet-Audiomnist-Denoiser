"""Compare trained models and classical baselines on the held-out test speakers.

    python -m denoiser.evaluate --checkpoint runs/unet_complex/best.pt runs/unet_magnitude/best.pt

Writes ``metrics.json``, ``results.md``, figures and listening samples to ``--out``.
Every system sees exactly the same noisy test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import metrics, viz
from .audio import STFT, save_audio
from .baselines import BASELINES
from .config import Config
from .data import build_datasets, load_speaker_metadata, make_loader
from .model import load_checkpoint

System = Callable[[torch.Tensor], torch.Tensor]
REPORTED = ("si_sdr", "si_sdr_i", "snr", "stoi", "pesq")


def snr_bucket(snr_db: float, width: int = 5) -> str:
    low = width * math.floor(snr_db / width)
    return f"{low} to {low + width}"


def age_band(age) -> str:
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "unknown"
    return "under 25" if age < 25 else "25-34" if age < 35 else "35 and over"


def realtime_factor(system: System, seconds: float, sample_rate: int, repeats: int = 5) -> float:
    """Processing time divided by audio duration, on CPU (median of ``repeats`` runs)."""
    wav = torch.randn(1, int(seconds * sample_rate)) * 0.1
    system(wav)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        system(wav)
        times.append(time.perf_counter() - start)
    return float(np.median(times)) / seconds


def mean_table(rows: list[dict], key: Callable[[dict], str], metric: str = "si_sdr_i") -> dict[str, dict[str, float]]:
    """``{group: {system: mean metric}}``."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[key(r)][r["system"]].append(r[metric])
    return {g: {s: float(np.nanmean(v)) for s, v in by_sys.items()} for g, by_sys in acc.items()}


@torch.no_grad()
def evaluate(
    checkpoints: dict[str, str],
    cfg: Config | None,
    overrides: list[str],
    out_dir: Path,
    limit: int = 0,
    with_stoi: bool = True,
    with_pesq: bool = False,
    num_samples: int = 5,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, path in checkpoints.items():
        model, model_cfg, _ = load_checkpoint(path, device)
        models[name] = model
        if cfg is None:  # the first checkpoint defines the test conditions
            cfg = model_cfg
        elif model_cfg.audio != cfg.audio:
            raise ValueError(f"{name} was trained with different audio settings; cannot compare on one test set")
    cfg = (cfg or Config()).override(overrides)
    sr = cfg.audio.sample_rate

    stft = STFT(cfg.audio.n_fft, cfg.audio.hop_length).to(device)
    systems: dict[str, System] = {name: (lambda x, fn=fn: fn(x, stft)) for name, fn in BASELINES.items()}
    systems.update(models)

    test = build_datasets(cfg)["test"]
    if limit:
        test.utterances = test.utterances[:: max(len(test) // limit, 1)][:limit]
    loader = make_loader(test, cfg, shuffle=False)

    rows: list[dict] = []
    examples: list[dict[str, np.ndarray]] = []
    captions: list[str] = []
    seen_kinds: set[str] = set()
    for batch in tqdm(loader, desc="evaluating", unit="batch"):
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        outputs = {"noisy": noisy, **{name: system(noisy) for name, system in systems.items()}}
        input_si_sdr = metrics.si_sdr(noisy, clean)
        clean_np = clean.cpu().numpy()
        for name, estimate in outputs.items():
            si_sdr, snr = metrics.si_sdr(estimate, clean), metrics.snr(estimate, clean)
            estimate_np = estimate.cpu().numpy()
            for i in range(len(noisy)):
                rows.append({
                    "system": name,
                    "path": batch["path"][i],
                    "speaker": batch["speaker"][i],
                    "noise_type": batch["noise_type"][i],
                    "snr_in": float(batch["snr_db"][i]),
                    "si_sdr": si_sdr[i].item(),
                    "si_sdr_i": (si_sdr[i] - input_si_sdr[i]).item(),
                    "snr": snr[i].item(),
                    "stoi": metrics.stoi(estimate_np[i], clean_np[i], sr) if with_stoi else float("nan"),
                    "pesq": metrics.pesq(estimate_np[i], clean_np[i], sr) if with_pesq else float("nan"),
                })
        for i in range(len(noisy)):
            kind = batch["noise_type"][i]
            if len(examples) < num_samples and kind not in seen_kinds:
                seen_kinds.add(kind)
                if seen_kinds >= set(batch["noise_type"]) | set(cfg.data.noise_types) - {"paired"}:
                    seen_kinds.clear()  # every type shown once; allow repeats
                examples.append({**{k: v[i].cpu().numpy() for k, v in outputs.items()}, "clean": clean_np[i]})
                captions.append(f"{batch['noise_type'][i]}, {float(batch['snr_db'][i]):.0f} dB")

    names = list(outputs)
    overall: dict[str, dict] = {}
    for name in names:
        overall[name] = {}
        for metric in REPORTED:
            values = np.array([r[metric] for r in rows if r["system"] == name])
            overall[name][metric] = None if np.all(np.isnan(values)) else float(np.nanmean(values))
    cpu_stft = STFT(cfg.audio.n_fft, cfg.audio.hop_length)
    for name in systems:
        if name in models:
            cpu_system: System = copy.deepcopy(models[name]).cpu()
        else:
            cpu_system = lambda x, fn=BASELINES[name]: fn(x, cpu_stft)  # noqa: E731
        overall[name]["rtf_cpu"] = realtime_factor(cpu_system, cfg.audio.segment_seconds, sr)
        overall[name]["parameters"] = models[name].num_parameters() if name in models else 0

    enhanced = [r for r in rows if r["system"] != "noisy"]
    by_noise = mean_table(enhanced, lambda r: r["noise_type"])
    by_snr = mean_table(enhanced, lambda r: snr_bucket(r["snr_in"]))
    by_snr = dict(sorted(by_snr.items(), key=lambda kv: int(kv[0].split()[0])))

    # Per-speaker-group breakdown for the primary (first) model, or the Wiener filter if none.
    primary = next(iter(models), "wiener")
    meta = load_speaker_metadata(cfg.data.root)
    demographics: dict[str, dict[str, dict]] = {}
    if meta:
        attributes = {
            "gender": lambda m: str(m.get("gender", "unknown")).lower(),
            "native speaker": lambda m: str(m.get("native speaker", "unknown")).lower(),
            "accent": lambda m: str(m.get("accent", "unknown")).lower(),
            "age": lambda m: age_band(m.get("age")),
        }
        for attribute, getter in attributes.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                if r["system"] == primary:
                    groups[getter(meta.get(r["speaker"], {}))].append(r)
            demographics[attribute] = {
                g: {
                    "speakers": len({r["speaker"] for r in rs}),
                    "utterances": len(rs),
                    "si_sdr_i_mean": float(np.mean([r["si_sdr_i"] for r in rs])),
                    "si_sdr_i_std": float(np.std([r["si_sdr_i"] for r in rs])),
                }
                for g, rs in sorted(groups.items())
            }

    result = {
        "test_utterances": len(test),
        "test_speakers": sorted({u.speaker for u in test.utterances}),
        "noise_source": cfg.data.noise_source,
        "sample_rate": sr,
        "checkpoints": checkpoints,
        "overall": overall,
        "si_sdr_i_by_noise_type": by_noise,
        "si_sdr_i_by_input_snr": by_snr,
        "primary_system": primary,
        "demographics": demographics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    plotted = names[1:]
    viz.plot_by_snr(by_snr, plotted, out_dir / "si_sdr_by_snr.png")
    viz.plot_by_group(by_noise, plotted, "Improvement by noise type", out_dir / "si_sdr_by_noise.png")
    if examples:
        columns = ["noisy", "wiener", *models, "clean"]
        viz.plot_spectrograms([{c: e[c] for c in columns} for e in examples], captions, sr,
                              out_dir / "spectrograms.png", cfg.audio.n_fft, cfg.audio.hop_length)
        for i, example in enumerate(examples):
            gain = 0.9 / max(float(np.abs(example["noisy"]).max()), 1e-8)  # one gain per example
            for name, wav in example.items():
                save_audio(out_dir / "samples" / f"{i:02d}_{name}.wav", wav * gain, sr)
    print(render_markdown(result))
    print(f"\nwrote {out_dir.resolve()}")
    return result


def _fmt(value, pattern: str = "{:.2f}") -> str:
    return "n/a" if value is None else pattern.format(value)


def render_markdown(result: dict) -> str:
    overall = result["overall"]
    has_pesq = any(v["pesq"] is not None for v in overall.values())
    lines = [
        f"Test set: {result['test_utterances']} utterances from {len(result['test_speakers'])} held-out speakers, "
        f"{result['noise_source']} noise, {result['sample_rate']} Hz.",
        "",
        "| System | SI-SDR (dB) | SI-SDR improvement (dB) | SNR (dB) | STOI |"
        + (" PESQ |" if has_pesq else "") + " Parameters | Real-time factor (CPU) |",
        "|---|---:|---:|---:|---:|" + ("---:|" if has_pesq else "") + "---:|---:|",
    ]
    for name, m in overall.items():
        params = f"{m['parameters'] / 1e6:.2f} M" if m.get("parameters") else "-"
        lines.append(
            f"| {viz.label(name)} | {_fmt(m['si_sdr'])} | {_fmt(m['si_sdr_i'], '{:+.2f}')} | {_fmt(m['snr'])} | "
            f"{_fmt(m['stoi'], '{:.3f}')} |" + (f" {_fmt(m['pesq'])} |" if has_pesq else "")
            + f" {params} | {_fmt(m.get('rtf_cpu'), '{:.3f}') if 'rtf_cpu' in m else '-'} |"
        )
    for title, table in (
        ("SI-SDR improvement (dB) by noise type", result["si_sdr_i_by_noise_type"]),
        ("SI-SDR improvement (dB) by input SNR (dB)", result["si_sdr_i_by_input_snr"]),
    ):
        systems = [s for s in overall if s != "noisy"]
        lines += ["", f"**{title}**", "", "| | " + " | ".join(viz.label(s) for s in systems) + " |",
                  "|---|" + "---:|" * len(systems)]
        lines += [f"| {g} | " + " | ".join(f"{vals[s]:+.2f}" for s in systems) + " |" for g, vals in table.items()]
    for attribute, groups in result["demographics"].items():
        lines += ["", f"**{viz.label(result['primary_system'])}: SI-SDR improvement by {attribute}**", "",
                  "| Group | Speakers | Utterances | Mean (dB) | Std (dB) |", "|---|---:|---:|---:|---:|"]
        lines += [
            f"| {g} | {v['speakers']} | {v['utterances']} | {v['si_sdr_i_mean']:+.2f} | {v['si_sdr_i_std']:.2f} |"
            for g, v in groups.items()
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="*", default=[], metavar="[NAME=]PATH",
                        help="trained models to compare; the name defaults to the run folder")
    parser.add_argument("--config", help="YAML config (only needed when evaluating baselines without a checkpoint)")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="config overrides")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--limit", type=int, default=0, help="evaluate on at most this many utterances")
    parser.add_argument("--no-stoi", action="store_true", help="skip STOI (faster)")
    parser.add_argument("--pesq", action="store_true", help="also compute PESQ (slow; needs `pip install pesq`)")
    parser.add_argument("--samples", type=int, default=5, help="number of listening examples to export")
    args = parser.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = Path(item).resolve().parent.name, item
        checkpoints[name] = path
    cfg = Config.load(args.config) if args.config or not checkpoints else None
    evaluate(checkpoints, cfg, args.set, Path(args.out), args.limit, not args.no_stoi, args.pesq, args.samples)


if __name__ == "__main__":
    main()
