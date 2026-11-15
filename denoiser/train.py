"""Train a denoiser.

    python -m denoiser.train --config configs/unet_complex.yaml
    python -m denoiser.train --config configs/unet_complex.yaml --set train.epochs=5 data.root=D:/AudioMNIST/data
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .data import build_datasets, make_loader
from .losses import DenoiseLoss, si_sdr
from .model import Denoiser, save_checkpoint
from .viz import plot_history


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: Denoiser, loader, criterion: DenoiseLoss, device: torch.device) -> dict[str, float]:
    """Mean loss, SI-SDR and SI-SDR improvement over the (deterministic) validation set."""
    model.eval()
    loss_sum, sdr_sum, gain_sum, count = 0.0, 0.0, 0.0, 0
    for batch in loader:
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        estimate = model(noisy)
        sdr = si_sdr(estimate, clean)
        loss_sum += criterion(estimate, clean, noisy).item() * len(noisy)
        sdr_sum += sdr.sum().item()
        gain_sum += (sdr - si_sdr(noisy, clean)).sum().item()
        count += len(noisy)
    return {"val_loss": loss_sum / count, "val_si_sdr": sdr_sum / count, "val_si_sdr_i": gain_sum / count}


def amp_is_safe(model: Denoiser, noisy: torch.Tensor) -> bool:
    """Check that a half-precision forward pass stays finite.

    Some GPUs (notably the GTX 16xx series) return NaN from float16
    convolutions. The check runs on a copy so a bad pass cannot poison the
    batch-norm statistics of the real model.
    """
    probe = copy.deepcopy(model).train()
    with torch.no_grad(), torch.autocast(noisy.device.type):
        return bool(torch.isfinite(probe(noisy)).all())


def train(cfg: Config, resume: str | None = None) -> Path:
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(cfg)
    train_loader = make_loader(datasets["train"], cfg, shuffle=True)
    val_loader = make_loader(datasets["val"], cfg, shuffle=False)
    if len(train_loader) == 0:
        raise ValueError("training set is smaller than one batch; lower train.batch_size")

    model = Denoiser(cfg).to(device)
    criterion = DenoiseLoss(cfg.loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(train_loader)
    warmup_steps = max(cfg.train.warmup_epochs * len(train_loader), 1)

    def lr_factor(step: int) -> float:  # linear warm-up, then cosine decay
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    amp_checked = False
    history: list[dict] = []
    start_epoch, best, stale = 0, -float("inf"), 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, best = state["history"], state["epoch"], state["best"]

    print(
        f"device={device}  parameters={model.num_parameters() / 1e6:.2f}M  "
        f"train={len(datasets['train'])}  val={len(datasets['val'])}  test={len(datasets['test'])} utterances"
    )
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.train.epochs}", unit="batch")
        for batch in progress:
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            if use_amp and not amp_checked:
                use_amp, amp_checked = amp_is_safe(model, noisy), True
                if not use_amp:
                    print("mixed precision gives non-finite values on this GPU; training in full precision")
                    scaler = torch.amp.GradScaler(enabled=False)
            with torch.autocast(device.type, enabled=use_amp):
                estimate = model(noisy)
            loss = criterion(estimate, clean, noisy)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}; lower train.lr")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(noisy)
            seen += len(noisy)
            progress.set_postfix(loss=f"{running / seen:.4f}")

        record = {
            "epoch": epoch + 1,
            "train_loss": running / seen,
            **validate(model, val_loader, criterion, device),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"  train_loss={record['train_loss']:.4f}  val_loss={record['val_loss']:.4f}  "
            f"val SI-SDR={record['val_si_sdr']:.2f} dB  improvement={record['val_si_sdr_i']:+.2f} dB"
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_history(history, out_dir / "history.png")

        improved = record["val_si_sdr_i"] > best
        if improved:
            best, stale = record["val_si_sdr_i"], 0
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch=epoch + 1, val_si_sdr_i=best)
        else:
            stale += 1
        save_checkpoint(
            out_dir / "last.pt",
            model,
            cfg,
            epoch=epoch + 1,
            best=best,
            history=history,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        )
        if stale >= cfg.train.patience:
            print(f"no improvement for {stale} epochs, stopping early")
            break

    print(f"best validation SI-SDR improvement: {best:+.2f} dB -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config file (defaults are used if omitted)")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="config overrides")
    parser.add_argument("--resume", help="path to a last.pt checkpoint to continue from")
    args = parser.parse_args()
    train(Config.load(args.config, args.set), args.resume)


if __name__ == "__main__":
    main()
