"""Waveform-domain evaluation metrics.

SI-SDR and SNR live in :mod:`denoiser.losses` (they are differentiable and
shared with training). STOI and PESQ come from optional packages and return
NaN when they cannot be computed, so callers should aggregate with nanmean.
"""

from __future__ import annotations

import warnings

import numpy as np

from .losses import si_sdr, snr

__all__ = ["si_sdr", "snr", "stoi", "pesq"]


def stoi(estimate: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    """Short-time objective intelligibility in [0, 1] (Taal et al., 2011).

    Needs about 0.4 s of active speech; shorter utterances return NaN.
    """
    try:
        from pystoi import stoi as _stoi
    except ImportError:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(_stoi(reference, estimate, sample_rate, extended=False))
    return value if value > 1e-4 else float("nan")  # pystoi returns 1e-5 when it has too few frames


def pesq(estimate: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    """Wide-band PESQ (ITU-T P.862.2). Requires the optional ``pesq`` package and 16 kHz audio."""
    try:
        from pesq import pesq as _pesq
    except ImportError:
        return float("nan")
    if sample_rate != 16000:
        return float("nan")
    try:
        return float(_pesq(sample_rate, reference, estimate, "wb"))
    except Exception:  # pesq raises on clips with no detectable utterance
        return float("nan")
