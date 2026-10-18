"""Optional vocal isolation for the restoration pipeline.

The core pipeline is deliberately DSP-only.  When the user asks to attenuate or
remove vocals, this module uses a pretrained Demucs model (``htdemucs``) to
separate the mixture into drums/bass/other/vocals and subtracts the requested
amount of the vocal stem (130 % for ``remove``, which also cancels the residual
voice left in the other stems).  Demucs is an optional dependency: when it is not
installed (or the model cannot be loaded) stereo files fall back to a
conservative mid/side centre-channel reduction, and mono files are returned
unchanged with an explicit reason in the report.

The public entry point is :func:`suppress_vocals`.  It never raises on a
missing optional dependency; callers get the original audio plus a report that
says what happened.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

from . import dsp

# Vocal-stem subtraction factor for each UI mode.  "remove" is deliberately
# above 1.0: Demucs leaves a small amount of the voice in the non-vocal stems,
# so a plain 1.0 subtraction still leaves audible syllables.  The extra 30 %
# removes that residual; on instrumental passages the vocal stem is more than
# 20 dB below the mix, so the over-subtraction is inaudible.
AMOUNTS: dict[str, float] = {
    "keep": 0.0,
    "reduce": 0.55,
    "remove": 1.30,
}

MODEL_NAME = "htdemucs"

_MODEL: Any = None
_MODEL_ERROR: str | None = None
# Loading the model and running inference are serialized: the Flask demo is
# threaded and a single PyTorch module is not safe to call concurrently.
_MODEL_LOCK = threading.Lock()


def amount_for(mode: str) -> float:
    """Return the vocal-stem subtraction amount for ``mode`` (0 for keep)."""
    return float(AMOUNTS.get(str(mode).lower(), 0.0))


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Polyphase resampling along the time axis, used to match Demucs' 44.1 kHz."""
    if int(sr) == int(target):
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(int(sr), int(target))
    return signal.resample_poly(audio, int(target) // divisor,
                                int(sr) // divisor, axis=0)


def _load_model() -> Any:
    """Load and cache ``htdemucs``; raise if Demucs/Torch are unavailable."""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        raise RuntimeError(_MODEL_ERROR)
    try:
        from demucs.pretrained import get_model
    except Exception as exc:  # ImportError, TorchCodec issues, ...
        _MODEL_ERROR = f"{exc.__class__.__name__}: {exc}"
        raise RuntimeError(_MODEL_ERROR) from exc
    try:
        model = get_model(MODEL_NAME)
        model.eval()
    except Exception as exc:  # model download/loading failure
        _MODEL_ERROR = f"{exc.__class__.__name__}: {exc}"
        raise RuntimeError(_MODEL_ERROR) from exc
    _MODEL = model
    return model


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad a resampled signal to the exact input length."""
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    pad = length - len(audio)
    if audio.ndim == 1:
        return np.pad(audio, (0, pad))
    return np.pad(audio, ((0, pad), (0, 0)))


def _demucs_suppress(audio: np.ndarray, sr: int, amount: float
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract ``amount`` of the Demucs vocal stem.  May raise."""
    import torch
    from demucs.apply import apply_model

    model = _load_model()
    target_sr = int(model.samplerate)
    channels = int(audio.shape[1])

    if channels == 1:
        analysis = audio[:, 0]
    elif channels == 2:
        analysis = audio
    else:
        # Multichannel layouts are rare here: separate the mono downmix and
        # subtract the resulting vocal estimate from every channel.
        analysis = audio.mean(axis=1)

    analysis = _resample(np.asarray(analysis, dtype=np.float64), sr, target_sr)
    if analysis.ndim == 1:
        stereo = np.repeat(analysis[:, None], 2, axis=1)
    else:
        stereo = analysis
        if stereo.shape[1] == 1:
            stereo = np.repeat(stereo, 2, axis=1)
        elif stereo.shape[1] > 2:
            stereo = np.repeat(stereo.mean(axis=1, keepdims=True), 2, axis=1)

    # Demucs normalizes by the standard deviation of the mono downmix, not by
    # the per-channel means (which are almost equal on a centred recording).
    reference = stereo.mean(axis=1)
    scale = float(reference.std())
    if not np.isfinite(scale) or scale < 1e-8:
        return audio.copy(), {
            "applied": False,
            "backend": "demucs",
            "reason": "silent_input",
        }

    mix = (stereo - reference.mean()) / scale
    tensor = torch.from_numpy(np.ascontiguousarray(mix.T)[None]).float()
    with _MODEL_LOCK:
        sources = apply_model(
            model, tensor, device="cpu", shifts=0, split=True,
            overlap=0.25, progress=False,
        )[0].cpu().numpy()
    sources = sources * scale + reference.mean()

    names = list(getattr(model, "sources", []))
    if "vocals" in names:
        vocal_index = names.index("vocals")
    else:  # defensive: htdemucs always has a vocals stem
        vocal_index = len(sources) - 1

    # Demucs returns (sources, channels, samples); the rest of this project
    # uses (samples, channels).
    vocals = np.asarray(sources[vocal_index])
    if vocals.ndim == 2:
        vocals = vocals.T
    if channels == 1:
        vocal_mono = vocals.mean(axis=1) if vocals.ndim > 1 else vocals
        vocal_mono = _resample(vocal_mono, target_sr, sr)
        vocal_mono = _match_length(vocal_mono, len(audio))
        restored = audio - amount * vocal_mono[:, None]
    elif channels == 2:
        restored = stereo - amount * vocals
        restored = _resample(restored, target_sr, sr)
        restored = _match_length(restored, len(audio))
        if restored.ndim == 1:
            restored = restored[:, None]
    else:
        vocal_mono = vocals.mean(axis=1) if vocals.ndim > 1 else vocals
        vocal_mono = _resample(vocal_mono, target_sr, sr)
        vocal_mono = _match_length(vocal_mono, len(audio))
        restored = audio - amount * vocal_mono[:, None]

    stats = {
        "applied": True,
        "backend": "demucs",
        "model": MODEL_NAME,
        "amount": amount,
        "vocals_rms_db": dsp.to_db(dsp.rms(vocals)),
        "sources": names,
    }
    return np.asarray(restored, dtype=np.float64), stats


def _mid_side_suppress(audio: np.ndarray, sr: int, amount: float
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Fallback for stereo files: attenuate centre-panned vocal-band content.

    This is the classic karaoke method.  It works when the voice is panned to
    the centre, but it also touches centre-panned instruments, so it is only a
    fallback for machines without Demucs.
    """
    if audio.shape[1] < 2:
        return audio.copy(), {
            "applied": False,
            "backend": "mid/side",
            "reason": "mono_without_demucs",
        }

    n_fft, hop = dsp.stft_params(sr)
    left = audio[:, 0]
    right = audio[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    stft_l = librosa.stft(left, n_fft=n_fft, hop_length=hop)
    stft_r = librosa.stft(right, n_fft=n_fft, hop_length=hop)
    stft_mid = librosa.stft(mid, n_fft=n_fft, hop_length=hop)
    stft_side = librosa.stft(side, n_fft=n_fft, hop_length=hop)

    # Coherence is ~1 for a perfectly centre-panned component and ~0 for
    # uncorrelated content.  Smoothing avoids isolated spectral holes.
    cross = 2.0 * np.abs(stft_l * np.conj(stft_r))
    energy = np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2
    coherence = cross / (energy + dsp.EPS)
    coherence = uniform_filter(coherence, size=(3, 3), mode="nearest")

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (dsp._smoothstep(freqs, 140.0, 260.0)
            * (1.0 - dsp._smoothstep(freqs, 6500.0, 8500.0)))
    gain = np.clip(float(amount) * coherence * band[:, None], 0.0, 1.0)

    new_mid = stft_mid * (1.0 - gain)
    new_left = librosa.istft(new_mid + stft_side, hop_length=hop, length=len(audio))
    new_right = librosa.istft(new_mid - stft_side, hop_length=hop, length=len(audio))
    restored = audio.copy()
    restored[:, 0] = new_left
    restored[:, 1] = new_right
    return restored, {
        "applied": True,
        "backend": "mid/side",
        "amount": float(amount),
        "band_hz": [140.0, 8500.0],
    }


def suppress_vocals(audio: np.ndarray, sr: int, mode: str = "keep",
                    backend: str = "auto") -> tuple[np.ndarray, dict[str, Any]]:
    """Keep, attenuate or remove vocals.

    Parameters
    ----------
    audio : array_like
        Mono or multichannel signal in ``(samples, channels)`` layout.
    sr : int
        Sample rate of ``audio``.
    mode : {"keep", "reduce", "remove"}
        ``keep`` is a no-op, ``reduce`` subtracts 55 % of the vocal stem and
        ``remove`` subtracts 130 % of it to cancel the residual voice that
        Demucs leaves in the other stems.
    backend : {"auto", "demucs", "mid_side"}
        ``auto`` tries Demucs and falls back to mid/side.  ``demucs`` reports
        an error instead of falling back; ``mid_side`` skips Demucs entirely.
    """
    x = dsp.as_2d(audio).copy()
    amount = amount_for(mode)
    started = time.perf_counter()
    stats: dict[str, Any] = {
        "mode": str(mode).lower(),
        "amount": amount,
        "applied": False,
        "backend": None,
    }
    if amount <= 0.0 or len(x) == 0:
        stats["reason"] = "not_requested"
        stats["elapsed_s"] = round(time.perf_counter() - started, 3)
        return x, stats

    if backend in {"auto", "demucs"}:
        try:
            restored, info = _demucs_suppress(x, sr, amount)
            stats.update(info)
            stats["elapsed_s"] = round(time.perf_counter() - started, 3)
            return restored, stats
        except Exception as exc:  # optional dependency / model failure
            stats["demucs_error"] = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            if backend == "demucs":
                stats["reason"] = "demucs_unavailable"
                stats["elapsed_s"] = round(time.perf_counter() - started, 3)
                return x, stats

    restored, info = _mid_side_suppress(x, sr, amount)
    stats.update(info)
    stats["elapsed_s"] = round(time.perf_counter() - started, 3)
    return restored, stats
