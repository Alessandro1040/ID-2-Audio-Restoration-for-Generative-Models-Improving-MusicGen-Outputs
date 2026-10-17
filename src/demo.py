"""Synthetic demo material for the restoration pipeline.

The generator creates a short, deterministic chord progression with bass and
light percussion.  It is then degraded in the same way generative models tend
to fail: band limitation, broadband hiss, low-bit quantization and clipping.
Because the clean signal is known, reference metrics (SI-SDR, LSD) can be
computed without downloading any audio.

The sounds are intentionally simple; they are a reproducible smoke test, not a
replacement for a real MusicGen benchmark.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SR = 32000


def synth_music(sr: int = SR, duration: float = 12.0, seed: int = 7) -> np.ndarray:
    """Return a short mono "musical" signal with chords, bass and percussion."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    x = np.zeros(n)
    roots = [130.81, 110.00, 87.31, 98.00]       # C3, A2, F2, G2
    thirds = [164.81, 130.81, 110.00, 123.47]
    fifths = [196.00, 164.81, 130.81, 146.83]
    segment = duration / len(roots)

    for i, (root, third, fifth) in enumerate(zip(roots, thirds, fifths)):
        start = i * segment
        for freq in (root, third, fifth):
            for harmonic, amplitude in enumerate((1.0, 0.45, 0.22, 0.10, 0.05), start=1):
                if freq * harmonic > sr / 2:
                    continue
                local = np.arange(int(segment * sr)) / sr
                envelope = np.exp(-local * 2.2) * (1.0 - np.exp(-local * 60.0))
                vibrato = 1.0 + 0.002 * np.sin(2.0 * np.pi * 5.2 * local)
                note = amplitude * envelope * np.sin(
                    2.0 * np.pi * freq * harmonic * vibrato * local)
                offset = int(start * sr)
                room = max(0, min(len(note), n - offset))
                x[offset:offset + room] += note[:room]

    for i, root in enumerate(roots):
        local = np.arange(int(segment * sr)) / sr
        envelope = np.exp(-local * 1.1) * (1.0 - np.exp(-local * 30.0))
        offset = int(i * segment * sr)
        note = 0.35 * envelope * np.sin(2.0 * np.pi * root / 2.0 * local)
        room = max(0, min(len(note), n - offset))
        x[offset:offset + room] += note[:room]

    beat = 0.5  # 120 bpm
    for k in range(int(duration / beat)):
        offset = int(k * beat * sr)
        length = int(0.08 * sr)
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.015 * sr))
        burst = signal.sosfilt(
            signal.butter(2, [1500, 6000], "bandpass", fs=sr, output="sos"), burst)
        room = max(0, min(length, n - offset))
        x[offset:offset + room] += 0.08 * burst[:room]

    x /= np.max(np.abs(x)) + 1e-9
    return (0.5 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float64)


def corrupt(x: np.ndarray, sr: int = SR, noise_db: float = 12.0,
            bits: int = 8, drive: float = 2.4, band: float = 8000.0,
            seed: int = 11) -> np.ndarray:
    """Band-limit, add hiss, quantize and clip a clean signal."""
    rng = np.random.default_rng(seed)
    if band < sr / 2.0:
        down = max(1000, int(band * 2))
        y = signal.resample_poly(x, down, sr)
        y = signal.resample_poly(y, sr, down)
    else:
        y = x.copy()

    noise = rng.standard_normal(len(y))
    noise = signal.sosfilt(
        signal.butter(2, 40.0, "highpass", fs=sr, output="sos"), noise)
    power = float(np.mean(y ** 2))
    noise *= np.sqrt(power / (np.mean(noise ** 2) * 10.0 ** (noise_db / 10.0)))
    y = y + noise

    scale = 2.0 ** (int(bits) - 1)
    y = np.round(y * scale) / scale
    y = np.clip(y * drive, -1.0, 1.0)
    return y.astype(np.float64)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` and add it to ``signal`` at the requested SNR."""
    noise = noise[:len(signal)]
    signal_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    gain = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0) + 1e-12))
    return signal + gain * noise
