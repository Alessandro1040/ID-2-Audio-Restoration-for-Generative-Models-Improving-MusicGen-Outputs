"""DSP building blocks used by the restoration pipeline.

The module intentionally depends only on numpy, scipy, librosa and soundfile.
Every stage is deterministic (except dithering, which accepts a seed) and works
on mono ``(n,)`` or multichannel ``(n, channels)`` float arrays.

Design notes
------------
* Noise is estimated per frequency bin from a low percentile of the power
  spectrogram.  On continuous music the quietest frames are the best available
  proxy for the noise floor; a minimum-statistics tracker was tested as well but
  did not outperform the percentile estimator on the benchmark.
* The gain is a decision-directed Wiener gain.  A frequency-dependent gain
  floor protects the low band (where music lives) and allows deeper attenuation
  above 1 kHz (where hiss and codec noise live).
* A transient protection term restores gain on onsets so that drums and piano
  attacks are not softened.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import librosa
import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def as_2d(audio: np.ndarray) -> np.ndarray:
    """Return ``(n, channels)`` float64, accepting ``(n,)`` and ``(channels, n)``."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {x.shape}")
    # soundfile/librosa return (n, ch); accept (ch, n) only when unambiguous.
    if x.shape[0] < x.shape[1] and x.shape[0] <= 8:
        x = x.T
    return x


def as_mono(audio: np.ndarray) -> np.ndarray:
    x = as_2d(audio)
    return x.mean(axis=1)


def to_db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def from_db(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def rms(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if np.size(audio) else 0.0


def sanitize(audio: np.ndarray) -> np.ndarray:
    """Replace non-finite values and remove DC per channel."""
    x = as_2d(audio).copy()
    x[~np.isfinite(x)] = 0.0
    x -= x.mean(axis=0, keepdims=True)
    return x


def stft_params(sr: int, n_fft: int | None = None, hop: int | None = None) -> tuple[int, int]:
    if n_fft is None:
        if sr >= 32000:
            n_fft = 2048
        elif sr >= 16000:
            n_fft = 1024
        else:
            n_fft = 512
    n_fft = int(n_fft)
    hop = int(hop) if hop else n_fft // 4
    return n_fft, hop


def _per_channel(x: np.ndarray, fn):
    """Apply a mono function to every channel of an ``(n, ch)`` array."""
    cols = [fn(x[:, c]) for c in range(x.shape[1])]
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------- #
# Spectral denoising
# --------------------------------------------------------------------------- #
def estimate_noise_psd(magnitude: np.ndarray, percentile: float = 15.0,
                       bias: float = 2.0, smooth_bins: int = 3) -> np.ndarray:
    """Estimate a stationary noise PSD from a magnitude spectrogram.

    Parameters
    ----------
    magnitude : (freqs, frames)
        Magnitude STFT of the mixture.
    percentile : float
        Which percentile of the per-bin temporal distribution to use.  Lower
        values are more conservative (less noise removed, less risk to music).
    bias : float
        Multiplier compensating the downward bias of a low percentile of an
        exponentially distributed periodogram.  Tuned on real music; it is not
        meant to recover the statistical mean.
    smooth_bins : int
        Frequency smoothing of the estimate.
    """
    power = np.asarray(magnitude, dtype=np.float64) ** 2
    if power.ndim != 2:
        raise ValueError("magnitude must be 2-D (freqs, frames)")
    if power.shape[1] < 8:
        # Too few frames for a percentile: use a robust minimum.
        noise = np.min(power, axis=1) * bias * 4.0
    else:
        q = float(np.clip(percentile, 1.0, 50.0))
        noise = np.percentile(power, q, axis=1) * bias
    noise = uniform_filter1d(np.maximum(noise, EPS), max(1, int(smooth_bins)))
    return noise


def _floor_curve(freqs: np.ndarray, floor_low_db: float, floor_high_db: float) -> np.ndarray:
    """Gain floor in dB, gentle at 250 Hz and stricter above 8 kHz."""
    f = np.maximum(freqs, 20.0)
    t = np.clip(np.log2(f / 1000.0) / 3.0, 0.0, 1.0)
    return floor_low_db + (floor_high_db - floor_low_db) * t


def wiener_gain(magnitude: np.ndarray, noise_psd: np.ndarray,
                floor_low_db: float = -8.0, floor_high_db: float = -22.0,
                alpha: float = 0.98, beta: float = 0.5,
                smooth_bins: int = 3, protect: float = 0.5,
                freqs: np.ndarray | None = None) -> np.ndarray:
    """Decision-directed Wiener gain with a frequency-dependent floor.

    ``alpha`` is the decision-directed smoothing factor, ``beta`` smooths the
    gain over time.  ``protect`` (0-1) raises the gain on spectral onsets.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    noise = np.maximum(np.asarray(noise_psd, dtype=np.float64), EPS)[:, None]
    gamma = mag ** 2 / noise

    if freqs is None:
        floor_db = np.full(mag.shape[0], floor_low_db, dtype=np.float64)
    else:
        floor_db = _floor_curve(freqs, floor_low_db, floor_high_db)
    floor = (10.0 ** (floor_db / 20.0))[:, None]

    gain = np.empty_like(mag)
    g_prev = np.ones(mag.shape[0])
    gamma_prev = np.ones(mag.shape[0])
    for i in range(mag.shape[1]):
        xi = alpha * (g_prev ** 2) * gamma_prev
        xi += (1.0 - alpha) * np.maximum(gamma[:, i] - 1.0, 0.0)
        g = np.maximum(xi / (1.0 + xi), floor[:, 0])
        gain[:, i] = g
        g_prev, gamma_prev = g, gamma[:, i]

    gain = uniform_filter1d(gain, max(1, int(smooth_bins)), axis=0)
    # Start the one-pole smoother at the first frame instead of at zero, which
    # would otherwise attenuate the beginning of every block.
    coeff = 1.0 - float(beta)
    z = signal.lfilter_zi([coeff], [1.0, -float(beta)])[:, None] * gain[:, :1]
    gain = signal.lfilter([coeff], [1.0, -float(beta)], gain, axis=1, zi=z)[0]
    gain = np.clip(gain, floor, 1.0)

    if protect > 0.0 and mag.shape[1] > 1:
        flux = np.maximum(0.0, mag - np.roll(mag, 1, axis=1)).sum(axis=0)
        flux[0] = 0.0
        flux /= np.percentile(flux, 95) + EPS
        gain += (1.0 - gain) * np.clip(flux, 0.0, 1.0)[None, :] ** 2 * float(protect)

    return np.clip(gain, 0.0, 1.0)


def spectral_denoise(audio: np.ndarray, sr: int, strength: float = 1.0,
                     n_fft: int | None = None, hop: int | None = None,
                     percentile: float = 15.0, bias: float = 2.0,
                     floor_low_db: float = -8.0, floor_high_db: float = -22.0,
                     protect: float = 0.5) -> np.ndarray:
    """Remove stationary broadband noise from mono or multichannel audio.

    A single mask is estimated from the average power of all channels and applied
    to every channel, which keeps the stereo image coherent.
    """
    x = as_2d(audio)
    strength = float(np.clip(strength, 0.0, 1.5))
    if strength <= 0.0 or len(x) < 64:
        return x.copy()

    n_fft, hop = stft_params(sr, n_fft, hop)
    spectra = np.stack([librosa.stft(x[:, c], n_fft=n_fft, hop_length=hop)
                        for c in range(x.shape[1])], axis=0)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    noise = estimate_noise_psd(np.sqrt(power), percentile=percentile, bias=bias)
    # strength scales the amount of noise we believe is present
    noise = noise * (0.35 + 0.65 * strength)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    gain = wiener_gain(np.sqrt(power), noise,
                       floor_low_db=floor_low_db * min(1.0, strength),
                       floor_high_db=floor_high_db * min(1.0, strength),
                       protect=protect, freqs=freqs)

    out = []
    for c in range(x.shape[1]):
        clean = librosa.istft(spectra[c] * gain, hop_length=hop, length=len(x))
        out.append(clean)
    return np.stack(out, axis=1)


# --------------------------------------------------------------------------- #
# Clipping repair
# --------------------------------------------------------------------------- #
def declip(audio: np.ndarray, sr: int, threshold: float = 0.999,
           max_run_ms: float = 8.0, pad_ms: float = 2.0,
           max_ratio: float = 0.05) -> tuple[np.ndarray, dict]:
    """Repair flat-topped (clipped) samples with shape-preserving interpolation.

    Only short runs are repaired: long runs carry too little information and
    interpolating them would invent audio.  Returns the audio and a stats dict.
    """
    x = as_2d(audio).copy()
    n, channels = x.shape
    max_run = max(1, int(sr * max_run_ms / 1000.0))
    pad = max(2, int(sr * pad_ms / 1000.0))
    clipped_before = int(np.sum(np.abs(x) >= threshold))
    ratio_before = clipped_before / max(1, x.size)

    stats = {"ratio_before": ratio_before, "runs_fixed": 0, "runs_skipped": 0,
             "samples_fixed": 0, "applied": False}
    if clipped_before == 0:
        return x, stats
    if ratio_before > max_ratio:
        stats["skipped_reason"] = "clipped_ratio_above_limit"
        return x, stats

    for c in range(channels):
        clipped = np.abs(x[:, c]) >= threshold
        if not clipped.any():
            continue
        edges = np.diff(np.concatenate(([0], clipped.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            if end - start > max_run:
                stats["runs_skipped"] += 1
                continue
            lo, hi = max(0, start - pad), min(n, end + pad)
            idx = np.arange(lo, hi)
            clean = idx[~clipped[lo:hi]]
            if len(clean) < 4:
                stats["runs_skipped"] += 1
                continue
            try:
                interp = PchipInterpolator(clean, x[clean, c], extrapolate=True)
            except ValueError:
                stats["runs_skipped"] += 1
                continue
            x[start:end, c] = interp(np.arange(start, end))
            stats["samples_fixed"] += int(end - start)
            stats["runs_fixed"] += 1

    stats["applied"] = stats["samples_fixed"] > 0
    stats["ratio_after"] = float(np.sum(np.abs(x) >= threshold)) / max(1, x.size)
    return x, stats


# --------------------------------------------------------------------------- #
# Filters and EQ
# --------------------------------------------------------------------------- #
def highpass(audio: np.ndarray, sr: int, cutoff: float = 25.0, order: int = 2) -> np.ndarray:
    x = as_2d(audio)
    cutoff = float(min(max(cutoff, 5.0), 0.45 * sr))
    sos = signal.butter(int(order), cutoff, btype="highpass", fs=sr, output="sos")
    try:
        # Zero-phase filtering avoids rotating the bass and moving transients,
        # which matters when the result is compared with the input.
        return signal.sosfiltfilt(sos, x, axis=0)
    except ValueError:
        return signal.sosfilt(sos, x, axis=0)


def _band_prominence(freqs: np.ndarray, pxx: np.ndarray, f0: float,
                     half_width: float = 2.0, ref_width: float = 12.0) -> float:
    peak_mask = (freqs >= f0 - half_width) & (freqs <= f0 + half_width)
    ref_mask = (((freqs >= f0 - ref_width) & (freqs <= f0 - 4.0)) |
                ((freqs >= f0 + 4.0) & (freqs <= f0 + ref_width)))
    if not peak_mask.any() or not ref_mask.any():
        return 0.0
    peak = float(np.max(pxx[peak_mask]))
    ref = float(np.median(pxx[ref_mask]))
    if ref <= 0.0 or peak <= 0.0:
        return 0.0
    return 10.0 * math.log10(peak / ref)


def detect_hum(audio: np.ndarray, sr: int, candidates: Sequence[float] = (50.0, 60.0),
               max_freq: float = 500.0, min_prominence_db: float = 8.0) -> list[float]:
    """Detect mains hum and its harmonics.  Returns a list of frequencies in Hz."""
    x = as_mono(audio)
    if len(x) < max(2048, int(0.5 * sr)):
        return []
    freqs, pxx = signal.welch(x, fs=sr, nperseg=min(16384, len(x)))
    found: list[float] = []
    for base in candidates:
        if base >= sr / 2:
            continue
        if _band_prominence(freqs, pxx, base, ref_width=10.0) < min_prominence_db:
            continue
        found.append(float(base))
        harmonic = 2.0 * base
        while harmonic <= min(max_freq, 0.45 * sr):
            if _band_prominence(freqs, pxx, harmonic, ref_width=8.0) >= min_prominence_db - 2.0:
                found.append(float(harmonic))
            harmonic += base
    return sorted(set(round(f, 2) for f in found))


def remove_hum(audio: np.ndarray, sr: int, frequencies: Iterable[float],
               q: float = 30.0) -> np.ndarray:
    x = as_2d(audio).copy()
    for f0 in frequencies:
        if not (10.0 < f0 < 0.49 * sr):
            continue
        b, a = signal.iirnotch(w0=float(f0), Q=float(q), fs=sr)
        if len(x) > 3 * max(len(a), len(b)):
            x = signal.filtfilt(b, a, x, axis=0)
        else:
            x = signal.lfilter(b, a, x, axis=0)
    return x


def _peaking_biquad(f0: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, 0.49 * sr) / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b = np.array([1.0 + alpha * a_gain, -2.0 * cos_w0, 1.0 - alpha * a_gain])
    a = np.array([1.0 + alpha / a_gain, -2.0 * cos_w0, 1.0 - alpha / a_gain])
    return signal.tf2sos(b / a[0], a / a[0])


def _shelf_biquad(f0: float, gain_db: float, sr: int, high: bool,
                  slope: float = 1.0) -> np.ndarray:
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, 0.49 * sr) / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a_gain + 1.0 / a_gain) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a_gain) * alpha

    if high:
        b = np.array([a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + two_sqrt_a_alpha),
                      -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0),
                      a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - two_sqrt_a_alpha)])
        a = np.array([(a_gain + 1) - (a_gain - 1) * cos_w0 + two_sqrt_a_alpha,
                      2 * ((a_gain - 1) - (a_gain + 1) * cos_w0),
                      (a_gain + 1) - (a_gain - 1) * cos_w0 - two_sqrt_a_alpha])
    else:
        b = np.array([a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 + two_sqrt_a_alpha),
                      2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cos_w0),
                      a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 - two_sqrt_a_alpha)])
        a = np.array([(a_gain + 1) + (a_gain - 1) * cos_w0 + two_sqrt_a_alpha,
                      -2 * ((a_gain - 1) + (a_gain + 1) * cos_w0),
                      (a_gain + 1) + (a_gain - 1) * cos_w0 - two_sqrt_a_alpha])
    return signal.tf2sos(b / a[0], a / a[0])


def apply_eq(audio: np.ndarray, sr: int, low_shelf_db: float = 0.0,
             low_shelf_hz: float = 120.0, presence_db: float = 0.0,
             presence_hz: float = 3000.0, presence_q: float = 0.8,
             air_db: float = 0.0, air_hz: float = 10000.0) -> np.ndarray:
    """Subtle tonal shaping with RBJ biquads.

    The previous implementation summed four band-pass filters, which changed the
    gain by several dB and introduced phase cancellation.  Shelves and a
    peaking filter keep the level under control.
    """
    x = as_2d(audio)
    sections = []
    if abs(low_shelf_db) > 0.01:
        sections.append(_shelf_biquad(low_shelf_hz, low_shelf_db, sr, high=False))
    if abs(presence_db) > 0.01:
        sections.append(_peaking_biquad(presence_hz, presence_db, presence_q, sr))
    if abs(air_db) > 0.01 and air_hz < 0.49 * sr:
        sections.append(_shelf_biquad(air_hz, air_db, sr, high=True))
    if not sections:
        return x.copy()
    sos = np.vstack(sections)
    return signal.sosfilt(sos, x, axis=0)


def compress(audio: np.ndarray, sr: int, threshold_db: float = -18.0,
             ratio: float = 2.0, attack_ms: float = 20.0,
             release_ms: float = 200.0, makeup_db: float = 0.0) -> np.ndarray:
    """Gentle feed-forward compressor with a linked stereo detector.

    The gain curve is computed on 5 ms frames and interpolated back to samples;
    this keeps the implementation fast for long files.
    """
    x = as_2d(audio)
    if x.size == 0 or ratio <= 1.0:
        return x.copy()
    frame = max(1, int(0.005 * sr))
    frames = int(math.ceil(len(x) / frame))
    padded = np.pad(x, ((0, frames * frame - len(x)), (0, 0)))
    level = np.max(np.abs(padded.reshape(frames, frame, x.shape[1])), axis=(1, 2))
    level_db = 20.0 * np.log10(np.maximum(level, EPS))
    over = np.maximum(0.0, level_db - threshold_db)
    target_gr = over * (1.0 - 1.0 / ratio)

    frame_rate = sr / frame
    attack = math.exp(-1.0 / max(1.0, frame_rate * attack_ms / 1000.0))
    release = math.exp(-1.0 / max(1.0, frame_rate * release_ms / 1000.0))
    gr = np.zeros(frames)
    prev = 0.0
    for i in range(frames):
        coeff = attack if target_gr[i] > prev else release
        prev = coeff * prev + (1.0 - coeff) * target_gr[i]
        gr[i] = prev

    gain_db = -gr + float(makeup_db)
    gain = 10.0 ** (gain_db / 20.0)
    sample_positions = np.arange(len(x)) / frame
    gain_samples = np.interp(sample_positions, np.arange(frames), gain)
    return x * gain_samples[:, None]


# --------------------------------------------------------------------------- #
# Loudness (ITU-R BS.1770-4), limiting and dithering
# --------------------------------------------------------------------------- #
_K_SHELF = signal.tf2sos(
    [1.53512485958697, -2.69169618940638, 1.19839281085285],
    [1.0, -1.69065929318241, 0.73248077421585],
)
_K_HIGHPASS = signal.tf2sos(
    [1.0, -2.0, 1.0],
    [1.0, -1.99004745483398, 0.99007225036621],
)
_K_SOS = np.vstack([_K_SHELF, _K_HIGHPASS])


def _resample_to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 48000:
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(int(sr), 48000)
    return signal.resample_poly(audio, 48000 // divisor, int(sr) // divisor, axis=0)


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """Gated integrated loudness in LUFS (BS.1770-4).  ``-inf`` for silence."""
    x = as_2d(audio)
    if x.size == 0 or rms(x) < 1e-7:
        return float("-inf")
    y = signal.sosfilt(_K_SOS, _resample_to_48k(x, sr), axis=0)
    block = int(0.4 * 48000)
    hop = int(0.1 * 48000)
    if len(y) < block:
        starts = [0]
        block = len(y)
    else:
        starts = range(0, len(y) - block + 1, hop)
    powers = np.array([np.mean(y[s:s + block] ** 2, axis=0).sum() for s in starts])
    powers = np.maximum(powers, EPS)
    levels = -0.691 + 10.0 * np.log10(powers)
    gated = levels > -70.0
    if not gated.any():
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(np.mean(powers[gated])) - 10.0
    gated &= levels > relative
    if not gated.any():
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(powers[gated])))


def true_peak_db(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    x = as_2d(audio)
    if x.size == 0:
        return float("-inf")
    y = signal.resample_poly(x, int(oversample), 1, axis=0)
    return to_db(np.max(np.abs(y)))


def _measure_true_peak(audio: np.ndarray, sr: int) -> float:
    """Alias used where a ``true_peak_db`` parameter would shadow the function."""
    return true_peak_db(audio, sr)


def limit(audio: np.ndarray, sr: int, ceiling_db: float = -1.0,
          lookahead_ms: float = 5.0, release_ms: float = 80.0
          ) -> tuple[np.ndarray, float]:
    """Look-ahead brick-wall limiter.  Returns (audio, max gain reduction).

    The gain drops over the look-ahead window (so the peak never arrives before
    the gain is low) and recovers with a one-pole release.  A final clip only
    catches numerical overshoot; it is not part of the normal operation.
    """
    x = as_2d(audio)
    if x.size == 0:
        return x.copy(), 0.0
    ceiling = from_db(ceiling_db)
    lookahead = max(1, int(sr * lookahead_ms / 1000.0))
    envelope = maximum_filter1d(np.max(np.abs(x), axis=1), size=2 * lookahead + 1,
                                mode="nearest")
    target = np.minimum(1.0, ceiling / np.maximum(envelope, EPS))
    # Hold the gain low across the look-ahead window (no overshoot on attack).
    target = minimum_filter1d(target, size=lookahead + 1, mode="nearest")
    # Instant attack, smooth release (standard feed-forward limiter behaviour).
    coeff = math.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    b, a = [1.0 - coeff], [1.0, -coeff]
    state = signal.lfilter_zi(b, a) * float(target[0])
    released, _ = signal.lfilter(b, a, target, zi=state)
    gain = np.minimum(target, released)
    y = np.clip(x * gain[:, None], -ceiling, ceiling)
    # Report positive gain reduction in dB (0 means the limiter did nothing).
    reduction_db = -to_db(np.min(gain)) if gain.size else 0.0
    return y, reduction_db


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -14.0,
                       true_peak_db: float = -1.0, max_gain_db: float = 30.0
                       ) -> tuple[np.ndarray, dict]:
    """Normalize to a loudness target and guarantee the true-peak ceiling."""
    x = as_2d(audio)
    before = integrated_loudness(x, sr)
    stats = {"lufs_before": before, "lufs_after": before, "gain_db": 0.0,
             "limiter_reduction_db": 0.0, "target_lufs": target_lufs}
    if not math.isfinite(before):
        return x.copy(), stats

    gain_db = float(np.clip(target_lufs - before, -max_gain_db, max_gain_db))
    y = x * from_db(gain_db)
    y, reduction = limit(y, sr, ceiling_db=true_peak_db)
    after = integrated_loudness(y, sr)
    # The limiter can shave a few tenths of a dB; one corrective pass is enough.
    if math.isfinite(after) and after < target_lufs - 0.5:
        extra = min(target_lufs - after, 3.0)
        y, reduction2 = limit(y * from_db(extra), sr, ceiling_db=true_peak_db)
        gain_db += extra
        reduction = max(reduction, reduction2)
        after = integrated_loudness(y, sr)

    # The limiter works on sample peaks; inter-sample peaks can still exceed the
    # ceiling.  Apply one small gain trim so the reported dBTP is guaranteed.
    measured_peak = _measure_true_peak(y, sr)
    if math.isfinite(measured_peak) and measured_peak > true_peak_db:
        correction = true_peak_db - measured_peak
        y = y * from_db(correction)
        gain_db += correction
        after = integrated_loudness(y, sr)

    stats.update({"lufs_after": after, "gain_db": gain_db,
                  "limiter_reduction_db": reduction})
    return y, stats


def dither(audio: np.ndarray, bits: int = 16, seed: int = 0) -> np.ndarray:
    """Add TPDF dither for the target integer bit depth (no-op above 16 bits)."""
    if bits > 16:
        return np.asarray(audio, dtype=np.float64)
    lsb = 2.0 / (2 ** int(bits))
    rng = np.random.default_rng(seed)
    x = as_2d(audio)
    noise = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    return x + noise




