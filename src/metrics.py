"""Objective audio-quality metrics for restoration evaluation.

The first version of this module had a broken ``compute_all`` (it called
``compute_snr`` with a single argument) and a misleading SNR that compared the
restored signal with the *noisy* input without a reference.  This rewrite keeps
the reference-free descriptors but separates them from reference-based metrics:

* **reference-free** (spectral flatness, bandwidth, roll-off, noise floor,
  clipping, DC offset, ...) can be computed on any file;
* **reference-based** (SI-SDR, SNR, LSD, segmental SNR, and optionally PESQ /
  STOI) require the clean signal and are only reported when one is available.

Higher/lower-is-better directions are declared in ``METRIC_DIRECTION`` so the
CLI and the comparison table never mark a regression as an improvement.

Author: Alessandro Lo Curcio
Course: Machine Learning 2024/2025, Sapienza University of Rome
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
from scipy import signal

from . import dsp

LOWER_IS_BETTER = {
    "Spectral Flatness", "Zero Crossing Rate", "LSD (dB)",
    "Clipping (%)", "Noise Floor (dBFS)", "DC Offset",
}
HIGHER_IS_BETTER = {
    "SI-SDR (dB)", "SNR (dB)", "Segmental SNR (dB)", "PESQ", "STOI",
}
NEUTRAL = {"Dynamic Range (dB)", "Spectral Centroid (Hz)",
           "Bandwidth (Hz)", "Spectral Rolloff (Hz)", "THD (%)",
           "Integrated Loudness (LUFS)", "True Peak (dBTP)"}
METRIC_DIRECTION = {
    **{k: "lower" for k in LOWER_IS_BETTER},
    **{k: "higher" for k in HIGHER_IS_BETTER},
    **{k: "neutral" for k in NEUTRAL},
}


class AudioQualityMetrics:
    """Compute objective audio-quality metrics.

    Parameters
    ----------
    sample_rate : int
        Analysis sample rate.  Files are loaded at this rate by
        :meth:`evaluate_file`.
    """

    def __init__(self, sample_rate: int = 32000):
        self.sr = int(sample_rate)

    # ------------------------------------------------------------------ #
    # Reference-based metrics
    # ------------------------------------------------------------------ #
    def compute_snr(self, original: np.ndarray, restored: np.ndarray) -> float:
        """SNR in dB between a reference and an estimate.

        The estimate is scaled to the reference energy first so that a simple
        volume change is not counted as noise.  Returns 100 dB for identical
        signals (instead of infinity, to keep CSV/JSON simple).
        """
        ref, est = self._align(original, restored)
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        est_rms = float(np.sqrt(np.mean(est ** 2)))
        if est_rms < 1e-12:
            return 0.0
        est = est * (ref_rms / est_rms)
        noise = ref - est
        noise_power = float(np.mean(noise ** 2))
        if noise_power < 1e-10:
            return 100.0
        return float(10.0 * np.log10(np.mean(ref ** 2) / noise_power))

    def compute_si_sdr(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        """Scale-invariant signal-to-distortion ratio (SI-SDR) in dB."""
        ref, est = self._align(reference, estimate)
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
        target = alpha * ref
        error = est - target
        ratio = float(np.sum(target ** 2) / (np.sum(error ** 2) + 1e-12))
        return float(10.0 * np.log10(max(ratio, 1e-12)))

    def compute_segmental_snr(self, reference: np.ndarray, estimate: np.ndarray,
                              frame_ms: float = 20.0) -> float:
        """Average frame-wise SNR in dB, clamped to [-10, 35] dB."""
        ref, est = self._align(reference, estimate)
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        est_rms = float(np.sqrt(np.mean(est ** 2)))
        if est_rms < 1e-12:
            return -10.0
        est = est * (ref_rms / est_rms)
        frame = max(1, int(self.sr * frame_ms / 1000.0))
        count = len(ref) // frame
        if count == 0:
            return self.compute_snr(ref, est)
        ref_f = ref[:count * frame].reshape(count, frame)
        est_f = est[:count * frame].reshape(count, frame)
        noise = np.mean((ref_f - est_f) ** 2, axis=1)
        signal_power = np.mean(ref_f ** 2, axis=1)
        snr = 10.0 * np.log10((signal_power + 1e-12) / (noise + 1e-12))
        return float(np.mean(np.clip(snr, -10.0, 35.0)))

    def compute_lsd(self, reference: np.ndarray, estimate: np.ndarray,
                    fmin: float = 200.0, fmax: float = 7000.0) -> float:
        """Log-spectral distance (dB), optionally restricted to a band.

        The default band (200 Hz - 7 kHz) avoids bins where a band-limited
        reference has almost no energy, which makes full-band LSD unstable.
        """
        ref, est = self._align(reference, estimate)
        n_fft = 2048 if self.sr >= 32000 else 1024
        hop = n_fft // 4
        a = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
        b = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop))
        frames = min(a.shape[1], b.shape[1])
        a, b = a[:, :frames], b[:, :frames]
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        upper = fmax if fmax else self.sr / 2
        band = (freqs >= fmin) & (freqs <= min(upper, self.sr / 2))
        if not band.any():
            return 0.0
        a, b = a[band], b[band]
        # Ignore bins where the reference has essentially no content: a single
        # near-zero bin would otherwise dominate the log distance.
        active = a.max(axis=1) > max(float(a.max()) * 1e-3, 1e-8)
        if active.any():
            a, b = a[active], b[active]
        floor = max(float(a.max()) * 1e-4, 1e-8)
        a = np.maximum(a, floor)
        b = np.maximum(b, floor)
        diff = 20.0 * np.log10(a) - 20.0 * np.log10(b)
        diff = np.clip(diff, -60.0, 60.0)
        return float(np.mean(np.sqrt(np.mean(diff ** 2, axis=0))))

    def compute_pesq(self, reference: np.ndarray, estimate: np.ndarray
                     ) -> Optional[float]:
        """Optional PESQ (wide-band, 16 kHz).  ``None`` if not installed."""
        try:
            from pesq import pesq  # type: ignore
        except ImportError:
            return None
        ref, est = self._align(reference, estimate)
        if self.sr != 16000:
            ref = signal.resample_poly(ref, 16000, self.sr)
            est = signal.resample_poly(est, 16000, self.sr)
        try:
            return float(pesq(16000, ref, est, "wb"))
        except Exception:
            return None

    def compute_stoi(self, reference: np.ndarray, estimate: np.ndarray
                     ) -> Optional[float]:
        """Optional STOI (16 kHz).  ``None`` if not installed."""
        try:
            from pystoi import stoi  # type: ignore
        except ImportError:
            return None
        ref, est = self._align(reference, estimate)
        if self.sr != 16000:
            ref = signal.resample_poly(ref, 16000, self.sr)
            est = signal.resample_poly(est, 16000, self.sr)
        try:
            return float(stoi(ref, est, 16000, extended=False))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Reference-free descriptors
    # ------------------------------------------------------------------ #
    def compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """Spectral flatness in [0, 1]; lower means more tonal."""
        flatness = librosa.feature.spectral_flatness(y=self._mono(audio))
        return float(np.mean(flatness))

    def compute_bandwidth(self, audio: np.ndarray) -> float:
        """Mean spectral bandwidth in Hz."""
        bandwidth = librosa.feature.spectral_bandwidth(y=self._mono(audio), sr=self.sr)
        return float(np.mean(bandwidth))

    def compute_spectral_rolloff(self, audio: np.ndarray,
                                 roll_percent: float = 0.85) -> float:
        """Frequency below which ``roll_percent`` of the energy is contained."""
        rolloff = librosa.feature.spectral_rolloff(
            y=self._mono(audio), sr=self.sr, roll_percent=roll_percent)
        return float(np.mean(rolloff))

    def compute_dynamic_range(self, audio: np.ndarray) -> float:
        """Crest factor (peak minus RMS) in dB."""
        x = self._mono(audio)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        level = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        if level < 1e-10:
            return 0.0
        return float(20.0 * np.log10((peak + 1e-10) / (level + 1e-10)))

    def compute_thd(self, audio: np.ndarray, n_harmonics: int = 5) -> float:
        """Total harmonic distortion (%) for a tonal signal.

        The fundamental is the strongest spectral peak above 20 Hz; harmonics
        are measured at the nearest bin to ``n * f0``.  On polyphonic music this
        number is only a rough diagnostic.
        """
        x = self._mono(audio)
        if x.size < 64:
            return 0.0
        window = np.hanning(len(x))
        magnitude = np.abs(np.fft.rfft(x * window))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.sr)
        valid = freqs >= 20.0
        if not valid.any():
            return 0.0
        fundamental_idx = int(np.flatnonzero(valid)[np.argmax(magnitude[valid])])
        fundamental_power = float(magnitude[fundamental_idx] ** 2)
        if fundamental_power < 1e-12:
            return 0.0
        harmonic_power = 0.0
        for harmonic in range(2, int(n_harmonics) + 1):
            target = harmonic * freqs[fundamental_idx]
            lo = max(0, int(np.searchsorted(freqs, target - 2.0 * (freqs[1] - freqs[0]))))
            hi = min(len(magnitude), int(np.searchsorted(freqs, target + 2.0 * (freqs[1] - freqs[0]))) + 1)
            if hi > lo:
                harmonic_power += float(np.max(magnitude[lo:hi]) ** 2)
        return float(math.sqrt(harmonic_power / fundamental_power) * 100.0)

    def compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Spectral centroid in Hz ("brightness")."""
        centroid = librosa.feature.spectral_centroid(y=self._mono(audio), sr=self.sr)
        return float(np.mean(centroid))

    def compute_zcr(self, audio: np.ndarray) -> float:
        """Mean zero-crossing rate."""
        return float(np.mean(librosa.feature.zero_crossing_rate(self._mono(audio))))

    def compute_noise_floor(self, audio: np.ndarray,
                            percentile: float = 10.0) -> float:
        """Noise floor in dBFS, estimated from quiet frame RMS values."""
        x = self._mono(audio)
        frame = max(1, int(0.02 * self.sr))
        count = len(x) // frame
        if count == 0:
            return -120.0
        frames = x[:count * frame].reshape(count, frame)
        levels = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
        return float(np.clip(np.percentile(levels, percentile), -120.0, 0.0))

    def compute_clipping_ratio(self, audio: np.ndarray,
                               threshold: float = 0.999) -> float:
        """Percentage of samples at or above the clipping threshold."""
        x = self._mono(audio)
        return float(np.mean(np.abs(x) >= threshold) * 100.0)

    def compute_dc_offset(self, audio: np.ndarray) -> float:
        """Absolute DC offset."""
        return float(abs(np.mean(self._mono(audio))))

    def compute_loudness(self, audio: np.ndarray) -> float:
        """Integrated loudness in LUFS (BS.1770-4), -120 for silence."""
        value = dsp.integrated_loudness(self._mono(audio), self.sr)
        return float(max(value, -120.0))

    def compute_true_peak(self, audio: np.ndarray) -> float:
        """True peak in dBTP, -120 for silence."""
        value = dsp.true_peak_db(self._mono(audio), self.sr)
        return float(max(value, -120.0))

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #
    def compute_all(self, audio: np.ndarray,
                    reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all metrics.  Reference-based ones need ``reference``."""
        x = self._mono(audio)
        metrics: Dict[str, float] = {
            "Spectral Flatness": self.compute_spectral_flatness(x),
            "Bandwidth (Hz)": self.compute_bandwidth(x),
            "Spectral Rolloff (Hz)": self.compute_spectral_rolloff(x),
            "Dynamic Range (dB)": self.compute_dynamic_range(x),
            "THD (%)": self.compute_thd(x),
            "Spectral Centroid (Hz)": self.compute_spectral_centroid(x),
            "Zero Crossing Rate": self.compute_zcr(x),
            "Noise Floor (dBFS)": self.compute_noise_floor(x),
            "Clipping (%)": self.compute_clipping_ratio(x),
            "DC Offset": self.compute_dc_offset(x),
            "Integrated Loudness (LUFS)": self.compute_loudness(x),
            "True Peak (dBTP)": self.compute_true_peak(x),
        }
        if reference is not None:
            ref = self._mono(reference)
            metrics["SI-SDR (dB)"] = self.compute_si_sdr(ref, x)
            metrics["SNR (dB)"] = self.compute_snr(ref, x)
            metrics["Segmental SNR (dB)"] = self.compute_segmental_snr(ref, x)
            metrics["LSD (dB)"] = self.compute_lsd(ref, x)
            pesq = self.compute_pesq(ref, x)
            if pesq is not None:
                metrics["PESQ"] = pesq
            stoi = self.compute_stoi(ref, x)
            if stoi is not None:
                metrics["STOI"] = stoi
        return metrics

    def evaluate_file(self, filepath: str,
                      reference_path: Optional[str] = None) -> Dict[str, float]:
        """Load a file (mono at ``self.sr``) and compute its metrics."""
        audio, _ = librosa.load(filepath, sr=self.sr, mono=True)
        reference = None
        if reference_path is not None:
            reference, _ = librosa.load(reference_path, sr=self.sr, mono=True)
        return self.compute_all(audio, reference)

    def compare_files(self, original_path: str, restored_path: str,
                      reference_path: Optional[str] = None
                      ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Optional[float]]]:
        """Compare original and restored files.

        If ``reference_path`` (the clean target) is given, reference-based
        metrics are computed for both the original and the restored file.
        """
        original = self.evaluate_file(original_path, reference_path)
        restored = self.evaluate_file(restored_path, reference_path)
        improvements: Dict[str, Optional[float]] = {}
        for key in original:
            a, b = original.get(key), restored.get(key)
            improvements[key] = None if a is None or b is None else b - a
        return original, restored, improvements

    def compare_arrays(self, reference: np.ndarray, original: np.ndarray,
                       restored: np.ndarray
                       ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Optional[float]]]:
        """Like :meth:`compare_files`, for in-memory signals."""
        original_metrics = self.compute_all(original, reference)
        restored_metrics = self.compute_all(restored, reference)
        improvements: Dict[str, Optional[float]] = {}
        for key in original_metrics:
            a, b = original_metrics.get(key), restored_metrics.get(key)
            improvements[key] = None if a is None or b is None else b - a
        return original_metrics, restored_metrics, improvements

    # ------------------------------------------------------------------ #
    # Reporting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_improvement(metric: str, delta: Optional[float]) -> Optional[bool]:
        if delta is None:
            return None
        direction = METRIC_DIRECTION.get(metric, "neutral")
        if direction == "lower":
            return delta < 0
        if direction == "higher":
            return delta > 0
        return None

    def print_comparison(self, original_metrics: Dict[str, float],
                         restored_metrics: Dict[str, float],
                         improvements: Dict[str, Optional[float]]) -> None:
        """Print a direction-aware comparison table."""
        print("\n" + "=" * 82)
        print(f"{'Metric':<30} {'Original':>13} {'Restored':>13} {'Delta':>13}  Verdict")
        print("=" * 82)
        for name, original in original_metrics.items():
            restored = restored_metrics.get(name)
            delta = improvements.get(name)
            if original is None or restored is None or delta is None:
                verdict = "n/a"
                print(f"{name:<30} {'n/a':>13} {'n/a':>13} {'n/a':>13}  {verdict}")
                continue
            good = self.is_improvement(name, delta)
            verdict = "~" if good is None else ("improved" if good else "worse")
            if "Hz" in name and "LSD" not in name:
                print(f"{name:<30} {original:>13.0f} {restored:>13.0f} "
                      f"{delta:>+13.0f}  {verdict}")
            else:
                print(f"{name:<30} {original:>13.3f} {restored:>13.3f} "
                      f"{delta:>+13.3f}  {verdict}")
        print("=" * 82)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        return dsp.as_mono(audio)

    @staticmethod
    def _align(reference: np.ndarray, estimate: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        ref = dsp.as_mono(reference)
        est = dsp.as_mono(estimate)
        length = min(len(ref), len(est))
        if length == 0:
            raise ValueError("empty audio signal")
        return ref[:length], est[:length]


def main() -> None:
    """Small self-test that needs no external file."""
    sr = 32000
    t = np.arange(int(2.0 * sr)) / sr
    reference = 0.4 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(0)
    noisy = reference + 0.02 * rng.standard_normal(len(reference))
    metrics = AudioQualityMetrics(sample_rate=sr)
    print("Noisy signal vs clean reference")
    for name, value in metrics.compute_all(noisy, reference).items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
