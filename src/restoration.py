"""Audio restoration pipeline for generative music (MusicGen and similar).

This is a **zero-shot, reference-free** post-processor: it never needs a clean
target and does not modify the semantic content of the track.  The pipeline is:

    declip -> hum removal -> spectral denoising -> high-pass -> tonal EQ
           -> optional compression -> loudness normalization + limiter -> dither

Compared with the first version of this project, the rewrite

* preserves the input sample rate and channel layout (MusicGen outputs 32 kHz;
  the old code silently resampled to 44.1 kHz and downmixed to mono);
* replaces the hand-tuned "anti-vocal" gate and 2x spectral subtraction, which
  removed large parts of the music, with a decision-directed Wiener filter;
* replaces the sum of four band-pass filters (which changed the level and the
  phase response) with RBJ shelves and a peaking filter;
* adds proper LUFS normalization, a true-peak limiter and dithering only when
  the output is actually quantized;
* processes long files in overlapping blocks to bound memory usage.

Author: Alessandro Lo Curcio
Course: Machine Learning 2024/2025, Sapienza University of Rome
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from . import dsp

SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


@dataclass
class RestorationConfig:
    """Parameters of the restoration pipeline.

    Use :meth:`from_preset` unless you need full control.
    """

    preset: str = "music"
    denoise: bool = True
    denoise_strength: float = 1.0
    noise_percentile: float = 15.0
    noise_bias: float = 2.0
    floor_low_db: float = -8.0
    floor_high_db: float = -22.0
    transient_protection: float = 0.5

    declip: bool = True
    clip_threshold: float = 0.999
    declip_max_run_ms: float = 8.0

    remove_hum: bool = False
    hum_min_prominence_db: float = 8.0

    highpass_hz: float = 0.0
    eq: bool = True
    low_shelf_db: float = -0.5
    low_shelf_hz: float = 120.0
    presence_db: float = 0.8
    presence_hz: float = 3000.0
    presence_q: float = 0.8
    air_db: float = 0.5
    air_hz: float = 10000.0

    compress: bool = False
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 2.0
    compressor_attack_ms: float = 20.0
    compressor_release_ms: float = 200.0

    normalize: bool = True
    target_lufs: float | None = None
    true_peak_db: float = -1.0
    max_gain_db: float = 30.0

    target_sample_rate: int | None = None
    mono: bool = False
    bit_depth: int = 24
    max_block_seconds: float = 120.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.denoise_strength = float(np.clip(self.denoise_strength, 0.0, 1.5))
        self.noise_percentile = float(np.clip(self.noise_percentile, 1.0, 50.0))
        self.noise_bias = float(np.clip(self.noise_bias, 0.5, 10.0))
        self.floor_low_db = float(np.clip(self.floor_low_db, -40.0, 0.0))
        self.floor_high_db = float(np.clip(self.floor_high_db, -60.0, 0.0))
        self.transient_protection = float(np.clip(self.transient_protection, 0.0, 1.0))
        self.bit_depth = int(self.bit_depth if self.bit_depth in SUBTYPES else 24)
        if self.target_lufs is not None:
            self.target_lufs = float(np.clip(self.target_lufs, -40.0, -5.0))
        self.true_peak_db = float(np.clip(self.true_peak_db, -10.0, 0.0))
        if self.max_block_seconds is not None:
            self.max_block_seconds = float(max(1.0, self.max_block_seconds))

    @classmethod
    def from_preset(cls, preset: str = "music", **overrides: Any) -> "RestorationConfig":
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        params = dict(PRESETS[preset])
        params.update(overrides)
        params["preset"] = preset
        return cls(**params)


PRESETS: dict[str, dict[str, Any]] = {
    # Balanced default for MusicGen-style instrumental output.
    "music": {
        "denoise_strength": 1.0,
        "noise_percentile": 15.0,
        "noise_bias": 2.0,
        "floor_low_db": -8.0,
        "floor_high_db": -22.0,
        "transient_protection": 0.5,
        "highpass_hz": 0.0,
        "low_shelf_db": -0.5,
        "presence_db": 0.8,
        "air_db": 0.5,
        "compress": False,
        "target_lufs": None,
    },
    # Conservative: use when the source is already clean or when in doubt.
    "gentle": {
        "denoise_strength": 0.55,
        "noise_percentile": 10.0,
        "noise_bias": 1.7,
        "floor_low_db": -6.0,
        "floor_high_db": -14.0,
        "transient_protection": 0.6,
        "highpass_hz": 0.0,
        "low_shelf_db": -0.3,
        "presence_db": 0.4,
        "air_db": 0.3,
        "compress": False,
        "target_lufs": None,
    },
    # For noisy, hissy or low-bitrate sources.  May soften dense mixes.
    "strong": {
        "denoise_strength": 1.0,
        "noise_percentile": 15.0,
        "noise_bias": 2.0,
        "floor_low_db": -10.0,
        "floor_high_db": -26.0,
        "transient_protection": 0.5,
        "highpass_hz": 0.0,
        "low_shelf_db": -0.8,
        "presence_db": 1.0,
        "air_db": 0.8,
        "compress": False,
        "target_lufs": None,
    },
    # Speech / podcast: more low-cut, gentle compression, -16 LUFS.
    "voice": {
        "denoise_strength": 1.15,
        "noise_percentile": 18.0,
        "noise_bias": 2.0,
        "floor_low_db": -12.0,
        "floor_high_db": -26.0,
        "transient_protection": 0.35,
        "highpass_hz": 80.0,
        "low_shelf_db": -2.5,
        "low_shelf_hz": 180.0,
        "presence_db": 2.5,
        "presence_hz": 2800.0,
        "air_db": 0.8,
        "compress": True,
        "remove_hum": True,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 2.0,
        "target_lufs": -16.0,
    },
}


class AudioRestorer:
    """Multi-stage restoration pipeline.

    Parameters
    ----------
    sample_rate : int | None
        Optional output sample rate.  ``None`` (default) preserves the input
        rate, which is the right choice for MusicGen (32 kHz).
    preset : {"music", "gentle", "strong", "voice"}
        Starting point for the parameters.
    verbose : bool
        Print the progress of the stages.
    config : RestorationConfig | None
        Full configuration; when provided, ``preset`` and ``**overrides`` are
        ignored.
    **overrides
        Any field of :class:`RestorationConfig`, e.g. ``denoise_strength=1.2``.
    """

    def __init__(self, sample_rate: int | None = None, preset: str = "music",
                 verbose: bool = True, config: RestorationConfig | None = None,
                 **overrides: Any):
        if config is not None:
            self.config = config
        else:
            self.config = RestorationConfig.from_preset(
                preset, target_sample_rate=sample_rate, **overrides)
        self.verbose = verbose
        # Kept for backwards compatibility with the first version.
        self.sr = sample_rate

    def restore_array(self, audio: np.ndarray, sr: int,
                      config: RestorationConfig | None = None
                      ) -> tuple[np.ndarray, dict]:
        """Restore an in-memory signal.  Returns ``(audio, report)``."""
        cfg = config or self.config
        started = time.perf_counter()
        x = dsp.sanitize(audio)
        input_sr = int(sr)
        if cfg.mono and x.shape[1] > 1:
            x = x.mean(axis=1, keepdims=True)
        x, sr = self._resample(x, sr, cfg.target_sample_rate)

        report: dict[str, Any] = {
            "preset": cfg.preset,
            "input_sample_rate": input_sr,
            "sample_rate": int(sr),
            "channels": int(x.shape[1]),
            "duration_s": round(len(x) / float(sr), 4) if sr else 0.0,
            "stages": {},
        }
        if x.size == 0 or dsp.peak(x) < 1e-7:
            report["silent"] = True
            report["loudness"] = {}
            report["elapsed_s"] = round(time.perf_counter() - started, 4)
            return x.copy(), report

        report["loudness"] = {
            "lufs_before": self._finite(dsp.integrated_loudness(x, sr)),
            "true_peak_before_db": self._finite(dsp.true_peak_db(x, sr)),
        }

        if cfg.declip:
            x, stats = dsp.declip(x, sr, threshold=cfg.clip_threshold,
                                  max_run_ms=cfg.declip_max_run_ms)
            report["stages"]["declip"] = stats

        if cfg.remove_hum:
            hum = dsp.detect_hum(x, sr, min_prominence_db=cfg.hum_min_prominence_db)
            report["stages"]["hum"] = {"detected_hz": hum, "removed": bool(hum)}
            if hum:
                x = dsp.remove_hum(x, sr, hum)

        if cfg.denoise and cfg.denoise_strength > 0:
            if self._log_enabled:
                n = len(x) / float(sr)
                print(f"  Denoising ({n:.1f}s, preset={cfg.preset})...")
            x, blocks = self._denoise(x, sr, cfg)
            report["stages"]["denoise"] = {
                "strength": cfg.denoise_strength,
                "percentile": cfg.noise_percentile,
                "bias": cfg.noise_bias,
                "floor_db": [cfg.floor_low_db, cfg.floor_high_db],
                "blocks": blocks,
            }

        if cfg.highpass_hz and cfg.highpass_hz > 0:
            x = dsp.highpass(x, sr, cutoff=cfg.highpass_hz)
            report["stages"]["highpass"] = {"cutoff_hz": cfg.highpass_hz}

        if cfg.eq:
            x = dsp.apply_eq(
                x, sr,
                low_shelf_db=cfg.low_shelf_db, low_shelf_hz=cfg.low_shelf_hz,
                presence_db=cfg.presence_db, presence_hz=cfg.presence_hz,
                presence_q=cfg.presence_q,
                air_db=cfg.air_db, air_hz=cfg.air_hz,
            )
            report["stages"]["eq"] = {
                "low_shelf_db": cfg.low_shelf_db,
                "presence_db": cfg.presence_db,
                "air_db": cfg.air_db,
            }

        if cfg.compress:
            x = dsp.compress(
                x, sr,
                threshold_db=cfg.compressor_threshold_db,
                ratio=cfg.compressor_ratio,
                attack_ms=cfg.compressor_attack_ms,
                release_ms=cfg.compressor_release_ms,
            )
            report["stages"]["compressor"] = {
                "threshold_db": cfg.compressor_threshold_db,
                "ratio": cfg.compressor_ratio,
            }

        if cfg.normalize:
            if cfg.target_lufs is None:
                if dsp.peak(x) > dsp.from_db(cfg.true_peak_db):
                    x, reduction = dsp.limit(x, sr, ceiling_db=cfg.true_peak_db)
                    # Trim inter-sample overshoot left by the sample-peak limiter.
                    measured = dsp.true_peak_db(x, sr)
                    if math.isfinite(measured) and measured > cfg.true_peak_db:
                        x *= dsp.from_db(cfg.true_peak_db - measured)
                else:
                    reduction = 0.0
                report["loudness"].update({
                    "gain_db": 0.0,
                    "limiter_reduction_db": self._finite(reduction),
                    "lufs_after": self._finite(dsp.integrated_loudness(x, sr)),
                })
            else:
                x, stats = dsp.normalize_loudness(
                    x, sr, target_lufs=cfg.target_lufs,
                    true_peak_db=cfg.true_peak_db, max_gain_db=cfg.max_gain_db)
                report["loudness"].update({k: self._finite(v) for k, v in stats.items()})

        if cfg.bit_depth <= 16:
            x = dsp.dither(x, bits=cfg.bit_depth, seed=cfg.seed)
            report["stages"]["dither"] = {"bits": cfg.bit_depth, "type": "TPDF"}

        x = np.clip(x, -1.0, 1.0)
        report["loudness"]["true_peak_after_db"] = self._finite(dsp.true_peak_db(x, sr))
        report["elapsed_s"] = round(time.perf_counter() - started, 4)
        return x, report

    def restore_file(self, input_path: str | os.PathLike,
                     output_path: str | os.PathLike,
                     config: RestorationConfig | None = None,
                     overwrite: bool = True) -> dict:
        """Restore ``input_path`` and write ``output_path``.  Returns a report."""
        cfg = config or self.config
        source = Path(input_path)
        target = Path(output_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists")
        try:
            audio, sr = sf.read(str(source), dtype="float64", always_2d=True)
        except Exception:
            # libsndfile does not decode every format; librosa/audioread can help.
            data, sr = librosa.load(str(source), sr=None, mono=False)
            audio = data.T if data.ndim > 1 else data[:, None]
        restored, report = self.restore_array(audio, sr, cfg)
        target.parent.mkdir(parents=True, exist_ok=True)
        output_sr = int(report.get("sample_rate", sr))
        sf.write(str(target), restored, output_sr, subtype=SUBTYPES[cfg.bit_depth])
        report.update({
            "input": str(source),
            "output": str(target),
            "bit_depth": cfg.bit_depth,
            "subtype": SUBTYPES[cfg.bit_depth],
        })
        return report

    def restore(self, input_path: str, output_path: str,
                denoise_method: str | None = None,
                noise_reduction: float | None = None) -> np.ndarray:
        """Backwards-compatible wrapper returning the restored samples."""
        cfg = self.config
        if noise_reduction is not None:
            cfg = replace(cfg, denoise_strength=float(noise_reduction))
        self.restore_file(input_path, output_path, cfg)
        audio, _ = sf.read(output_path, dtype="float64", always_2d=True)
        return audio[:, 0] if audio.shape[1] == 1 else audio

    def batch_restore(self, input_files: list[str], output_dir: str,
                      prefix: str = "restored_") -> list[str]:
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for input_path in input_files:
            target = os.path.join(output_dir, prefix + os.path.basename(input_path))
            try:
                self.restore_file(input_path, target)
                outputs.append(target)
            except Exception as exc:  # pragma: no cover - logging path
                print(f"Error processing {input_path}: {exc}")
        return outputs

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @property
    def _log_enabled(self) -> bool:
        return bool(self.verbose)

    @staticmethod
    def _finite(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _resample(audio: np.ndarray, sr: int, target: int | None
                  ) -> tuple[np.ndarray, int]:
        if target is None or int(target) == int(sr):
            return audio, int(sr)
        target = int(target)
        divisor = math.gcd(int(sr), target)
        return signal.resample_poly(audio, target // divisor, int(sr) // divisor,
                                    axis=0), target

    def _denoise(self, audio: np.ndarray, sr: int,
                 cfg: RestorationConfig) -> tuple[np.ndarray, int]:
        block = int(cfg.max_block_seconds * sr) if cfg.max_block_seconds else 0
        if block <= 0 or len(audio) <= block:
            return dsp.spectral_denoise(
                audio, sr,
                strength=cfg.denoise_strength,
                percentile=cfg.noise_percentile,
                bias=cfg.noise_bias,
                floor_low_db=cfg.floor_low_db,
                floor_high_db=cfg.floor_high_db,
                protect=cfg.transient_protection,
            ), 1

        fade = min(int(0.5 * sr), block // 4)
        out = np.zeros_like(audio)
        start, count, first = 0, 0, True
        while start < len(audio):
            end = min(len(audio), start + block)
            clean = dsp.spectral_denoise(
                audio[start:end], sr,
                strength=cfg.denoise_strength,
                percentile=cfg.noise_percentile,
                bias=cfg.noise_bias,
                floor_low_db=cfg.floor_low_db,
                floor_high_db=cfg.floor_high_db,
                protect=cfg.transient_protection,
            )
            if first:
                out[start:end] = clean
                first = False
            else:
                overlap = min(fade, end - start)
                if overlap > 0:
                    w = np.linspace(0.0, 1.0, overlap)[:, None]
                    out[start:start + overlap] = (out[start:start + overlap] * (1.0 - w)
                                                  + clean[:overlap] * w)
                    out[start + overlap:end] = clean[overlap:]
                else:
                    out[start:end] = clean
            count += 1
            if end >= len(audio):
                break
            start = end - fade
        return out, count


# The notebook in the first release used this name for the same class.
AggressiveAudioRestorer = AudioRestorer


def write_report(report: dict, path: str | os.PathLike) -> None:
    """Write a JSON report, converting non-finite numbers to ``null``."""
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean(report), handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Command-line entry point: ``python -m src.restoration input.wav``."""
    import argparse

    parser = argparse.ArgumentParser(description="Restore a noisy audio file")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--preset", default="music", choices=sorted(PRESETS))
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32])
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-declip", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    restorer = AudioRestorer(
        sample_rate=args.sample_rate,
        preset=args.preset,
        verbose=True,
        denoise=not args.no_denoise,
        declip=not args.no_declip,
        bit_depth=args.bit_depth,
    )
    report = restorer.restore_file(args.input, args.output)
    if args.report:
        write_report(report, args.report)
    print(json.dumps(report.get("loudness", {}), indent=2))


if __name__ == "__main__":
    main()
