#!/usr/bin/env python3
"""Create a reproducible before/after demo and the figures used in the README.

The script needs no network and no audio input: it synthesizes a short track,
degrades it (band limitation + hiss + 8-bit quantization + clipping), restores
it and reports SI-SDR/LSD plus spectral descriptors.

    python scripts/make_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import librosa.display  # noqa: E402
from src.demo import corrupt, synth_music  # noqa: E402
from src.dsp import as_mono  # noqa: E402
from src.metrics import AudioQualityMetrics  # noqa: E402
from src.restoration import AudioRestorer, write_report  # noqa: E402

SR = 32000
COLORS = {"clean": "#2ecc71", "dirty": "#e74c3c", "restored": "#3498db"}


def save_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_24")


def plot_waveforms(clean: np.ndarray, dirty: np.ndarray, restored: np.ndarray,
                   sr: int, path: Path) -> None:
    time = np.arange(len(clean)) / sr
    step = max(1, len(clean) // 4000)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, data, title, color in zip(
            axes,
            (clean, dirty, restored),
            ("Clean reference", "Degraded input (hiss, 8-bit, clipping)", "Restored"),
            (COLORS["clean"], COLORS["dirty"], COLORS["restored"])):
        data = as_mono(data)
        ax.plot(time[::step], data[::step], color=color, linewidth=0.6)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveform comparison", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(clean: np.ndarray, dirty: np.ndarray, restored: np.ndarray,
                  sr: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, color in (
            (clean, "Clean", COLORS["clean"]),
            (dirty, "Degraded", COLORS["dirty"]),
            (restored, "Restored", COLORS["restored"])):
        data = as_mono(data)
        freqs, power = signal.welch(data, fs=sr, nperseg=4096)
        ax.semilogx(freqs[1:], 10 * np.log10(power[1:] + 1e-12),
                    label=label, color=color, linewidth=1.3)
    ax.set_xlim(40, sr / 2)
    ax.set_ylim(-125, -20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Average spectrum")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_spectrograms(clean: np.ndarray, dirty: np.ndarray, restored: np.ndarray,
                      sr: int, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    for ax, data, title in zip(
            axes,
            (clean, dirty, restored),
            ("Clean reference", "Degraded input", "Restored")):
        data = as_mono(data)
        spectrum = librosa.amplitude_to_db(
            np.abs(librosa.stft(data, n_fft=2048, hop_length=512)), ref=np.max)
        image = librosa.display.specshow(
            spectrum, sr=sr, hop_length=512, x_axis="time", y_axis="log",
            ax=ax, cmap="magma", vmin=-80, vmax=0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=axes, format="%+2.0f dB", fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Spectrogram comparison (log frequency)", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(path: Path, preset: str) -> None:
    stages = [
        ("Input", "noisy / band-limited / clipped"),
        ("De-clip", "short flat-topped runs only"),
        ("Spectral denoise", "decision-directed Wiener + transient protection"),
        ("Tonal EQ", "subtle shelves (optional)"),
        ("Hum notch", "detected mains harmonics (optional)"),
        ("Loudness / limiter", "optional LUFS target + -1 dBTP ceiling"),
        ("Output", "24-bit WAV, sample rate preserved"),
    ]
    height = len(stages)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, height + 1)
    ax.axis("off")
    for index, (title, subtitle) in enumerate(stages):
        y = height - index
        color = "#2ecc71" if index in (0, len(stages) - 1) else "#3498db"
        box = FancyBboxPatch((1.2, y - 0.38), 7.6, 0.76,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor=color, edgecolor="#1b2b34", linewidth=1.2,
                             alpha=0.88)
        ax.add_patch(box)
        ax.text(5.0, y + 0.12, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(5.0, y - 0.16, subtitle, ha="center", va="center",
                fontsize=9, color="white")
        if index < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - 0.42), (5.0, y - 0.92),
                                         arrowstyle="-|>", mutation_scale=16,
                                         linewidth=1.6, color="#34495e"))
    ax.set_title(f"Restoration pipeline (preset: {preset})", fontsize=14, pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)



def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the restoration demo")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--preset", default="music",
                        choices=["music", "gentle", "strong", "voice"])
    parser.add_argument("--demo-dir", default="results/demo")
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--metrics", default="results/metrics/demo_results.json")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    plots_dir = Path(args.plots_dir)
    clean = synth_music(SR, duration=args.duration)
    dirty = corrupt(clean)

    restorer = AudioRestorer(preset=args.preset, verbose=True)
    restored, report = restorer.restore_array(dirty, SR)
    metrics_calc = AudioQualityMetrics(sample_rate=SR)
    original_metrics, restored_metrics, improvements = metrics_calc.compare_arrays(
        clean, dirty, restored)

    save_wav(demo_dir / "clean.wav", clean)
    save_wav(demo_dir / "dirty.wav", dirty)
    save_wav(demo_dir / "restored.wav", restored)

    plot_waveforms(clean, dirty, restored, SR, plots_dir / "waveforms_comparison.png")
    plot_spectrograms(clean, dirty, restored, SR, plots_dir / "spectrograms_comparison.png")
    plot_spectrum(clean, dirty, restored, SR, plots_dir / "frequency_response.png")
    plot_pipeline(plots_dir / "pipeline_diagram.png", args.preset)

    payload = {
        "preset": args.preset,
        "duration_s": args.duration,
        "metrics": {
            "degraded": original_metrics,
            "restored": restored_metrics,
            "improvements": improvements,
        },
        "report": report,
        "config": asdict(restorer.config),
    }
    write_report(payload, args.metrics)

    metrics_calc.print_comparison(original_metrics, restored_metrics, improvements)
    print(f"\nAudio files : {demo_dir}")
    print(f"Figures     : {plots_dir}")
    print(f"Metrics     : {args.metrics}")
    print(f"SI-SDR      : {original_metrics['SI-SDR (dB)']:.2f} -> "
          f"{restored_metrics['SI-SDR (dB)']:.2f} dB")
    print(f"LSD         : {original_metrics['LSD (dB)']:.2f} -> "
          f"{restored_metrics['LSD (dB)']:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

